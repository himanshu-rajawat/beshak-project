import logging
import os
from typing import Any

import anthropic

from app.llm_providers.base import BaseLLMProvider, parse_json_response, register

_CLAUDE_SONNET = "claude-sonnet-4-6"
_CLAUDE_HAIKU = "claude-haiku-4-5-20251001"

_logger = logging.getLogger(__name__)


@register("claude")
class ClaudeProvider(BaseLLMProvider):

    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY is not set in environment")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def chat(self, messages: list[dict], system: str, max_tokens: int = 2048) -> str:
        response = self._get_client().messages.create(
            model=_CLAUDE_SONNET,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def chat_json(self, messages: list[dict], system: str, max_tokens: int = 2048) -> dict:
        text = self.chat(messages, system, max_tokens)
        return parse_json_response(text, fallback={})

    def chat_lightweight(self, messages: list[dict], system: str, max_tokens: int = 600) -> str:
        response = self._get_client().messages.create(
            model=_CLAUDE_HAIKU,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def run_react_loop(
        self,
        vector_store: Any,
        messages_history: list[dict],
        user_message: str,
        max_iterations: int = 6,
    ) -> tuple[str, str]:
        """
        ReAct agent loop using Claude tool use.

        Claude iteratively calls search_document until it has enough context,
        then returns a final JSON answer. Each search and its chunks are logged
        to the terminal in real time.
        """
        from app.vector_store import search_chunks
        from app.llm_providers.tool_schemas import CHAT_REACT_SYSTEM, SEARCH_DOCUMENT_TOOL

        client = self._get_client()
        # Plain-text history + current user turn
        messages: list = list(messages_history) + [{"role": "user", "content": user_message}]

        for iteration in range(1, max_iterations + 1):
            _logger.info("[ReAct] Iteration %d — calling Claude with tools", iteration)

            response = client.messages.create(
                model=_CLAUDE_SONNET,
                max_tokens=2048,
                system=CHAT_REACT_SYSTEM,
                messages=messages,
                tools=[SEARCH_DOCUMENT_TOOL],
            )

            _logger.info("[ReAct] stop_reason=%s", response.stop_reason)

            # ── Final answer ──────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                text_parts = [
                    block.text for block in response.content if block.type == "text"
                ]
                raw_text = "".join(text_parts)
                result = parse_json_response(raw_text, fallback={})
                if isinstance(result, dict) and result:
                    return (
                        result.get("answer", "I could not find an answer."),
                        result.get("confidence", "low"),
                    )
                return raw_text or "I was unable to find a clear answer.", "low"

            # ── Tool use ─────────────────────────────────────────────────
            if response.stop_reason == "tool_use":
                # Append assistant message with full content (TextBlock + ToolUseBlock)
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    query = block.input.get("query", "")
                    k = max(1, min(6, int(block.input.get("k", 3))))

                    print(
                        f"\n[ReAct iter={iteration}] Searching: '{query}'  (k={k})",
                        flush=True,
                    )
                    _logger.info("[ReAct] Tool call — query=%r k=%d", query, k)

                    chunks = search_chunks(vector_store, query, k=k)
                    for i, chunk in enumerate(chunks, 1):
                        preview = chunk[:200].replace("\n", " ")
                        print(f"  └─ Chunk {i}: {preview}", flush=True)
                        _logger.info("[ReAct] Chunk %d: %s", i, preview[:120])

                    result_text = (
                        "\n\n---\n\n".join(chunks)
                        if chunks
                        else "No relevant content found for this query."
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                # Feed results back to Claude
                messages.append({"role": "user", "content": tool_results})

            else:
                _logger.warning(
                    "[ReAct] Unexpected stop_reason=%s — stopping", response.stop_reason
                )
                break

        _logger.warning("[ReAct] Max iterations (%d) reached without final answer", max_iterations)
        return "I was unable to find a definitive answer after multiple searches.", "low"
