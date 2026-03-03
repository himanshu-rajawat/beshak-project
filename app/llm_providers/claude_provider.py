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
