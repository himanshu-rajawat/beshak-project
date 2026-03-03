import json
import re
from abc import ABC, abstractmethod
from typing import Any

# Registry maps provider name → provider class
_REGISTRY: dict[str, type["BaseLLMProvider"]] = {}


def register(name: str):
    """Class decorator that registers a provider under the given name."""
    def decorator(cls: type["BaseLLMProvider"]):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_provider(name: str = "openai") -> "BaseLLMProvider":
    name = name.lower()
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown LLM provider '{name}'. Available: {available}")
    return _REGISTRY[name]()


def available_providers() -> list[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Shared parsing helpers (module-level so llm_tasks can import them directly)
# ---------------------------------------------------------------------------

def parse_json_response(text: str, fallback):
    """
    Parse a JSON object from raw LLM text.
    Tries the full text first, then extracts the first {...} block.
    Returns fallback if neither succeeds.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return fallback


def parse_numbered_contexts(text: str, expected: int) -> list[str]:
    """
    Parse "1. <context>\\n2. <context>\\n..." LLM output into a list.
    Falls back to every non-empty line if numbered parsing yields too few results.
    Pads with "" or truncates to match expected length.
    """
    contexts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if m:
            contexts.append(m.group(1).strip())

    if len(contexts) < max(1, expected // 2):
        contexts = [l.strip() for l in text.splitlines() if l.strip()]

    while len(contexts) < expected:
        contexts.append("")
    return contexts[:expected]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """
    Thin LLM primitive interface. Providers implement only these three methods.
    All application logic (extraction, chat-with-RAG, enrichment) lives in
    app/llm_tasks.py so it can evolve without touching providers.

    To add a new provider:
      1. Create app/llm_providers/<name>_provider.py
      2. Subclass BaseLLMProvider and apply @register("<name>")
      3. Import it in app/llm_providers/__init__.py
    """

    @abstractmethod
    def chat(self, messages: list[dict], system: str, max_tokens: int = 2048) -> str:
        """Single LLM call using the main model. Returns raw text."""
        ...

    @abstractmethod
    def chat_json(self, messages: list[dict], system: str, max_tokens: int = 2048) -> dict:
        """Single LLM call requesting structured JSON. Returns parsed dict."""
        ...

    @abstractmethod
    def chat_lightweight(self, messages: list[dict], system: str, max_tokens: int = 600) -> str:
        """Single call using the provider's cheapest/fastest model. Returns raw text."""
        ...

    def run_react_loop(
        self,
        vector_store: Any,
        messages_history: list[dict],
        user_message: str,
        max_iterations: int = 6,
    ) -> tuple[str, str]:
        """
        ReAct agent loop using the search_document tool.
        Returns (answer, confidence).

        Default implementation: single-shot RAG (for providers that don't support tool use).
        Override in providers that support tool-use (e.g. ClaudeProvider).
        """
        from app.vector_store import search_chunks
        from app.llm_providers.tool_schemas import CHAT_SYSTEM

        chunks = search_chunks(vector_store, user_message, k=4)
        context = "\n\n---\n\n".join(chunks) if chunks else "No relevant content found."
        augmented = (
            f"Document excerpts:\n<context>\n{context}\n</context>\n\n"
            f"Question: {user_message}"
        )
        messages = list(messages_history) + [{"role": "user", "content": augmented}]
        result = self.chat_json(messages, system=CHAT_SYSTEM)
        if result:
            return result.get("answer", "I could not find an answer."), result.get("confidence", "low")
        return "I was unable to find a clear answer in the document.", "low"
