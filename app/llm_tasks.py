"""
Application-level LLM tasks built on top of BaseLLMProvider primitives.

If business logic changes tomorrow (different prompts, RAG strategy, output
shape), update here only — providers stay untouched.
"""
from typing import Any

from app.llm_providers.base import BaseLLMProvider, parse_numbered_contexts
from app.llm_providers.tool_schemas import ENRICH_SYSTEM, EXTRACTION_SYSTEM

_EXTRACTION_FALLBACK = {
    "proposer": {}, "policy": {}, "insured_members": [], "nominee": {},
    "error": "Failed to parse LLM response.",
}


def run_extraction(provider: BaseLLMProvider, text: str, filename: str) -> dict:
    """Extract structured policy data from document text via a single LLM call."""
    messages = [{
        "role": "user",
        "content": (
            f"Extract all insurance policy information from the document '{filename}'.\n\n"
            f"<document>\n{text}\n</document>"
        ),
    }]
    result = provider.chat_json(messages, system=EXTRACTION_SYSTEM, max_tokens=4096)
    return result or _EXTRACTION_FALLBACK


def run_chat(
    provider: BaseLLMProvider,
    vector_store: Any,
    messages_history: list[dict],
    user_message: str,
) -> tuple[str, str]:
    """
    Answer a user question using the provider's ReAct loop (tool-use if supported,
    single-shot RAG otherwise). Returns (answer, confidence).
    """
    return provider.run_react_loop(vector_store, messages_history, user_message)


def enrich_chunk_batch(
    provider: BaseLLMProvider,
    chunks: list[str],
    doc_preview: str,
) -> list[str]:
    """
    Generate one context sentence per chunk using the provider's lightweight model.
    Returns a list of context strings (empty string = no context for that chunk).
    """
    n = len(chunks)
    numbered_chunks = "\n\n".join(f"{i + 1}. {chunk[:400]}" for i, chunk in enumerate(chunks))
    prompt = (
        f"Document preview:\n{doc_preview}\n\n"
        f"Return EXACTLY {n} numbered lines (1 to {n}).\n"
        "Each line must follow this format:\nN. <context sentence>\n\n"
        "No blank lines. No extra text.\n\n"
        f"Chunks:\n{numbered_chunks}"
    )
    try:
        raw = provider.chat_lightweight([{"role": "user", "content": prompt}], system=ENRICH_SYSTEM)
        return parse_numbered_contexts(raw, n)
    except Exception:
        return [""] * n
