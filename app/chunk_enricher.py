"""
LLM-based chunk enrichment for insurance documents.

Each chunk gets a concise context sentence prepended so that FAISS embeddings
capture document-level semantics, not just local text similarity.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.llm_tasks import enrich_chunk_batch

if TYPE_CHECKING:
    from app.llm_providers.base import BaseLLMProvider

_BATCH_SIZE = 10       # chunks per lightweight-model call
_DOC_PREVIEW_LEN = 2000  # chars of document used as context for the LLM


def enrich_chunks_with_context(
    chunks: list[str],
    full_text: str,
    provider: "BaseLLMProvider",
) -> list[str]:
    """
    Return a new list where each chunk has a one-sentence LLM context prepended.
    Format:  "[Context: <sentence>]\\n\\n<original chunk>"
    Chunks whose context is empty are returned as-is.
    """
    if not chunks:
        return chunks

    doc_preview = full_text[:_DOC_PREVIEW_LEN].strip()
    enriched: list[str] = []

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start: batch_start + _BATCH_SIZE]
        contexts = enrich_chunk_batch(provider, batch, doc_preview)

        # Defensive: ensure we have exactly one context per chunk
        while len(contexts) < len(batch):
            contexts.append("")
        contexts = contexts[: len(batch)]

        for chunk, ctx in zip(batch, contexts):
            enriched.append(f"[Context: {ctx}]\n\n{chunk}" if ctx else chunk)

    return enriched
