import threading
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def build_vectorstore(chunks: list[str]) -> Any:
    """Build and return a FAISS vector store from text chunks."""
    embeddings = _get_embeddings()
    return FAISS.from_texts(chunks, embeddings)


def search_chunks(vector_store: Any, query: str, k: int = 4) -> list[str]:
    """Return top-k most relevant chunks for the query."""
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def search_chunks_with_scores(
    vector_store: Any, query: str, k: int = 4
) -> tuple[list[str], list[float]]:
    """Return top-k chunks and their L2 distances (lower = more similar)."""
    results = vector_store.similarity_search_with_score(query, k=k)
    chunks = [doc.page_content for doc, _ in results]
    scores = [float(score) for _, score in results]
    return chunks, scores
