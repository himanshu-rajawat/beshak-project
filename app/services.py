"""
Business logic layer. Routes call these functions; they call lower-level modules.
Raises plain Python exceptions (never Flask responses):
  ValueError   → caller maps to 422
  LookupError  → caller maps to 404
  RuntimeError → caller maps to 409
"""
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from app.chunk_enricher import enrich_chunks_with_context
from app.chunker import chunk_text
from app.data_store import DataStore, DocumentRecord
from app.llm_providers import get_provider
from app.llm_tasks import run_chat, run_extraction
from app.regex_nlp_extractor import extract_policy_fields
from app.text_extractor import extract_text
from app.vector_store import build_vectorstore

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _ext(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text → run regex+NLP extraction → save record.
    Does NOT chunk or build a vector store (call index_document for that).
    """
    ext = _ext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: pdf, txt, docx")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File exceeds 20 MB limit")

    logger.info("Uploading '%s' (%.1f KB)", filename, len(file_bytes) / 1024)

    text, extraction_method = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError("Could not extract any text from the file")
    logger.info("Text extracted via '%s' — %d chars", extraction_method, len(text))

    extracted_data = extract_policy_fields(text)
    logger.info("Regex/NLP extraction complete for '%s'", filename)

    file_id = str(uuid.uuid4())
    record = DocumentRecord(
        file_id=file_id,
        filename=filename,
        text=text,
        chunks=[],
        vector_store=None,
        extracted_json=extracted_data,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        extraction_method=extraction_method,
    )
    DataStore().save_document(record)
    logger.info("Document saved — file_id=%s", file_id)

    return {
        "file_id": file_id,
        "extracted_data": extracted_data,
        "extraction_method": extraction_method,
        "data_extraction_method": "regex_nlp",
        "indexed": False,
        "message": (
            "File uploaded and processed successfully. "
            "Call POST /api/files/<file_id>/index before chatting."
        ),
    }


def llm_extract_document(file_id: str, client: str) -> dict:
    """
    Run LLM extraction on an already-uploaded document and overwrite extracted_json.
    Raises LookupError if the document is not found.
    """
    store = DataStore()
    doc = store.get_document(file_id)
    if doc is None:
        raise LookupError(f"Document '{file_id}' not found")

    logger.info("LLM extraction — file_id=%s client=%s", file_id, client)
    provider = get_provider(client)
    extracted_data = run_extraction(provider, doc.text, doc.filename)
    store.update_document_json(file_id, extracted_data)
    logger.info("LLM extraction complete — file_id=%s", file_id)

    return {
        "file_id": file_id,
        "extracted_data": extracted_data,
        "data_extraction_method": "llm",
        "llm_client": client,
    }


def index_document(file_id: str, client: str) -> dict:
    """
    Chunk → enrich → FAISS build → persist on record.
    Raises LookupError if file not found, RuntimeError if already indexed.
    """
    store = DataStore()
    doc = store.get_document(file_id)
    if doc is None:
        raise LookupError(f"Document '{file_id}' not found")
    if doc.vector_store is not None:
        raise RuntimeError(f"Document '{file_id}' is already indexed")

    logger.info("Indexing '%s' (file_id=%s) with client=%s", doc.filename, file_id, client)
    provider = get_provider(client)
    chunks = chunk_text(doc.text)
    logger.info("Chunked into %d chunks", len(chunks))
    enriched_chunks = enrich_chunks_with_context(chunks, doc.text, provider)
    logger.info("Chunk enrichment complete — building FAISS index")
    vector_store = build_vectorstore(enriched_chunks)
    store.update_document_index(file_id, chunks, vector_store)
    logger.info("FAISS index ready — file_id=%s", file_id)

    return {
        "file_id": file_id,
        "indexed": True,
        "chunk_count": len(chunks),
        "message": "Document indexed successfully. You can now chat with it.",
    }


def chat_with_document(
    file_id: str,
    user_message: str,
    conv_id: str | None,
    client: str,
) -> dict:
    """
    Get/create conversation → run chat agent → persist messages.
    Raises LookupError if file/conversation not found, RuntimeError if not indexed,
    ValueError if conversation belongs to a different document.
    """
    store = DataStore()
    doc = store.get_document(file_id)
    if doc is None:
        raise LookupError(f"Document '{file_id}' not found")
    if doc.vector_store is None:
        raise RuntimeError(
            f"Document '{file_id}' has not been indexed. "
            "Call POST /api/files/<file_id>/index first."
        )

    if conv_id:
        conv = store.get_conversation(conv_id)
        if conv is None:
            raise LookupError(f"Conversation '{conv_id}' not found")
        if conv.file_id != file_id:
            raise ValueError("Conversation belongs to a different document")
        logger.info("Resuming conversation conv_id=%s (turn %d)", conv_id, len(conv.messages) // 2 + 1)
    else:
        conv = store.create_conversation(file_id)
        logger.info("New conversation conv_id=%s for file_id=%s", conv.conv_id, file_id)

    logger.info("Chat — file_id=%s client=%s | user: %s", file_id, client, user_message[:80])
    provider = get_provider(client)
    answer, confidence = run_chat(provider, doc.vector_store, conv.messages, user_message)
    logger.info("Chat response [%s confidence] — %s", confidence, answer[:120])

    store.append_message(conv.conv_id, "user", user_message)
    store.append_message(conv.conv_id, "assistant", answer)

    return {
        "answer": answer,
        "conv_id": conv.conv_id,
        "confidence": confidence,
        "llm_client": client,
    }
