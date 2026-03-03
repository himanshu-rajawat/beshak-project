import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class DocumentRecord:
    file_id: str
    filename: str
    text: str
    chunks: list[str]
    vector_store: Optional[Any]  # FAISS instance — never serialized; None until indexed
    extracted_json: Optional[dict]
    uploaded_at: str
    extraction_method: str


@dataclass
class ConversationRecord:
    conv_id: str
    file_id: str
    messages: list[dict] = field(default_factory=list)  # [{role, content}] plain strings


class DataStore:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._documents: dict[str, DocumentRecord] = {}
                    instance._conversations: dict[str, ConversationRecord] = {}
                    instance._lock = threading.RLock()
                    cls._instance = instance
        return cls._instance

    # --- Documents ---

    def save_document(self, record: DocumentRecord) -> None:
        with self._lock:
            self._documents[record.file_id] = record

    def get_document(self, file_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            return self._documents.get(file_id)

    def update_document_json(self, file_id: str, extracted_json: dict) -> None:
        with self._lock:
            doc = self._documents.get(file_id)
            if doc:
                doc.extracted_json = extracted_json

    def update_document_index(
        self, file_id: str, chunks: list[str], vector_store: Any
    ) -> None:
        with self._lock:
            doc = self._documents.get(file_id)
            if doc:
                doc.chunks = chunks
                doc.vector_store = vector_store

    def list_documents(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "file_id": doc.file_id,
                    "filename": doc.filename,
                    "uploaded_at": doc.uploaded_at,
                    "extraction_method": doc.extraction_method,
                    "chunk_count": len(doc.chunks),
                    "indexed": doc.vector_store is not None,
                    "has_extracted_json": doc.extracted_json is not None,
                }
                for doc in self._documents.values()
            ]

    # --- Conversations ---

    def create_conversation(self, file_id: str) -> ConversationRecord:
        with self._lock:
            conv = ConversationRecord(
                conv_id=str(uuid.uuid4()),
                file_id=file_id,
            )
            self._conversations[conv.conv_id] = conv
            return conv

    def get_conversation(self, conv_id: str) -> Optional[ConversationRecord]:
        with self._lock:
            return self._conversations.get(conv_id)

    def append_message(self, conv_id: str, role: str, content: str) -> None:
        with self._lock:
            conv = self._conversations.get(conv_id)
            if conv:
                conv.messages.append({"role": role, "content": content})
