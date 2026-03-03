# BeshakPdfExtractor

A Flask REST API that extracts structured data from insurance policy documents (PDF) using regex+spacy/LLM and answers free-form questions about them via RAG + LLM.

---

## Prerequisites

| Dependency | Purpose | Install |
|------------|---------|---------|
| **Python 3.11+** | Runtime | [python.org](https://python.org) |
| **Poppler** | PDF → image conversion (OCR path only) | [Windows builds](https://github.com/oschwartz10612/poppler-windows/releases) — extract zip, set `POPPLER_PATH` |
| **Tesseract** | OCR engine (scanned PDFs only) | [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) — set `TESSERACT_CMD` |

> Poppler and Tesseract are only needed for scanned PDFs. Digital PDFs (selectable text) are handled by pdfplumber without them.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The HuggingFace embedding model (`all-MiniLM-L6-v2`, ~22 MB) downloads automatically on the first `/index` call.

### 2. Configure environment

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows
```

Edit `.env`:

```env
# Required for whichever provider(s) you use
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Windows only — skip if already on PATH
POPPLER_PATH=C:\Users\you\poppler-windows\Library\bin
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 3. Run

```bash
python run.py
```

Server starts at `http://localhost:5000`.

---

## Workflow

```
POST /api/upload              → extract text + regex/NLP extraction → returns file_id
POST /api/files/<id>/index    → chunk + enrich + build FAISS index  → required before chat
POST /api/chat                → RAG + LLM agent → answer + confidence

POST /api/files/<id>/extract  → (optional) overwrite extracted_data with LLM extraction
```

---

## API Reference

### `GET /api/providers`

List registered LLM providers.

```json
{ "providers": ["claude", "openai"] }
```

---

### `POST /api/upload`

Upload a document. Extracts text and runs regex/NLP field extraction. Does **not** build a vector index.

- **Content-Type:** `multipart/form-data`
- **Fields:** `file` — `.pdf`, `.txt`, or `.docx`, max 20 MB *(required)*

**Response `201`:**
```json
{
  "file_id": "uuid",
  "extraction_method": "pdfplumber | ocr | docx | txt",
  "data_extraction_method": "regex_nlp",
  "indexed": false,
  "message": "File uploaded and processed successfully. Call POST /api/files/<file_id>/index before chatting.",
  "extracted_data": {
    "proposer": { "name": "...", "email": "...", "phone": "...", "address": "..." },
    "policy": {
      "number": "...", "type": "...", "issue_date": "...", "renewal_date": "...",
      "period": "...", "premiums": "...", "sum_insured": "...", "bonus": "...",
      "limit_of_coverage": "...", "recharge_benefit": "...", "payment_frequency": "..."
    },
    "insured_members": [
      { "name": "...", "gender": "...", "dob": "...", "age": "...", "relationship": "...", "pre_existing_disease": "..." }
    ],
    "nominee": { "name": "...", "relationship": "...", "percentage": "..." }
  }
}
```

---

### `POST /api/files/<file_id>/index`

Chunk the document, enrich chunks with context (LLM), and build the FAISS vector store. **Required before `/api/chat`.**

- **Content-Type:** `application/json`

**Request:**
```json
{ "client": "claude" }
```
`client` is optional (default: `claude`).

**Response `200`:**
```json
{
  "file_id": "uuid",
  "indexed": true,
  "chunk_count": 42,
  "message": "Document indexed successfully. You can now chat with it."
}
```

Returns `409` if the document is already indexed.

---

### `POST /api/files/<file_id>/extract`

Re-run extraction using an LLM and overwrite `extracted_data`. Optional — use when regex/NLP output is insufficient.

- **Content-Type:** `application/json`

**Request:**
```json
{ "client": "openai" }
```
`client` is optional (default: `openai`).

**Response `200`:**
```json
{
  "file_id": "uuid",
  "data_extraction_method": "llm",
  "llm_client": "openai",
  "extracted_data": { ... }
}
```

---

### `POST /api/chat`

Ask a question about an uploaded and indexed document.

- **Content-Type:** `application/json`

**Request:**
```json
{
  "file_id": "uuid",
  "message": "What is the sum insured?",
  "client": "claude",
  "conv_id": "uuid (optional — omit to start a new conversation)"
}
```
`client` is optional (default: `claude`).

**Response `200`:**
```json
{
  "answer": "The sum insured is ₹5,00,000.",
  "conv_id": "uuid",
  "confidence": "high | medium | low",
  "llm_client": "claude"
}
```

Pass the returned `conv_id` in follow-up requests to maintain conversation context. Returns `409` if the document has not been indexed yet.

---

### `GET /api/files`

List all uploaded documents.

```json
[
  {
    "file_id": "uuid",
    "filename": "policy.pdf",
    "uploaded_at": "2024-01-01T10:00:00+00:00",
    "extraction_method": "pdfplumber",
    "chunk_count": 42,
    "has_extracted_json": true
  }
]
```

---

### `GET /api/files/<file_id>`

Get full details for a document.

```json
{
  "file_id": "uuid",
  "filename": "policy.pdf",
  "uploaded_at": "2024-01-01T10:00:00+00:00",
  "extraction_method": "pdfplumber",
  "chunk_count": 42,
  "indexed": true,
  "extracted_data": { ... }
}
```

---

## Adding a new LLM provider

1. Create `app/llm_providers/<name>_provider.py`
2. Subclass `BaseLLMProvider`, apply `@register("<name>")`, implement `chat`, `chat_json`, and `chat_lightweight`
3. Add one import line to `app/llm_providers/__init__.py`

`client=<name>` will work immediately after that.

---

## Caveats

- **In-memory store** — all data is lost on restart. Replace `DataStore` with a Redis-backed store for production.
- **Single process** — run with `python run.py`. Multi-process WSGI workers will not share the in-memory store.
- **API costs** — `/index` makes one LLM call per chunk for context enrichment. `/chat` makes 1–8 LLM calls depending on how many document searches the ReAct agent needs.
