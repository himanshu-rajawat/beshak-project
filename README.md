# BeshakPdfExtractor

A Flask REST API that extracts structured data from insurance policy documents (PDF, DOCX, TXT) using LLM agents + RAG (FAISS).

Supports **Claude (claude-sonnet-4-6)** and **OpenAI GPT-4o** out of the box. The provider is selected per request via a `client` field, making it easy to add new LLMs.

---

## Features

- **Smart text extraction** — pdfplumber for digital PDFs, Tesseract OCR fallback for scanned documents
- **Structured extraction** — agentic loop searches the document and returns a normalized JSON schema (proposer, policy details, insured members, nominee)
- **Conversational Q&A** — ReAct chatbot agent answers free-form questions with confidence scoring
- **Multi-turn conversations** — conversation history is preserved across requests via `conv_id`
- **Pluggable LLM providers** — choose Claude or OpenAI per request; adding a new provider takes ~50 lines

---

## Project Structure

```
BeshakPdfExtractor/
├── app/
│   ├── __init__.py                    # Flask app factory
│   ├── data_store.py                  # In-memory singleton (documents + conversations)
│   ├── text_extractor.py              # PDF/DOCX/TXT extraction with OCR fallback
│   ├── chunker.py                     # Text chunking (LangChain RecursiveCharacterTextSplitter)
│   ├── vector_store.py                # FAISS vector store + HuggingFace embeddings
│   ├── agents.py                      # Thin delegator → llm_providers
│   ├── routes.py                      # API endpoints
│   └── llm_providers/
│       ├── __init__.py                # Re-exports + triggers provider registration
│       ├── base.py                    # BaseLLMProvider + @register decorator + factory
│       ├── claude_provider.py         # Anthropic Claude implementation
│       └── openai_provider.py         # OpenAI GPT-4o implementation
├── run.py                             # Entry point
├── requirements.txt
├── .env.example
│
│   # Original standalone scripts (untouched, not imported by the Flask app)
├── pdf_loader.py
├── chunking_strategy.py
└── extract_json_re.py
```

---

## Prerequisites

| Dependency | Purpose | Install |
|------------|---------|---------|
| **Poppler** | PDF → image conversion (OCR path) | [Windows builds](https://github.com/oschwartz10612/poppler-windows/releases) — download the `Release-*.zip` (not Source code), extract, set `POPPLER_PATH` |
| **Tesseract** | OCR engine | [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) — run the `.exe` installer, set `TESSERACT_CMD` |
| **Python 3.11+** | Runtime | [python.org](https://python.org) |

> Poppler and Tesseract are only needed if your PDFs are scanned images. Digital PDFs (where text is selectable) are handled by pdfplumber without either tool.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> The HuggingFace embedding model (`all-MiniLM-L6-v2`, ~22 MB) downloads automatically on the first upload request.

### 2. Configure environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Edit `.env` — set at least one LLM key:

```env
# Required: set whichever provider(s) you want to use
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Windows only — skip if already on system PATH
POPPLER_PATH=C:\Users\you\Downloads\poppler-windows-25.12.0-0\Library\bin
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 3. Run the server

```bash
python run.py
```

Server starts at `http://localhost:5000`.

---

## LLM Providers

| `client` value | Model | Key required |
|----------------|-------|-------------|
| `claude` (default) | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o | `OPENAI_API_KEY` |

Pass `client` as a form field on upload or a JSON field on chat. Omit it to use Claude.

### Adding a new provider

1. Create `app/llm_providers/<name>_provider.py`
2. Subclass `BaseLLMProvider` and implement `run_extraction_agent` and `run_chat_agent`
3. Decorate the class with `@register("<name>")`
4. Add one import line to `app/llm_providers/__init__.py`

```python
# app/llm_providers/gemini_provider.py
from app.llm_providers.base import BaseLLMProvider, register

@register("gemini")
class GeminiProvider(BaseLLMProvider):
    def run_extraction_agent(self, vector_store, filename): ...
    def run_chat_agent(self, vector_store, messages_history, user_message): ...
```

```python
# app/llm_providers/__init__.py  — add one line:
from app.llm_providers import claude_provider, openai_provider, gemini_provider  # noqa
```

That's it — `client=gemini` will work immediately.

---

## API Reference

### `GET /api/providers`

List all registered LLM providers.

```json
{ "providers": ["claude", "openai"] }
```

---

### `POST /api/upload`

Upload an insurance document for extraction.

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file` — `.pdf`, `.txt`, or `.docx`, max 20 MB *(required)*
  - `client` — `claude` or `openai` *(optional, default: `claude`)*

**Response:**
```json
{
  "file_id": "uuid",
  "llm_client": "claude",
  "extraction_method": "pdfplumber | ocr | docx | txt",
  "message": "File uploaded and processed successfully.",
  "extracted_data": {
    "proposer": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "9876543210",
      "address": "123 Main St, Mumbai"
    },
    "policy": {
      "number": "P/123456/01/2024/001",
      "type": "Health Insurance",
      "issue_date": "01/01/2024",
      "renewal_date": "31/12/2024",
      "period": "1 year",
      "premiums": "₹12,000",
      "sum_insured": "₹5,00,000",
      "bonus": "N/A",
      "limit_of_coverage": "N/A",
      "recharge_benefit": "N/A",
      "payment_frequency": "Annual"
    },
    "insured_members": [
      {
        "name": "John Doe",
        "gender": "Male",
        "dob": "01/01/1985",
        "age": "39",
        "relationship": "Self",
        "pre_existing_disease": "None"
      }
    ],
    "nominee": {
      "name": "Jane Doe",
      "relationship": "Spouse",
      "percentage": "100%"
    }
  }
}
```

---

### `POST /api/chat`

Ask a question about an uploaded document.

- **Content-Type:** `application/json`

**Request:**
```json
{
  "file_id": "uuid",
  "message": "What is the sum insured?",
  "client": "openai",
  "conv_id": "uuid (optional — omit to start a new conversation)"
}
```

**Response:**
```json
{
  "answer": "The sum insured is ₹5,00,000.",
  "conv_id": "uuid",
  "confidence": "high | medium | low",
  "llm_client": "openai"
}
```

Pass the returned `conv_id` in follow-up requests to maintain conversation context. You can switch `client` between turns — the vector store is provider-agnostic.

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

Get full details for a specific document, including extracted data.

---

## Postman Quick-Start

```
1. GET  /api/providers
   → confirm which clients are available

2. POST /api/upload
   form-data: file=<policy.pdf>, client=claude
   → copy file_id

3. GET  /api/files/<file_id>
   → verify extracted data

4. POST /api/chat
   {"file_id": "...", "message": "What is the policy number?", "client": "claude"}
   → copy conv_id

5. POST /api/chat  (follow-up, switch to OpenAI)
   {"file_id": "...", "message": "Who are the insured members?", "conv_id": "...", "client": "openai"}
```

---

## Notes

- **In-memory store** — data is lost on restart. For production with multiple workers, replace `DataStore` with a Redis-backed store.
- **Single worker** — run with `python run.py`. Do not use multi-process WSGI workers without a shared store.
- **API costs** — each `/api/upload` triggers multiple LLM calls (the extraction agentic loop). Each `/api/chat` triggers 1–8 calls depending on how many document searches the agent needs.
