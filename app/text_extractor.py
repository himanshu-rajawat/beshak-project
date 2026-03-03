import io
import os
import string

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from docx import Document


import re

def is_usable_text(text: str) -> bool:
    if not text or len(text) < 100:
        return False

    words = text.split()
    sample = words[:200]

    # 1. readability
    valid_words = sum(1 for w in sample if re.match(r"^[A-Za-z]{3,}$", w))
    readability = valid_words / len(sample) if sample else 0

    # 2. symbol noise
    symbols = len(re.findall(r"[^A-Za-z0-9\s]", text))
    symbol_ratio = symbols / len(text)

    # 3. keyword presence
    text_lower = text.lower()
    has_keywords = any(k in text_lower for k in ["policy", "name", "email", "insured"])

    # decision
    if readability < 0.6:
        return False
    if symbol_ratio > 0.1:
        return False

    return True

def _word_readability_score(text):
    words = text.split()
    if not words:
        return 0

    valid = sum(1 for w in words[:200] if re.match(r"^[A-Za-z]{3,}$", w))
    return valid / len(words[:200])

def _symbol_ratio(text):
    symbols = re.findall(r"[^A-Za-z0-9\s]", text)
    return len(symbols) / len(text) if text else 1


def _pdf_via_ocr(file_bytes: bytes) -> str:
    poppler_path = os.getenv("POPPLER_PATH") or None
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = convert_from_bytes(file_bytes, poppler_path=poppler_path)
    pages = [pytesseract.image_to_string(img) for img in images]
    return "\n\n".join(pages)


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text from PDF, DOCX, or TXT bytes.
    Returns (text, method) where method is one of:
      'pdfplumber', 'ocr', 'docx', 'txt'
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(pages).strip()
            if is_usable_text(text):
                return text, "pdfplumber"
        except Exception:
            pass
        # Fall back to OCR
        text = _pdf_via_ocr(file_bytes)
        return text, "ocr"

    if lower.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs), "docx"

    # .txt (or unknown — treat as plain text)
    return file_bytes.decode("utf-8", errors="replace"), "txt"
