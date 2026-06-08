"""
modules/pdf_loader.py
─────────────────────
Handles PDF upload validation and text extraction using PyMuPDF (fitz).
Preserves section structure and page metadata.
"""

import os
import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from config import UPLOAD_DIR, MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int          # 1-indexed
    text: str
    char_count: int = 0
    has_images: bool = False
    has_tables: bool = False

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class DocumentContent:
    """Represents a fully parsed PDF document."""
    filename: str
    filepath: str
    file_hash: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.total_pages > 0


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_pdf(filepath: str, filename: str) -> tuple[bool, str]:
    """
    Validate uploaded PDF file.
    Returns (is_valid, error_message).
    """
    # Check extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '.{ext}'. Only PDF files are accepted."

    # Check file exists
    if not os.path.exists(filepath):
        return False, "File not found on disk."

    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB."

    # Check it's a real PDF (magic bytes)
    try:
        with open(filepath, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            return False, "File does not appear to be a valid PDF (invalid header)."
    except OSError as e:
        return False, f"Cannot read file: {e}"

    # Try opening with PyMuPDF
    try:
        doc = fitz.open(filepath)
        if doc.page_count == 0:
            doc.close()
            return False, "PDF has no pages."
        doc.close()
    except fitz.FileDataError:
        return False, "PDF is corrupted or password-protected."
    except Exception as e:
        return False, f"Failed to open PDF: {e}"

    return True, ""


# ─── File Hashing ─────────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """SHA-256 hash of file contents for deduplication."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove excessive whitespace / hyphenation artifacts
    - Normalize line breaks
    - Remove header/footer noise
    """
    # Fix hyphenated line breaks (word wrap in PDFs)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize various whitespace
    text = re.sub(r" +", " ", text)           # multiple spaces → one
    text = re.sub(r"\t", " ", text)            # tabs → space
    text = re.sub(r"\n{3,}", "\n\n", text)     # 3+ newlines → 2

    # Remove lines that look like page numbers (isolated digits)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Remove journal header/footer lines
    text = re.sub(r"Scientific Reports.*?\n", "", text)
    text = re.sub(r"Vol\.?:?\(?\d+\)?.*?\n", "", text)
    text = re.sub(r"www\.nature\.com.*?\n", "", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ─── Section Detection (heuristic) ────────────────────────────────────────────

_SECTION_PATTERNS = [
    re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z ]+)$", re.MULTILINE),           # "1. Introduction"
    re.compile(r"^([A-Z][A-Z ]{3,})$", re.MULTILINE),                     # "ABSTRACT"
    re.compile(r"^(Abstract|Introduction|Related Work|Methodology|"
               r"Experiments?|Results?|Discussion|Conclusion|References?)$",
               re.MULTILINE | re.IGNORECASE),
]

def detect_sections(text: str) -> list[str]:
    """Return a list of detected section headings (for metadata)."""
    sections = []
    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            heading = match.group(0).strip()
            if heading not in sections:
                sections.append(heading)
    return sections


# ─── Core Extraction ──────────────────────────────────────────────────────────

def extract_pdf(filepath: str, filename: str) -> DocumentContent:
    """
    Extract text and metadata from a PDF file.

    Args:
        filepath: Absolute path to the PDF.
        filename: Original filename (for display).

    Returns:
        DocumentContent with all pages and metadata.
    """
    file_hash = compute_file_hash(filepath)
    doc_content = DocumentContent(
        filename=filename,
        filepath=filepath,
        file_hash=file_hash,
        total_pages=0,
    )

    try:
        doc = fitz.open(filepath)
        doc_content.total_pages = doc.page_count

        # Extract PDF-level metadata
        meta = doc.metadata or {}
        doc_content.metadata = {
            "title": meta.get("title", filename),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "pages": doc.page_count,
        }

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract text (preserving reading order)
            raw_text = page.get_text("text", sort=True)
# For first page, also try extracting blocks to catch title
            if page_num == 0:
                blocks = page.get_text("blocks", sort=True)
                block_texts = [b[4] for b in blocks if len(b[4].strip()) > 10]
                raw_text = "\n".join(block_texts) if block_texts else raw_text
            cleaned = clean_text(raw_text)

            # Detect images / tables (approximate)
            has_images = len(page.get_images()) > 0
            has_tables = bool(re.search(r"\|\s*\|", cleaned))  # simple heuristic

            page_content = PageContent(
                page_number=page_num + 1,   # 1-indexed
                text=cleaned,
                has_images=has_images,
                has_tables=has_tables,
            )
            doc_content.pages.append(page_content)

        doc.close()
        logger.info(
            "Extracted %d pages from '%s' (%d chars total)",
            doc_content.total_pages,
            filename,
            doc_content.total_chars,
        )

    except fitz.FileDataError as e:
        doc_content.error = f"Corrupted PDF: {e}"
        logger.error("Corrupted PDF '%s': %s", filename, e)
    except Exception as e:
        doc_content.error = f"Extraction failed: {e}"
        logger.error("Failed to extract '%s': %s", filename, e)

    return doc_content


# ─── Save Uploaded File ───────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> tuple[str, str]:
    """
    Save a Streamlit UploadedFile to UPLOAD_DIR.

    Returns:
        (filepath, filename)
    """
    filename = uploaded_file.name
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return filepath, filename


# ─── Public API ───────────────────────────────────────────────────────────────

def load_pdf(uploaded_file) -> DocumentContent:
    """
    Full pipeline: save → validate → extract.
    Intended to be called from the Streamlit app.

    Returns:
        DocumentContent (check .is_valid and .error)
    """
    filepath, filename = save_uploaded_file(uploaded_file)

    is_valid, error_msg = validate_pdf(filepath, filename)
    if not is_valid:
        return DocumentContent(
            filename=filename,
            filepath=filepath,
            file_hash="",
            total_pages=0,
            error=error_msg,
        )

    return extract_pdf(filepath, filename)
