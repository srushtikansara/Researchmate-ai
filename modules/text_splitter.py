"""
modules/text_splitter.py
─────────────────────────
Splits extracted document text into overlapping chunks
using LangChain's RecursiveCharacterTextSplitter.
Each chunk carries full metadata for citation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
from modules.pdf_loader import DocumentContent, PageContent

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk with full provenance metadata."""
    chunk_id: str               # unique identifier
    text: str                   # chunk content
    source_file: str            # original filename
    page_number: int            # page where chunk starts
    chunk_index: int            # position within the document
    char_start: int = 0         # character offset in original page text
    total_chars: int = 0        # length of chunk
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.total_chars = len(self.text)

    def to_metadata_dict(self) -> dict:
        """Flat dict suitable for FAISS / LangChain metadata storage."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source_file,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "total_chars": self.total_chars,
            **self.metadata,
        }


# ─── Splitter Factory ─────────────────────────────────────────────────────────

def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )


# ─── Chunking Logic ───────────────────────────────────────────────────────────

def chunk_document(doc: DocumentContent) -> list[TextChunk]:
    """
    Split a DocumentContent into TextChunks, preserving page attribution.

    Strategy:
    - Process each page individually so that chunk ↔ page mapping is exact.
    - If a page produces no text, skip it.
    - Chunk IDs are deterministic: {filename}_{page}_{chunk_idx}

    Returns:
        Ordered list of TextChunk objects.
    """
    if not doc.is_valid:
        logger.warning("Skipping chunking for invalid doc '%s': %s", doc.filename, doc.error)
        return []

    splitter = build_splitter()
    all_chunks: list[TextChunk] = []
    global_chunk_idx = 0

    for page in doc.pages:
        if not page.text.strip():
            continue

        # Split this page's text
        raw_splits = splitter.create_documents(
            texts=[page.text],
            metadatas=[{"source": doc.filename, "page": page.page_number}],
        )

        for split in raw_splits:
            chunk_id = f"{doc.filename}_p{page.page_number}_c{global_chunk_idx}"
            char_start = split.metadata.get("start_index", 0)

            chunk = TextChunk(
                chunk_id=chunk_id,
                text=split.page_content,
                source_file=doc.filename,
                page_number=page.page_number,
                chunk_index=global_chunk_idx,
                char_start=char_start,
                metadata={
                    "has_images": page.has_images,
                    "has_tables": page.has_tables,
                    "doc_title": doc.metadata.get("title", doc.filename),
                    "doc_author": doc.metadata.get("author", "Unknown"),
                },
            )
            all_chunks.append(chunk)
            global_chunk_idx += 1

    logger.info(
        "Chunked '%s': %d pages → %d chunks",
        doc.filename,
        doc.total_pages,
        len(all_chunks),
    )
    return all_chunks


def chunk_documents(docs: list[DocumentContent]) -> list[TextChunk]:
    """
    Chunk multiple documents. Deduplicates by file hash.

    Returns:
        Combined list of chunks across all documents.
    """
    seen_hashes: set[str] = set()
    all_chunks: list[TextChunk] = []

    for doc in docs:
        if doc.file_hash in seen_hashes:
            logger.info("Skipping duplicate document '%s'", doc.filename)
            continue
        seen_hashes.add(doc.file_hash)
        all_chunks.extend(chunk_document(doc))

    logger.info("Total chunks across %d documents: %d", len(docs), len(all_chunks))
    return all_chunks


# ─── Utility ──────────────────────────────────────────────────────────────────

def get_chunk_stats(chunks: list[TextChunk]) -> dict:
    """Return summary statistics about a set of chunks."""
    if not chunks:
        return {"total": 0}
    lengths = [c.total_chars for c in chunks]
    sources = {c.source_file for c in chunks}
    return {
        "total": len(chunks),
        "sources": list(sources),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
