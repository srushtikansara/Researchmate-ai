"""
modules/retriever.py
─────────────────────
Top-level retrieval interface.
Formats retrieved chunks into context strings for the LLM.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from config import TOP_K_RETRIEVAL
from modules.vector_db import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Encapsulates retrieval output for a single query."""
    query: str
    chunks: list[tuple[Document, float]]        # (doc, similarity_score)
    retrieval_time_ms: float
    stats: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def format_context(self, include_citations: bool = True) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        Args:
            include_citations: If True, prepend each chunk with source info.

        Returns:
            Formatted context string.
        """
        if self.is_empty:
            return ""

        parts = []
        for i, (doc, score) in enumerate(self.chunks, 1):
            meta = doc.metadata
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")

            if include_citations:
                header = f"[Source {i}: {source}, Page {page}]"
                parts.append(f"{header}\n{doc.page_content}")
            else:
                parts.append(doc.page_content)

        return "\n\n---\n\n".join(parts)

    def get_citations(self) -> list[dict]:
        """Return structured citation info for each chunk."""
        citations = []
        for i, (doc, score) in enumerate(self.chunks, 1):
            meta = doc.metadata
            citations.append({
                "index": i,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "?"),
                "score": round(float(score), 4),
                "preview": doc.page_content[:200] + ("…" if len(doc.page_content) > 200 else ""),
                "full_text": doc.page_content,
            })
        return citations


# ─── Retriever Class ──────────────────────────────────────────────────────────

class PaperRetriever:
    """
    High-level retriever that wraps VectorStore.
    Supports filtering by document and configurable K.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        filter_files: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-K relevant chunks for a query.

        Args:
            query: User's natural language question.
            k: Number of chunks to retrieve.
            filter_files: Restrict results to specific uploaded PDFs.

        Returns:
            RetrievalResult with chunks and metadata.
        """
        start = time.time()

        chunks = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_files=filter_files,
        )
        elapsed_ms = (time.time() - start) * 1000
        stats = self.vector_store.get_retrieval_stats(chunks)

        logger.info(
            "Retrieved %d chunks in %.1f ms for query: '%s'",
            len(chunks),
            elapsed_ms,
            query[:60],
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            retrieval_time_ms=round(elapsed_ms, 1),
            stats=stats,
        )

    def retrieve_for_comparison(
        self,
        query: str,
        paper_a: str,
        paper_b: str,
        k_per_paper: int = 3,
    ) -> tuple[RetrievalResult, RetrievalResult]:
        """
        Retrieve chunks separately from two papers for comparison.

        Returns:
            (result_a, result_b)
        """
        result_a = self.retrieve(query, k=k_per_paper, filter_files=[paper_a])
        result_b = self.retrieve(query, k=k_per_paper, filter_files=[paper_b])
        return result_a, result_b


# ─── Module-Level Singleton ───────────────────────────────────────────────────

_retriever: Optional[PaperRetriever] = None


def get_retriever() -> PaperRetriever:
    """Return the singleton PaperRetriever."""
    global _retriever
    if _retriever is None:
        _retriever = PaperRetriever()
    return _retriever
