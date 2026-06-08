"""
modules/hybrid_retriever.py
────────────────────────────
Hybrid retrieval: FAISS (semantic) + BM25 (keyword) + CrossEncoder reranking.
Significantly improves retrieval quality over pure vector search.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from config import TOP_K_RETRIEVAL
from modules.vector_db import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


# ─── BM25 Retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Keyword-based BM25 retrieval over indexed documents.
    Falls back gracefully if rank_bm25 is not installed.
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._documents: list[Document] = []
        self._bm25 = None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import rank_bm25
            return True
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 retrieval disabled. Run: pip install rank-bm25")
            return False

    def index_documents(self, documents: list[Document]):
        """Index documents for BM25 search."""
        if not self._available:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._documents = documents
            self._corpus = [doc.page_content.lower().split() for doc in documents]
            self._bm25 = BM25Okapi(self._corpus)
            logger.info("BM25 indexed %d documents", len(documents))
        except Exception as e:
            logger.error("BM25 indexing failed: %s", e)
            self._available = False

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        """Retrieve top-K documents using BM25."""
        if not self._available or self._bm25 is None:
            return []
        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k]
            return [self._documents[i] for i in top_indices if scores[i] > 0]
        except Exception as e:
            logger.error("BM25 retrieval failed: %s", e)
            return []

    @property
    def is_ready(self) -> bool:
        return self._available and self._bm25 is not None


# ─── CrossEncoder Reranker ────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    Uses: cross-encoder/ms-marco-MiniLM-L-6-v2
    Falls back gracefully if sentence_transformers not available.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder reranker '%s' …", self.MODEL_NAME)
            self._model = CrossEncoder(self.MODEL_NAME)
            self._available = True
            logger.info("CrossEncoder reranker loaded successfully")
        except Exception as e:
            logger.warning("CrossEncoder not available: %s. Using score-based ranking.", e)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents using CrossEncoder.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            List of (document, score) tuples sorted by relevance.
        """
        if not self._available or not documents:
            return [(doc, 1.0) for doc in documents[:top_k]]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._model.predict(pairs)
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(doc, float(score)) for doc, score in ranked[:top_k]]
        except Exception as e:
            logger.error("CrossEncoder reranking failed: %s", e)
            return [(doc, 1.0) for doc in documents[:top_k]]

    @property
    def is_ready(self) -> bool:
        return self._available


# ─── Hybrid Retrieval Result ──────────────────────────────────────────────────

@dataclass
class HybridRetrievalResult:
    """Result from hybrid retrieval pipeline."""
    query: str
    chunks: list[tuple[Document, float]]
    retrieval_time_ms: float
    used_bm25: bool = False
    used_reranking: bool = False
    stats: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def format_context(self, include_citations: bool = True) -> str:
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


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS semantic search + BM25 keyword search,
    then reranks using CrossEncoder for best results.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        use_reranking: bool = True,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker() if use_reranking else None
        self._bm25_indexed = False

    def _ensure_bm25_indexed(self):
        """Index all documents in vector store for BM25."""
        if self._bm25_indexed or not self.bm25.is_ready:
            return
        try:
            if self.vector_store._store:
                docs = list(self.vector_store._store.docstore._dict.values())
                if docs:
                    self.bm25.index_documents(docs)
                    self._bm25_indexed = True
        except Exception as e:
            logger.warning("Could not index for BM25: %s", e)

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        filter_files: Optional[list[str]] = None,
    ) -> HybridRetrievalResult:
        """
        Hybrid retrieval: FAISS + BM25 + CrossEncoder reranking.

        Args:
            query: User's question.
            k: Final number of chunks to return.
            filter_files: Restrict to specific PDFs.

        Returns:
            HybridRetrievalResult with best chunks.
        """
        start = time.time()
        used_bm25 = False
        used_reranking = False

        # 1. FAISS semantic search (get more candidates for reranking)
        faiss_k = k * 3
        faiss_results = self.vector_store.similarity_search(
            query=query,
            k=faiss_k,
            filter_files=filter_files,
        )
        faiss_docs = [doc for doc, _ in faiss_results]

        # 2. BM25 keyword search
        self._ensure_bm25_indexed()
        bm25_docs = []
        if self.bm25.is_ready:
            bm25_raw = self.bm25.retrieve(query, k=k * 2)
            # Apply file filter
            if filter_files:
                bm25_docs = [
                    d for d in bm25_raw
                    if d.metadata.get("source") in filter_files
                ]
            else:
                bm25_docs = bm25_raw
            used_bm25 = bool(bm25_docs)

        # 3. Merge results (deduplicate by chunk_id)
        seen_ids = set()
        merged_docs = []
        for doc in faiss_docs + bm25_docs:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_docs.append(doc)

        if not merged_docs:
            elapsed_ms = (time.time() - start) * 1000
            return HybridRetrievalResult(
                query=query,
                chunks=[],
                retrieval_time_ms=round(elapsed_ms, 1),
            )

        # 4. CrossEncoder reranking
        if self.reranker and self.reranker.is_ready and len(merged_docs) > k:
            reranked = self.reranker.rerank(query, merged_docs, top_k=k)
            used_reranking = True
        else:
            # Fall back to FAISS scores
            score_map = {
                doc.metadata.get("chunk_id", doc.page_content[:50]): score
                for doc, score in faiss_results
            }
            reranked = []
            for doc in merged_docs[:k]:
                cid = doc.metadata.get("chunk_id", doc.page_content[:50])
                score = score_map.get(cid, 0.5)
                reranked.append((doc, float(score)))

        elapsed_ms = (time.time() - start) * 1000

        # Build stats
        sources = list({doc.metadata.get("source", "?") for doc, _ in reranked})
        pages = sorted({doc.metadata.get("page", 0) for doc, _ in reranked})
        scores = [round(s, 4) for _, s in reranked]

        stats = {
            "count": len(reranked),
            "sources": sources,
            "pages": pages,
            "scores": scores,
            "faiss_count": len(faiss_docs),
            "bm25_count": len(bm25_docs),
            "merged_count": len(merged_docs),
        }

        logger.info(
            "Hybrid retrieval: %d results (FAISS:%d BM25:%d Reranked:%s) in %.1f ms",
            len(reranked),
            len(faiss_docs),
            len(bm25_docs),
            used_reranking,
            elapsed_ms,
        )

        return HybridRetrievalResult(
            query=query,
            chunks=reranked,
            retrieval_time_ms=round(elapsed_ms, 1),
            used_bm25=used_bm25,
            used_reranking=used_reranking,
            stats=stats,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(use_reranking: bool = True) -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(use_reranking=use_reranking)
    return _hybrid_retriever
