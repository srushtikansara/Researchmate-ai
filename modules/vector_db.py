"""
modules/vector_db.py
─────────────────────
FAISS-based vector store management.
Handles indexing, persistence, and deduplication.
"""

import os
import json
import logging
import time
from typing import Optional

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import FAISS_INDEX_PATH
from modules.embeddings import get_embedding_model
from modules.text_splitter import TextChunk

logger = logging.getLogger(__name__)


# ─── FAISS Store Manager ──────────────────────────────────────────────────────

class VectorStore:
    """
    Manages a FAISS vector store with add/search/persist operations.
    Tracks indexed document hashes to avoid re-indexing.
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.meta_path = index_path + "_meta.json"
        self._store: Optional[FAISS] = None
        self._indexed_hashes: set[str] = set()
        self._indexed_files: set[str] = set()
        self._load_meta()

    # ── Metadata Persistence ──────────────────────────────────────────────────

    def _load_meta(self):
        """Load indexed hash tracking metadata."""
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                self._indexed_hashes = set(meta.get("hashes", []))
                self._indexed_files = set(meta.get("files", []))
                logger.info(
                    "Loaded index metadata: %d hashes, %d files",
                    len(self._indexed_hashes),
                    len(self._indexed_files),
                )
            except Exception as e:
                logger.warning("Could not load index metadata: %s", e)

    def _save_meta(self):
        """Persist indexed hash tracking metadata."""
        try:
            with open(self.meta_path, "w") as f:
                json.dump(
                    {
                        "hashes": list(self._indexed_hashes),
                        "files": list(self._indexed_files),
                    },
                    f,
                )
        except Exception as e:
            logger.warning("Could not save index metadata: %s", e)

    # ── Store Lifecycle ───────────────────────────────────────────────────────

    def load_existing(self) -> bool:
        """
        Load a persisted FAISS index from disk.
        Returns True if successfully loaded.
        """
        if not os.path.exists(self.index_path):
            return False
        try:
            embeddings = get_embedding_model()
            self._store = FAISS.load_local(
                self.index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded existing FAISS index from '%s'", self.index_path)
            return True
        except Exception as e:
            logger.warning("Could not load existing index: %s", e)
            return False

    def save(self):
        """Persist the current FAISS index to disk."""
        if self._store is None:
            return
        try:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            self._store.save_local(self.index_path)
            self._save_meta()
            logger.info("FAISS index saved to '%s'", self.index_path)
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)

    def is_loaded(self) -> bool:
        return self._store is not None

    def is_file_indexed(self, filename: str) -> bool:
        return filename in self._indexed_files

    def is_hash_indexed(self, file_hash: str) -> bool:
        return file_hash in self._indexed_hashes

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[TextChunk],
        file_hash: str,
        filename: str,
        force: bool = False,
    ) -> int:
        """
        Add text chunks to the vector store.

        Args:
            chunks: List of TextChunk to index.
            file_hash: SHA-256 of the source PDF (deduplication key).
            filename: Original filename.
            force: Re-index even if already indexed.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            logger.warning("No chunks to add for '%s'", filename)
            return 0

        if not force and self.is_hash_indexed(file_hash):
            logger.info("'%s' already indexed, skipping.", filename)
            return 0

        # Convert TextChunks → LangChain Documents
        documents = [
            Document(
                page_content=chunk.text,
                metadata=chunk.to_metadata_dict(),
            )
            for chunk in chunks
        ]

        embeddings = get_embedding_model()
        start = time.time()

        if self._store is None:
            self._store = FAISS.from_documents(documents, embeddings)
        else:
            self._store.add_documents(documents)

        elapsed = time.time() - start
        logger.info(
            "Indexed %d chunks from '%s' in %.2f s",
            len(chunks),
            filename,
            elapsed,
        )

        self._indexed_hashes.add(file_hash)
        self._indexed_files.add(filename)
        self.save()
        return len(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_files: Optional[list[str]] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-K similar chunks for a query.

        Args:
            query: Natural language question.
            k: Number of results to return.
            filter_files: If set, restrict search to these filenames.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if self._store is None:
            logger.warning("Vector store is empty. Upload and process PDFs first.")
            return []

        start = time.time()

        try:
            results = self._store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error("Similarity search failed: %s", e)
            return []

        elapsed = time.time() - start
        logger.debug("Retrieval took %.3f s for query: '%s'", elapsed, query[:60])

        # Optional file filter
        if filter_files:
            results = [
                (doc, score)
                for doc, score in results
                if doc.metadata.get("source") in filter_files
            ]

        return results

    def get_retrieval_stats(self, results: list[tuple[Document, float]]) -> dict:
        """Return metadata about a retrieval result."""
        if not results:
            return {"count": 0}
        sources = list({doc.metadata.get("source", "unknown") for doc, _ in results})
        pages = sorted({doc.metadata.get("page", 0) for doc, _ in results})
        scores = [round(float(score), 4) for _, score in results]
        return {
            "count": len(results),
            "sources": sources,
            "pages": pages,
            "scores": scores,
            "avg_score": round(sum(scores) / len(scores), 4),
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear the vector store and all metadata."""
        self._store = None
        self._indexed_hashes.clear()
        self._indexed_files.clear()

        for path in [self.index_path, self.meta_path]:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    logger.warning("Could not remove '%s': %s", path, e)

        logger.info("Vector store reset.")


# ─── Module-Level Singleton ───────────────────────────────────────────────────

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore, loading from disk if available."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.load_existing()
    return _vector_store
