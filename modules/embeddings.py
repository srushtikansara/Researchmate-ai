"""
modules/embeddings.py
──────────────────────
Embedding model management using HuggingFace sentence-transformers.
Supports caching to avoid redundant computation.
"""

import logging
import time
from functools import lru_cache
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


# ─── Singleton Embedding Model ────────────────────────────────────────────────

_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    Downloads the model on first call, returns cached on subsequent calls.

    Args:
        model_name: HuggingFace model identifier.
        device: 'cpu' or 'cuda'.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    logger.info("Loading embedding model '%s' on %s …", model_name, device)
    start = time.time()

    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True,   # BGE models benefit from normalization
        "batch_size": EMBEDDING_BATCH_SIZE,
    }

    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    elapsed = time.time() - start
    logger.info("Embedding model loaded in %.2f s", elapsed)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (one per text).
    """
    model = get_embedding_model()
    start = time.time()
    vectors = model.embed_documents(texts)
    elapsed = time.time() - start
    logger.debug("Embedded %d texts in %.2f s", len(texts), elapsed)
    return vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    BGE models prefix the query with "Represent this sentence:".

    Args:
        query: The user's question.

    Returns:
        Embedding vector.
    """
    model = get_embedding_model()
    return model.embed_query(query)


def get_embedding_dimension() -> int:
    """Return the vector dimension of the current model."""
    model = get_embedding_model()
    # Embed a dummy text to get dimension
    sample = model.embed_documents(["test"])
    return len(sample[0])
