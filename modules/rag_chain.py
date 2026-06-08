"""
modules/rag_chain.py
─────────────────────
RAG chain: retrieves context and generates answers via LLM.
Supports Ollama (local) and HuggingFace Hub backends.
Enforces strict grounding — no hallucination.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Generator

from config import (
    LLM_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    HF_MODEL_ID,
    HF_API_TOKEN,
    RAG_SYSTEM_PROMPT,
    TOP_K_RETRIEVAL,
)
from modules.retriever import PaperRetriever, RetrievalResult, get_retriever

logger = logging.getLogger(__name__)

# Sentinel returned when no relevant context is found
NO_INFO_RESPONSE = "I could not find this information in the uploaded paper."


# ─── LLM Backends ─────────────────────────────────────────────────────────────

def _build_ollama_llm():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.1,
            num_predict=4096,
            num_ctx=8192,
        )
    except Exception as e:
        logger.error("Failed to build Ollama LLM: %s", e)
        raise


def _build_huggingface_llm():
    """Build a HuggingFace Hub LLM using LangChain."""
    try:
        from langchain_community.llms import HuggingFaceHub
        if not HF_API_TOKEN:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN environment variable is not set. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        return HuggingFaceHub(
            repo_id=HF_MODEL_ID,
            huggingfacehub_api_token=HF_API_TOKEN,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 1024},
        )
    except Exception as e:
        logger.error("Failed to build HuggingFace LLM: %s", e)
        raise


def _build_llm():
    """Return the configured LLM based on LLM_BACKEND setting."""
    if LLM_BACKEND == "ollama":
        return _build_ollama_llm()
    elif LLM_BACKEND == "huggingface":
        return _build_huggingface_llm()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: '{LLM_BACKEND}'. Use 'ollama' or 'huggingface'.")


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Complete response from the RAG chain."""
    question: str
    answer: str
    retrieval: RetrievalResult
    response_time_ms: float
    total_time_ms: float
    citations: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_grounded(self) -> bool:
        """True if the answer was derived from retrieved context."""
        return self.answer != NO_INFO_RESPONSE and self.error is None

    @property
    def metrics(self) -> dict:
        return {
            "retrieval_time_ms": self.retrieval.retrieval_time_ms,
            "response_time_ms": self.response_time_ms,
            "total_time_ms": self.total_time_ms,
            "chunks_retrieved": len(self.retrieval.chunks),
            "sources": self.retrieval.stats.get("sources", []),
            "pages": self.retrieval.stats.get("pages", []),
        }


# ─── RAG Chain ────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Main RAG pipeline:
    1. Retrieve relevant chunks from FAISS
    2. Format prompt with context
    3. Generate answer via LLM
    4. Return grounded, cited response
    """

    def __init__(self, retriever: Optional[PaperRetriever] = None):
        self.retriever = retriever or get_retriever()
        self._llm = None   # lazy-loaded

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _build_llm()
        return self._llm

    def _build_prompt(self, context: str, question: str) -> str:
        """Construct the full RAG prompt."""
        return RAG_SYSTEM_PROMPT.format(context=context, question=question)

    def _generate(self, prompt: str) -> tuple[str, float]:
        """
        Call the LLM and return (response_text, elapsed_ms).
        """
        start = time.time()
        try:
            response = self.llm.invoke(prompt)
            elapsed_ms = (time.time() - start) * 1000
            return response.strip(), elapsed_ms
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("LLM generation failed: %s", e)
            raise

    def answer(
        self,
        question: str,
        k: int = TOP_K_RETRIEVAL,
        filter_files: Optional[list[str]] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline for a single question.

        Args:
            question: User's question.
            k: Chunks to retrieve.
            filter_files: Restrict to specific PDFs.
            chat_history: Previous Q&A pairs for context (unused in base prompt).

        Returns:
            RAGResponse with answer, citations, and metrics.
        """
        pipeline_start = time.time()

        # 1. Retrieve
        retrieval = self.retriever.retrieve(question, k=k, filter_files=filter_files)

        # 2. Guard: nothing retrieved
        if retrieval.is_empty:
            total_ms = (time.time() - pipeline_start) * 1000
            return RAGResponse(
                question=question,
                answer=NO_INFO_RESPONSE,
                retrieval=retrieval,
                response_time_ms=0,
                total_time_ms=round(total_ms, 1),
                citations=[],
            )

        # 3. Format context
        context = retrieval.format_context(include_citations=True)

        # 4. Optionally inject recent chat history
        if chat_history:
            history_text = self._format_history(chat_history[-4:])  # last 2 turns
            context = history_text + "\n\n" + context

        # 5. Generate answer
        prompt = self._build_prompt(context=context, question=question)
        try:
            answer_text, response_ms = self._generate(prompt)
        except Exception as e:
            total_ms = (time.time() - pipeline_start) * 1000
            return RAGResponse(
                question=question,
                answer="",
                retrieval=retrieval,
                response_time_ms=0,
                total_time_ms=round(total_ms, 1),
                error=str(e),
            )

        total_ms = (time.time() - pipeline_start) * 1000
        citations = retrieval.get_citations()

        return RAGResponse(
            question=question,
            answer=answer_text,
            retrieval=retrieval,
            response_time_ms=round(response_ms, 1),
            total_time_ms=round(total_ms, 1),
            citations=citations,
        )

    def _format_history(self, history: list[dict]) -> str:
        """Format recent chat history for injection into prompt."""
        lines = ["Previous conversation:"]
        for turn in history:
            lines.append(f"Q: {turn.get('question', '')}")
            lines.append(f"A: {turn.get('answer', '')}")
        return "\n".join(lines)


# ─── Module-Level Singleton ───────────────────────────────────────────────────

_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Return the singleton RAGChain."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
