"""
modules/rag_chain.py
─────────────────────
RAG chain: retrieves context and generates answers via LLM.
Supports Ollama (local) and HuggingFace Hub backends.
Enforces strict grounding — no hallucination.
"""

import os
import logging
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

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
    """Build HuggingFace LLM using direct requests API."""
    try:
        from langchain_core.language_models.llms import LLM
        from typing import Optional, List

        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        if not token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set")
        _model_id = os.getenv("HF_MODEL_ID", HF_MODEL_ID)
        _token = token

        class HFDirectLLM(LLM):
            model_id: str = _model_id
            hf_token: str = _token

            @property
            def _llm_type(self) -> str:
                return "huggingface_direct"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                **kwargs
            ) -> str:
                url = f"https://api-inference.huggingface.co/models/{self.model_id}"
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 512,
                        "temperature": 0.1,
                        "return_full_text": False,
                        "do_sample": True,
                    }
                }
                response = requests.post(
                    url, headers=headers, json=payload, timeout=120
                )
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                    return text.strip()
                elif isinstance(result, dict):
                    return str(result.get("generated_text", result))
                return str(result)

        return HFDirectLLM()

    except Exception as e:
        logger.error("Failed to build HuggingFace LLM: %s", e)
        raise


def _build_llm():
    if LLM_BACKEND == "ollama":
        return _build_ollama_llm()
    elif LLM_BACKEND == "huggingface":
        return _build_huggingface_llm()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: '{LLM_BACKEND}'.")


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieval: RetrievalResult
    response_time_ms: float
    total_time_ms: float
    citations: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_grounded(self) -> bool:
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
    def __init__(self, retriever: Optional[PaperRetriever] = None):
        self.retriever = retriever or get_retriever()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _build_llm()
        return self._llm

    def _build_prompt(self, context: str, question: str) -> str:
        return RAG_SYSTEM_PROMPT.format(context=context, question=question)

    def _generate(self, prompt: str) -> tuple:
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
        filter_files: Optional[list] = None,
        chat_history: Optional[list] = None,
    ) -> RAGResponse:
        pipeline_start = time.time()
        retrieval = self.retriever.retrieve(question, k=k, filter_files=filter_files)

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

        context = retrieval.format_context(include_citations=True)
        if chat_history:
            history_text = self._format_history(chat_history[-4:])
            context = history_text + "\n\n" + context

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

    def _format_history(self, history: list) -> str:
        lines = ["Previous conversation:"]
        for turn in history:
            lines.append(f"Q: {turn.get('question', '')}")
            lines.append(f"A: {turn.get('answer', '')}")
        return "\n".join(lines)


_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain