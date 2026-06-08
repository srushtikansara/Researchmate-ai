"""
modules/comparison.py
──────────────────────
Multi-paper comparison and research gap detection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import COMPARISON_PROMPT, RESEARCH_GAP_PROMPT, TOP_K_RETRIEVAL
from modules.retriever import PaperRetriever, get_retriever
from modules.rag_chain import RAGChain, get_rag_chain

logger = logging.getLogger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Side-by-side comparison of two or more papers."""
    paper_names: list[str]
    comparison_text: str
    generation_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.comparison_text) and self.error is None


@dataclass
class ResearchInsights:
    """Research gaps and future directions for a paper."""
    filename: str
    insights_text: str
    research_gaps: list[str] = field(default_factory=list)
    future_directions: list[str] = field(default_factory=list)
    generation_time_s: float = 0.0
    error: Optional[str] = None


# ─── Paper Comparator ─────────────────────────────────────────────────────────

class PaperComparator:
    """
    Compares multiple papers by retrieving relevant context from each
    and generating a structured comparison via the LLM.
    """

    COMPARISON_ASPECTS = [
        "problem statement and motivation",
        "methodology and technical approach",
        "key contributions and novelty",
        "experimental results and evaluation metrics",
        "limitations and future work",
    ]

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        retriever: Optional[PaperRetriever] = None,
    ):
        self.rag_chain = rag_chain or get_rag_chain()
        self.retriever = retriever or get_retriever()

    def compare(
        self,
        paper_a: str,
        paper_b: str,
        custom_query: Optional[str] = None,
        k_per_paper: int = 5,
    ) -> ComparisonResult:
        """
        Compare two papers.

        Args:
            paper_a: Filename of first paper.
            paper_b: Filename of second paper.
            custom_query: Optional custom focus for comparison.
            k_per_paper: Chunks to retrieve per paper.

        Returns:
            ComparisonResult with structured comparison.
        """
        start = time.time()
        paper_names = [paper_a, paper_b]

        # Build comparison query
        base_query = custom_query or (
            "Compare the methodology, contributions, results, and limitations "
            "of these research papers."
        )

        # Retrieve context from both papers
        context_parts = []
        for paper in paper_names:
            result = self.retriever.retrieve(
                query=base_query,
                k=k_per_paper,
                filter_files=[paper],
            )
            if not result.is_empty:
                ctx = result.format_context(include_citations=True)
                context_parts.append(f"=== {paper} ===\n{ctx}")

        if not context_parts:
            return ComparisonResult(
                paper_names=paper_names,
                comparison_text="No content could be retrieved from the selected papers.",
                error="No context retrieved",
            )

        combined_context = "\n\n".join(context_parts)

        # Build prompt
        prompt = COMPARISON_PROMPT.format(
            context=combined_context,
            paper_names=", ".join(paper_names),
        )

        try:
            comparison_text, _ = self.rag_chain._generate(prompt)
        except Exception as e:
            return ComparisonResult(
                paper_names=paper_names,
                comparison_text="",
                error=str(e),
            )

        elapsed = round(time.time() - start, 2)
        return ComparisonResult(
            paper_names=paper_names,
            comparison_text=comparison_text,
            generation_time_s=elapsed,
        )

    def compare_aspect(
        self,
        paper_a: str,
        paper_b: str,
        aspect: str,
        k_per_paper: int = 4,
    ) -> str:
        """
        Compare two papers on a specific aspect.

        Args:
            paper_a: Filename of first paper.
            paper_b: Filename of second paper.
            aspect: Comparison dimension (e.g., "methodology", "results").

        Returns:
            Comparison text for the given aspect.
        """
        query = f"What is the {aspect} of this paper?"
        result = self.compare(paper_a, paper_b, custom_query=query, k_per_paper=k_per_paper)
        return result.comparison_text


# ─── Research Gap Detector ────────────────────────────────────────────────────

class ResearchGapDetector:
    """
    Identifies research gaps, limitations, and future directions
    from one or more papers.
    """

    def __init__(self, rag_chain: Optional[RAGChain] = None, retriever: Optional[PaperRetriever] = None):
        self.rag_chain = rag_chain or get_rag_chain()
        self.retriever = retriever or get_retriever()

    def detect_gaps(self, filename: str, k: int = 8) -> ResearchInsights:
        """
        Detect research gaps and future directions in a paper.

        Args:
            filename: Paper to analyze.
            k: Chunks to retrieve.

        Returns:
            ResearchInsights with structured analysis.
        """
        start = time.time()

        query = (
            "What are the limitations, research gaps, open problems, "
            "future work, and potential improvements mentioned in this paper?"
        )

        result = self.retriever.retrieve(query=query, k=k, filter_files=[filename])

        if result.is_empty:
            return ResearchInsights(
                filename=filename,
                insights_text="No relevant content retrieved.",
                error="Empty retrieval",
            )

        context = result.format_context(include_citations=True)
        prompt = RESEARCH_GAP_PROMPT.format(context=context)

        try:
            insights_text, _ = self.rag_chain._generate(prompt)
        except Exception as e:
            return ResearchInsights(
                filename=filename,
                insights_text="",
                error=str(e),
            )

        elapsed = round(time.time() - start, 2)
        return ResearchInsights(
            filename=filename,
            insights_text=insights_text,
            generation_time_s=elapsed,
        )


# ─── Module-Level Singletons ──────────────────────────────────────────────────

_comparator: Optional[PaperComparator] = None
_gap_detector: Optional[ResearchGapDetector] = None


def get_comparator() -> PaperComparator:
    global _comparator
    if _comparator is None:
        _comparator = PaperComparator()
    return _comparator


def get_gap_detector() -> ResearchGapDetector:
    global _gap_detector
    if _gap_detector is None:
        _gap_detector = ResearchGapDetector()
    return _gap_detector
