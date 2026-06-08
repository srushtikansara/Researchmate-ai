"""
modules/trend_analyzer.py
──────────────────────────
Analyzes trends across multiple uploaded research papers.
Identifies common datasets, models, metrics, and techniques.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.rag_chain import RAGChain, get_rag_chain
from modules.retriever import PaperRetriever, get_retriever

logger = logging.getLogger(__name__)

TREND_PROMPT = """Based on the research papers provided in the context below, analyze and identify trends.

Context:
{context}

Papers analyzed: {paper_names}

Please provide a detailed trend analysis covering:

1. **Frequently Used Datasets**: List all datasets mentioned across papers with frequency
2. **Popular Models & Architectures**: What models/architectures appear most often?
3. **Common Evaluation Metrics**: What metrics are used to evaluate performance?
4. **Frequently Cited Techniques**: What techniques or methods appear across multiple papers?
5. **Common Challenges**: What challenges are repeatedly mentioned?
6. **Overall Research Direction**: What is the general direction of research in this area?

Be specific and cite which papers mention each trend.

Analysis:"""


@dataclass
class TrendReport:
    """Trend analysis report across multiple papers."""
    paper_names: list
    trend_text: str
    generation_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.trend_text) and self.error is None


class TrendAnalyzer:
    """Analyzes research trends across multiple papers."""

    TREND_QUERIES = [
        "datasets used for training and evaluation experiments",
        "model architecture neural network deep learning methods",
        "evaluation metrics accuracy performance results benchmark",
        "techniques methods approaches proposed in the paper",
        "challenges limitations problems in the research area",
    ]

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        retriever: Optional[PaperRetriever] = None,
    ):
        self.rag_chain = rag_chain or get_rag_chain()
        self.retriever = retriever or get_retriever()

    def analyze(
        self,
        paper_names: list,
        k_per_query: int = 4,
    ) -> TrendReport:
        """
        Analyze trends across multiple papers.

        Args:
            paper_names: List of paper filenames to analyze.
            k_per_query: Chunks to retrieve per trend query.

        Returns:
            TrendReport with structured trend analysis.
        """
        import time
        start = time.time()

        # Collect context from multiple queries
        all_context_parts = []
        for query in self.TREND_QUERIES:
            result = self.retriever.retrieve(
                query=query,
                k=k_per_query,
                filter_files=paper_names if paper_names else None,
            )
            if not result.is_empty:
                ctx = result.format_context(include_citations=True)
                all_context_parts.append(ctx)

        if not all_context_parts:
            return TrendReport(
                paper_names=paper_names,
                trend_text="No content could be retrieved from the selected papers.",
                error="Empty retrieval",
            )

        combined_context = "\n\n---\n\n".join(all_context_parts[:8])  # limit context

        prompt = TREND_PROMPT.format(
            context=combined_context,
            paper_names=", ".join(paper_names),
        )

        try:
            trend_text, _ = self.rag_chain._generate(prompt)
        except Exception as e:
            return TrendReport(
                paper_names=paper_names,
                trend_text="",
                error=str(e),
            )

        elapsed = round(time.time() - start, 2)
        return TrendReport(
            paper_names=paper_names,
            trend_text=trend_text,
            generation_time_s=elapsed,
        )


_trend_analyzer: Optional[TrendAnalyzer] = None


def get_trend_analyzer() -> TrendAnalyzer:
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    return _trend_analyzer
