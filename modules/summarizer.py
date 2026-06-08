"""
modules/summarizer.py
──────────────────────
Generates structured summaries of research papers using RAG.
Produces: executive summary, key contributions, methodology,
          results, and limitations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import SUMMARY_PROMPTS, TOP_K_RETRIEVAL
from modules.retriever import PaperRetriever, get_retriever
from modules.rag_chain import RAGChain, get_rag_chain

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class PaperSummary:
    """Structured summary of a single research paper."""
    filename: str
    executive_summary: str = ""
    key_contributions: str = ""
    methodology: str = ""
    results: str = ""
    limitations: str = ""
    generation_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return bool(self.executive_summary) and self.error is None

    def to_markdown(self) -> str:
        """Render the summary as a Markdown string."""
        lines = [
            f"# Summary: {self.filename}",
            "",
            "## Executive Summary",
            self.executive_summary or "_Not available_",
            "",
            "## Key Contributions",
            self.key_contributions or "_Not available_",
            "",
            "## Methodology",
            self.methodology or "_Not available_",
            "",
            "## Results",
            self.results or "_Not available_",
            "",
            "## Limitations",
            self.limitations or "_Not available_",
        ]
        return "\n".join(lines)


# ─── Summarizer ───────────────────────────────────────────────────────────────

class PaperSummarizer:
    """
    Generates structured paper summaries by running targeted RAG queries
    for each summary section.
    """

    # Section queries — designed to retrieve the most relevant chunks
    SECTION_QUERIES = {
    "executive_summary": (
        "Provide a comprehensive detailed executive summary of this entire paper. "
        "Include: what problem it solves, what method is proposed, what datasets were used, "
        "what results were achieved, and what is the significance of this work. "
        "Write at least 8-10 sentences with specific details from the paper."
    ),
    "key_contributions": (
        "List ALL key contributions, novelties, and innovations of this paper in detail. "
        "Include every unique aspect: new algorithms, new techniques, new datasets, "
        "new architectures, improvements over existing methods. "
        "Write each contribution as a detailed numbered point."
    ),
    "methodology": (
        "Explain the complete detailed methodology of this paper. "
        "Include: model architecture, number of layers, filters, training process, "
        "preprocessing steps, feature extraction methods, classification approach, "
        "and all technical details mentioned. Be very specific and detailed."
    ),
    "results": (
        "List ALL results and performance numbers from this paper. "
        "Include every accuracy percentage, every dataset result, every comparison table value, "
        "every benchmark score mentioned. Be very specific with exact numbers."
    ),
    "limitations": (
        "List ALL limitations, weaknesses, failure cases, constraints, and future work "
        "mentioned in this paper. Include every single limitation the authors acknowledge. "
        "Write each as a detailed point."
    ),
}

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        retriever: Optional[PaperRetriever] = None,
    ):
        self.rag_chain = rag_chain or get_rag_chain()

    def summarize(
        self,
        filename: str,
        k: int = 10,
    ) -> PaperSummary:
        """
        Generate a full structured summary for a single paper.

        Args:
            filename: The paper to summarize (must be indexed).
            k: Chunks to retrieve per section query.

        Returns:
            PaperSummary with all sections populated.
        """
        summary = PaperSummary(filename=filename)
        start = time.time()

        for section, query in self.SECTION_QUERIES.items():
            logger.info("Generating section '%s' for '%s' …", section, filename)
            try:
                response = self.rag_chain.answer(
                    question=query,
                    k=k,
                    filter_files=[filename],
                )
                setattr(summary, section, response.answer)
            except Exception as e:
                logger.error("Error generating '%s': %s", section, e)
                setattr(summary, section, f"Error generating this section: {e}")

        summary.generation_time_s = round(time.time() - start, 2)
        logger.info(
            "Summary for '%s' generated in %.1f s",
            filename,
            summary.generation_time_s,
        )
        return summary

    def summarize_all(self, filenames: list[str]) -> list[PaperSummary]:
        """Generate summaries for a list of papers."""
        return [self.summarize(fn) for fn in filenames]


# ─── Module-Level Singleton ───────────────────────────────────────────────────

_summarizer: Optional[PaperSummarizer] = None


def get_summarizer() -> PaperSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = PaperSummarizer()
    return _summarizer
