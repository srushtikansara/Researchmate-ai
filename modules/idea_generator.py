"""
modules/idea_generator.py
──────────────────────────
Generates novel research ideas based on uploaded papers.
Identifies gaps and suggests future research directions.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.rag_chain import RAGChain, get_rag_chain
from modules.retriever import PaperRetriever, get_retriever

logger = logging.getLogger(__name__)

IDEA_GENERATION_PROMPT = """You are an expert research advisor. Based on the research papers provided below, generate novel and actionable research ideas.

Context from research papers:
{context}

Papers analyzed: {paper_names}

Generate 5 specific, novel research ideas. For each idea provide:

**Idea [N]: [Title of the idea]**

**Problem it solves:** [What gap or limitation does this address?]

**Proposed Approach:** [How would you implement this?]

**Evidence from papers:** [Which paper/finding motivates this idea?]

**Expected Impact:** [What improvement or contribution would this make?]

---

Focus on:
- Combining techniques from different papers
- Addressing stated limitations
- Using newer technologies not yet applied in this area
- Improving datasets or evaluation methods
- Real-world applications

Research Ideas:"""

RECOMMENDATION_PROMPT = """Based on the research topics and content from the uploaded papers, provide research recommendations.

Context:
{context}

Current research topics: {paper_names}

Provide:

1. **Related Research Areas** (5 areas closely related to these papers)
2. **Suggested Next Papers to Read** (types of papers that would complement this research)
3. **Future Exploration Directions** (3-5 specific directions worth exploring)
4. **Skill/Technology Gaps** (what techniques or tools would help advance this research)
5. **Potential Collaborations** (what other fields could contribute to this research)

Be specific and actionable.

Recommendations:"""


@dataclass
class ResearchIdeas:
    """Generated research ideas from papers."""
    paper_names: list
    ideas_text: str
    generation_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.ideas_text) and self.error is None


@dataclass
class ResearchRecommendations:
    """Research recommendations based on papers."""
    paper_names: list
    recommendations_text: str
    generation_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.recommendations_text) and self.error is None


class IdeaGenerator:
    """Generates novel research ideas from uploaded papers."""

    CONTEXT_QUERIES = [
        "limitations challenges future work open problems",
        "methodology approach model architecture proposed",
        "results accuracy performance evaluation metrics",
        "dataset training evaluation experimental setup",
        "contributions novelty improvements over existing methods",
    ]

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        retriever: Optional[PaperRetriever] = None,
    ):
        self.rag_chain = rag_chain or get_rag_chain()
        self.retriever = retriever or get_retriever()

    def _gather_context(self, paper_names: list, k: int = 5) -> str:
        """Gather relevant context from papers for idea generation."""
        all_parts = []
        for query in self.CONTEXT_QUERIES:
            result = self.retriever.retrieve(
                query=query,
                k=k,
                filter_files=paper_names if paper_names else None,
            )
            if not result.is_empty:
                all_parts.append(result.format_context(include_citations=True))
        return "\n\n---\n\n".join(all_parts[:6])

    def generate_ideas(
        self,
        paper_names: list,
        k: int = 5,
    ) -> ResearchIdeas:
        """
        Generate novel research ideas from uploaded papers.

        Args:
            paper_names: Papers to base ideas on.
            k: Chunks to retrieve per query.

        Returns:
            ResearchIdeas with generated ideas.
        """
        import time
        start = time.time()

        context = self._gather_context(paper_names, k)
        if not context:
            return ResearchIdeas(
                paper_names=paper_names,
                ideas_text="No content could be retrieved.",
                error="Empty retrieval",
            )

        prompt = IDEA_GENERATION_PROMPT.format(
            context=context,
            paper_names=", ".join(paper_names),
        )

        try:
            ideas_text, _ = self.rag_chain._generate(prompt)
        except Exception as e:
            return ResearchIdeas(
                paper_names=paper_names,
                ideas_text="",
                error=str(e),
            )

        elapsed = round(time.time() - start, 2)
        return ResearchIdeas(
            paper_names=paper_names,
            ideas_text=ideas_text,
            generation_time_s=elapsed,
        )

    def generate_recommendations(
        self,
        paper_names: list,
        k: int = 5,
    ) -> ResearchRecommendations:
        """
        Generate research recommendations based on uploaded papers.

        Args:
            paper_names: Papers to base recommendations on.
            k: Chunks to retrieve per query.

        Returns:
            ResearchRecommendations with suggested directions.
        """
        import time
        start = time.time()

        context = self._gather_context(paper_names, k)
        if not context:
            return ResearchRecommendations(
                paper_names=paper_names,
                recommendations_text="No content could be retrieved.",
                error="Empty retrieval",
            )

        prompt = RECOMMENDATION_PROMPT.format(
            context=context,
            paper_names=", ".join(paper_names),
        )

        try:
            rec_text, _ = self.rag_chain._generate(prompt)
        except Exception as e:
            return ResearchRecommendations(
                paper_names=paper_names,
                recommendations_text="",
                error=str(e),
            )

        elapsed = round(time.time() - start, 2)
        return ResearchRecommendations(
            paper_names=paper_names,
            recommendations_text=rec_text,
            generation_time_s=elapsed,
        )


_idea_generator: Optional[IdeaGenerator] = None


def get_idea_generator() -> IdeaGenerator:
    global _idea_generator
    if _idea_generator is None:
        _idea_generator = IdeaGenerator()
    return _idea_generator
