"""
utils/helpers.py
────────────────
Shared utility functions for ResearchMate.
"""

import os
import time
import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def format_time_ms(ms: float) -> str:
    """Format milliseconds as a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def format_file_size(path: str) -> str:
    """Return human-readable file size."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "unknown size"
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def truncate_text(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def sanitize_filename(filename: str) -> str:
    """Remove unsafe characters from filenames."""
    return re.sub(r"[^\w\-. ]", "_", filename)


def timestamp() -> str:
    """Return a compact timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def highlight_query_terms(text: str, query: str, max_preview: int = 400) -> str:
    """
    Return a snippet of text around the first query term match.
    (Used for citation previews.)
    """
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    text_lower = text.lower()

    best_pos = len(text)
    for term in query_terms:
        pos = text_lower.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(text):
        # No match found — return beginning
        return truncate_text(text, max_preview)

    start = max(0, best_pos - 100)
    snippet = text[start : start + max_preview]
    if start > 0:
        snippet = "…" + snippet
    if start + max_preview < len(text):
        snippet += "…"
    return snippet
