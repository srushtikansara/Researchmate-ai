"""
app.py — ResearchMate AI v2.0
──────────────────────────────
Upgraded Research Paper Assistant with:
- Improved Sidebar with stats
- Research Gap Finder
- Multi-Paper Comparison
- Research Trend Analysis
- Novel Research Idea Generator
- Research Recommendation Engine
- Hybrid Search (FAISS + BM25 + CrossEncoder)

Run: streamlit run app.py
"""

import os
import time
import logging
import streamlit as st

st.set_page_config(
    page_title="ResearchMate AI – Research Paper Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import TOP_K_RETRIEVAL, MAX_FILES, LLM_BACKEND, OLLAMA_MODEL, HF_MODEL_ID, RAG_SYSTEM_PROMPT
from modules.pdf_loader import load_pdf
from modules.text_splitter import chunk_document, get_chunk_stats
from modules.vector_db import get_vector_store
from modules.rag_chain import get_rag_chain
from modules.summarizer import get_summarizer
from modules.comparison import get_comparator, get_gap_detector
from modules.trend_analyzer import get_trend_analyzer
from modules.idea_generator import get_idea_generator
from modules.hybrid_retriever import get_hybrid_retriever
from utils.helpers import format_time_ms, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary: #4F6AF5;
    --primary-light: #7B93FF;
    --accent: #F5A623;
    --success: #27AE60;
    --danger: #E74C3C;
    --bg-card: #1E1E2E;
    --bg-input: #2A2A3E;
    --teal: #00BCD4;
    --purple: #9C27B0;
    --green: #4CAF50;
}
.main .block-container { padding-top: 1.2rem; }
.chat-user {
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    color: white; border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; margin: 8px 0 8px 48px;
    font-size: 0.95rem; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(79,106,245,0.3);
}
.chat-assistant {
    background: var(--bg-card); border: 1px solid #333;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 16px; margin: 8px 48px 8px 0;
    font-size: 0.95rem; line-height: 1.6;
}
.citation-card {
    background: var(--bg-input); border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 6px 0;
    font-size: 0.82rem; color: #ccc;
}
.citation-header { font-weight: 600; color: var(--accent); margin-bottom: 4px; }
.metric-badge {
    display: inline-block; background: rgba(79,106,245,0.15);
    border: 1px solid rgba(79,106,245,0.4); border-radius: 20px;
    padding: 2px 10px; font-size: 0.78rem; color: var(--primary-light); margin: 2px;
}
.hybrid-badge {
    display: inline-block; background: rgba(0,188,212,0.15);
    border: 1px solid rgba(0,188,212,0.4); border-radius: 20px;
    padding: 2px 10px; font-size: 0.78rem; color: var(--teal); margin: 2px;
}
.idea-card {
    background: var(--bg-card); border-left: 4px solid var(--teal);
    border-radius: 0 12px 12px 0; padding: 16px; margin: 12px 0;
}
.trend-card {
    background: var(--bg-card); border-left: 4px solid var(--purple);
    border-radius: 0 12px 12px 0; padding: 16px; margin: 12px 0;
}
.rec-card {
    background: var(--bg-card); border-left: 4px solid var(--green);
    border-radius: 0 12px 12px 0; padding: 16px; margin: 12px 0;
}
.tip-box {
    background: rgba(245,166,35,0.08); border: 1px solid rgba(245,166,35,0.3);
    border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; color: #ccc; margin: 8px 0;
}
[data-testid="stSidebar"] { background: #13131F; }
[data-testid="stSidebar"] hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "documents": {},
        "chat_history": [],
        "processing_log": [],
        "active_files": [],
        "summaries": {},
        "insights": {},
        "comparison_result": None,
        "trend_report": None,
        "research_ideas": None,
        "recommendations": None,
        "use_hybrid": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def log_status(msg, level="info"):
    st.session_state.processing_log.append({"msg": msg, "level": level, "ts": time.time()})
    logger.info(msg) if level == "info" else logger.error(msg)

def get_indexed_files():
    return list(get_vector_store()._indexed_files)

def llm_label():
    return f"Ollama / {OLLAMA_MODEL}" if LLM_BACKEND == "ollama" else f"HuggingFace / {HF_MODEL_ID.split('/')[-1]}"

def get_page_count_from_store(filename):
    """Get page count from vector store metadata when session state is unavailable."""
    try:
        if get_vector_store()._store:
            pg_set = set(
                d.metadata.get("page", 0)
                for d in get_vector_store()._store.docstore._dict.values()
                if hasattr(d, "metadata") and d.metadata.get("source") == filename
            )
            return max(pg_set) if pg_set else "?"
    except Exception:
        pass
    return "?"

def get_total_pages_from_store(indexed):
    """Get total pages across all indexed papers."""
    try:
        if get_vector_store()._store:
            all_pages = set()
            for d in get_vector_store()._store.docstore._dict.values():
                if hasattr(d, "metadata"):
                    pg = d.metadata.get("page", 0)
                    src = d.metadata.get("source", "")
                    if src in indexed and pg:
                        all_pages.add((src, pg))
            return len(all_pages)
    except Exception:
        pass
    return 0

def get_total_chunks():
    """Get total chunk count from vector store."""
    try:
        if get_vector_store()._store:
            return len(get_vector_store()._store.docstore._dict)
    except Exception:
        pass
    return 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:4px 0 12px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <div style="width:34px;height:34px;background:#4F6AF5;border-radius:8px;
                     display:flex;align-items:center;justify-content:center;font-size:18px;">
                    🔬
                </div>
                <span style="font-size:17px;font-weight:500;color:var(--color-text-primary)">
                    Research<span style="color:#4F6AF5">Mate</span>
                </span>
            </div>
            <span style="font-size:11px;padding:2px 8px;background:rgba(79,106,245,0.15);
                  color:#7B93FF;border-radius:20px;font-weight:500;">AI v2.0 · RAG + Hybrid</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Upload ─────────────────────────────────────────────────
        st.markdown("<p style='font-size:11px;font-weight:500;text-transform:uppercase;"
                    "letter-spacing:0.06em;color:#888;margin-bottom:8px'>"
                    "Upload Papers</p>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "PDF files", type=["pdf"],
            accept_multiple_files=True,
            help=f"Up to {MAX_FILES} PDFs · 200 MB each",
            label_visibility="collapsed",
        )
        if uploaded_files:
            for uf in uploaded_files[:MAX_FILES]:
                if uf.name not in st.session_state.documents:
                    _process_uploaded_file(uf)

        st.divider()

        # ── Indexed Papers ─────────────────────────────────────────
        indexed = get_indexed_files()
        st.markdown("<p style='font-size:11px;font-weight:500;text-transform:uppercase;"
                    "letter-spacing:0.06em;color:#888;margin-bottom:8px'>"
                    "Indexed Papers</p>", unsafe_allow_html=True)

        if indexed:
            colors = ["#4F6AF5", "#27AE60", "#F5A623", "#E74C3C",
                      "#00BCD4", "#9C27B0", "#FF5722", "#607D8B"]

            for i, fn in enumerate(indexed):
                color = colors[i % len(colors)]
                # Get pages from session or from store
                doc = st.session_state.documents.get(fn)
                if doc:
                    pages = doc.total_pages
                else:
                    pages = get_page_count_from_store(fn)

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                     background:rgba(255,255,255,0.04);border:0.5px solid #333;
                     border-radius:8px;margin-bottom:6px;">
                    <div style="width:8px;height:8px;border-radius:50%;
                         background:{color};flex-shrink:0;"></div>
                    <span style="font-size:12px;color:#ddd;flex:1;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                          title="{fn}">{fn[:24]}{'…' if len(fn)>24 else ''}</span>
                    <span style="font-size:11px;color:#666;flex-shrink:0;">{pages}pg</span>
                </div>
                """, unsafe_allow_html=True)

            # Stats
            total_pages = sum(
                st.session_state.documents[f].total_pages
                for f in indexed if f in st.session_state.documents
            )
            if total_pages == 0:
                total_pages = get_total_pages_from_store(indexed)

            total_chunks = get_total_chunks()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Papers", len(indexed))
            with col2:
                st.metric("Pages", total_pages if total_pages else "?")
            with col3:
                st.metric("Chunks", total_chunks)

            st.divider()

            # Filter
            st.markdown("<p style='font-size:12px;color:#888;margin-bottom:6px'>"
                        "Filter chat to:</p>", unsafe_allow_html=True)
            st.session_state.active_files = st.multiselect(
                "Papers", options=indexed, default=[],
                label_visibility="collapsed",
                help="Leave empty to search ALL papers",
            )
        else:
            st.markdown("""
            <div style="padding:20px 16px;text-align:center;
                 border:0.5px dashed #333;border-radius:8px;">
                <div style="font-size:28px;margin-bottom:8px;">📄</div>
                <div style="font-size:12px;color:#666;line-height:1.6;">
                    No papers indexed yet.<br>Upload PDFs above to get started.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Processing Log ─────────────────────────────────────────
        if st.session_state.processing_log:
            with st.expander("📋 Processing Log", expanded=False):
                for entry in st.session_state.processing_log[-8:]:
                    icon = "✅" if entry["level"] == "info" else "❌"
                    st.markdown(f"<small style='color:#aaa'>{icon} {entry['msg']}</small>",
                                unsafe_allow_html=True)
            st.divider()

        # ── Settings ───────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:5px 0;border-bottom:0.5px solid #333;">
                    <span style="font-size:12px;color:#aaa;">🤖 LLM</span>
                    <span style="font-size:11px;color:#27AE60;font-weight:500;">● Online</span>
                </div>
                <div style="font-size:11px;color:#666;margin-top:6px;line-height:1.8;">
                    Model: <span style="color:#aaa">{llm_label()}</span><br>
                    Embeddings: <span style="color:#aaa">BGE-small-en-v1.5</span><br>
                    Top-K: <span style="color:#aaa">{TOP_K_RETRIEVAL}</span> ·
                    Chunk: <span style="color:#aaa">1000</span> ·
                    Overlap: <span style="color:#aaa">200</span>
                </div>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;">
                <span style="font-size:10px;padding:2px 7px;border-radius:20px;
                      background:rgba(0,188,212,0.12);color:#00BCD4;
                      border:0.5px solid rgba(0,188,212,0.3)">🔀 FAISS + BM25</span>
                <span style="font-size:10px;padding:2px 7px;border-radius:20px;
                      background:rgba(156,39,176,0.12);color:#CE93D8;
                      border:0.5px solid rgba(156,39,176,0.3)">🎯 CrossEncoder</span>
                <span style="font-size:10px;padding:2px 7px;border-radius:20px;
                      background:rgba(79,106,245,0.12);color:#7B93FF;
                      border:0.5px solid rgba(79,106,245,0.3)">🧠 RAG</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Reset Vector Store", use_container_width=True, key="reset_vs"):
                get_vector_store().reset()
                for k in ["documents", "chat_history", "summaries", "insights",
                          "comparison_result", "trend_report", "research_ideas",
                          "recommendations", "processing_log", "active_files"]:
                    st.session_state[k] = (
                        {} if k in ["documents", "summaries", "insights"]
                        else ([] if k in ["chat_history", "processing_log", "active_files"]
                              else None)
                    )
                st.success("Reset complete!")
                st.rerun()

        # ── Footer ─────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:8px;">
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
                <span style="font-size:10px;padding:2px 6px;border-radius:20px;
                      background:#1a1a2e;border:0.5px solid #333;color:#555">LangChain</span>
                <span style="font-size:10px;padding:2px 6px;border-radius:20px;
                      background:#1a1a2e;border:0.5px solid #333;color:#555">FAISS</span>
                <span style="font-size:10px;padding:2px 6px;border-radius:20px;
                      background:#1a1a2e;border:0.5px solid #333;color:#555">BM25</span>
                <span style="font-size:10px;padding:2px 6px;border-radius:20px;
                      background:#1a1a2e;border:0.5px solid #333;color:#555">PyMuPDF</span>
                <span style="font-size:10px;padding:2px 6px;border-radius:20px;
                      background:#1a1a2e;border:0.5px solid #333;color:#555">Ollama</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _process_uploaded_file(uploaded_file):
    """Extract, chunk, embed, and index an uploaded PDF."""
    filename = uploaded_file.name
    with st.spinner(f"Processing `{filename}` …"):
        log_status(f"Extracting '{filename}' …")
        doc = load_pdf(uploaded_file)
        if not doc.is_valid:
            log_status(f"Failed: {doc.error}", level="error")
            st.sidebar.error(f"❌ {doc.error}")
            return
        log_status(f"Extracted {doc.total_pages} pages, {doc.total_chars:,} chars")

        chunks = chunk_document(doc)
        stats = get_chunk_stats(chunks)
        log_status(f"Created {stats['total']} chunks (avg {stats['avg_chars']:.0f} chars)")

        vs = get_vector_store()
        added = vs.add_chunks(chunks, file_hash=doc.file_hash, filename=filename)
        if added > 0:
            log_status(f"Indexed {added} chunks from '{filename}'")
            st.sidebar.success(f"✅ `{filename}` — {doc.total_pages} pages, {added} chunks")
        else:
            log_status(f"'{filename}' already indexed (skipped)")
            st.sidebar.info(f"ℹ️ `{filename}` already indexed")

        # Reset BM25 index to pick up new docs
        try:
            from modules.hybrid_retriever import _hybrid_retriever
            if _hybrid_retriever:
                _hybrid_retriever._bm25_indexed = False
        except Exception:
            pass

        st.session_state.documents[filename] = doc


# ─── Retrieval Helper ─────────────────────────────────────────────────────────
def do_retrieval(question, k=None, filter_files=None):
    """Use hybrid or standard retrieval based on settings."""
    k = k or TOP_K_RETRIEVAL
    if st.session_state.get("use_hybrid", True):
        retriever = get_hybrid_retriever()
        return retriever.retrieve(question, k=k, filter_files=filter_files)
    else:
        from modules.retriever import get_retriever
        return get_retriever().retrieve(question, k=k, filter_files=filter_files)


def _run_rag(question, filter_files=None, chat_history=None):
    """Core RAG pipeline — retrieve + generate."""
    retrieval = do_retrieval(question, filter_files=filter_files)
    rag = get_rag_chain()
    if retrieval.is_empty:
        return (
            "I could not find this information in the uploaded paper.",
            [], {}, False, False,
        )
    context = retrieval.format_context()
    prompt = RAG_SYSTEM_PROMPT.format(context=context, question=question)
    answer_text, response_ms = rag._generate(prompt)
    citations = retrieval.get_citations()
    metrics = {
        "retrieval_time_ms": retrieval.retrieval_time_ms,
        "response_time_ms": response_ms,
        "chunks_retrieved": len(retrieval.chunks),
        "pages": retrieval.stats.get("pages", []),
    }
    used_hybrid = hasattr(retrieval, "used_bm25")
    used_reranking = getattr(retrieval, "used_reranking", False)
    return answer_text, citations, metrics, used_hybrid, used_reranking


# ─── Tab 1: Chat ──────────────────────────────────────────────────────────────
def render_chat_tab():
    st.markdown("### 💬 Chat with Your Papers")
    indexed = get_indexed_files()
    if not indexed:
        st.markdown('<div class="tip-box">👈 Upload research PDFs in the sidebar to get started.</div>',
                    unsafe_allow_html=True)
        return

    # Quick question buttons
    st.markdown("**Quick Questions:**")
    quick_questions = [
        "What is the methodology used?",
        "What are the key contributions?",
        "What are the limitations?",
        "What dataset was used?",
        "Summarize the paper.",
        "What results were achieved?",
    ]
    cols = st.columns(3)
    for i, qq in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(qq, use_container_width=True, key=f"qq_{i}"):
                st.session_state["_pending_q"] = qq

    st.divider()

    # Chat history display
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">🧑‍🔬 {turn["question"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chat-assistant">📄 {turn["answer"]}</div>',
                    unsafe_allow_html=True)
        if turn.get("citations"):
            with st.expander(f"📎 {len(turn['citations'])} Source(s)", expanded=False):
                for cit in turn["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            [{cit['index']}] {cit['source']} · Page {cit['page']}
                            <span class="metric-badge">Score: {cit['score']}</span>
                        </div>
                        {cit['preview']}
                    </div>""", unsafe_allow_html=True)
        if turn.get("metrics"):
            m = turn["metrics"]
            hybrid_info = ""
            if turn.get("used_hybrid"):
                hybrid_info = '<span class="hybrid-badge">🔀 Hybrid</span>'
                if turn.get("used_reranking"):
                    hybrid_info += '<span class="hybrid-badge">🎯 Reranked</span>'
            st.markdown(
                f'<span class="metric-badge">⏱ {format_time_ms(m.get("retrieval_time_ms",0))}</span>'
                f'<span class="metric-badge">🤖 {format_time_ms(m.get("response_time_ms",0))}</span>'
                f'<span class="metric-badge">📦 {m.get("chunks_retrieved",0)} chunks</span>'
                + hybrid_info, unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # Handle quick button click
    pending_q = st.session_state.get("_pending_q", "")
    if pending_q:
        st.session_state.pop("_pending_q")
        filter_files = st.session_state.active_files or None
        with st.spinner(f"Answering: {pending_q}"):
            answer, citations, metrics, used_hybrid, used_reranking = _run_rag(
                pending_q, filter_files=filter_files,
            )
        st.session_state.chat_history.append({
            "question": pending_q, "answer": answer,
            "citations": citations, "metrics": metrics,
            "used_hybrid": used_hybrid, "used_reranking": used_reranking,
        })
        st.rerun()

    # Manual input
    user_input = st.text_area(
        "Your question:", value="", height=80,
        placeholder="e.g. What model architecture is proposed?",
        label_visibility="collapsed", key="chat_input_box",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_clicked = st.button("🔍 Ask ResearchMate", use_container_width=True, key="ask_btn")
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked and user_input.strip():
        question = user_input.strip()
        filter_files = st.session_state.active_files or None
        with st.spinner("Retrieving context and generating answer …"):
            answer, citations, metrics, used_hybrid, used_reranking = _run_rag(
                question, filter_files=filter_files,
            )
        st.session_state.chat_history.append({
            "question": question, "answer": answer,
            "citations": citations, "metrics": metrics,
            "used_hybrid": used_hybrid, "used_reranking": used_reranking,
        })
        st.rerun()


# ─── Tab 2: Summary ───────────────────────────────────────────────────────────
def render_summary_tab():
    st.markdown("### 📝 Automatic Paper Summarization")
    indexed = get_indexed_files()
    if not indexed:
        st.info("Upload and index papers first.")
        return

    selected = st.selectbox("Choose a paper:", indexed, key="sum_select")
    if st.button("🔄 Generate Summary", use_container_width=True, key="gen_sum_btn"):
        if selected in st.session_state.summaries:
            del st.session_state.summaries[selected]
        with st.spinner(f"Generating summary for `{selected}` … (2-5 min on CPU)"):
            summary = get_summarizer().summarize(selected)
            st.session_state.summaries[selected] = summary

    if selected in st.session_state.summaries:
        summary = st.session_state.summaries[selected]
        st.markdown(f"**Generated in:** {summary.generation_time_s:.1f} s")
        tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
            "📋 Executive Summary", "🌟 Key Contributions",
            "🔬 Methodology", "📊 Results", "⚠️ Limitations",
        ])
        with tab_a: st.markdown(summary.executive_summary or "_Not available_")
        with tab_b: st.markdown(summary.key_contributions or "_Not available_")
        with tab_c: st.markdown(summary.methodology or "_Not available_")
        with tab_d: st.markdown(summary.results or "_Not available_")
        with tab_e: st.markdown(summary.limitations or "_Not available_")
        st.download_button(
            "⬇️ Download Summary (Markdown)",
            data=summary.to_markdown(),
            file_name=f"summary_{selected.replace('.pdf','')}.md",
            mime="text/markdown",
            use_container_width=True, key="dl_sum",
        )


# ─── Tab 3: Research Gap Finder ───────────────────────────────────────────────
def render_gap_finder_tab():
    st.markdown("### 🔭 Research Gap Finder")
    indexed = get_indexed_files()
    if not indexed:
        st.info("Upload papers first.")
        return

    st.markdown(
        '<div class="tip-box">🔍 Automatically identifies research gaps, open problems, '
        'limitations, and future work from your papers.</div>',
        unsafe_allow_html=True,
    )

    selected = st.selectbox("Select paper to analyze:", indexed, key="gap_select")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🧠 Find Research Gaps", use_container_width=True, key="gap_btn"):
            with st.spinner(f"Analyzing gaps in `{selected}` …"):
                insights = get_gap_detector().detect_gaps(selected)
                st.session_state.insights[selected] = insights

    with col2:
        custom_gap_q = st.text_input(
            "Custom question:",
            placeholder="e.g. What future work is suggested?",
            key="custom_gap_q",
            label_visibility="collapsed",
        )
        if st.button("🔍 Ask Custom", use_container_width=True, key="custom_gap_btn"):
            if custom_gap_q.strip():
                with st.spinner("Analyzing …"):
                    answer, citations, metrics, _, _ = _run_rag(
                        custom_gap_q, filter_files=[selected],
                    )
                    st.markdown("**Answer:**")
                    st.markdown(f'<div class="trend-card">{answer}</div>',
                                unsafe_allow_html=True)
                    if citations:
                        with st.expander(f"📎 {len(citations)} Source(s)", expanded=False):
                            for cit in citations:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        [{cit['index']}] {cit['source']} · Page {cit['page']}
                                        <span class="metric-badge">Score: {cit['score']}</span>
                                    </div>
                                    {cit['preview']}
                                </div>""", unsafe_allow_html=True)

    if selected in st.session_state.insights:
        insights = st.session_state.insights[selected]
        if insights.error:
            st.error(f"Analysis failed: {insights.error}")
        else:
            st.markdown(f"**Analysis time:** {insights.generation_time_s:.1f} s")
            st.markdown("---")
            st.markdown(f'<div class="trend-card">{insights.insights_text}</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Gap Analysis",
                data=f"# Research Gap Analysis\n\n{insights.insights_text}",
                file_name=f"gaps_{selected.replace('.pdf','')}.md",
                mime="text/markdown",
                use_container_width=True, key="dl_gaps",
            )


# ─── Tab 4: Paper Comparison ──────────────────────────────────────────────────
def render_comparison_tab():
    st.markdown("### ⚖️ Multi-Paper Comparison")
    indexed = get_indexed_files()
    if len(indexed) < 2:
        st.info("Index at least 2 papers to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        paper_a = st.selectbox("Paper A:", indexed, key="cmp_a")
    with col2:
        paper_b = st.selectbox("Paper B:", [f for f in indexed if f != paper_a], key="cmp_b")

    compare_aspects = st.multiselect(
        "Comparison aspects:",
        ["Methodology", "Dataset", "Model Architecture", "Results & Accuracy",
         "Key Contributions", "Limitations", "Future Work", "Evaluation Metrics"],
        default=["Methodology", "Dataset", "Results & Accuracy", "Limitations"],
        key="cmp_aspects",
    )

    custom_focus = st.text_input(
        "Custom focus (optional):",
        placeholder="e.g. Compare training strategies and performance",
        key="cmp_focus",
    )

    if st.button("🔄 Compare Papers", use_container_width=True, key="cmp_btn"):
        focus = custom_focus or f"Compare {', '.join(compare_aspects)}"
        with st.spinner(f"Comparing `{paper_a}` vs `{paper_b}` … (30-60s on CPU)"):
            result = get_comparator().compare(
                paper_a=paper_a, paper_b=paper_b,
                custom_query=focus,
            )
            st.session_state.comparison_result = result

    if st.session_state.comparison_result:
        result = st.session_state.comparison_result
        if result.is_valid:
            st.markdown(f"**Generated in:** {result.generation_time_s:.1f} s")
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**📄 {result.paper_names[0]}**")
            with col_b:
                st.markdown(f"**📄 {result.paper_names[1]}**")
            st.markdown(result.comparison_text)
            st.download_button(
                "⬇️ Download Comparison (Markdown)",
                data=f"# Paper Comparison\n\n**Papers:** {' vs '.join(result.paper_names)}\n\n{result.comparison_text}",
                file_name="comparison.md",
                mime="text/markdown",
                use_container_width=True, key="dl_cmp",
            )
        else:
            st.error(f"Comparison failed: {result.error}")


# ─── Tab 5: Trend Analysis ────────────────────────────────────────────────────
def render_trend_tab():
    st.markdown("### 📈 Research Trend Analysis")
    indexed = get_indexed_files()
    if not indexed:
        st.info("Upload papers first.")
        return

    st.markdown(
        '<div class="tip-box">📊 Analyzes all uploaded papers to identify common '
        'datasets, models, metrics, and research techniques.</div>',
        unsafe_allow_html=True,
    )

    selected_papers = st.multiselect(
        "Select papers to analyze:",
        options=indexed, default=indexed, key="trend_papers",
    )

    if st.button("📈 Analyze Trends", use_container_width=True, key="trend_btn"):
        if not selected_papers:
            st.warning("Select at least one paper.")
            return
        with st.spinner(f"Analyzing trends across {len(selected_papers)} paper(s) … (2-3 min)"):
            report = get_trend_analyzer().analyze(selected_papers)
            st.session_state.trend_report = report

    if st.session_state.trend_report:
        report = st.session_state.trend_report
        if report.is_valid:
            st.markdown(f"**Generated in:** {report.generation_time_s:.1f} s · "
                        f"**Papers analyzed:** {len(report.paper_names)}")
            st.divider()
            st.markdown(f'<div class="trend-card">{report.trend_text}</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Trend Report",
                data=f"# Research Trend Analysis\n\n{report.trend_text}",
                file_name="trend_report.md",
                mime="text/markdown",
                use_container_width=True, key="dl_trend",
            )
        else:
            st.error(f"Analysis failed: {report.error}")


# ─── Tab 6: Research Ideas ────────────────────────────────────────────────────
def render_ideas_tab():
    st.markdown("### 💡 Novel Research Ideas & Recommendations")
    indexed = get_indexed_files()
    if not indexed:
        st.info("Upload papers first.")
        return

    st.markdown(
        '<div class="tip-box">🚀 Generates novel research ideas and recommendations '
        'based on gaps and limitations found in your papers.</div>',
        unsafe_allow_html=True,
    )

    selected_papers = st.multiselect(
        "Select papers:",
        options=indexed, default=indexed, key="idea_papers",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Generate Research Ideas", use_container_width=True, key="idea_btn"):
            if not selected_papers:
                st.warning("Select at least one paper.")
            else:
                with st.spinner("Generating novel research ideas … (2-3 min)"):
                    ideas = get_idea_generator().generate_ideas(selected_papers)
                    st.session_state.research_ideas = ideas

    with col2:
        if st.button("🗺️ Get Recommendations", use_container_width=True, key="rec_btn"):
            if not selected_papers:
                st.warning("Select at least one paper.")
            else:
                with st.spinner("Generating recommendations …"):
                    recs = get_idea_generator().generate_recommendations(selected_papers)
                    st.session_state.recommendations = recs

    # Show Ideas
    if st.session_state.research_ideas:
        ideas = st.session_state.research_ideas
        if ideas.is_valid:
            st.markdown(f"#### 💡 Research Ideas  *(generated in {ideas.generation_time_s:.1f}s)*")
            st.markdown(f'<div class="idea-card">{ideas.ideas_text}</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Ideas",
                data=f"# Novel Research Ideas\n\n{ideas.ideas_text}",
                file_name="research_ideas.md",
                mime="text/markdown",
                use_container_width=True, key="dl_ideas",
            )

    # Show Recommendations
    if st.session_state.recommendations:
        recs = st.session_state.recommendations
        if recs.is_valid:
            st.markdown(f"#### 🗺️ Recommendations  *(generated in {recs.generation_time_s:.1f}s)*")
            st.markdown(f'<div class="rec-card">{recs.recommendations_text}</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Recommendations",
                data=f"# Research Recommendations\n\n{recs.recommendations_text}",
                file_name="recommendations.md",
                mime="text/markdown",
                use_container_width=True, key="dl_recs",
            )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 0.4rem;">
        <h1 style="font-size:2rem; font-weight:800; margin:0;">
            🔬 Research<span style="color:#4F6AF5;">Mate</span>
            <span style="font-size:1rem; color:#00BCD4; font-weight:400;"> AI v2.0</span>
        </h1>
        <p style="color:#888; margin:4px 0 0; font-size:0.95rem;">
            Research Paper Assistant · Hybrid RAG · Gap Finder · Idea Generator
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat",
        "📝 Summary",
        "🔭 Gap Finder",
        "⚖️ Comparison",
        "📈 Trends",
        "💡 Ideas",
    ])

    with tab1: render_chat_tab()
    with tab2: render_summary_tab()
    with tab3: render_gap_finder_tab()
    with tab4: render_comparison_tab()
    with tab5: render_trend_tab()
    with tab6: render_ideas_tab()


if __name__ == "__main__":
    main()