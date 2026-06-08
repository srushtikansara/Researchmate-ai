"""
ResearchMate Configuration
Central configuration for all modules.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_ALT = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"          # change to "cuda" if GPU available
EMBEDDING_BATCH_SIZE = 32

# ─── Vector Database ──────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
TOP_K_RETRIEVAL = 8

# ─── LLM ──────────────────────────────────────────────────────────────────────
# Options: "ollama" (local) or "huggingface" (HuggingFace Hub)
LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface")

# Ollama settings (run: ollama pull llama3)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# HuggingFace Hub settings (free tier available)
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# ─── RAG Prompt ───────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are ResearchMate, an expert research paper assistant.

Answer the question based ONLY on the provided context. Be DETAILED and COMPREHENSIVE.
Write complete, thorough answers. Do NOT give short answers.
Always include specific details, numbers, and technical terms from the context.
If asked to summarize or list, provide AT LEAST 5-8 detailed points.

RULES:
1. Answer ONLY from the provided context.
2. Be detailed - write long, thorough answers.
3. Include specific numbers, percentages, and technical details.
4. If information is not in context, say: "I could not find this information in the uploaded paper."
5. Never hallucinate or make up information.

Context from research papers:
{context}

Question: {question}

Detailed Answer:"""

# ─── Summarization Prompts ────────────────────────────────────────────────────
SUMMARY_PROMPTS = {
    "executive_summary": "Provide a detailed executive summary (8-10 sentences) of this research paper covering the problem, approach, and findings.",
    "key_contributions": "List ALL key contributions, novelties and innovations of this paper in detail as numbered points. Be thorough and specific.",
    "methodology": "Explain in detail the complete methodology, model architecture, algorithms, datasets, and experimental setup used in this paper.",
    "results": "List ALL results, accuracy numbers, performance metrics, and evaluation outcomes reported in this paper with specific numbers.",
    "limitations": "List all limitations, weaknesses, failure cases, and future work mentioned in this paper in detail.",
}

# ─── Comparison Prompt ────────────────────────────────────────────────────────
COMPARISON_PROMPT = """Compare and contrast these research papers based on the provided context:

Context:
{context}

Papers being compared: {paper_names}

Provide a structured comparison covering:
1. Problem Statement
2. Methodology
3. Key Contributions
4. Results & Performance
5. Limitations
6. Novelty

Answer:"""

# ─── Research Insights Prompt ─────────────────────────────────────────────────
RESEARCH_GAP_PROMPT = """Based on the research paper content provided:

Context:
{context}

Identify and elaborate on:
1. **Research Gaps**: What problems remain unsolved?
2. **Potential Improvements**: How could the methods be improved?
3. **Future Directions**: What future work is suggested or implied?
4. **Open Questions**: What questions does this work raise?

Provide specific, actionable insights grounded in the paper's content.

Answer:"""

# ─── UI Settings ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 200
ALLOWED_EXTENSIONS = ["pdf"]
MAX_FILES = 10

# Streamlit page config
PAGE_TITLE = "ResearchMate – Research Paper Assistant"
PAGE_ICON = "📄"
LAYOUT = "wide"
