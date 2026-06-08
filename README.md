<<<<<<< HEAD
# 📄 ResearchMate — Research Paper Assistant

> **A production-ready RAG application** for interacting with research papers using natural language.  
> Upload PDFs → Ask questions → Get grounded, cited answers.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?logo=chainlink)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-1.8%2B-blue)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Features

| Feature | Description |
|---|---|
| **PDF Upload** | Single or multi-PDF upload (up to 10 files, 200 MB each) |
| **RAG Pipeline** | Retrieve-then-Generate with strict grounding (no hallucination) |
| **Semantic Search** | FAISS vector store with BGE embeddings |
| **Multi-Paper Chat** | Query across all indexed papers simultaneously |
| **Automatic Summary** | Executive summary, key contributions, methodology, results, limitations |
| **Research Gaps** | AI-powered detection of gaps, improvements, and future directions |
| **Paper Comparison** | Side-by-side structured comparison of two papers |
| **Citations** | Every answer cites source paper + page number + relevance score |
| **Metrics** | Retrieval time, LLM time, chunks used, pages referenced |
| **Deduplication** | SHA-256 hash prevents re-indexing the same PDF |
| **Conversation Memory** | Recent chat history injected into RAG context |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   Retriever  │────▶│    FAISS    │
│     UI      │     │  (Top-K=5)   │     │ Vector Store│
└─────────────┘     └──────────────┘     └─────────────┘
     │                      │
     │              Retrieved Chunks
     │                      │
     ▼                      ▼
┌─────────────┐     ┌──────────────┐
│  RAG Chain  │────▶│  LLM (Llama3 │
│   Prompt    │     │  / Mistral)  │
│  Builder   │     └──────────────┘
└─────────────┘             │
                      Grounded Answer
                      + Citations
```

### Folder Structure

```
ResearchMate/
│
├── app.py                  # Streamlit UI — main entry point
├── config.py               # All configuration constants
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── data/
│   ├── uploads/            # Saved PDF files
│   └── vectorstore/        # Persisted FAISS index
│
├── modules/
│   ├── pdf_loader.py       # PDF validation & text extraction (PyMuPDF)
│   ├── text_splitter.py    # RecursiveCharacterTextSplitter chunking
│   ├── embeddings.py       # HuggingFace embedding model (BGE)
│   ├── vector_db.py        # FAISS vector store CRUD
│   ├── retriever.py        # Top-K similarity retrieval
│   ├── rag_chain.py        # RAG prompt builder + LLM integration
│   ├── summarizer.py       # Automated paper summarization
│   └── comparison.py       # Multi-paper comparison + gap detection
│
└── utils/
    └── helpers.py          # Shared utility functions
```

---

## ⚡ Installation

### Prerequisites

- Python 3.10+
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ResearchMate.git
cd ResearchMate
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the BGE embedding model (~130 MB). This is cached locally for subsequent runs.

### Step 4: Set Up Your LLM

#### Option A — Ollama (Recommended, Free, Local)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3 (8B model, ~4.7 GB)
ollama pull llama3

# OR use Mistral 7B
ollama pull mistral
```

Ollama runs automatically on `http://localhost:11434`.

#### Option B — HuggingFace Hub (Free API Tier)

```bash
cp .env.example .env
# Edit .env and set:
# LLM_BACKEND=huggingface
# HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Step 5: Launch ResearchMate

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage Guide

### 1. Upload Papers

- Click **Upload PDFs** in the sidebar
- Select one or more research papers (.pdf)
- ResearchMate automatically extracts, chunks, embeds, and indexes them

### 2. Chat with Papers

Navigate to the **💬 Chat with Papers** tab.

**Example questions:**
```
What is the methodology used in this paper?
What dataset was used for training and evaluation?
What model architecture is proposed?
What are the key contributions?
What are the limitations of this approach?
What future work do the authors suggest?
What evaluation metrics were used?
Summarize the paper in simple terms.
What results were achieved on the benchmark?
```

**Multi-paper queries:**
```
Compare Paper A and Paper B on methodology.
Which paper achieves better performance on ImageNet?
What is the common limitation across all papers?
```

### 3. Auto-Summary

Go to **📝 Paper Summary** → select a paper → click **Generate Summary**.

Generates structured sections:
- Executive Summary
- Key Contributions  
- Methodology
- Results
- Limitations

### 4. Research Insights

Go to **🔭 Research Insights** → select a paper → click **Detect Research Gaps**.

Outputs:
- Identified research gaps
- Potential improvements
- Future directions
- Open questions

### 5. Paper Comparison

Go to **⚖️ Paper Comparison** → select two papers → click **Compare Papers**.

Compares across:
- Problem statement & motivation
- Methodology & technical approach
- Key contributions & novelty
- Results & evaluation
- Limitations & future work

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Chunking
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between chunks

# Retrieval
TOP_K_RETRIEVAL = 10        # chunks retrieved per query

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Alternative: "sentence-transformers/all-MiniLM-L6-v2"

# LLM
LLM_BACKEND = "ollama"     # "ollama" | "huggingface"
OLLAMA_MODEL = "llama3"    # any model pulled in Ollama
```

---

## 📊 Evaluation Metrics

ResearchMate displays for every response:

| Metric | Description |
|---|---|
| **Retrieval Time** | Time to retrieve top-K chunks from FAISS |
| **LLM Time** | Time for the model to generate the answer |
| **Chunks Retrieved** | Number of context chunks used |
| **Source Pages** | Which pages the answer references |
| **Similarity Score** | Cosine similarity of each retrieved chunk |

---

## 🚀 Deployment

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository, set `app.py` as the entry point
4. Add secrets under **Settings → Secrets**:
   ```
   HUGGINGFACEHUB_API_TOKEN = "hf_xxx"
   LLM_BACKEND = "huggingface"
   ```
5. Deploy!

> Note: Use HuggingFace backend for cloud deployment (Ollama requires a local server).

### HuggingFace Spaces

1. Create a new Space with **Streamlit** SDK
2. Upload all project files
3. Add `HUGGINGFACEHUB_API_TOKEN` as a Space Secret
4. Update `config.py`: set `LLM_BACKEND = "huggingface"`

### Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect GitHub repo
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variables in the Render dashboard

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit 1.35+ |
| **PDF Extraction** | PyMuPDF (fitz) |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace) |
| **Vector DB** | FAISS (CPU) |
| **LLM (local)** | Llama 3 / Mistral via Ollama |
| **LLM (API)** | Mistral-7B via HuggingFace Hub |
| **Orchestration** | LangChain |
| **Language** | Python 3.10+ |

---

## 🧪 Development

### Running Tests

```bash
pip install pytest
pytest tests/
```

### Code Style

```bash
pip install black isort
black .
isort .
```

### Adding a New LLM Backend

1. Add your backend function in `modules/rag_chain.py`:
   ```python
   def _build_my_llm():
       from langchain_community.llms import MyLLM
       return MyLLM(...)
   ```
2. Add to `_build_llm()` dispatch logic
3. Update `config.py` with your backend name and settings

---

## ❓ FAQ

**Q: The embedding model download is slow. Can I use a cached version?**  
A: Yes. HuggingFace caches models in `~/.cache/huggingface/`. On first run it downloads once, subsequent runs use the cache.

**Q: Can I use OpenAI GPT-4 instead of Llama?**  
A: Yes. Add a new backend in `rag_chain.py` using `langchain_openai.ChatOpenAI` and set `OPENAI_API_KEY` in your `.env`.

**Q: How many papers can I index?**  
A: FAISS is highly scalable. Hundreds of papers with thousands of chunks each will work fine on CPU. For very large collections, consider using FAISS GPU or a managed vector DB like Pinecone.

**Q: What if the answer says "I could not find this information"?**  
A: The relevant content wasn't in the top-5 retrieved chunks. Try rephrasing your question or increasing `TOP_K_RETRIEVAL` in `config.py`.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) for the RAG orchestration framework  
- [FAISS](https://faiss.ai) by Meta AI for efficient vector search  
- [PyMuPDF](https://pymupdf.readthedocs.io) for fast PDF processing  
- [BAAI](https://huggingface.co/BAAI) for the BGE embedding models  
- [Ollama](https://ollama.ai) for local LLM serving  
- [Streamlit](https://streamlit.io) for the UI framework  

---

*Built with ❤️ for researchers and students who want to understand papers faster.*
=======
# Researchmate-ai
🔬 ResearchMate AI – Intelligent Research Paper Assistant powered by RAG, Hybrid Search, and LLM. Upload PDFs, chat, summarize, find research gaps, compare papers, and generate novel research ideas.
>>>>>>> 1ba3a91cc3317230708092dd1209a620ab48df55
