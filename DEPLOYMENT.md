# ResearchMate Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Streamlit Cloud](#streamlit-cloud)
3. [HuggingFace Spaces](#huggingface-spaces)
4. [Render](#render)
5. [Docker](#docker)
6. [Environment Variables Reference](#environment-variables-reference)

---

## Local Development

```bash
# Clone and set up
git clone https://github.com/yourusername/ResearchMate.git
cd ResearchMate
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install Ollama + pull model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3

# Run
streamlit run app.py
# → http://localhost:8501
```

---

## Streamlit Cloud

**Cost:** Free tier available  
**LLM:** Must use HuggingFace Hub (Ollama requires a local server)

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/ResearchMate.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   - Click **New app** → connect your repo
   - Main file path: `app.py`
   - Python version: `3.11`

3. **Set Secrets** (Settings → Secrets)
   ```toml
   HUGGINGFACEHUB_API_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
   LLM_BACKEND = "huggingface"
   ```

4. **Update config.py for cloud**
   ```python
   LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface")
   HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
   ```

5. Click **Deploy** ✅

---

## HuggingFace Spaces

**Cost:** Free CPU tier  
**LLM:** HuggingFace Hub (same account, free inference API)

### Steps

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Space name: `ResearchMate`
3. SDK: **Streamlit**
4. Upload all project files

5. Create `app.py` entry point (already done)

6. Add Secrets (Settings → Variables and Secrets)
   ```
   HUGGINGFACEHUB_API_TOKEN = hf_xxx
   LLM_BACKEND = huggingface
   ```

7. Add `packages.txt` for system dependencies:
   ```
   libgl1-mesa-glx
   ```

8. Push and Space will auto-build ✅

---

## Render

**Cost:** Free tier (750 hrs/month)  
**LLM:** HuggingFace Hub

### Steps

1. Sign up at [render.com](https://render.com)
2. **New → Web Service** → connect GitHub repo

3. Configure:
   | Field | Value |
   |---|---|
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
   | Environment | `Python 3` |

4. Environment Variables:
   ```
   HUGGINGFACEHUB_API_TOKEN = hf_xxx
   LLM_BACKEND = huggingface
   ```

5. Click **Create Web Service** ✅

---

## Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false"]
```

```bash
# Build and run
docker build -t ResearchMate .
docker run -p 8501:8501 \
  -e HUGGINGFACEHUB_API_TOKEN=hf_xxx \
  -e LLM_BACKEND=huggingface \
  ResearchMate
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `"ollama"` or `"huggingface"` |
| `HUGGINGFACEHUB_API_TOKEN` | — | Required for HuggingFace backend |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model name in Ollama |

---

## Performance Tips

- **GPU acceleration**: Replace `faiss-cpu` with `faiss-gpu` in requirements.txt
- **Faster embeddings**: Set `EMBEDDING_DEVICE = "cuda"` in config.py
- **Larger model**: Use `ollama pull llama3:70b` for better accuracy (requires ~40 GB RAM)
- **Production FAISS**: For thousands of papers, use Pinecone, Weaviate, or Chroma instead of FAISS
