# RAG Pipeline 📚🔍

A **modular retrieval-augmented generation (RAG)** pipeline that ingests documents (PDF, images), builds vector indexes using FAISS, and exposes a FastAPI endpoint for querying via a chat-based LLM.

---

## 🚀 Features

* 🔄 **Modular ingestion pipeline**

  * Converts PDF or image documents to Markdown using ByteDance Dolphin (Vision-Language Model)
  * Normalizes HTML tables, splits into sections, chunks into parent and sentence-level documents
  * Skips re-ingestion of unchanged files using SHA-256 file tracking
  * Stores parent chunks in SQLite and maintains FAISS indexes for retrieval
* 🤖 **Retrieval and generation**

  * FastAPI `/query` endpoint that retrieves context and calls LLM for structured answers
  * Supports rate-limiting and token logging for OpenAI-compatible LLMs
* 📦 **DevOps-ready**

  * Fully containerizable via Docker
  * Configurable via TOML (`dynaconf`)
  * Logging to file and console with a structured format

---

## 📁 Repository Structure

```text
├── README.md                                   # Project overview and setup
├── app                                         # API & core runtime
│   ├── api.py                                  # FastAPI entry‑point
│   ├── core/
│   │   └── generator.py                        # Generic abstraction class for LLM API Calls
│   ├── docstore.py                             # SQLite doc blob store
│   ├── generator.py                            # (wrapper for generator)
│   ├── index_loader.py                         # Load FAISS indexes
│   ├── main.py                                 # CLI/demo
│   ├── prompt_templates.py                     # Prompt templates for RAG/LLM
│   ├── ratelimiter.py                          # Request/token limiter
│   ├── retriever.py                            # Multi-level retriever logic
│   └── tokenlogger.py                          # Token usage tracking
├── config/
│   ├── RagConfig.toml                          # Main settings (paths, models)
│   └── .secrtets.toml                          # Secrets (API keys, tokens)
├── docs/                                       # Sample docs to ingest
├── hf_model/                                   # Local Dolphin VLM files
├── ingestion_pipeline/                         # Ingestion & chunking
│   ├── chunker.py
│   ├── document_parser_save_dir/               # Saved markdown & images
│   ├── documentparser.py
│   ├── filetracker.py
│   ├── indexer.py
│   ├── ingest.py
│   ├── ingestion_databases/                    # SQLite + FAISS data
│   ├── loader.py
│   ├── store.py
│   └── utils/
├── pyproject.toml                              # Build & dependency file
├── Dockerfile                                  # Docker build recipe
```

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.10+
* Docker (optional)
* CUDA (for GPU acceleration)

### 1. Clone & Install

Quick setup with [uv](https://github.com/astral-sh/uv) package manager:

```bash
# Install uv globally (one‑time)
pip install uv

# Clone repo
git clone https://github.com/suyashhchougule/rag_pipeline.git
cd rag_pipeline

# Create env & install deps in one go
uv venv
source .venv/bin/activate
uv sync
```

---

### 2. Configure

1. **Create your Cerebras API key**  
   - Go to the [Cerebras Cloud platform](https://cloud.cerebras.ai/platform/org_4myhnjpmnjyx3mrjt3nwmy33/apikeys):  
   - Log in (or sign up if you don’t have an account).  
   - Click **“Create API Key”**, copy the generated key.

2. **Populate your secrets file**  
   ```bash
   config/.secrets.toml
3. Configure config/RagConfig.toml, if required. 

### 3. Run Ingestion

```bash
python ingestion_pipeline/ingest.py --input_folder ./docs
```

This will parse your files, chunk them, build/update vector indexes, and store everything.

### 4. Launch the API

```bash
cd app
uvicorn app.api:app --reload --port 8000
```

Then query via HTTP:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{ "question": "Define affinity fraud" }'
```

---

## 🐳 Docker Setup

### Build image:

```bash
docker build -t rag-api .
```

### Run container (Option A: built-in model download):

```bash
docker run -p 8000:8000 rag-api
```
---
## ❤️ Acknowledgements

Powered by:

* ByteDance **Dolphin** (VLM)
* FAISS
* FastAPI
* Opens AI-compatible **LLaMA**–like models
* Docker

---
