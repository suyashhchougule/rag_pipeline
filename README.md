# RAG Pipeline 📚🔍

A **modular retrieval-augmented generation (RAG)** pipeline that ingests documents (PDF, images), builds vector indexes, and exposes a FastAPI endpoint for querying via a chat-based LLMs.

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

  * Quick setup UV.
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
│   └── .secrtets.toml                          # template for Secrets (API keys, tokens)
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
```

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.10+
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
3. Configure `config/RagConfig.toml`, if required. 

### 3. Run Ingestion

Download Dolphin Model

```huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model```


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
###  Architecture Overview: 

## 📥 Ingestion & Document Understanding

Uses **[ByteDance Dolphin](https://huggingface.co/ByteDance/Dolphin)** ([GitHub](https://github.com/bytedance/Dolphin)), a state-of-the-art vision–language model that converts both **scanned** *and* **digital PDFs** into richly structured Markdown including headings, sections, paragraphs, tables, figures.

* **Why Dolphin?**  
  *Heterogeneous-Anchor Prompting* lets the model detect layout first, then extract text, giving higher fidelity than single-pass VLMs.  
* **Benefits for RAG**  
  - Eliminates the OCR vs. digital PDF fork.  
  - Emits clean block-level structure, which downstream chunkers can trust.  
  - Handles noisy scans without template tuning.

The extractor runs once per document during ingestion; unchanged files are skipped via SHA-256 digests, keeping re-ingestion idempotent.

**Alternatives evaluated**

|  | 
|-----------|
| **docling** — <https://github.com/docling-project/docling> | 
| **MinerU** — <https://github.com/opendatalab/MinerU> |


---


## 🔪 Chunking Strategy

We combine **hierarchical** *and* **structure-aware** chunking because no single split size works for all queries.

| Level            | Method                     | Size / Overlap          | Purpose                        |
|------------------|----------------------------|-------------------------|--------------------------------|
| **Sentence**     | Sentence splitter          | ≈ 1–2 sentences, 0 ovl  | Precise keyword hits           |
| **Parent**       | Recursive splitter         | **2048 tokens**, 128 ovl| Larger context               |
| **Markdown**     | Header-aware splitter      | `<h1…h6>` boundaries    | Preserve author organisation   |

1. **Hierarchical Re-ranker Retriever (HRR)**  
   HRR reranks intermediate-size chunks against the query, then surfaces their parent context, lifting recall without flooding the prompt.  
   *Reference:* “Hierarchical Retrieval for RAG” ([arXiv 2503.02401](https://arxiv.org/pdf/2503.02401)).

2. **Document-Structure-Aware Chunking**  
   When Markdown headings exist, we split on those natural boundaries instead of fixed windows, boosting retrieval accuracy by **5–10 %** in finance & tech docs.  
   *References:*  
   • [Snowflake Engineering – Impact of Retrieval Chunking in Finance RAG](https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/)  
   • [Milvus – What Chunking Strategies Work Best?](https://milvus.io/ai-quick-reference/what-chunking-strategies-work-best-for-document-indexing)

> **Trade-off:** Hierarchical splitting increases index size, but HRR avoids extra LLM calls, so latency stays < 500 ms. Structure-aware splits fall back to recursive rules when documents lack headings, ensuring robustness.

## 🧬 Embeddings & Vector Index

### Embedding Model  

Embeds both queries and chunks with **[Qwen 3 Embedding 0.6 B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** — a 600 M-parameter model that currently sits **4ᵗʰ on the [MTEB leaderboard](<https://huggingface.co/spaces/mteb/leaderboard>)**.  
The model offers near-state-of-the-art semantic retrieval quality while remaining lightweight enough to run on CPU or a single consumer GPU, making it a pragmatic “minimal yet robust” choice.


### FAISS Index  

Uses **FAISS** for its zero-friction setup and tight local latency.  
Because the retriever is abstracted behind a thin `VectorStore` interface, FAISS can be swapped for a production-grade service (Azure Search, OpenSearch, etc.).

**Hierarchical Re-ranker support**  
Maintains **two indexes**:  

| Index | Granularity | Purpose |
|-------|-------------|---------|
| `faiss_sentence.index` | **Sentence-level embeddings** | High-precision keyword hits and their corresponding parent retrieval|
| `faiss_parent.index`   | **Parent (≈2 048-token) embeddings** | Broader context for answer grounding |

This dual-index layout powers the **Hierarchical Re-ranker Retriever (HRR)**:  

## 🔍 Retrieval Strategy (Multi-Level)

Uses a **two-index, multi-level retrieval flow** inspired by HRR:

1. **Parallel search**  
   * Query the **sentence index** for the top-**15** hits (`k_sentence = 15`).  
   * Query the **parent index** for the top-**15** hits (`k_parent = 15`).

2. **Parent mapping & merge**  
   * Each sentence-level hit is mapped to its `parent_id`.  
   * Direct parent hits keep their own `uid`.  
   * We retain the **single best similarity score** per parent.

3. **Score sort & cut-off**  
   * Merge the two parent lists.  
   * Sort descending by cosine similarity.  
   * Return the first **`max_return` parent chunks** (configurable, default **5**).

4. **Context assembly**  
   * Load the selected parent chunks from the SQLite doc-store.  
   * Pass them — plus their similarity scores — to the prompt builder.

> **Why this works**  
> • Sentence search captures fine-grained matches; parent search surfaces broader context.  
> • De-duplication keeps one score per parent, preventing prompt spam.  

## 📝 LLM & Prompt Design

### Generator Abstraction  
`app/generator.py` is a thin wrapper that can call **any OpenAI-compatible endpoint** by reading `endpoint_url` and `api_key` from the TOML config. Swap in Azure OpenAI, Google Vertex, Hugging Face Inference Endpoints, vLLM, or a self-hosted model without touching application code.

| Default model (demo) | Provider | Params | Endpoint |
|----------------------|----------|--------|----------|
| `llama-4-scout-17b-16e-instruct` | Cerebras | 17 B | `https://api.cerebras.ai/v1` |

### Rate Limiting  
A class-level **RateLimiter** enforces requests-per-minute *and* tokens-per-minute ceilings.  
Benefits:  
* Prevents hard 429s / quota bans.  
* Smooths traffic spikes (protects SLAs).  
* Caps runaway cost in pay-as-you-go billing.  
Values are configurable from config.

### JSON-Mode Support  
When the upstream model advertises “`mode: json`”, the generator sets `response_format={"type":"json_object"}`.  
*Guarantees* syntactically valid JSON, easing downstream parsing and evals.

### Prompt Template (COSTAR-inspired)
Uses **[COSTAR Prompt Technique](https://medium.com/@frugalzentennial/unlocking-the-power-of-costar-prompt-engineering-a-guide-and-example-on-converting-goals-into-dc5751ce9875)**
```text
## Persona
You are a helpful assistant trusted for accurate, reference-backed answers.

## Instruction
1. Answer the question **only** with facts you can locate in <context>.  
2. If the context does **not** contain the answer, reply exactly with “Insufficient context".  
3. Use complete sentences and provide enough detail for clarity.  
4. Do not invent or add information that is not present in the context.

## Context
<context>
{context}
</context>

## Question
{question}

## Tone
Adopt a concise and helpful tone, informative yet approachable.

## Audience
Respond to audience who may not share your background knowledge; keep jargon minimal.

## Output Format
Return a valid JSON object with these keys **and no additional keys**:

json
{{
  "answer": "<your best answer or '“Insufficient context'>",
}}
```

## 🔌 API Reference

### `POST /query`

#### Request body
| Field      | Type | Default | Description |
|------------|------|---------|-------------|
| `question` | `str` | — (required) | Natural-language query |

> Send `Content-Type: application/json`.

#### Successful response `200`
```jsonc
{
  "answer": {                     // JSON answer from the LLM
    "answer": "Affinity fraud is …"
  },
  "context_chunks": [             // Evidence passed to the LLM
    {
      "chunk_id": "chunk uid",
      "summary": "…text of the chunk…",
      "metadata": { "page": 7 },
      "score": 0.84
    }
  ],
  "meta": {                       // Trace metadata
    "embedding_model": "Qwen3-Embedding-0.6B",
    "generator_model": "llama-4-scout-17b-16e-instruct",
    "retriever": "MultiLevelRetriever",
    "prompt_template": "PROMPT_TMPL",
    "num_context_chunks": 3
  }
}
```

| Status | Condition                     | Body example                                 |
| ------ | ----------------------------- | -------------------------------------------- |
| `404`  | No relevant context found     | `{ "detail": "No relevant context found." }` |
| `500`  | Retriever / prompt / LLM fail | `{ "detail": "Generator error: …" }`         |

## 📈 Observability & Logging

| Aspect | Implementation |
|--------|----------------|
| **Library / format** | Python `logging` with a **structured plain-text formatter**<br>`"%(asctime)s %(levelname)s %(name)s %(message)s"` |
| **Per-request fields** | • timestamp • level • logger name • query text • retrieved-chunk count • token in/out • latency ms • exception detail |
| **Sinks** | 1. **stdout** (for Docker/ Kubernetes log scraping)<br>2. Rotating files: `rag_pipeline.log` (ingestion/runtime) and `rag_pipeline_api.log` (FastAPI endpoint) |
```
Example entry:
2025-07-17 15:21:20,635 INFO generator setting generation model (llama-4-scout-17b-16e-instruct)
2025-07-17 15:21:27,453 INFO api Received query: What are common investment scams on  social media?
2025-07-17 15:21:28,110 INFO api Retrieved 10 parent chunks for query.
2025-07-17 15:19:09,809 INFO httpx HTTP Request: POST https://api.cerebras.ai/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-17 15:19:09,859 INFO generator Model response (CEREBRAS_LLAMA4) in 0.96s
2025-07-17 15:19:09,859 INFO tokenlogger Token usage - prompt: 2097, completion: 126
2025-07-17 15:19:09,859 INFO api LLM generation complete for query.
```
## 🔐 Security & Secrets

| **Aspect**           | **Implementation**                                         | **Notes**                                                                                                                                     |
|----------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Config Loader**     | **Dynaconf** – Layered settings                             | Loads from `config/RagConfig.toml` (defaults) → `config/.secrets.toml` (runtime) → environment variables                                      |
| **Secrets File**      | **Template Only** committed: `config/.secrets.template.toml` | Copy and rename to `.secrets.toml`, then fill in secrets such as `API_KEY`, etc.                                                              |
| **Runtime Overrides** | Environment variables                                       | All secret keys (e.g., `OPENAI_API_KEY`, `ENDPOINT_URL`, etc.) can be injected via CI/CD pipelines or `docker-compose`                      |
| **Logging Hygiene**   | Secure logging practices                                    | Secrets are never logged; only safe metrics such as token counts and latency are recorded                                                    |
> Only the **template** for secrets is tracked in Git; actual credentials stay local or in your secret-manager of choice.

## 🚀 Advanced Features Implemented

**1. Vision-Language Ingestion, not just OCR**  
ByteDance Dolphin converts both scanned and native PDFs into structured Markdown, preserving tables, figures, and headings. That means the very first ingestion run already handles mixed document types without two separate pipelines.

**2. Hierarchical + document-structure-Aware Chunking**  
Sentence → parent token hierarchy captures micro and macro context, while header-aware splitting keeps logical sections intact. This hybrid approach mirrors the HRR paper’s recall gains without requiring an extra LLM re-rank call.

**3. Dual-Index Retrieval with Score-Level Fusion**  
A sentence-level FAISS index surfaces pinpoint matches; a parent-level index supplies broader context. At query time their scores are fused, de-duplicated on `parent_id`, and only the top five parents flow to the prompt—achieving high recall and low prompt cost.

**4. Generator Wrapper for “OpenAI Compatible APIS”**  
`OpenAIChatGenerator` can hit *any* OpenAI-compatible endpoint (OpenAI, Azure, Vertex, vLLM, Cerebras) by flipping two config keys. In the demo it calls the free **llama-4-scout-17B-16e** model on the Cerebras platform.

**5. Strict JSON-Mode Output**  
When the upstream model supports it, requests include `response_format={"type":"json_object"}`. That guarantees JSON-parsable answers and eliminates brittle regex post-processing.

**6. RateLimiter with RPM *and* TPM Controls**  
A class-level limiter enforces both requests-per-minute and tokens-per-minute, protecting upstream quotas, smoothing traffic spikes, and capping cost exposure—all configurable at runtime.

**7. Token Logger for Usage Analytics**  
`SimpleTokenLogger` records prompt and completion token counts per request, enabling accurate cost tracking and future per-tenant billing.

**8. Layered Secret Management**  
Only a **template** secrets file ships with the repo; real credentials live in a git-ignored `.secrets.toml` or environment variables. Dynaconf merges defaults → secrets → envs, so CI and Docker deployments stay credential-free.

**9. Structured Logging Ready for ELK/Grafana**  
Every log line carries a timestamp, level, module, query, chunk counts, similarity scores, latency, and token usage—already formatted for JSON ingestion while still human-readable.

## 🛠️ Next-Level Enhancements for Production, Scale, and Efficiency
---

### 1. Retrieval & Indexing
* **Cross-Encoder Re-ranking for Precision Boost**  
  Integrate a lightweight cross-encoder such as **Cohere Rerank** or **bge-reranker-large** after the initial out Multi-level retrieval step.  
  This allows re-ranking the top-30 dense hits with a task-specific relevance model, significantly improving answer precision, especially in long-form, multi-hop queries.
* **Hybrid Retrieval (BM25 + Vectors)**  
  Blend lexical BM25 scores with dense embeddings to handle rare terms and spelling variants without sacrificing semantic recall.
  
### 2. Parallelised Ingestion with Batched VLM Inference  
* Improve throughput by parallelising the ingestion pipeline to handle multiple documents concurrently using `ray` tool.  
* Additionally, AS ByteDance Dolphin VLM has [vllm serving support,](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md) We can serve it with **vLLM batched inference** for PDF text extraction, reducing latency and improving GPU utilisation.

### 3. Model serving
* **Local vLLM / SGLang Inference**  
  Serve the LLM in-house to avoid network hops and cut the cost per million tokens.
* We can host our models serverlessly on platforms like `Rupod` and `Google Cloud Run` to save always-on-GPU-VM cost.

### 4. Evaluation
For evaluation, we can use metrics like MRR, precision, and recall to evaluate Retrieval. 
For Generation eval, we can use the LLM‑as‑a‑judge framework. These evaluation methods can be implemented with custom scripts or by leveraging tools such as LangSmith or tools mentioned below.
* **Automated RAG Evaluation**  
  Run RAGAS/Bench-RAG on a synthetic Q-A set every commit; fail the build if precision/recall regress.
* **Canary & Blue-Green Deployments**  
  Roll new embedding models or prompt variants to 5 % traffic; monitor accuracy before 100 % cut-over.
* **Prompt Registry & Versioning**  
  Store templates in a git-tracked catalogue; tie each model release to a prompt hash for reproducibility.

### 5. Security & Compliance
* **API Key + RBAC**  
  Enforce per-tenant quotas and restrict document access scope.
---

> Used **GitHub Copilot** and **ChatGPT-4** for quick code refactors and debugging hints, Documentation.

## Acknowledgements

Powered by:

* ByteDance **Dolphin** (VLM)
* FAISS
* FastAPI
* Opens source **LLaMA**–like models
---
