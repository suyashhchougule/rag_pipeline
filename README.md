# RAG Pipeline 📚🔍

A **modular retrieval-augmented generation (RAG)** pipeline that ingests documents (PDF, images), builds vector indexes, and exposes a FastAPI endpoint for querying via both **simple RAG** and **agentic RAG** architectures.

---

## 🚀 Features

* 🔄 **Modular ingestion pipeline**

  * Converts PDF or image documents to Markdown using ByteDance Dolphin (Vision-Language Model)
  * Normalizes HTML tables, splits into sections, chunks into parent and sentence-level documents
  * Skips re-ingestion of unchanged files using SHA-256 file tracking
  * Stores parent chunks in SQLite and maintains FAISS indexes for retrieval
    
* 🤖 **Dual RAG architectures**

  * Simple RAG: Direct retrieval + generation pipeline with rate-limiting and token logging
  * Agentic RAG: LangGraph-powered multi-agent system with question decomposition and orchestration
  * Unified API endpoint with rag_type parameter to switch between architectures
  * Comprehensive metadata collection for both approaches

* 🎯 **Agent-based intelligence**

  * Planner Agent: **Decomposes complex questions into focused sub-questions and gives detailed answers**
  * Retriever Agent: Performs multi-level document retrieval with context gathering
  * Supervisor: Orchestrates agents and synthesizes final structured responses
  * Built on LangGraph with ReAct agent framework for reliable multi-step reasoning
 
* 📊 **Production-ready API**
  * FastAPI endpoints: /query, /health, /stats, /info
  * Comprehensive error handling and input validation
  * Structured JSON responses with detailed metadata

* 📦 **DevOps-ready**

  * Quick setup UV.
  * Configurable via TOML (`dynaconf`)
  * Logging to file and console with a structured format
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
uv sync
source .venv/bin/activate
```

---

### 2. Configure

1. **Create your Cerebras API key**  
   - Go to the [Cerebras Cloud platform](https://cloud.cerebras.ai/platform/org_4myhnjpmnjyx3mrjt3nwmy33/apikeys):  
   - Log in (or sign up if you don’t have an account).  
   - Click **“Create API Key”**, copy the generated key.
   - Please add GPT-4o or any reasoning model API key and corresponding credentials as well for the agent RAG to work.

2. **Populate your secrets file**  
   ```bash
   config/.secrets.toml
3. Configure `config/RagConfig.toml`, if required. 

### 3. Run Ingestion

Download Dolphin Model

```bash
huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model

python ingestion_pipeline/ingest.py --input_folder ./docs
```

This will parse your files, chunk them, build/update vector indexes, and store everything.

### 4. Launch the API

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then query via HTTP:

```bash
# Get system info
curl http://localhost:8000/api/v1/info

# Health check for simple RAG
curl "http://localhost:8000/api/v1/health?rag_type=simple"

# Health check for agentic RAG
curl "http://localhost:8000/api/v1/health?rag_type=agentic"

# Query simple RAG
curl -X POST "http://localhost:8000/api/v1/query?rag_type=simple" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Affinity fraud?", "k_sent": 20, "k_parent": 10, "max_return": 10}'

# Complex Query handling with agentic RAG  
curl -X POST "http://localhost:8000/api/v1/query?rag_type=agentic" \
     -H "Content-Type: application/json" \
     -d '{"question": "List the different types of frauds", "k_sent": 20, "k_parent": 10, "max_return": 10}'

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

## Agentic Architecture (LangGraph + ReAct)
For complex queries requiring decomposition, the system employs a **multi-agent architecture** powered by **LangGraph**:

### Agent Roles
| Agent | Purpose | Tools |
|-------|---------|-------|
| **Planner Expert** | Breaks down complex questions into sub-questions | `plan_sub_questions` |
| **Retriever Expert** | Fetches context using multi-level retrieval | `multi_level_retrieve` |
| **Supervisor** | Orchestrates agents and synthesizes final answers | Agent coordination |

### LangGraph Supervisor Pattern
- **Orchestration**: Supervisor manages agent handoffs and maintains conversation state
- **Tool Integration**: Each agent has specialized tools for their domain expertise  
- **Structured Output**: All agents use JSON mode for reliable inter-agent communication
- **Error Handling**: Graceful fallbacks when agents encounter issues

### Benefits Over Simple RAG
- **Complex Query Handling**: Decomposes multi-part questions automatically
- **Better Context Coverage**: Each sub-question gets dedicated retrieval attention
- **Reasoning Transparency**: Agent interactions provide explainable decision paths
- **Adaptive Processing**: Different strategies for simple vs. complex queries

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

**10. Multi-Agent RAG with LangGraph Orchestration**  
Agentic RAG implementation using LangGraph's supervisor pattern with specialized ReAct agents. The planner agent decomposes complex queries, the retriever agent handles context gathering with domain expertise, and the supervisor orchestrates the workflow while maintaining conversation state and synthesizing final answers.


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
