from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import logging
from pathlib import Path
from dynaconf import Dynaconf
from typing import List, Dict, Any, Optional
import json

from ratelimiter import RateLimiter
from tokenlogger import SimpleTokenLogger
from generator import OpenAIChatGenerator
from retriever import MultiLevelRetriever
from docstore import DocStore
from prompt_templates import PROMPT_TMPL
from index_loader import load_indexes
from core.pydantic_models import ChunkInfo, QueryRequest, QueryResponse
import sys

BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config" / "RagConfig.toml"
SECRETS_PATH = BASE_DIR / "config" / ".secrets.toml"
cfg = Dynaconf(settings_files=[CONFIG_PATH, SECRETS_PATH])

SENT_IDX_DIR = BASE_DIR / cfg.INGESTION.SENT_IDX_DIR
PARENT_IDX_DIR = BASE_DIR / cfg.INGESTION.PARENT_IDX_DIR
SQLITE_PATH = BASE_DIR / cfg.INGESTION.PARENTS_DB
EMBED_MODEL = cfg.INGESTION.EMBED_MODEL
LOGGER_FILE = cfg.LOGGER.API_FILE_PATH
RPM = int(cfg.RATELIMITER.RPM)
TPM = int(cfg.RATELIMITER.TPM)

def setup_logging():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    if log.hasHandlers():
        log.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    file_handler = logging.FileHandler(LOGGER_FILE, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    return logging.getLogger(__name__)

log = setup_logging()

# --- Load everything ONCE at startup ---
log.info("Initializing indexes, docstore, retriever, and generator...")
s_index, p_index = load_indexes(SENT_IDX_DIR, PARENT_IDX_DIR, EMBED_MODEL)
docstore = DocStore(SQLITE_PATH)
retriever = MultiLevelRetriever(
    s_index, 
    p_index, 
    docstore, 
    )
limiter = RateLimiter(rpm=RPM, tpm=TPM)
token_logger = SimpleTokenLogger()
generator = OpenAIChatGenerator(
    config=cfg,
    rate_limiter=limiter,
    token_logger=token_logger,
)

app = FastAPI()

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    log.info(f"Received query: {req.question}")
    try:
        # Input validation
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if req.k_sent <= 0 or req.k_parent <= 0 or req.max_return <= 0:
            raise HTTPException(status_code=400, detail="k_sent, k_parent, and max_return must be positive integers")
        
        try:
            parent_chunks = retriever.retrieve(
                req.question,
                k_sent=req.k_sent,
                k_parent=req.k_parent,
                max_return=req.max_return
            )
        except ValueError as e:
            log.error(f"Invalid retrieval parameters: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Retriever failed")
            raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)}")

        log.info(f"Retrieved {len(parent_chunks)} parent chunks for query.")
        if not parent_chunks:
            log.warning("No context found for query.")
            raise HTTPException(status_code=404, detail="No relevant context found.")

        chunk_infos = []
        for item in parent_chunks:
            try:
                doc = item["doc"]
                score = item.get("score")
                source = item.get("source", "unknown")
                summary = doc.page_content
                chunk_infos.append(ChunkInfo(
                    chunk_id=doc.metadata.get('uid', ''),
                    summary=summary,
                    metadata=doc.metadata,
                    score=score,
                    source=source
                ))
            except Exception as e:
                log.exception("Failed to process chunk info")
                continue

        try:
            context_blob = "\n---\n".join(item["doc"].page_content for item in parent_chunks)
            prompt = PROMPT_TMPL.format(context=context_blob, question=req.question)
        except Exception as e:
            log.exception("Failed to build prompt")
            raise HTTPException(status_code=500, detail=f"Prompt construction error: {str(e)}")
        
        try:
            answer = generator.generate_response(
                text=prompt,
                response_format="json_object",
                model_name=cfg.DEFAULT.GENERATOR_MODEL_SECTION
            )
        except Exception as e:
            log.exception("Generator failed")
            raise HTTPException(status_code=500, detail=f"Generator error: {str(e)}")

        log.info("LLM generation complete for query.")

        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except Exception:
                answer = {"answer": answer}

        # Enhanced metadata
        retriever_stats = retriever.get_retriever_stats()
        meta = {
            "embedding_model": EMBED_MODEL,
            "generator_model": cfg.DEFAULT.GENERATOR_MODEL_SECTION,
            "retriever": retriever.__class__.__name__,
            "similarity_type": retriever_stats.get("similarity_type", "unknown"),
            "sentence_index_vectors": retriever_stats.get("sentence_index_vectors", 0),
            "parent_index_vectors": retriever_stats.get("parent_index_vectors", 0),
            "prompt_template": PROMPT_TMPL,
            "num_context_chunks": len(chunk_infos),
            "retrieval_params": {
                "k_sent": req.k_sent,
                "k_parent": req.k_parent,
                "max_return": req.max_return
            }
        }

        return QueryResponse(answer=answer, context_chunks=chunk_infos, meta=meta)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add a new endpoint for retriever statistics
@app.get("/retriever/stats")
def get_retriever_stats():
    """Get statistics about the retriever indexes"""
    try:
        stats = retriever.get_retriever_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        log.exception("Failed to get retriever stats")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        stats = retriever.get_retriever_stats()
        return {
            "status": "healthy",
            "indexes_loaded": {
                "sentence_index": stats["sentence_index_vectors"] > 0,
                "parent_index": stats["parent_index_vectors"] > 0
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
