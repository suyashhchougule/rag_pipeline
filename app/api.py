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
import sys
BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config" / "RagConfig.toml"
SECRETS_PATH = BASE_DIR / "config" / ".secrtets.toml"
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
retriever = MultiLevelRetriever(s_index, p_index, docstore)
limiter = RateLimiter(rpm=RPM, tpm=TPM)
token_logger = SimpleTokenLogger()
generator = OpenAIChatGenerator(
    config=cfg,
    rate_limiter=limiter,
    token_logger=token_logger,
)

app = FastAPI()

class ChunkInfo(BaseModel):
    chunk_id: str
    summary: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None

class QueryRequest(BaseModel):
    question: str
    k_sent: Optional[int] = 20
    k_parent: Optional[int] = 10
    max_return: Optional[int] = 10

class QueryResponse(BaseModel):
    answer: dict
    context_chunks: List[ChunkInfo]
    meta: Dict[str, Any]

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    log.info(f"Received query: {req.question}")
    try:
        parent_chunks = retriever.retrieve(
            req.question,
            k_sent=req.k_sent,
            k_parent=req.k_parent,
            max_return=req.max_return
        )
        log.info(f"Retrieved {len(parent_chunks)} parent chunks for query.")
        if not parent_chunks:
            log.warning("No context found for query.")
            raise HTTPException(status_code=404, detail="No relevant context found.")

        # Now parent_chunks is a list of dicts: {"doc": ..., "score": ...}
        chunk_infos = []
        for item in parent_chunks:
            doc = item["doc"]
            score = item.get("score")
            summary = doc.page_content #[:120].replace('\n', ' ') + ('...' if len(doc.page_content) > 120 else '')
            chunk_infos.append(ChunkInfo(
                chunk_id=doc.metadata.get('uid', ''),
                summary=summary,
                metadata=doc.metadata,
                score=score,
            ))

        context_blob = "\n---\n".join(item["doc"].page_content for item in parent_chunks)
        prompt = PROMPT_TMPL.format(context=context_blob, question=req.question)

        print(context_blob)

        answer = generator.generate_response(
            text=prompt,
            response_format="json_object",
            model_name=cfg.DEFAULT.GENERATOR_MODEL
        )

        log.info("LLM generation complete for query.")

        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except Exception:
                answer = {"answer": answer}

        meta = {
            "embedding_model": EMBED_MODEL,
            "generator_model": cfg.DEFAULT.GENERATOR_MODEL,
            "retriever": retriever.__class__.__name__,
            "prompt_template": PROMPT_TMPL,
            "num_context_chunks": len(chunk_infos)
        }

        return QueryResponse(answer=answer, context_chunks=chunk_infos, meta=meta)
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail=str(e))

