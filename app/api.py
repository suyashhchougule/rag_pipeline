from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import logging
from pathlib import Path
from dynaconf import Dynaconf

from ratelimiter import RateLimiter
from tokenlogger import SimpleTokenLogger
from generator import OpenAIChatGenerator
from retriever import MultiLevelRetriever
from docstore import DocStore
from prompt_templates import PROMPT_TMPL
from index_loader import load_indexes

# Setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config" / "RagConfig.toml"
SECRETS_PATH = BASE_DIR / "config" / ".secrtets.toml"
cfg = Dynaconf(settings_files=[CONFIG_PATH, SECRETS_PATH])

SENT_IDX_DIR = BASE_DIR / cfg.INGESTION.SENT_IDX_DIR
PARENT_IDX_DIR = BASE_DIR / cfg.INGESTION.PARENT_IDX_DIR
SQLITE_PATH = BASE_DIR / cfg.INGESTION.PARENTS_DB
EMBED_MODEL = cfg.INGESTION.EMBED_MODEL
RPM = int(cfg.RATELIMITER.RPM)
TPM = int(cfg.RATELIMITER.TPM)

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

class QueryRequest(BaseModel):
    question: str
    k_sent: Optional[int] = 20
    k_parent: Optional[int] = 10
    max_return: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: dict   # The LLM's JSON output (parsed)

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        # Step 1: Retrieve context
        parent_chunks = retriever.retrieve(
            req.question,
            k_sent=req.k_sent,
            k_parent=req.k_parent,
            max_return=req.max_return
        )
        if not parent_chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        context_blob = "\n---\n".join(chunk.page_content for chunk in parent_chunks)
        prompt = PROMPT_TMPL.format(context=context_blob, question=req.question)

        # Step 2: Generate answer
        answer = generator.generate_response(
            text=prompt,
            response_format="json_object",
            model_name=cfg.DEFAULT.GENERATOR_MODEL
        )

        # (Optional: if answer is a string, parse to dict)
        import json as pyjson
        if isinstance(answer, str):
            try:
                answer = pyjson.loads(answer)
            except Exception:
                answer = {"answer": answer, "citations": []}

        return QueryResponse(answer=answer)
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail=str(e))
