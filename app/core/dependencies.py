import logging
from pathlib import Path
from functools import lru_cache
from dynaconf import Dynaconf
from langchain_openai import ChatOpenAI

from retriever import MultiLevelRetriever
from docstore import DocStore
from index_loader import load_indexes
from generator import OpenAIChatGenerator
from ratelimiter import RateLimiter
from tokenlogger import SimpleTokenLogger
from rag.simple_rag import SimpleRAG
from rag.agentic_rag import AgenticRAG

log = logging.getLogger(__name__)

@lru_cache()
def get_config():
    """Get configuration - cached for performance"""
    BASE_DIR = Path(__file__).parent.parent.parent.resolve()
    return Dynaconf(
        settings_files=[
            BASE_DIR / "config" / "RagConfig.toml",
            BASE_DIR / "config" / ".secrets.toml",
        ]
    )

@lru_cache()
def get_shared_components():
    """Initialize shared components - cached as singletons"""
    cfg = get_config()
    BASE_DIR = Path(__file__).parent.parent.parent.resolve()
    
    # Paths
    SENT_IDX_DIR = BASE_DIR / cfg.INGESTION.SENT_IDX_DIR
    PARENT_IDX_DIR = BASE_DIR / cfg.INGESTION.PARENT_IDX_DIR
    SQLITE_PATH = BASE_DIR / cfg.INGESTION.PARENTS_DB
    EMBED_MODEL = cfg.INGESTION.EMBED_MODEL
    
    log.info("Loading indexes and docstore...")
    sent_index, par_index = load_indexes(SENT_IDX_DIR, PARENT_IDX_DIR, EMBED_MODEL)
    docstore = DocStore(SQLITE_PATH)
    retriever = MultiLevelRetriever(sent_index, par_index, docstore)
    
    return cfg, retriever

@lru_cache()
def get_simple_rag():
    """Get SimpleRAG instance - cached singleton"""
    cfg, retriever = get_shared_components()
    
    # Initialize generator components
    limiter = RateLimiter(rpm=int(cfg.RATELIMITER.RPM), tpm=int(cfg.RATELIMITER.TPM))
    token_logger = SimpleTokenLogger()
    generator = OpenAIChatGenerator(
        config=cfg,
        rate_limiter=limiter,
        token_logger=token_logger,
    )
    
    return SimpleRAG(cfg, retriever, generator)

@lru_cache()
def get_agentic_rag():
    """Get AgenticRAG instance - cached singleton"""
    cfg, retriever = get_shared_components()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=cfg.AGENT.MODEL_ID,
        temperature=cfg.AGENT.TEMPERATURE,
        max_tokens=cfg.AGENT.MAX_TOKENS,
        timeout=None,
        max_retries=2,
        openai_api_key=cfg["KEYS"][cfg.AGENT.API_KEY],
        base_url=cfg.AGENT.ENDPOINT
    )
    
    return AgenticRAG(cfg, retriever, llm)
