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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config" / "RagConfig.toml"
SECRETS_PATH = BASE_DIR / "config" / ".secrtets.toml"
cfg = Dynaconf(settings_files=[CONFIG_PATH, SECRETS_PATH])

# Embedding/Index/Docstore paths from config
SENT_IDX_DIR = BASE_DIR / cfg.INGESTION.SENT_IDX_DIR
PARENT_IDX_DIR = BASE_DIR / cfg.INGESTION.PARENT_IDX_DIR
SQLITE_PATH = BASE_DIR / cfg.INGESTION.PARENTS_DB
EMBED_MODEL = cfg.INGESTION.EMBED_MODEL
RPM = int(cfg.RATELIMITER.RPM)
TPM = int(cfg.RATELIMITER.TPM)

def main(query=None):
    try:
        log.info("Loading FAISS indexes and docstore...")
        s_index, p_index = load_indexes(SENT_IDX_DIR, PARENT_IDX_DIR, EMBED_MODEL)
        docstore = DocStore(SQLITE_PATH)
        retriever = MultiLevelRetriever(s_index, p_index, docstore)

        if not query:
            # Default query if not provided via CLI
            query = "What is affinity fraud ?"
        log.info(f"User query: {query}")

        # Retrieve context chunks
        parent_chunks = retriever.retrieve(query, k_sent=20, k_parent=10, max_return=3)
        if not parent_chunks:
            log.warning("No context found for query. Aborting.")
            return

        print(f"\nTop {len(parent_chunks)} parent chunks for {query!r}\n")
        for i, doc in enumerate(parent_chunks, 1):
            print(f"── Parent {i} uid={doc.metadata.get('uid', 'NA')[:8]}")
            print(doc.page_content, "…\n")

        # Prepare context for LLM
        context_blob = "\n---\n".join(chunk.page_content for chunk in parent_chunks)
        prompt = PROMPT_TMPL.format(context=context_blob, question=query)

        # Rate limiter & token logger setup
        limiter = RateLimiter(rpm=RPM, tpm=TPM)
        token_logger = SimpleTokenLogger()

        generator = OpenAIChatGenerator(
            config=cfg,
            rate_limiter=limiter,
            token_logger=token_logger,
        )

        # Generate LLM response
        log.info("Querying LLM with retrieved context...")
        answer = generator.generate_response(
            text=prompt,
            response_format="json_object",  # or {"type": "json_object"}
            model_name=cfg.DEFAULT.GENERATOR_MODEL
        )
        log.info("LLM answer: %s", answer)
        print("\n=== LLM Response ===\n", answer)
    except Exception as e:
        log.exception("Fatal error during main execution:")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG query with retrieval and generation")
    parser.add_argument("--query", type=str, help="User question (defaults to demo query)")
    args = parser.parse_args()
    main(query=args.query)
