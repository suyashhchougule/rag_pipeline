#!/usr/bin/env python3
"""
Modular Document Ingestion Pipeline

Features:
- Discover & load documents from a folder (e.g., Markdown files)
- Normalize HTML tables in Markdown
- Split into sections (H1/H2)
- Chunk into parent (≈2048 chars) & sentence chunks
- Track ingested files via SHA-256 hashes (skip unchanged)
- Store parent chunks in SQLite as pickled blobs
- Build or update FAISS vector indexes for sentences & parents, persisted on disk
"""

import os
import logging
from filetracker import FileTracker
from loader import Loader
from chunker import Chunker
from store import SQLiteBlobStore
from indexer import Indexer
from dynaconf import Dynaconf
from pathlib import Path
from documentparser import DolphinDocumentParser
import shutil
BASE_DIR = Path(__file__).parent.parent.resolve()

def setup_logging():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    if log.hasHandlers():
        log.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    file_handler = logging.FileHandler("rag_pipeline.log", mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    return log

log = setup_logging()

class IngestionPipeline:
    """
    Orchestrates loading, chunking, storing, and indexing.
    """
    def __init__(self,
                 input_folder: str,
                 config_dict: str = None):

        log.info("Initializing IngestionPipeline...")
        tracker_db = BASE_DIR / config_dict['INGESTION']['TRACKER_DB']
        parents_db = BASE_DIR / config_dict['INGESTION']['PARENTS_DB']
        index_dir = BASE_DIR / config_dict['INGESTION']['INDEX_DIR']
        embed_model = config_dict['INGESTION']['EMBED_MODEL']

        model_path = BASE_DIR / config_dict['INGESTION']['DOLPHIN_MODEL_PATH']         
        self.save_dir = BASE_DIR / config_dict['INGESTION']['DOCUMENT_PARSER_SAVE_DIR']        
        max_batch_size = config_dict['INGESTION']['DOCUMENT_PARSER_BATCH_SIZE']               
        self.input_path = input_folder
        self.md_save_dir = os.path.join(self.save_dir, "markdown")
        self.rechunk = config_dict['INGESTION'].get('RECHUNK', False)

        # Clean indexes and parent db if rechunk is True
        if self.rechunk:
            sent_idx_dir = BASE_DIR / config_dict['INGESTION']['SENT_IDX_DIR']
            parent_idx_dir = BASE_DIR / config_dict['INGESTION']['PARENT_IDX_DIR']
            # Remove FAISS index directories
            for idx_dir in [sent_idx_dir, parent_idx_dir]:
                if idx_dir.exists():
                    shutil.rmtree(idx_dir)
                    log.info(f"Deleted FAISS index directory: {idx_dir}")
            # Remove SQLite parent db
            if parents_db.exists():
                parents_db.unlink()
                log.info(f"Deleted parent SQLite DB: {parents_db}")
        
        log.info(f"Model path: {model_path}")
        log.info(f"Markdown output directory: {self.md_save_dir}")
        log.info(f"Max batch size: {max_batch_size}")
        log.info(f"Using Embedding model:{embed_model}")
        
        self.parser = DolphinDocumentParser(model_path, self.save_dir, max_batch_size)
        self.tracker = FileTracker(tracker_db)
        self.loader  = Loader(self.md_save_dir, self.tracker, rechunk=self.rechunk)
        self.chunker = Chunker()
        self.store   = SQLiteBlobStore(parents_db)
        self.indexer = Indexer(embed_model, index_dir)

    def run(self):
        log.info(f"Processing folder: {self.input_path}")
        results = self.parser.parse_documents(self.input_path)
        log.info(f"Document parsing complete. Parsed {len(results)} files.")
        sections = self.loader.load()
        log.info(f"Loaded {len(sections)} sections from markdown.")
        parents, sentences = self.chunker.chunk(sections)
        log.info(f"Chunked into {len(parents)} parent and {len(sentences)} sentence chunks.")
        self.store.save_parents(parents)
        log.info(f"Saved parent chunks to SQLite.")
        self.indexer.add_documents(sentences, parents)
        log.info(f"Added documents to vector index.")
        self.indexer.save()
        log.info("Indexes saved to disk.")
        self.tracker.close()
        log.info(f"Ingestion complete: {len(parents)} parents and {len(sentences)} sentences ingested.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modular ingestion pipeline")
    parser.add_argument("--input_folder", help="Folder containing documents to ingest")

    args = parser.parse_args()

    config_file = BASE_DIR / "config" / "RagConfig.toml"
    secret_config_file = BASE_DIR / "config" / ".secrtets.toml"
    log.info(f"Loading configuration from {config_file} and {secret_config_file}")
    cfg = Dynaconf(settings_files=[config_file, secret_config_file])

    pipeline = IngestionPipeline(
        input_folder=args.input_folder,
        config_dict=cfg
    )
    pipeline.run()
