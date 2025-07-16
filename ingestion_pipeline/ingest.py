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
from filetracker import FileTracker
from loader import Loader
from chunker import Chunker
from store import SQLiteBlobStore
from indexer import Indexer
from dynaconf import Dynaconf
from pathlib import Path
from documentparser import DolphinDocumentParser
BASE_DIR = Path(__file__).parent.parent.resolve()


class IngestionPipeline:
    """
    Orchestrates loading, chunking, storing, and indexing.
    """
    def __init__(self,
                 input_folder: str,
                 config_dict: str = None):

        tracker_db = BASE_DIR / config_dict['INGESTION']['TRACKER_DB']
        parents_db = BASE_DIR / config_dict['INGESTION']['PARENTS_DB']
        index_dir = BASE_DIR / config_dict['INGESTION']['INDEX_DIR']
        embed_model = config_dict['INGESTION']['EMBED_MODEL']

        model_path = BASE_DIR / config_dict['INGESTION']['DOLPHIN_MODEL_PATH']         # Path to your HuggingFace model or model ID
        self.save_dir = BASE_DIR / config_dict['INGESTION']['DOCUMENT_PARSER_SAVE_DIR']        # Directory to save results (optional)
        max_batch_size = config_dict['INGESTION']['DOCUMENT_PARSER_BATCH_SIZE']               # Batch size for processing (optional)
        self.input_path = input_folder
        self.md_save_dir = os.path.join(self.save_dir, "markdown")
        print(model_path)
        print(self.md_save_dir)
        print(max_batch_size)
        
        self.parser = DolphinDocumentParser(model_path, self.save_dir, max_batch_size)
        self.tracker = FileTracker(tracker_db)
        self.loader  = Loader(self.md_save_dir, self.tracker)
        self.chunker = Chunker()
        self.store   = SQLiteBlobStore(parents_db)
        self.indexer = Indexer(embed_model, index_dir)

    def run(self):
        print("processing folder ", self.input_path )
        results = self.parser.parse_documents(self.input_path)
        sections = self.loader.load()
        parents, sentences = self.chunker.chunk(sections)
        self.store.save_parents(parents)
        self.indexer.add_documents(sentences, parents)
        self.indexer.save()
        self.tracker.close()
        print(f"Ingested {len(parents)} parents and {len(sentences)} sentences.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modular ingestion pipeline")
    parser.add_argument("--input_folder", default='/home/user/rag_pipeline/ingestion_pipeline/docs', help="Folder containing documents to ingest")

    args = parser.parse_args()

    config_file = BASE_DIR / "config" / "RagConfig.toml"
    secret_config_file = BASE_DIR / "config" / ".secrtets.toml"
    print(config_file)
    cfg = Dynaconf(settings_files=[config_file,secret_config_file])

    pipeline = IngestionPipeline(
        input_folder=args.input_folder,
        config_dict=cfg
    )
    pipeline.run()
