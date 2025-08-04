from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class Indexer:
    """
    Manages FAISS indexes for sentences and parents, persisted on disk.
    Supports both L2 distance and cosine similarity.
    """
    
    def __init__(self, embed_model: str, index_dir: str, use_cosine_similarity: bool = True):
        self.embed_model = embed_model
        self.index_dir = Path(index_dir)
        self.sent_dir = self.index_dir / "sentences"
        self.parent_dir = self.index_dir / "parents"
        self.use_cosine_similarity = use_cosine_similarity

        # embedding wrapper
        self.embed = HuggingFaceEmbeddings(model_name=self.embed_model)

        # placeholders—don't create an empty index here!
        self.s_index: Optional[FAISS] = None
        self.p_index: Optional[FAISS] = None

        # if existing indexes on disk, load them
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing indexes from disk if they exist"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.sent_dir.exists() and self.parent_dir.exists():
                self.s_index = FAISS.load_local(
                    str(self.sent_dir), 
                    self.embed, 
                    allow_dangerous_deserialization=True
                )
                self.p_index = FAISS.load_local(
                    str(self.parent_dir), 
                    self.embed, 
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing indexes: {self.get_num_vectors_in_sentence_index()} sentences, {self.get_num_vectors_in_parent_index()} parents")
        except Exception as e:
            print(f"Failed to load existing indexes: {e}")
            self.s_index = None
            self.p_index = None

    def get_num_vectors_in_sentence_index(self) -> int:
        """Get the number of vectors in the sentence index"""
        if self.s_index is not None:
            return self.s_index.index.ntotal
        return 0

    def get_num_vectors_in_parent_index(self) -> int:
        """Get the number of vectors in the parent index"""
        if self.p_index is not None:
            return self.p_index.index.ntotal
        return 0

    def get_index_status(self) -> Dict[str, Any]:
        """Get comprehensive status of both indexes"""
        return {
            "sentence_index": {
                "exists": self.s_index is not None,
                "vector_count": self.get_num_vectors_in_sentence_index()
            },
            "parent_index": {
                "exists": self.p_index is not None,
                "vector_count": self.get_num_vectors_in_parent_index()
            },
            "using_cosine_similarity": self.use_cosine_similarity
        }

    def add_documents(
        self,
        sentence_docs: List[Document],
        parent_docs: List[Document],
    ):
        """
        On the very first batch:
          • bootstrap via from_documents(...)
        On later batches:
          • call add_documents(...)
        """
        # ── sentences ─────────────────────────────
        if sentence_docs:
            if self.s_index is None:
                # first-ever add with similarity strategy
                if self.use_cosine_similarity:
                    self.s_index = FAISS.from_documents(
                        sentence_docs, 
                        self.embed,
                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                        normalize_L2=True
                    )
                else:
                    self.s_index = FAISS.from_documents(sentence_docs, self.embed)
            else:
                # subsequent adds
                self.s_index.add_documents(sentence_docs)

        # ── parents ───────────────────────────────
        if parent_docs:
            if self.p_index is None:
                # first-ever add with similarity strategy
                if self.use_cosine_similarity:
                    self.p_index = FAISS.from_documents(
                        parent_docs, 
                        self.embed,
                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                        normalize_L2=True
                    )
                else:
                    self.p_index = FAISS.from_documents(parent_docs, self.embed)
            else:
                # subsequent adds
                self.p_index.add_documents(parent_docs)

        print(f"After adding: {self.get_num_vectors_in_sentence_index()} sentence vectors")
        print(f"After adding: {self.get_num_vectors_in_parent_index()} parent vectors")

    def save(self):
        """Persist both indexes to disk so they can be re-loaded next time."""
        if self.s_index:
            self.s_index.save_local(str(self.sent_dir))
            print(f"Saved sentence index to {self.sent_dir}")
        if self.p_index:
            self.p_index.save_local(str(self.parent_dir))
            print(f"Saved parent index to {self.parent_dir}")

    def clear_indexes(self):
        """Clear both indexes from memory"""
        self.s_index = None
        self.p_index = None
        print("Cleared indexes from memory")

    def delete_persisted_indexes(self):
        """Delete saved indexes from disk"""
        if self.sent_dir.exists():
            shutil.rmtree(self.sent_dir)
            print(f"Deleted sentence index from {self.sent_dir}")
        if self.parent_dir.exists():
            shutil.rmtree(self.parent_dir)
            print(f"Deleted parent index from {self.parent_dir}")
        self.clear_indexes()