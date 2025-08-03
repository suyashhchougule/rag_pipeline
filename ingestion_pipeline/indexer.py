from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Indexer:
    """
    Manages FAISS indexes for sentences and parents, persisted on disk.
    """
    def __init__(self, embed_model: str, index_dir: str):
        self.embed_model = embed_model
        self.index_dir = Path(index_dir)
        self.sent_dir = self.index_dir / "sentences"
        self.parent_dir = self.index_dir / "parents"

        # embedding wrapper
        self.embed = HuggingFaceEmbeddings(model_name=self.embed_model)

        # placeholders—don’t create an empty index here!
        self.s_index: Optional[FAISS] = None
        self.p_index: Optional[FAISS] = None

        # if existing indexes on disk, load them
        self._load_if_exists()

    def _load_if_exists(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.sent_dir.exists() and self.parent_dir.exists():
            self.s_index = FAISS.load_local(str(self.sent_dir), self.embed, allow_dangerous_deserialization=True)
            self.p_index = FAISS.load_local(str(self.parent_dir), self.embed, allow_dangerous_deserialization=True)

    def get_num_vectors_in_sentence_index(self) -> int:
        if self.s_index is not None:
            return self.s_index.index.ntotal
        return 0  # or raise an error if preferred

    def get_num_vectors_in_parent_index(self) -> int:
        if self.p_index is not None:
            return self.p_index.index.ntotal
        return 0

    def add_documents(
        self,
        sentence_docs: List[Document],
        parent_docs:   List[Document],
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
                # first-ever add
                self.s_index = FAISS.from_documents(sentence_docs, self.embed)
            else:
                # subsequent adds
                self.s_index.add_documents(sentence_docs)

        # ── parents ───────────────────────────────
        if parent_docs:
            if self.p_index is None:
                self.p_index = FAISS.from_documents(parent_docs, self.embed)
            else:
                self.p_index.add_documents(parent_docs)

        print(f"After adding: {self.get_num_vectors_in_sentence_index()} sentence vectors")
        print(f"After adding: {self.get_num_vectors_in_parent_index()} parent vectors")

    def save(self):
        """Persist both indexes to disk so they can be re-loaded next time."""
        if self.s_index:
            self.s_index.save_local(str(self.sent_dir))
        if self.p_index:
            self.p_index.save_local(str(self.parent_dir))