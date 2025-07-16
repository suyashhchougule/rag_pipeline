from typing import List
from langchain_community.vectorstores import FAISS
from docstore import DocStore

class MultiLevelRetriever:
    def __init__(self, sentence_index: FAISS, parent_index: FAISS, docstore: DocStore):
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.docstore = docstore

    def retrieve(self, query: str, k_sent=15, k_parent=10, max_return=5) -> List:
        s_hits = self.sentence_index.similarity_search(query, k=k_sent)
        p_hits = self.parent_index.similarity_search(query, k=k_parent)
        ordered_ids, seen = [], set()
        for s in s_hits:
            pid = s.metadata["parent_id"]
            if pid not in seen:
                ordered_ids.append(pid)
                seen.add(pid)
        for p in p_hits:
            pid = p.metadata["uid"]
            if pid not in seen:
                ordered_ids.append(pid)
                seen.add(pid)
        return self.docstore.load_parents(ordered_ids[:max_return])
