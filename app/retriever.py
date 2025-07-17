from typing import List
from langchain_community.vectorstores import FAISS
from core.retriever import BaseRetriever
from docstore import DocStore
from typing import List, Dict, Any

class MultiLevelRetriever(BaseRetriever):
    def __init__(self, sentence_index: FAISS, parent_index: FAISS, docstore: DocStore):
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.docstore = docstore

    def retrieve(self, query: str, k_sent=15, k_parent=10, max_return=5) -> List[Dict[str, Any]]:
        # Use similarity_search_with_score to get scores
        s_hits = self.sentence_index.similarity_search_with_score(query, k=k_sent)
        p_hits = self.parent_index.similarity_search_with_score(query, k=k_parent)

        ordered = []
        seen = set()
        # Add sentence-level chunks
        for doc, score in s_hits:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                # Instead of appending the sentence doc, append the parent_id and score
                ordered.append({"parent_id": pid, "score": score})
                seen.add(pid)
        # Add parent-level chunks
        for doc, score in p_hits:
            pid = doc.metadata.get("uid")
            if pid and pid not in seen:
                ordered.append({"parent_id": pid, "score": score})
                seen.add(pid)
        # Truncate and load parent docs from docstore 
        ordered = sorted(ordered, key=lambda x: x["score"])
        top_ordered = ordered[:max_return]
        parent_ids = [item["parent_id"] for item in top_ordered]
        docs = self.docstore.load_parents(parent_ids)
        # Combine loaded docs with their scores
        results = []
        for i, item in enumerate(top_ordered):
            doc = docs[i] if i < len(docs) else None
            if doc is not None:
                results.append({"doc": doc, "score": item["score"]})
        return results
