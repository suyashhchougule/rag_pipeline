from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from core.retriever import BaseRetriever
from docstore import DocStore


class MultiLevelRetriever(BaseRetriever):
    def __init__(self, sentence_index: FAISS, parent_index: FAISS, docstore: DocStore, 
                 use_cosine_similarity: bool = True):
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.docstore = docstore
        self.use_cosine_similarity = use_cosine_similarity

    def retrieve(self, query: str, k_sent: int = 15, k_parent: int = 10, max_return: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multi-level strategy.
        
        Args:
            query: Search query string
            k_sent: Number of sentence-level results to retrieve
            k_parent: Number of parent-level results to retrieve  
            max_return: Maximum number of final results to return
            
        Returns:
            List of dictionaries containing documents and their scores
        """
        # Validate inputs
        if not query.strip():
            return []
        
        if k_sent <= 0 or k_parent <= 0 or max_return <= 0:
            raise ValueError("k_sent, k_parent, and max_return must be positive integers")

        # Use similarity_search_with_score to get scores
        s_hits = self.sentence_index.similarity_search_with_score(query, k=k_sent)
        p_hits = self.parent_index.similarity_search_with_score(query, k=k_parent)

        ordered = []
        seen = set()
        
        # Add sentence-level chunks
        for doc, score in s_hits:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                ordered.append({"parent_id": pid, "score": score, "source": "sentence"})
                seen.add(pid)
        
        # Add parent-level chunks
        for doc, score in p_hits:
            pid = doc.metadata.get("uid")
            if pid and pid not in seen:
                ordered.append({"parent_id": pid, "score": score, "source": "parent"})
                seen.add(pid)

        # Sort based on similarity type
        if self.use_cosine_similarity:
            # For cosine similarity: higher scores are better
            ordered = sorted(ordered, key=lambda x: x["score"], reverse=True)
        else:
            # For L2 distance: lower scores are better  
            ordered = sorted(ordered, key=lambda x: x["score"])
        
        # Truncate to max_return
        top_ordered = ordered[:max_return]
        
        # Extract parent_ids and load documents
        parent_ids = [item["parent_id"] for item in top_ordered]
        docs = self.docstore.load_parents(parent_ids)
        
        # Combine loaded docs with their scores
        results = []
        for i, item in enumerate(top_ordered):
            if i < len(docs) and docs[i] is not None:
                results.append({
                    "doc": docs[i], 
                    "score": item["score"],
                    "source": item["source"]  # Track whether it came from sentence or parent search
                })
        
        return results

    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever indexes"""
        return {
            "sentence_index_vectors": self.sentence_index.index.ntotal if self.sentence_index else 0,
            "parent_index_vectors": self.parent_index.index.ntotal if self.parent_index else 0,
            "similarity_type": "cosine" if self.use_cosine_similarity else "l2_distance"
        }
