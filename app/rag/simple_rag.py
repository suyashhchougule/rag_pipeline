import logging
import json
from typing import Dict, Any

from .base import BaseRAG
from core.pydantic_models import QueryRequest, QueryResponse, ChunkInfo
from prompt_templates import PROMPT_TMPL

log = logging.getLogger(__name__)

class SimpleRAG(BaseRAG):
    """Non-agentic RAG implementation"""
    
    def __init__(self, config, retriever, generator):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        
    def query(self, request: QueryRequest) -> QueryResponse:
        """Process query using simple RAG pipeline"""
        log.info(f"Processing simple RAG query: {request.question}")
        
        # Validate input
        if not request.question.strip():
            raise ValueError("Query cannot be empty")
        
        if request.k_sent <= 0 or request.k_parent <= 0 or request.max_return <= 0:
            raise ValueError("k_sent, k_parent, and max_return must be positive integers")
        
        # Retrieve context
        try:
            parent_chunks = self.retriever.retrieve(
                request.question,
                k_sent=request.k_sent,
                k_parent=request.k_parent,
                max_return=request.max_return
            )
        except Exception as e:
            log.exception("Retrieval failed")
            raise RuntimeError(f"Retriever error: {str(e)}")
        
        if not parent_chunks:
            raise ValueError("No relevant context found")
        
        # Process chunks
        chunk_infos = self._process_chunks(parent_chunks)
        
        # Generate response
        try:
            context_blob = "\n---\n".join(item["doc"].page_content for item in parent_chunks)
            prompt = PROMPT_TMPL.format(context=context_blob, question=request.question)
            answer = self.generator.generate_response(
                text=prompt,
                response_format="json_object",
                model_name=self.config.DEFAULT.GENERATOR_MODEL_SECTION
            )
        except Exception as e:
            log.exception("Generation failed")
            raise RuntimeError(f"Generator error: {str(e)}")
        
        # Parse answer
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                answer = {"answer": answer}
        
        # Collect metadata
        meta = self._collect_metadata(request, len(chunk_infos))
        
        return QueryResponse(
            answer=answer,
            context_chunks=chunk_infos,
            meta=meta
        )
    
    def _process_chunks(self, parent_chunks):
        """Process retrieved chunks into ChunkInfo objects"""
        chunk_infos = []
        for item in parent_chunks:
            try:
                doc = item["doc"]
                score = item.get("score")
                source = item.get("source", "unknown")
                summary = doc.page_content
                
                chunk_infos.append(ChunkInfo(
                    chunk_id=doc.metadata.get('uid', ''),
                    summary=summary,
                    metadata=doc.metadata,
                    score=score,
                    source=source
                ))
            except Exception as e:
                log.exception("Failed to process chunk info")
                continue
        return chunk_infos
    
    def _collect_metadata(self, request: QueryRequest, num_chunks: int) -> Dict[str, Any]:
        """Collect system metadata"""
        retriever_stats = self.retriever.get_retriever_stats()
        return {
            "embedding_model": self.config.INGESTION.EMBED_MODEL,
            "generator_model": self.config.DEFAULT.GENERATOR_MODEL_SECTION,
            "retriever": self.retriever.__class__.__name__,
            "similarity_type": retriever_stats.get("similarity_type", "unknown"),
            "sentence_index_vectors": retriever_stats.get("sentence_index_vectors", 0),
            "parent_index_vectors": retriever_stats.get("parent_index_vectors", 0),
            "architecture": "Simple RAG",
            "prompt_template": PROMPT_TMPL,
            "num_context_chunks": num_chunks,
            "retrieval_params": {
                "k_sent": request.k_sent,
                "k_parent": request.k_parent,
                "max_return": request.max_return
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status"""
        try:
            stats = self.retriever.get_retriever_stats()
            return {
                "status": "healthy",
                "architecture": "Simple RAG",
                "indexes_loaded": {
                    "sentence_index": stats["sentence_index_vectors"] > 0,
                    "parent_index": stats["parent_index_vectors"] > 0
                }
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics"""
        return self.retriever.get_retriever_stats()
