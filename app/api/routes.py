from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Literal
import logging

from core.pydantic_models import QueryRequest, QueryResponse
from core.dependencies import get_simple_rag, get_agentic_rag
from rag.base import BaseRAG

log = logging.getLogger(__name__)
router = APIRouter()

def get_rag_instance(rag_type: Literal["simple", "agentic"] = Query(...)) -> BaseRAG:
    """Dependency to get the appropriate RAG instance"""
    if rag_type == "simple":
        return get_simple_rag()
    elif rag_type == "agentic":
        return get_agentic_rag()
    else:
        raise HTTPException(status_code=400, detail="Invalid RAG type. Use 'simple' or 'agentic'")

@router.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    rag_instance: BaseRAG = Depends(get_rag_instance)
):
    """Main query endpoint supporting both simple and agentic RAG"""
    try:
        log.info(f"Processing {type(rag_instance).__name__} query: {request.question}")
        result = rag_instance.query(request)
        log.info(f"Successfully processed query with {len(result.context_chunks)} chunks")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error in /query endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
def health_check(rag_type: Literal["simple", "agentic"] = Query(...)):
    """Health check endpoint for specific RAG type"""
    try:
        rag_instance = get_rag_instance(rag_type)
        return rag_instance.get_health_status()
    except Exception as e:
        log.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e)}

@router.get("/stats")
def get_stats(rag_type: Literal["simple", "agentic"] = Query(...)):
    """Get statistics for specific RAG type"""
    try:
        rag_instance = get_rag_instance(rag_type)
        return {
            "status": "success",
            "rag_type": rag_type,
            "stats": rag_instance.get_stats()
        }
    except Exception as e:
        log.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/info")
def get_system_info():
    """Get information about available RAG types"""
    return {
        "available_rag_types": ["simple", "agentic"],
        "simple_rag": {
            "description": "Non-agentic RAG with direct retrieval and generation",
            "features": ["Multi-level retrieval", "Rate limiting", "Token logging"]
        },
        "agentic_rag": {
            "description": "LangGraph-based agentic RAG with question decomposition",
            "features": ["Question decomposition", "Agent orchestration", "Multi-step reasoning"]
        }
    }
