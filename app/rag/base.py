from abc import ABC, abstractmethod
from typing import Dict, Any
from core.pydantic_models import QueryRequest, QueryResponse

class BaseRAG(ABC):
    """Abstract base class for RAG implementations"""
    
    @abstractmethod
    def query(self, request: QueryRequest) -> QueryResponse:
        """Process a query and return response with metadata"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics"""
        pass
