from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """
    Abstract base class for retriever implementations.
    Enforces the contract for a `retrieve` method.
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents given a query.
        
        Args:
            query (str): The input query string.
            **kwargs: Additional retrieval parameters.
        
        Returns:
            List[Dict[str, Any]]: A list of dicts containing documents and optional scores.
        """
        pass
