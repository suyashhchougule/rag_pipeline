from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ChunkInfo(BaseModel):
    chunk_id: str
    summary: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None
    source: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    k_sent: Optional[int] = 20
    k_parent: Optional[int] = 10
    max_return: Optional[int] = 10

class QueryResponse(BaseModel):
    answer: dict
    meta: Dict[str, Any]
    context_chunks: List[ChunkInfo]