import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.logging import setup_logging

# Setup logging
setup_logging()
log = logging.getLogger(__name__)
    
# Create FastAPI app
app = FastAPI(
    title="Unified RAG API",
    description="Unified API supporting both simple and agentic RAG implementations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Unified RAG API",
        "endpoints": {
            "query": "/api/v1/query?rag_type=simple|agentic",
            "health": "/api/v1/health?rag_type=simple|agentic",
            "stats": "/api/v1/stats?rag_type=simple|agentic",
            "info": "/api/v1/info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
