import logging
import json
from typing import Dict, Any, List
from pathlib import Path

from .base import BaseRAG
from core.pydantic_models import QueryRequest, QueryResponse, ChunkInfo
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import convert_to_messages

log = logging.getLogger(__name__)

class AgenticRAG(BaseRAG):
    """Agentic RAG implementation using LangGraph"""
    
    def __init__(self, config, retriever, llm):
        self.config = config
        self.retriever = retriever
        self.llm = llm
        
        # Global variables for metadata collection
        self._current_retrieval_results = []
        self._current_retrieval_params = {"k_sent": 20, "k_parent": 10, "max_return": 10}
        
        # Initialize agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup LangGraph agents and supervisor"""
        # Create tools
        self.plan_tool = self._create_plan_tool()
        self.retrieve_tool = self._create_retrieve_tool()
        
        # Create agents
        self.planner_agent = create_react_agent(
            model=self.llm,
            tools=[self.plan_tool],
            name="planner_expert",
            prompt=(
                "You are a planning agent.\n"
                "Call plan_sub_questions tool once with the user's question.\n"
                "Return the tool's JSON response verbatim."
            ),
        )
        
        self.retriever_agent = create_react_agent(
            model=self.llm,
            tools=[self.retrieve_tool],
            name="retriever_expert",
            prompt=(
                "You are a retrieval expert with the multi_level_retrieve tool.\n"
                "Always call the tool to get context and respond with the retrieved information."
            ),
        )
        
        # Create supervisor
        self.supervisor = create_supervisor(
            agents=[self.planner_agent, self.retriever_agent],
            model=self.llm,
            prompt=(
                "You are the supervisor managing the planner and retriever agents.\n"
                "1. Use planner_expert to break down the question.\n"
                "2. Pass each sub-question to retriever_expert to fetch context.\n"
                "3. Synthesize a final answer based on retrieved information.\n"
                "4. Return the final answer with citation details in JSON format."
            ),
            output_mode="last_message",
            add_handoff_back_messages=True,
        ).compile()
    
    def _create_plan_tool(self):
        """Create the planning tool"""
        @tool
        def plan_sub_questions(question: str) -> str:
            """Break down the input question into sub-questions or return the original if simple."""
            analysis_raw = self.llm.invoke(
                f'Return "SIMPLE" or "COMPLEX" for this question:\nQUESTION: {question}'
            ).content.strip().upper()

            if analysis_raw == "SIMPLE":
                return json.dumps({"sub_questions": [question]})

            response = self.llm.invoke(
                f"""You are an expert question decomposer tasked with breaking down the question below.
Limit the number of sub-questions to at most 5.
List each clear and focused sub-question on a separate line without additional commentary.

QUESTION:
{question}
"""
            ).content

            sub_questions = [
                line.strip(" -•*") for line in response.splitlines() if len(line.strip()) > 5
            ]

            if not sub_questions:
                sub_questions = [question]

            log.info(f"Sub-questions generated: {sub_questions}")
            return json.dumps({"sub_questions": sub_questions})
        
        return plan_sub_questions
    
    def _create_retrieve_tool(self):
        """Create the retrieval tool"""
        @tool
        def multi_level_retrieve(query: str) -> str:
            """Retrieve relevant documents using a multi-level retriever based on the query."""
            k_sent = self._current_retrieval_params["k_sent"]
            k_parent = self._current_retrieval_params["k_parent"]
            max_return = self._current_retrieval_params["max_return"]
            
            results = self.retriever.retrieve(query, k_sent=k_sent, k_parent=k_parent, max_return=max_return)
            
            # Store retrieval results globally for metadata collection
            self._current_retrieval_results.extend(results)
            
            seen, texts = set(), []
            for item in results:
                doc = item.get("doc")
                if doc is None:
                    continue
                uid = doc.metadata.get("uid", None)
                if uid in seen:
                    continue
                seen.add(uid)
                texts.append(doc.page_content)

            return "\n---\n".join(texts) if texts else "NO_CONTEXT_FOUND"
        
        return multi_level_retrieve
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """Process query using agentic RAG pipeline"""
        log.info(f"Processing agentic RAG query: {request.question}")
        
        # Validate input
        if not request.question.strip():
            raise ValueError("Query cannot be empty")
        
        if request.k_sent <= 0 or request.k_parent <= 0 or request.max_return <= 0:
            raise ValueError("k_sent, k_parent, and max_return must be positive integers")
        
        # Reset and set current parameters
        self._current_retrieval_results = []
        self._current_retrieval_params = {
            "k_sent": request.k_sent,
            "k_parent": request.k_parent,
            "max_return": request.max_return
        }
        
        messages = [{"role": "user", "content": request.question}]
        
        try:
            # Run the supervisor
            result = self.supervisor.invoke({"messages": messages})
            
            # Extract the final answer
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            answer_content = final_message.content if final_message else "No answer generated"
            
            # Try to parse as JSON, fallback to text
            try:
                if isinstance(answer_content, str):
                    answer = json.loads(answer_content)
                else:
                    answer = answer_content
            except json.JSONDecodeError:
                answer = {"answer": answer_content}
            
            # Collect metadata from the retrieval results
            chunk_infos = self._collect_chunk_metadata(self._current_retrieval_results)
            meta = self._collect_system_metadata(request.k_sent, request.k_parent, request.max_return, len(chunk_infos))
            
            return QueryResponse(
                answer=answer,
                context_chunks=chunk_infos,
                meta=meta
            )
            
        except Exception as e:
            log.exception("Error in agentic RAG query")
            raise RuntimeError(f"Agentic RAG error: {str(e)}")
    
    def _collect_chunk_metadata(self, retrieval_results: List[Dict]) -> List[ChunkInfo]:
        """Collect per-chunk metadata from retrieval results"""
        chunk_infos = []
        seen = set()
        
        for item in retrieval_results:
            try:
                doc = item.get("doc")
                if doc is None:
                    continue
                    
                uid = doc.metadata.get('uid', '')
                if uid in seen:
                    continue
                seen.add(uid)
                
                score = item.get("score")
                source = item.get("source", "unknown")
                summary = doc.page_content
                
                chunk_infos.append(ChunkInfo(
                    chunk_id=uid,
                    summary=summary,
                    metadata=doc.metadata,
                    score=score,
                    source=source
                ))
            except Exception as e:
                log.exception("Failed to process chunk info")
                continue
        
        return chunk_infos
    
    def _collect_system_metadata(self, k_sent: int, k_parent: int, max_return: int, num_chunks: int) -> Dict[str, Any]:
        """Collect system-level metadata"""
        try:
            retriever_stats = self.retriever.get_retriever_stats()
        except Exception as e:
            log.warning(f"Failed to get retriever stats: {e}")
            retriever_stats = {}
        
        return {
            "embedding_model": self.config.INGESTION.EMBED_MODEL,
            "generator_model": self.config.AGENT.MODEL_ID,
            "retriever": self.retriever.__class__.__name__,
            "similarity_type": retriever_stats.get("similarity_type", "unknown"),
            "sentence_index_vectors": retriever_stats.get("sentence_index_vectors", 0),
            "parent_index_vectors": retriever_stats.get("parent_index_vectors", 0),
            "agent_architecture": "LangGraph Supervisor + ReAct",
            "agents_used": ["planner_expert", "retriever_expert", "supervisor"],
            "llm_config": {
                "model": self.config.AGENT.MODEL_ID,
                "temperature": self.config.AGENT.TEMPERATURE,
                "max_tokens": self.config.AGENT.MAX_TOKENS
            },
            "num_context_chunks": num_chunks,
            "retrieval_params": {
                "k_sent": k_sent,
                "k_parent": k_parent,
                "max_return": max_return
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status"""
        try:
            stats = self.retriever.get_retriever_stats()
            return {
                "status": "healthy",
                "architecture": "LangGraph Supervisor + ReAct",
                "indexes_loaded": {
                    "sentence_index": stats["sentence_index_vectors"] > 0,
                    "parent_index": stats["parent_index_vectors"] > 0
                },
                "agents_initialized": True,
                "retriever_type": self.retriever.__class__.__name__
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics"""
        return self.retriever.get_retriever_stats()
