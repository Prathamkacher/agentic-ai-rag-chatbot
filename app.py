"""
FastAPI Backend Application for Agentic AI Knowledge Assistant.
Exposes POST /chat REST endpoint.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import run_rag_pipeline

# Initialize FastAPI App
app = FastAPI(
    title="Agentic AI Knowledge Assistant API",
    description="Backend API for RAG-based Chatbot querying the Agentic AI eBook",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request payload for /chat endpoint."""
    question: str = Field(
        ...,
        min_length=2,
        description="The query string to search in the Agentic AI eBook.",
        example="What is Agentic AI according to the eBook?"
    )


class SourceChunk(BaseModel):
    """Metadata for retrieved chunk."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float


class ChatResponse(BaseModel):
    """Response payload for /chat endpoint."""
    final_answer: str
    retrieved_context: List[str]
    confidence_score: float
    retrieved_sources: Optional[List[SourceChunk]] = []


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Agentic AI Knowledge Assistant API",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """
    POST /chat Endpoint.
    Processes user query using LangGraph RAG pipeline and returns answer with context & score.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )
    
    try:
        result = run_rag_pipeline(question)
        return ChatResponse(
            final_answer=result["final_answer"],
            retrieved_context=result["retrieved_context"],
            confidence_score=result["confidence_score"],
            retrieved_sources=result.get("retrieved_sources", [])
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=True
    )
