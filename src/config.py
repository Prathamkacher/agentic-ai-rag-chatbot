"""
Configuration Module for Agentic AI Knowledge Assistant.
Handles environment variables and application-wide settings using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application Settings Configuration."""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # Vector Database Settings
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    
    # Models (Google Gemini)
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072  # Gemini embedding-001 actual output dimension
    
    # Text Splitting & Retrieval Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # Document Path
    PDF_PATH: str = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "agentic_ai_ebook.pdf"))
    
    # FastAPI Server Configuration
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "127.0.0.1")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))

    @classmethod
    def validate_keys(cls) -> None:
        """Validate critical environment variables."""
        missing = []
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your_gemini_api_key_here":
            missing.append("GEMINI_API_KEY")
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "your_pinecone_api_key_here":
            missing.append("PINECONE_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                f"Please configure them in your .env file."
            )


settings = Settings()
