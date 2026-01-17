"""
Configuration settings for the Threat Intelligence RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings."""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval Settings
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    
    # Hybrid Search Weights
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    # Vector Store
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = "threat_intel"
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    
    # Generation Settings
    max_tokens: int = 1024
    temperature: float = 0.1  # Low temperature for factual responses
    
    # Hallucination Detection
    hallucination_threshold: float = 0.7  # Confidence threshold
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()


# Prompt Templates
SYSTEM_PROMPT = """You are a cybersecurity expert assistant that answers questions about security vulnerabilities, CVEs, attack techniques, and threat intelligence.

IMPORTANT RULES:
1. ONLY use information from the provided context documents
2. If the context doesn't contain relevant information, say "I don't have information about this in my knowledge base"
3. Always cite your sources using [Source: X] format
4. Be precise and technical in your responses
5. If you're uncertain, express that uncertainty

Context documents will be provided in the user message."""

GENERATION_PROMPT = """Based on the following security threat intelligence documents, answer the user's question.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

Provide a detailed, accurate answer citing specific sources. Use [Source: document_id] to cite information."""

HALLUCINATION_CHECK_PROMPT = """Analyze the following answer and determine if each claim is supported by the provided sources.

SOURCES:
{sources}

ANSWER TO CHECK:
{answer}

For each factual claim in the answer, respond with:
1. The claim
2. Whether it's SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED by the sources
3. Which source supports it (if any)

Respond in JSON format:
{{
    "claims": [
        {{"claim": "...", "status": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED", "source": "..."}}
    ],
    "overall_faithfulness": 0.0 to 1.0
}}"""
