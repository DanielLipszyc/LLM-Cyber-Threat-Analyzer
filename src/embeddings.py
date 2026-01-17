"""
Embedding generation for RAG system.
Supports OpenAI and local sentence-transformer models.
"""

from typing import List, Optional
import numpy as np
from openai import OpenAI

from .config import settings


class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI or local models.
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        use_local: bool = False
    ):
        self.model = model or settings.embedding_model
        self.use_local = use_local
        
        if use_local:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
        else:
            self.client = OpenAI(api_key=api_key or settings.openai_api_key)
            # Dimensions for OpenAI models
            self.dimension = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.use_local:
            embedding = self.local_model.encode(text)
            return embedding.tolist()
        else:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Uses batching for efficiency.
        """
        if self.use_local:
            embeddings = self.local_model.encode(texts)
            return embeddings.tolist()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Sort by index to maintain order
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([e.embedding for e in batch_embeddings])
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Some models have different embeddings for queries vs documents.
        """
        # For OpenAI models, query and document embeddings are the same
        return self.embed_text(query)
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def batch_cosine_similarity(
        query_vec: List[float],
        doc_vecs: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between query and multiple documents."""
        query = np.array(query_vec)
        docs = np.array(doc_vecs)
        
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(docs_norm, query_norm)
        return similarities.tolist()
