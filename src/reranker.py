"""
Cross-encoder reranking for improved retrieval precision.
Uses a more powerful model to rerank the initial retrieval results.
"""

from typing import List, Tuple
from dataclasses import dataclass

from .config import settings
from .retriever import RetrievalResult


@dataclass
class RerankResult:
    """Result after reranking."""
    id: str
    content: str
    metadata: dict
    original_score: float
    rerank_score: float
    final_score: float


class CrossEncoderReranker:
    """
    Rerank retrieval results using a cross-encoder model.
    Cross-encoders are more accurate than bi-encoders but slower,
    so we use them only on the top-k results from initial retrieval.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: The search query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked results with updated scores
        """
        top_k = top_k or settings.top_k_rerank
        
        if not results:
            return []
        
        # Load model
        model = self._load_model()
        
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, result.content) for result in results]
        
        # Get cross-encoder scores
        scores = model.predict(pairs)
        
        # Combine with original scores
        reranked = []
        for i, result in enumerate(results):
            rerank_score = float(scores[i])
            
            # Weighted combination of original and rerank scores
            # Give more weight to reranker (it's more accurate)
            final_score = 0.3 * result.score + 0.7 * self._normalize_score(rerank_score)
            
            reranked.append(RerankResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                original_score=result.score,
                rerank_score=rerank_score,
                final_score=final_score
            ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        return reranked[:top_k]
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize cross-encoder score to [0, 1] range.
        Cross-encoder scores can be negative or > 1.
        """
        # Sigmoid-like normalization
        import math
        return 1 / (1 + math.exp(-score))


class LLMReranker:
    """
    Alternative reranker using LLM for scoring relevance.
    More expensive but can capture complex relevance signals.
    """
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RerankResult]:
        """Rerank using LLM relevance scoring."""
        top_k = top_k or settings.top_k_rerank
        
        if not results:
            return []
        
        reranked = []
        
        for result in results:
            # Ask LLM to score relevance
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.

Query: {query}

Document:
{result.content[:1000]}

Respond with only a number from 0-10."""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                
                score_text = response.choices[0].message.content.strip()
                rerank_score = float(score_text) / 10.0  # Normalize to 0-1
            except:
                rerank_score = 0.5  # Default if parsing fails
            
            final_score = 0.3 * result.score + 0.7 * rerank_score
            
            reranked.append(RerankResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                original_score=result.score,
                rerank_score=rerank_score,
                final_score=final_score
            ))
        
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked[:top_k]
