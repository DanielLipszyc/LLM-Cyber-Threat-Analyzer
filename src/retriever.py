"""
Hybrid retrieval combining dense (embedding) and sparse (BM25) retrieval.
Uses Reciprocal Rank Fusion to combine results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re

from .config import settings
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str  # "dense", "sparse", or "hybrid"


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval for better results.
    Uses Reciprocal Rank Fusion (RRF) to combine rankings.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        dense_weight: float = None,
        sparse_weight: float = None
    ):
        self.vector_store = vector_store
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight
        
        # BM25 index (built lazily)
        self._bm25_index = None
        self._bm25_corpus = None
        self._bm25_ids = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in vector store."""
        # Get all documents from vector store
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not all_data["ids"]:
            self._bm25_corpus = []
            self._bm25_ids = []
            self._bm25_index = None
            return
        
        self._bm25_ids = all_data["ids"]
        self._bm25_corpus = [
            {
                "id": all_data["ids"][i],
                "content": all_data["documents"][i] if all_data["documents"] else "",
                "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {}
            }
            for i in range(len(all_data["ids"]))
        ]
        
        # Tokenize corpus for BM25
        tokenized_corpus = [
            self._tokenize(doc["content"]) 
            for doc in self._bm25_corpus
        ]
        
        self._bm25_index = BM25Okapi(tokenized_corpus)
        print(f"Built BM25 index with {len(self._bm25_corpus)} documents")
    
    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25 sparse retrieval."""
        if self._bm25_index is None:
            self._build_bm25_index()
        
        if not self._bm25_corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc = self._bm25_corpus[idx]
                results.append(RetrievalResult(
                    id=doc["id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=scores[idx],
                    retrieval_method="sparse"
                ))
        
        return results
    
    def _dense_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Dense (embedding) retrieval."""
        results = self.vector_store.search(query, top_k=top_k)
        
        return [
            RetrievalResult(
                id=r["id"],
                content=r["content"],
                metadata=r["metadata"],
                score=r["score"],
                retrieval_method="dense"
            )
            for r in results
        ]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int = 60  # RRF constant
    ) -> List[RetrievalResult]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each ranking list
        """
        # Build ID to document mapping
        doc_map = {}
        
        for result in dense_results + sparse_results:
            if result.id not in doc_map:
                doc_map[result.id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "dense_rank": None,
                    "sparse_rank": None
                }
        
        # Assign ranks
        for rank, result in enumerate(dense_results):
            doc_map[result.id]["dense_rank"] = rank + 1
        
        for rank, result in enumerate(sparse_results):
            doc_map[result.id]["sparse_rank"] = rank + 1
        
        # Calculate RRF scores
        rrf_scores = []
        for doc_id, doc_data in doc_map.items():
            rrf_score = 0.0
            
            if doc_data["dense_rank"] is not None:
                rrf_score += self.dense_weight * (1.0 / (k + doc_data["dense_rank"]))
            
            if doc_data["sparse_rank"] is not None:
                rrf_score += self.sparse_weight * (1.0 / (k + doc_data["sparse_rank"]))
            
            rrf_scores.append(RetrievalResult(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                score=rrf_score,
                retrieval_method="hybrid"
            ))
        
        # Sort by RRF score
        rrf_scores.sort(key=lambda x: x.score, reverse=True)
        return rrf_scores
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Methods:
        - "dense": Only embedding-based retrieval
        - "sparse": Only BM25 retrieval
        - "hybrid": Combine both with RRF
        """
        top_k = top_k or settings.top_k_retrieval
        
        if method == "dense":
            return self._dense_search(query, top_k)
        elif method == "sparse":
            return self._sparse_search(query, top_k)
        elif method == "hybrid":
            # Get more results from each method for better fusion
            dense_results = self._dense_search(query, top_k * 2)
            sparse_results = self._sparse_search(query, top_k * 2)
            
            # Fuse results
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
            return fused[:top_k]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def refresh_index(self) -> None:
        """Refresh the BM25 index (call after adding new documents)."""
        self._build_bm25_index()
