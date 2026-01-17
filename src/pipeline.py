"""
Main RAG pipeline that orchestrates all components.
This is the primary interface for using the threat intelligence RAG system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from .config import settings
from .data_loader import DataLoader, Document
from .chunker import DocumentChunker, Chunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import HybridRetriever, RetrievalResult
from .reranker import CrossEncoderReranker, RerankResult
from .generator import ResponseGenerator, GeneratedResponse
from .hallucination import HallucinationDetector, HallucinationReport
from .database import DatabaseManager, DocumentRecord


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, str]]
    hallucination_warning: Optional[str]
    query_id: Optional[int] = None  # For feedback tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "citations": [
                {"document_id": c.document_id, "source": c.source, "title": c.document_title}
                for c in self.citations
            ] if self.citations and hasattr(self.citations[0], 'document_id') else self.citations,
            "hallucination_warning": self.hallucination_warning,
            "query_id": self.query_id,
            "metadata": self.metadata
        }


class ThreatIntelRAG:
    """
    Main RAG pipeline for threat intelligence queries.
    
    Usage:
        rag = ThreatIntelRAG()
        response = rag.query("What is CVE-2021-44228?")
        print(response.answer)
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator = None,
        vector_store: VectorStore = None,
        database: DatabaseManager = None,
        use_reranker: bool = True,
        check_hallucinations: bool = True
    ):
        # Initialize components
        self.embedder = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore(embedding_generator=self.embedder)
        self.retriever = HybridRetriever(self.vector_store)
        self.generator = ResponseGenerator()
        
        # Database for metadata and logging
        self.db = database or DatabaseManager()
        
        # Optional components
        self.use_reranker = use_reranker
        self.reranker = CrossEncoderReranker() if use_reranker else None
        
        self.check_hallucinations = check_hallucinations
        self.hallucination_detector = HallucinationDetector() if check_hallucinations else None
        
        # Data loader and chunker for ingestion
        self.data_loader = DataLoader()
        self.chunker = DocumentChunker()
    
    def ingest_documents(
        self,
        documents: List[Document] = None,
        chunking_strategy: str = "semantic"
    ) -> int:
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of documents to ingest. If None, loads default data.
            chunking_strategy: How to chunk documents ("semantic", "fixed", "sentence")
        
        Returns:
            Number of chunks created
        """
        # Load default data if none provided
        if documents is None:
            documents = self.data_loader.load_all_data()
        
        if not documents:
            print("No documents to ingest")
            return 0
        
        # Store document metadata in SQL database
        doc_records = []
        for doc in documents:
            doc_records.append(DocumentRecord(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                doc_type=doc.metadata.get("type", "unknown"),
                severity=doc.metadata.get("severity"),
                cvss_score=doc.metadata.get("cvss_score"),
                published_date=doc.metadata.get("published"),
                content_preview=doc.content[:500]
            ))
        
        self.db.insert_documents_batch(doc_records)
        print(f"Stored {len(doc_records)} document records in SQL database")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents, strategy=chunking_strategy)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        # Refresh BM25 index
        self.retriever.refresh_index()
        
        print(f"Ingested {len(documents)} documents as {len(chunks)} chunks")
        return len(chunks)
    
    def query(
        self,
        question: str,
        top_k: int = None,
        retrieval_method: str = "hybrid",
        filter_severity: str = None,
        filter_doc_type: str = None,
        filter_min_cvss: float = None,
        filter_after_date: str = None
    ) -> RAGResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            retrieval_method: "dense", "sparse", or "hybrid"
            filter_severity: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
            filter_doc_type: Filter by type (cve, mitre_attack)
            filter_min_cvss: Filter by minimum CVSS score
            filter_after_date: Filter by date (YYYY-MM-DD)
        
        Returns:
            RAGResponse with answer, sources, and confidence
        """
        start_time = time.time()
        
        # Step 0: Pre-filter documents using SQL if filters provided
        filter_ids = None
        if any([filter_severity, filter_doc_type, filter_min_cvss, filter_after_date]):
            filter_ids = self.db.get_document_ids_by_filter(
                severity=filter_severity,
                doc_type=filter_doc_type,
                min_cvss=filter_min_cvss,
                after_date=filter_after_date
            )
            if not filter_ids:
                return RAGResponse(
                    answer="No documents match the specified filters.",
                    confidence=0.0,
                    sources=[],
                    citations=[],
                    hallucination_warning=None,
                    query_id=None,
                    metadata={"filters_applied": True, "matching_docs": 0}
                )
        
        # Step 1: Retrieve relevant documents
        top_k = top_k or settings.top_k_retrieval
        retrieval_top_k = top_k * 3 if filter_ids else top_k  # Get 3x more if filtering
        
        retrieval_results = self.retriever.retrieve(
            question,
            top_k=retrieval_top_k,
            method=retrieval_method
        )
        
        # Apply SQL filters to retrieval results if needed
        if filter_ids:
            filter_ids_set = set(filter_ids)
            retrieval_results = [
                r for r in retrieval_results 
                if r.metadata.get("document_id", r.id.split("_chunk_")[0]) in filter_ids_set
            ][:top_k]
            
        retrieval_time = time.time() - start_time
        
        # Step 2: Rerank results (optional)
        if self.use_reranker and retrieval_results:
            rerank_results = self.reranker.rerank(
                question,
                retrieval_results,
                top_k=settings.top_k_rerank
            )
        else:
            # Convert RetrievalResult to RerankResult format
            rerank_results = [
                RerankResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    original_score=r.score,
                    rerank_score=r.score,
                    final_score=r.score
                )
                for r in retrieval_results[:settings.top_k_rerank]
            ]
        
        rerank_time = time.time() - start_time - retrieval_time
        
        # Step 3: Generate response
        if not rerank_results:
            return RAGResponse(
                answer="I couldn't find any relevant information in my knowledge base to answer this question.",
                confidence=0.0,
                sources=[],
                citations=[],
                hallucination_warning=None,
                query_id=None,
                metadata={"retrieval_time_ms": retrieval_time * 1000}
            )
        
        generated = self.generator.generate(question, rerank_results)
        
        generation_time = time.time() - start_time - retrieval_time - rerank_time
        
        # Step 4: Check for hallucinations (optional)
        hallucination_warning = None
        confidence = 0.8  # Default confidence
        
        if self.check_hallucinations:
            hall_report = self.hallucination_detector.check_response(
                generated,
                rerank_results
            )
            confidence = hall_report.confidence_score
            hallucination_warning = self.hallucination_detector.get_warning_message(hall_report)
        
        hallucination_time = time.time() - start_time - retrieval_time - rerank_time - generation_time
        
        # Format sources for response
        sources = [
            {
                "id": r.id,
                "document_id": r.metadata.get("document_id", r.id),
                "title": r.metadata.get("document_title", ""),
                "source": r.metadata.get("source", ""),
                "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "relevance_score": r.final_score
            }
            for r in rerank_results
        ]
        
        # Format citations
        citations = [
            {
                "document_id": c.document_id,
                "title": c.document_title,
                "source": c.source
            }
            for c in generated.citations
        ]
        
        total_time = time.time() - start_time
        
        # Log query to SQL database
        retrieved_doc_ids = [
            (r.metadata.get("document_id", r.id), i+1, r.final_score)
            for i, r in enumerate(rerank_results)
        ]
        
        query_id = self.db.log_query(
            query_text=question,
            response_text=generated.answer,
            confidence=confidence,
            latency_ms=int(total_time * 1000),
            tokens_used=generated.tokens_used,
            retrieval_method=retrieval_method,
            docs_retrieved=len(rerank_results),
            retrieved_doc_ids=retrieved_doc_ids
        )
        
        return RAGResponse(
            answer=generated.answer,
            confidence=confidence,
            sources=sources,
            citations=citations,
            hallucination_warning=hallucination_warning,
            query_id=query_id,
            metadata={
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "rerank_time_ms": round(rerank_time * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "hallucination_check_time_ms": round(hallucination_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2),
                "model": generated.model,
                "tokens_used": generated.tokens_used,
                "documents_retrieved": len(retrieval_results),
                "documents_after_rerank": len(rerank_results),
                "filters_applied": filter_ids is not None
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        store_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": store_stats,
            "settings": {
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "chunk_size": settings.chunk_size,
                "top_k_retrieval": settings.top_k_retrieval,
                "top_k_rerank": settings.top_k_rerank,
                "use_reranker": self.use_reranker,
                "check_hallucinations": self.check_hallucinations
            }
        }
    
    def add_feedback(self, query_id: int, rating: int, comment: str = None) -> int:
        """
        Add user feedback for a query response.
        
        Args:
            query_id: ID of the query (from RAGResponse.query_id)
            rating: 1-5 rating
            comment: Optional text feedback
        
        Returns:
            Feedback ID
        """
        return self.db.add_feedback(query_id, rating, comment)
    
    def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics summary from SQL database.
        
        Returns metrics like:
        - Total queries
        - Average confidence
        - Average latency
        - Queries by method
        - Top queried documents
        - Low-rated queries for review
        """
        return self.db.get_analytics_summary(days)
    
    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent queries from the database."""
        queries = self.db.get_recent_queries(limit)
        return [
            {
                "id": q.id,
                "query": q.query_text,
                "response": q.response_text[:200] + "..." if len(q.response_text) > 200 else q.response_text,
                "confidence": q.confidence,
                "latency_ms": q.latency_ms,
                "created_at": str(q.created_at)
            }
            for q in queries
        ]
    
    def filter_documents_sql(
        self,
        severity: str = None,
        doc_type: str = None,
        min_cvss: float = None,
        after_date: str = None,
        search_title: str = None
    ) -> List[Dict[str, Any]]:
        """
        Query documents using SQL filters (without RAG).
        Useful for browsing/exploring the knowledge base.
        """
        docs = self.db.filter_documents(
            severity=severity,
            doc_type=doc_type,
            min_cvss=min_cvss,
            after_date=after_date,
            search_title=search_title
        )
        return [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "severity": d.severity,
                "cvss_score": d.cvss_score,
                "published_date": d.published_date
            }
            for d in docs
        ]
    
    def clear(self) -> None:
        """Clear all data from the RAG system."""
        self.vector_store.clear()
        self.retriever.refresh_index()
        self.db.clear_all()
        print("RAG system cleared")


# Convenience function for quick usage
def create_rag_pipeline(**kwargs) -> ThreatIntelRAG:
    """Create and return a configured RAG pipeline."""
    return ThreatIntelRAG(**kwargs)
