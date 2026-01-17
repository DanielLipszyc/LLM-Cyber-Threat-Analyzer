"""
PostgreSQL database integration for metadata, query logging, and analytics.
Handles all structured data while ChromaDB handles vectors.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# Use SQLite for easy local development, can swap to PostgreSQL for production
import sqlite3
from contextlib import contextmanager

from .config import settings


@dataclass
class DocumentRecord:
    """Structured metadata for a document."""
    id: str
    title: str
    source: str
    doc_type: str  # 'cve' or 'mitre_attack'
    severity: Optional[str]
    cvss_score: Optional[float]
    published_date: Optional[str]
    content_preview: str
    created_at: datetime = None


@dataclass
class QueryRecord:
    """Record of a user query."""
    id: int
    query_text: str
    response_text: str
    confidence: float
    latency_ms: int
    tokens_used: int
    retrieval_method: str
    docs_retrieved: int
    created_at: datetime


@dataclass
class FeedbackRecord:
    """User feedback on a response."""
    id: int
    query_id: int
    rating: int  # 1-5
    comment: Optional[str]
    created_at: datetime


class DatabaseManager:
    """
    Manages PostgreSQL/SQLite database for structured data.
    
    Handles:
    - Document metadata
    - Query logging
    - User feedback
    - Analytics
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", "./threat_intel.db")
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(100) PRIMARY KEY,
                    title TEXT,
                    source VARCHAR(100),
                    doc_type VARCHAR(50),
                    severity VARCHAR(20),
                    cvss_score REAL,
                    published_date DATE,
                    content_preview TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    response_text TEXT,
                    confidence REAL,
                    latency_ms INTEGER,
                    tokens_used INTEGER,
                    retrieval_method VARCHAR(20),
                    docs_retrieved INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER REFERENCES queries(id),
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Query-document join table (which docs were retrieved for which query)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_documents (
                    query_id INTEGER REFERENCES queries(id),
                    document_id VARCHAR(100) REFERENCES documents(id),
                    rank INTEGER,
                    relevance_score REAL,
                    PRIMARY KEY (query_id, document_id)
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_severity ON documents(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON documents(published_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_date ON queries(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
    
    # ==================== Document Operations ====================
    
    def insert_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, source, doc_type, severity, cvss_score, published_date, content_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id,
                doc.title,
                doc.source,
                doc.doc_type,
                doc.severity,
                doc.cvss_score,
                doc.published_date,
                doc.content_preview,
                doc.created_at or datetime.now()
            ))
    
    def insert_documents_batch(self, docs: List[DocumentRecord]) -> None:
        """Batch insert documents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO documents 
                (id, title, source, doc_type, severity, cvss_score, published_date, content_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (d.id, d.title, d.source, d.doc_type, d.severity, d.cvss_score, 
                 d.published_date, d.content_preview, d.created_at or datetime.now())
                for d in docs
            ])
    
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get a document by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_document(row)
        return None
    
    def filter_documents(
        self,
        severity: str = None,
        doc_type: str = None,
        min_cvss: float = None,
        after_date: str = None,
        before_date: str = None,
        search_title: str = None,
        limit: int = 100
    ) -> List[DocumentRecord]:
        """
        Filter documents by structured fields.
        Returns document IDs that can be used to filter RAG retrieval.
        """
        conditions = []
        params = []
        
        if severity:
            conditions.append("UPPER(severity) = ?")
            params.append(severity.upper())
        
        if doc_type:
            conditions.append("doc_type = ?")
            params.append(doc_type)
        
        if min_cvss is not None:
            conditions.append("cvss_score >= ?")
            params.append(min_cvss)
        
        if after_date:
            conditions.append("published_date >= ?")
            params.append(after_date)
        
        if before_date:
            conditions.append("published_date <= ?")
            params.append(before_date)
        
        if search_title:
            conditions.append("title LIKE ?")
            params.append(f"%{search_title}%")
        
        query = "SELECT * FROM documents"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_document(row) for row in cursor.fetchall()]
    
    def get_document_ids_by_filter(self, **filters) -> List[str]:
        """Get just document IDs matching filters (for RAG pre-filtering)."""
        docs = self.filter_documents(**filters)
        return [d.id for d in docs]
    
    # ==================== Query Logging ====================
    
    def log_query(
        self,
        query_text: str,
        response_text: str,
        confidence: float,
        latency_ms: int,
        tokens_used: int,
        retrieval_method: str,
        docs_retrieved: int,
        retrieved_doc_ids: List[tuple] = None  # [(doc_id, rank, score), ...]
    ) -> int:
        """Log a query and its response. Returns query ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert query
            cursor.execute("""
                INSERT INTO queries 
                (query_text, response_text, confidence, latency_ms, tokens_used, retrieval_method, docs_retrieved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (query_text, response_text, confidence, latency_ms, tokens_used, retrieval_method, docs_retrieved))
            
            query_id = cursor.lastrowid
            
            # Log which documents were retrieved
            if retrieved_doc_ids:
                cursor.executemany("""
                    INSERT OR IGNORE INTO query_documents (query_id, document_id, rank, relevance_score)
                    VALUES (?, ?, ?, ?)
                """, [(query_id, doc_id, rank, score) for doc_id, rank, score in retrieved_doc_ids])
            
            return query_id
    
    def get_recent_queries(self, limit: int = 50) -> List[QueryRecord]:
        """Get recent queries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM queries ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [self._row_to_query(row) for row in cursor.fetchall()]
    
    # ==================== Feedback ====================
    
    def add_feedback(self, query_id: int, rating: int, comment: str = None) -> int:
        """Add user feedback for a query."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (query_id, rating, comment)
                VALUES (?, ?, ?)
            """, (query_id, rating, comment))
            return cursor.lastrowid
    
    def get_feedback_for_query(self, query_id: int) -> List[FeedbackRecord]:
        """Get all feedback for a query."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE query_id = ?", (query_id,))
            return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    # ==================== Analytics ====================
    
    def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for the dashboard."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total queries
            cursor.execute(
                "SELECT COUNT(*) FROM queries WHERE created_at >= ?", 
                (cutoff,)
            )
            total_queries = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute(
                "SELECT AVG(confidence) FROM queries WHERE created_at >= ?",
                (cutoff,)
            )
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Average latency
            cursor.execute(
                "SELECT AVG(latency_ms) FROM queries WHERE created_at >= ?",
                (cutoff,)
            )
            avg_latency = cursor.fetchone()[0] or 0
            
            # Average rating
            cursor.execute("""
                SELECT AVG(f.rating) FROM feedback f
                JOIN queries q ON f.query_id = q.id
                WHERE q.created_at >= ?
            """, (cutoff,))
            avg_rating = cursor.fetchone()[0] or 0
            
            # Queries by retrieval method
            cursor.execute("""
                SELECT retrieval_method, COUNT(*) 
                FROM queries WHERE created_at >= ?
                GROUP BY retrieval_method
            """, (cutoff,))
            by_method = dict(cursor.fetchall())
            
            # Documents by severity
            cursor.execute("""
                SELECT severity, COUNT(*) FROM documents
                WHERE severity IS NOT NULL
                GROUP BY severity
            """)
            by_severity = dict(cursor.fetchall())
            
            # Most queried documents
            cursor.execute("""
                SELECT d.id, d.title, COUNT(*) as query_count
                FROM query_documents qd
                JOIN documents d ON qd.document_id = d.id
                JOIN queries q ON qd.query_id = q.id
                WHERE q.created_at >= ?
                GROUP BY d.id
                ORDER BY query_count DESC
                LIMIT 10
            """, (cutoff,))
            top_docs = [{"id": r[0], "title": r[1], "count": r[2]} for r in cursor.fetchall()]
            
            # Low-rated queries (for review)
            cursor.execute("""
                SELECT q.query_text, q.response_text, f.rating, f.comment
                FROM queries q
                JOIN feedback f ON q.id = f.query_id
                WHERE f.rating <= 2 AND q.created_at >= ?
                ORDER BY f.created_at DESC
                LIMIT 10
            """, (cutoff,))
            low_rated = [
                {"query": r[0], "response": r[1][:200], "rating": r[2], "comment": r[3]}
                for r in cursor.fetchall()
            ]
            
            return {
                "period_days": days,
                "total_queries": total_queries,
                "avg_confidence": round(avg_confidence, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_rating": round(avg_rating, 2),
                "queries_by_method": by_method,
                "documents_by_severity": by_severity,
                "top_queried_documents": top_docs,
                "low_rated_queries": low_rated
            }
    
    def get_queries_over_time(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get query counts by day for charting."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM queries
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (cutoff,))
            return [{"date": r[0], "count": r[1]} for r in cursor.fetchall()]
    
    def get_confidence_distribution(self) -> List[Dict[str, Any]]:
        """Get distribution of confidence scores."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN confidence >= 0.8 THEN 'High (80-100%)'
                        WHEN confidence >= 0.5 THEN 'Medium (50-79%)'
                        ELSE 'Low (0-49%)'
                    END as bucket,
                    COUNT(*) as count
                FROM queries
                GROUP BY bucket
            """)
            return [{"bucket": r[0], "count": r[1]} for r in cursor.fetchall()]
    
    # ==================== Helpers ====================
    
    def _row_to_document(self, row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            doc_type=row["doc_type"],
            severity=row["severity"],
            cvss_score=row["cvss_score"],
            published_date=row["published_date"],
            content_preview=row["content_preview"],
            created_at=row["created_at"]
        )
    
    def _row_to_query(self, row) -> QueryRecord:
        return QueryRecord(
            id=row["id"],
            query_text=row["query_text"],
            response_text=row["response_text"],
            confidence=row["confidence"],
            latency_ms=row["latency_ms"],
            tokens_used=row["tokens_used"],
            retrieval_method=row["retrieval_method"],
            docs_retrieved=row["docs_retrieved"],
            created_at=row["created_at"]
        )
    
    def _row_to_feedback(self, row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["id"],
            query_id=row["query_id"],
            rating=row["rating"],
            comment=row["comment"],
            created_at=row["created_at"]
        )
    
    def clear_all(self):
        """Clear all data (for testing)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feedback")
            cursor.execute("DELETE FROM query_documents")
            cursor.execute("DELETE FROM queries")
            cursor.execute("DELETE FROM documents")
