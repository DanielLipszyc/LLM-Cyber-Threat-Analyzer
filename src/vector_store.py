"""
Vector store implementation using ChromaDB.
Handles storage and retrieval of document embeddings.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .chunker import Chunk
from .embeddings import EmbeddingGenerator


class VectorStore:
    """
    ChromaDB-based vector store for document chunks.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_generator: EmbeddingGenerator = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        
        # Initialize embedding generator
        self.embedder = embedding_generator or EmbeddingGenerator()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        """
        Add document chunks to the vector store.
        Generates embeddings and stores with metadata.
        """
        if not chunks:
            return
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = [chunk.id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    **{k: str(v) for k, v in chunk.metadata.items()}
                }
                for chunk in batch
            ]
            
            # Generate embeddings
            embeddings = self.embedder.embed_texts(documents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"Added batch {i // batch_size + 1}, total: {min(i + batch_size, len(chunks))}/{len(chunks)}")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using dense retrieval.
        
        Returns list of results with:
        - id: chunk ID
        - content: chunk text
        - metadata: chunk metadata
        - score: similarity score (higher is better)
        """
        top_k = top_k or settings.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Build where clause for filtering
        where = filter_metadata if filter_metadata else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": similarity
                })
        
        return formatted_results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID."""
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
        return None
    
    def get_all_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks belonging to a specific document."""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append({
                    "id": chunk_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })
        
        return sorted(chunks, key=lambda x: x["metadata"].get("chunk_index", 0))
    
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.count()
        
        # Get unique document count
        all_data = self.collection.get(include=["metadatas"])
        unique_docs = set()
        sources = {}
        
        if all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                doc_id = metadata.get("document_id", "unknown")
                source = metadata.get("source", "unknown")
                unique_docs.add(doc_id)
                sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_chunks": count,
            "unique_documents": len(unique_docs),
            "sources": sources
        }
