"""
Document chunking strategies for RAG.
Implements various chunking methods optimized for threat intelligence content.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re
import tiktoken

from .config import settings
from .data_loader import Document


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    content: str
    document_id: str
    document_title: str
    source: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source": self.source,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }


class DocumentChunker:
    """
    Chunks documents for embedding and retrieval.
    Supports multiple chunking strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_document(self, document: Document, strategy: str = "semantic") -> List[Chunk]:
        """
        Chunk a document using the specified strategy.
        
        Strategies:
        - "fixed": Fixed-size chunks based on token count
        - "semantic": Split on semantic boundaries (paragraphs, sections)
        - "sentence": Split on sentence boundaries
        """
        if strategy == "fixed":
            return self._fixed_chunk(document)
        elif strategy == "semantic":
            return self._semantic_chunk(document)
        elif strategy == "sentence":
            return self._sentence_chunk(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_chunk(self, document: Document) -> List[Chunk]:
        """Fixed-size chunking based on token count."""
        text = document.content
        tokens = self.encoding.encode(text)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            
            # Decode chunk tokens back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content=chunk_text.strip(),
                document_id=document.id,
                document_title=document.title,
                source=document.source,
                metadata={**document.metadata, "chunk_strategy": "fixed"},
                chunk_index=chunk_index,
                total_chunks=-1  # Will be updated after all chunks are created
            )
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _semantic_chunk(self, document: Document) -> List[Chunk]:
        """
        Semantic chunking that respects document structure.
        Splits on section boundaries while maintaining chunk size limits.
        """
        text = document.content
        
        # Split on common section patterns in security documents
        section_patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=[A-Z][a-z]+:)',  # Lines starting with "Label:"
            r'\n(?=CVE-\d)',  # CVE IDs
            r'\n(?=\d+\.)',  # Numbered sections
            r'\n(?=•|▪|◦|-\s)',  # Bullet points
        ]
        
        # First, split by major sections
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([p.strip() for p in parts if p.strip()])
            sections = new_sections
        
        # Now combine sections into chunks that fit within token limit
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            current_tokens = self.count_tokens(current_chunk)
            
            # If section alone exceeds chunk size, split it further
            if section_tokens > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        document, current_chunk.strip(), chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = ""
                
                # Split large section using fixed chunking
                temp_doc = Document(
                    id=f"{document.id}_temp",
                    title=document.title,
                    content=section,
                    source=document.source,
                    metadata=document.metadata
                )
                sub_chunks = self._fixed_chunk(temp_doc)
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{document.id}_chunk_{chunk_index}"
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.metadata["chunk_strategy"] = "semantic"
                    chunks.append(sub_chunk)
                    chunk_index += 1
            
            # If adding section would exceed limit, save current and start new
            elif current_tokens + section_tokens > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        document, current_chunk.strip(), chunk_index
                    ))
                    chunk_index += 1
                current_chunk = section
            
            # Otherwise, add section to current chunk
            else:
                current_chunk = f"{current_chunk}\n\n{section}" if current_chunk else section
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                document, current_chunk.strip(), chunk_index
            ))
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _sentence_chunk(self, document: Document) -> List[Chunk]:
        """Chunk by sentences, combining until chunk size is reached."""
        text = document.content
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            
            if self.count_tokens(test_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        document, current_chunk.strip(), chunk_index
                    ))
                    chunk_index += 1
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                document, current_chunk.strip(), chunk_index
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _create_chunk(self, document: Document, content: str, index: int) -> Chunk:
        """Helper to create a Chunk object."""
        return Chunk(
            id=f"{document.id}_chunk_{index}",
            content=content,
            document_id=document.id,
            document_title=document.title,
            source=document.source,
            metadata={**document.metadata, "chunk_strategy": "semantic"},
            chunk_index=index,
            total_chunks=-1
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = "semantic"
    ) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc, strategy)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
