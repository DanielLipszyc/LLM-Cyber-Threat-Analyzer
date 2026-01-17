"""
LLM response generation with citation tracking.
Generates answers grounded in retrieved documents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI

from .config import settings, SYSTEM_PROMPT, GENERATION_PROMPT
from .reranker import RerankResult


@dataclass
class Citation:
    """A citation to a source document."""
    document_id: str
    document_title: str
    source: str
    chunk_id: str
    relevant_text: str


@dataclass
class GeneratedResponse:
    """Response from the generator with citations."""
    answer: str
    citations: List[Citation]
    sources_used: List[Dict[str, Any]]
    model: str
    tokens_used: int


class ResponseGenerator:
    """
    Generate responses using LLM with retrieved context.
    Includes citation tracking and source attribution.
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.llm_model
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
    
    def _format_context(self, results: List[RerankResult]) -> str:
        """Format retrieved documents as context for the LLM."""
        context_parts = []
        
        for i, result in enumerate(results):
            doc_id = result.metadata.get("document_id", result.id)
            source = result.metadata.get("source", "Unknown")
            title = result.metadata.get("document_title", "Untitled")
            
            context_parts.append(f"""
[Document {i+1}]
ID: {doc_id}
Source: {source}
Title: {title}
Content:
{result.content}
---""")
        
        return "\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        retrieved_results: List[RerankResult],
        max_tokens: int = None,
        temperature: float = None
    ) -> GeneratedResponse:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: User's question
            retrieved_results: Reranked retrieval results
            max_tokens: Maximum tokens in response
            temperature: LLM temperature (lower = more deterministic)
        
        Returns:
            GeneratedResponse with answer and citations
        """
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature if temperature is not None else settings.temperature
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_results)
        
        # Build prompt
        user_prompt = GENERATION_PROMPT.format(
            context=context,
            question=query
        )
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Extract citations from the response
        citations = self._extract_citations(answer, retrieved_results)
        
        # Track which sources were actually used
        sources_used = [
            {
                "id": result.id,
                "document_id": result.metadata.get("document_id", ""),
                "title": result.metadata.get("document_title", ""),
                "source": result.metadata.get("source", ""),
                "relevance_score": result.final_score
            }
            for result in retrieved_results
        ]
        
        return GeneratedResponse(
            answer=answer,
            citations=citations,
            sources_used=sources_used,
            model=self.model,
            tokens_used=tokens_used
        )
    
    def _extract_citations(
        self,
        answer: str,
        results: List[RerankResult]
    ) -> List[Citation]:
        """Extract citations from the generated answer."""
        import re
        
        citations = []
        
        # Look for citation patterns like [Source: X] or [Document X]
        citation_patterns = [
            r'\[Source:\s*([^\]]+)\]',
            r'\[Document\s*(\d+)\]',
            r'\[(\w+-\d+-\d+)\]',  # CVE format
            r'\[(T\d+(?:\.\d+)?)\]',  # MITRE ATT&CK format
        ]
        
        found_refs = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer)
            found_refs.update(matches)
        
        # Map references to actual documents
        for ref in found_refs:
            # Try to match with retrieved documents
            for result in results:
                doc_id = result.metadata.get("document_id", result.id)
                
                # Check if reference matches document ID or index
                if (ref.lower() in doc_id.lower() or 
                    ref.isdigit() and int(ref) <= len(results)):
                    
                    citations.append(Citation(
                        document_id=doc_id,
                        document_title=result.metadata.get("document_title", ""),
                        source=result.metadata.get("source", ""),
                        chunk_id=result.id,
                        relevant_text=result.content[:200] + "..."
                    ))
                    break
        
        return citations
    
    def generate_with_query_expansion(
        self,
        query: str,
        retrieved_results: List[RerankResult]
    ) -> GeneratedResponse:
        """
        Generate response with automatic query expansion.
        First expands the query, then generates the answer.
        """
        # Expand query
        expansion_prompt = f"""Generate 3 related search queries that would help answer this security question:

Question: {query}

Respond with just the queries, one per line."""
        
        expansion_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        expanded_queries = expansion_response.choices[0].message.content.strip()
        
        # Add expanded context to the main generation
        enhanced_prompt = f"""Original question: {query}

Related aspects to consider:
{expanded_queries}

Please answer the original question comprehensively."""
        
        return self.generate(enhanced_prompt, retrieved_results)
