"""
Hallucination detection for RAG responses.
Verifies that generated claims are supported by source documents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from openai import OpenAI

from .config import settings, HALLUCINATION_CHECK_PROMPT
from .generator import GeneratedResponse
from .reranker import RerankResult


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    status: str  # "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"
    supporting_source: Optional[str]
    confidence: float


@dataclass
class HallucinationReport:
    """Complete hallucination analysis report."""
    claims: List[ClaimVerification]
    overall_faithfulness: float  # 0-1, higher is better
    hallucination_rate: float  # 0-1, lower is better
    flagged_claims: List[str]  # Claims that need attention
    confidence_score: float  # Overall confidence in the response


class HallucinationDetector:
    """
    Detect hallucinations in generated responses by verifying
    claims against source documents.
    """
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.threshold = settings.hallucination_threshold
    
    def check_response(
        self,
        response: GeneratedResponse,
        retrieved_results: List[RerankResult]
    ) -> HallucinationReport:
        """
        Check a generated response for hallucinations.
        
        Args:
            response: The generated response to check
            retrieved_results: The source documents used for generation
        
        Returns:
            HallucinationReport with detailed analysis
        """
        # Format sources for the prompt
        sources_text = self._format_sources(retrieved_results)
        
        # Build verification prompt
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            sources=sources_text,
            answer=response.answer
        )
        
        # Get LLM verification
        llm_response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )
        
        result_text = llm_response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"claims": [], "overall_faithfulness": 0.5}
        except json.JSONDecodeError:
            # Fallback if parsing fails
            result = {"claims": [], "overall_faithfulness": 0.5}
        
        # Process claims
        claims = []
        flagged_claims = []
        
        for claim_data in result.get("claims", []):
            claim = ClaimVerification(
                claim=claim_data.get("claim", ""),
                status=claim_data.get("status", "UNKNOWN"),
                supporting_source=claim_data.get("source"),
                confidence=self._status_to_confidence(claim_data.get("status", ""))
            )
            claims.append(claim)
            
            if claim.status in ["NOT_SUPPORTED", "PARTIALLY_SUPPORTED"]:
                flagged_claims.append(claim.claim)
        
        # Calculate metrics
        overall_faithfulness = result.get("overall_faithfulness", 0.5)
        
        if claims:
            supported_count = sum(
                1 for c in claims if c.status == "SUPPORTED"
            )
            hallucination_rate = 1 - (supported_count / len(claims))
        else:
            hallucination_rate = 0.0
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            overall_faithfulness,
            hallucination_rate,
            len(retrieved_results)
        )
        
        return HallucinationReport(
            claims=claims,
            overall_faithfulness=overall_faithfulness,
            hallucination_rate=hallucination_rate,
            flagged_claims=flagged_claims,
            confidence_score=confidence_score
        )
    
    def _format_sources(self, results: List[RerankResult]) -> str:
        """Format source documents for verification prompt."""
        sources = []
        for i, result in enumerate(results):
            doc_id = result.metadata.get("document_id", result.id)
            sources.append(f"[Source {i+1}: {doc_id}]\n{result.content}\n")
        return "\n---\n".join(sources)
    
    def _status_to_confidence(self, status: str) -> float:
        """Convert status to confidence score."""
        return {
            "SUPPORTED": 1.0,
            "PARTIALLY_SUPPORTED": 0.6,
            "NOT_SUPPORTED": 0.0,
            "UNKNOWN": 0.5
        }.get(status.upper(), 0.5)
    
    def _calculate_confidence(
        self,
        faithfulness: float,
        hallucination_rate: float,
        num_sources: int
    ) -> float:
        """
        Calculate overall confidence score.
        
        Factors:
        - Faithfulness to sources
        - Hallucination rate
        - Number of supporting sources
        """
        # More sources = more confidence (up to a point)
        source_factor = min(num_sources / 5, 1.0)
        
        # Combine factors
        confidence = (
            0.5 * faithfulness +
            0.3 * (1 - hallucination_rate) +
            0.2 * source_factor
        )
        
        return round(confidence, 3)
    
    def quick_check(self, answer: str, sources: List[str]) -> float:
        """
        Quick heuristic check for potential hallucinations.
        Returns a confidence score without full LLM verification.
        """
        # Check for hedging language (often indicates uncertainty)
        hedging_phrases = [
            "I think", "probably", "might be", "could be",
            "I'm not sure", "it's possible", "may be"
        ]
        hedge_count = sum(1 for phrase in hedging_phrases if phrase.lower() in answer.lower())
        
        # Check for specific citations
        citation_pattern = r'\[(?:Source|Document|CVE|T\d+)[^\]]*\]'
        citations = re.findall(citation_pattern, answer)
        
        # Check if key terms from sources appear in answer
        source_text = " ".join(sources).lower()
        answer_lower = answer.lower()
        
        # Extract key technical terms
        tech_terms = re.findall(r'\b(?:CVE-\d+-\d+|T\d+\.\d+|[A-Z]{2,})\b', source_text)
        matching_terms = sum(1 for term in tech_terms if term.lower() in answer_lower)
        
        # Calculate quick confidence
        citation_score = min(len(citations) / 3, 1.0) * 0.4
        term_score = min(matching_terms / 5, 1.0) * 0.4
        hedge_penalty = min(hedge_count * 0.1, 0.3)
        
        confidence = citation_score + term_score + 0.2 - hedge_penalty
        return max(0, min(1, confidence))
    
    def get_warning_message(self, report: HallucinationReport) -> Optional[str]:
        """Generate a warning message if hallucination risk is high."""
        if report.confidence_score < self.threshold:
            warnings = []
            
            if report.hallucination_rate > 0.3:
                warnings.append(
                    f"⚠️ {int(report.hallucination_rate * 100)}% of claims may not be "
                    f"fully supported by the source documents."
                )
            
            if report.flagged_claims:
                warnings.append(
                    f"⚠️ The following claims need verification:\n" +
                    "\n".join(f"  - {claim}" for claim in report.flagged_claims[:3])
                )
            
            if warnings:
                return "\n".join(warnings)
        
        return None
