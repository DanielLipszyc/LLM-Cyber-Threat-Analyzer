"""
Evaluation framework for RAG system.
Measures retrieval quality, answer relevance, and faithfulness.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from openai import OpenAI

from .config import settings


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    query: str
    retrieved_ids: List[str]
    expected_ids: List[str]
    answer: str
    expected_answer: Optional[str]
    metrics: Dict[str, float]


@dataclass
class EvalSummary:
    """Summary of evaluation across multiple queries."""
    num_queries: int
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_answer_relevance: float
    avg_faithfulness: float
    avg_latency_ms: float
    individual_results: List[EvalResult]


class RAGEvaluator:
    """
    Evaluate RAG system performance using various metrics.
    """
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
    
    def retrieval_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate retrieval precision.
        Precision = |retrieved ∩ relevant| / |retrieved|
        """
        if not retrieved_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_set.intersection(relevant_set)
        return len(intersection) / len(retrieved_set)
    
    def retrieval_recall(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate retrieval recall.
        Recall = |retrieved ∩ relevant| / |relevant|
        """
        if not relevant_ids:
            return 1.0  # If nothing is relevant, we got everything
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_set.intersection(relevant_set)
        return len(intersection) / len(relevant_set)
    
    def retrieval_f1(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate F1 score for retrieval."""
        precision = self.retrieval_precision(retrieved_ids, relevant_ids)
        recall = self.retrieval_recall(retrieved_ids, relevant_ids)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        MRR = 1 / rank of first relevant document
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        """
        import math
        
        k = k or len(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1 if doc_id in relevant_set else 0
            dcg += rel / math.log2(i + 2)  # +2 because index starts at 0
        
        # Calculate ideal DCG
        ideal_relevance = [1] * min(len(relevant_ids), k) + [0] * (k - len(relevant_ids))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance) if rel)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate how well the answer addresses the query using LLM.
        """
        prompt = f"""Rate how well this answer addresses the question on a scale of 0-10.

Question: {query}

Answer: {answer}

Consider:
- Does the answer directly address what was asked?
- Is the answer complete?
- Is the answer accurate and specific?

Respond with only a number from 0-10."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return score / 10.0  # Normalize to 0-1
        except:
            return 0.5
    
    def faithfulness(
        self,
        answer: str,
        sources: List[str]
    ) -> float:
        """
        Evaluate if the answer is faithful to the source documents.
        """
        sources_text = "\n---\n".join(sources[:5])  # Limit sources for prompt
        
        prompt = f"""Evaluate if the answer is faithful to the source documents.

Sources:
{sources_text}

Answer:
{answer}

Rate faithfulness from 0-10:
- 10: Every claim in the answer is directly supported by sources
- 5: Some claims are supported, some are not
- 0: The answer contains information not in the sources

Respond with only a number from 0-10."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return score / 10.0
        except:
            return 0.5
    
    def evaluate_query(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        relevant_doc_ids: List[str] = None,
        expected_answer: str = None
    ) -> EvalResult:
        """
        Evaluate a single query-answer pair.
        """
        retrieved_ids = [doc.get("id", "") for doc in retrieved_docs]
        retrieved_contents = [doc.get("content", "") for doc in retrieved_docs]
        
        metrics = {}
        
        # Retrieval metrics (if ground truth is provided)
        if relevant_doc_ids:
            metrics["precision"] = self.retrieval_precision(retrieved_ids, relevant_doc_ids)
            metrics["recall"] = self.retrieval_recall(retrieved_ids, relevant_doc_ids)
            metrics["f1"] = self.retrieval_f1(retrieved_ids, relevant_doc_ids)
            metrics["mrr"] = self.mrr(retrieved_ids, relevant_doc_ids)
            metrics["ndcg"] = self.ndcg(retrieved_ids, relevant_doc_ids)
        
        # Answer quality metrics
        metrics["answer_relevance"] = self.answer_relevance(query, answer)
        metrics["faithfulness"] = self.faithfulness(answer, retrieved_contents)
        
        return EvalResult(
            query=query,
            retrieved_ids=retrieved_ids,
            expected_ids=relevant_doc_ids or [],
            answer=answer,
            expected_answer=expected_answer,
            metrics=metrics
        )
    
    def evaluate_dataset(
        self,
        eval_data: List[Dict[str, Any]],
        rag_pipeline  # ThreatIntelRAG instance
    ) -> EvalSummary:
        """
        Evaluate RAG system on a dataset of queries.
        
        eval_data format:
        [
            {
                "query": "What is CVE-2021-44228?",
                "relevant_doc_ids": ["CVE-2021-44228"],
                "expected_answer": "..."  # Optional
            },
            ...
        ]
        """
        import time
        
        results = []
        total_latency = 0
        
        for item in eval_data:
            query = item["query"]
            relevant_ids = item.get("relevant_doc_ids", [])
            expected_answer = item.get("expected_answer")
            
            # Time the query
            start = time.time()
            response = rag_pipeline.query(query)
            latency = (time.time() - start) * 1000  # ms
            total_latency += latency
            
            # Format retrieved docs for evaluation
            retrieved_docs = [
                {"id": s["id"], "content": s.get("content", "")}
                for s in response.sources
            ]
            
            result = self.evaluate_query(
                query=query,
                retrieved_docs=retrieved_docs,
                answer=response.answer,
                relevant_doc_ids=relevant_ids,
                expected_answer=expected_answer
            )
            results.append(result)
        
        # Calculate averages
        def safe_avg(key):
            values = [r.metrics.get(key, 0) for r in results if key in r.metrics]
            return sum(values) / len(values) if values else 0.0
        
        return EvalSummary(
            num_queries=len(results),
            avg_retrieval_precision=safe_avg("precision"),
            avg_retrieval_recall=safe_avg("recall"),
            avg_answer_relevance=safe_avg("answer_relevance"),
            avg_faithfulness=safe_avg("faithfulness"),
            avg_latency_ms=total_latency / len(results) if results else 0,
            individual_results=results
        )
    
    def print_summary(self, summary: EvalSummary) -> None:
        """Print evaluation summary in a readable format."""
        print("\n" + "=" * 50)
        print("RAG EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Queries evaluated: {summary.num_queries}")
        print(f"\nRetrieval Metrics:")
        print(f"  Precision:  {summary.avg_retrieval_precision:.2%}")
        print(f"  Recall:     {summary.avg_retrieval_recall:.2%}")
        print(f"\nAnswer Quality:")
        print(f"  Relevance:   {summary.avg_answer_relevance:.2%}")
        print(f"  Faithfulness: {summary.avg_faithfulness:.2%}")
        print(f"\nPerformance:")
        print(f"  Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print("=" * 50)
