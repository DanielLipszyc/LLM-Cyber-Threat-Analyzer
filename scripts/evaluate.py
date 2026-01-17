#!/usr/bin/env python3
"""
Evaluation script for the Threat Intelligence RAG system.
Runs evaluation metrics on a test dataset.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --queries eval_queries.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ThreatIntelRAG
from src.evaluator import RAGEvaluator


# Default evaluation queries
DEFAULT_EVAL_DATA = [
    {
        "query": "What is CVE-2021-44228?",
        "relevant_doc_ids": ["CVE-2021-44228"],
        "expected_keywords": ["Log4j", "JNDI", "remote code execution"]
    },
    {
        "query": "How does the Zerologon vulnerability work?",
        "relevant_doc_ids": ["CVE-2020-1472"],
        "expected_keywords": ["Netlogon", "domain controller", "NTLM"]
    },
    {
        "query": "What is the HTTP/2 Rapid Reset attack?",
        "relevant_doc_ids": ["CVE-2023-44487"],
        "expected_keywords": ["denial of service", "stream", "reset"]
    },
    {
        "query": "How can attackers steal credentials using LSASS?",
        "relevant_doc_ids": ["T1003.001"],
        "expected_keywords": ["LSASS", "Mimikatz", "credential"]
    },
    {
        "query": "What techniques are used for ransomware attacks?",
        "relevant_doc_ids": ["T1486"],
        "expected_keywords": ["encrypt", "ransom", "backup"]
    },
    {
        "query": "How do attackers use PowerShell for execution?",
        "relevant_doc_ids": ["T1059.001"],
        "expected_keywords": ["PowerShell", "script", "execution"]
    },
    {
        "query": "What is spearphishing and how to detect it?",
        "relevant_doc_ids": ["T1566.001"],
        "expected_keywords": ["email", "attachment", "social engineering"]
    },
    {
        "query": "How does PrintNightmare vulnerability work?",
        "relevant_doc_ids": ["CVE-2021-34527"],
        "expected_keywords": ["Print Spooler", "remote code execution", "SYSTEM"]
    }
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSON file with evaluation queries"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Threat Intelligence RAG - Evaluation")
    print("=" * 50)
    
    # Load evaluation data
    if args.queries:
        print(f"\nLoading queries from: {args.queries}")
        with open(args.queries) as f:
            eval_data = json.load(f)
    else:
        print("\nUsing default evaluation queries")
        eval_data = DEFAULT_EVAL_DATA
    
    print(f"Evaluating {len(eval_data)} queries...")
    
    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = ThreatIntelRAG(use_reranker=False, check_hallucinations=True)
    
    # Check if data is loaded
    stats = rag.get_stats()
    if stats["vector_store"]["total_chunks"] == 0:
        print("No data loaded. Running ingestion first...")
        rag.ingest_documents()
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = RAGEvaluator()
    
    results = []
    for i, item in enumerate(eval_data):
        print(f"\n[{i+1}/{len(eval_data)}] Query: {item['query'][:50]}...")
        
        response = rag.query(item["query"])
        
        # Get retrieved doc IDs
        retrieved_ids = [s["document_id"] for s in response.sources]
        
        # Evaluate
        result = evaluator.evaluate_query(
            query=item["query"],
            retrieved_docs=[
                {"id": s["document_id"], "content": s["content"]}
                for s in response.sources
            ],
            answer=response.answer,
            relevant_doc_ids=item.get("relevant_doc_ids", [])
        )
        
        results.append({
            "query": item["query"],
            "retrieved_ids": retrieved_ids,
            "expected_ids": item.get("relevant_doc_ids", []),
            "metrics": result.metrics,
            "confidence": response.confidence
        })
        
        # Print per-query results
        print(f"  Precision: {result.metrics.get('precision', 'N/A'):.2%}")
        print(f"  Relevance: {result.metrics.get('answer_relevance', 0):.2%}")
        print(f"  Faithfulness: {result.metrics.get('faithfulness', 0):.2%}")
        print(f"  Confidence: {response.confidence:.2%}")
    
    # Calculate summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    def avg(key):
        values = [r["metrics"].get(key, 0) for r in results]
        return sum(values) / len(values) if values else 0
    
    print(f"\nRetrieval Metrics:")
    print(f"  Average Precision:    {avg('precision'):.2%}")
    print(f"  Average Recall:       {avg('recall'):.2%}")
    print(f"  Average F1:           {avg('f1'):.2%}")
    print(f"  Average MRR:          {avg('mrr'):.2%}")
    
    print(f"\nAnswer Quality:")
    print(f"  Average Relevance:    {avg('answer_relevance'):.2%}")
    print(f"  Average Faithfulness: {avg('faithfulness'):.2%}")
    
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    print(f"\nOverall Confidence:     {avg_confidence:.2%}")
    
    # Save results if output path provided
    if args.output:
        output_data = {
            "summary": {
                "num_queries": len(results),
                "avg_precision": avg("precision"),
                "avg_recall": avg("recall"),
                "avg_relevance": avg("answer_relevance"),
                "avg_faithfulness": avg("faithfulness"),
                "avg_confidence": avg_confidence
            },
            "individual_results": results
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
