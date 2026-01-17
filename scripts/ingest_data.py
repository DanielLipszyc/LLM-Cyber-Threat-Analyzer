#!/usr/bin/env python3
"""
Data ingestion script for the Threat Intelligence RAG system.
Loads threat intel data into the vector store.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --input custom_data.json
    python scripts/ingest_data.py --clear  # Clear existing data first
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ThreatIntelRAG
from src.data_loader import DataLoader, Document


def main():
    parser = argparse.ArgumentParser(description="Ingest threat intelligence data")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to custom JSON data file to ingest"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["semantic", "fixed", "sentence"],
        help="Chunking strategy to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Threat Intelligence RAG - Data Ingestion")
    print("=" * 50)
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    rag = ThreatIntelRAG(use_reranker=False, check_hallucinations=False)
    
    # Clear if requested
    if args.clear:
        print("Clearing existing data...")
        rag.clear()
    
    # Load data
    if args.input:
        print(f"\nLoading custom data from: {args.input}")
        data_loader = DataLoader()
        documents = data_loader.load_json_file(Path(args.input))
    else:
        print("\nLoading default threat intelligence data...")
        documents = None  # Will load default data
    
    # Ingest
    print(f"\nIngesting documents with '{args.strategy}' chunking strategy...")
    num_chunks = rag.ingest_documents(
        documents=documents,
        chunking_strategy=args.strategy
    )
    
    # Print stats
    stats = rag.get_stats()
    print("\n" + "=" * 50)
    print("Ingestion Complete!")
    print("=" * 50)
    print(f"Total chunks: {stats['vector_store']['total_chunks']}")
    print(f"Unique documents: {stats['vector_store']['unique_documents']}")
    print("\nSources:")
    for source, count in stats['vector_store']['sources'].items():
        print(f"  - {source}: {count} chunks")
    
    print("\nRun 'streamlit run app.py' to start the UI")


if __name__ == "__main__":
    main()
