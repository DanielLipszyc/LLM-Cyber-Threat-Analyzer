# 🔒 RAG-Powered Security Threat Intelligence Assistant

A Retrieval-Augmented Generation (RAG) system for querying security threat intelligence data. Ask questions about CVEs, attack techniques, malware, and vulnerabilities in natural language.

## Features

- **Hybrid Retrieval**: Combines dense embeddings + BM25 sparse retrieval for better accuracy
- **Cross-Encoder Reranking**: Reranks results for improved precision
- **Hallucination Detection**: Flags responses that may not be grounded in retrieved documents
- **Citation Tracking**: Every claim links back to source documents
- **Evaluation Framework**: Measure retrieval precision, answer relevance, and faithfulness

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│            "What vulnerabilities affect Log4j?"                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Query Processing                           │
│  • Query expansion (generate related terms)                     │
│  • Query embedding (dense vector)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Retrieval                            │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │ Dense Retrieval │         │ Sparse (BM25)   │                │
│  │   (Embeddings)  │         │   Retrieval     │                │
│  └─────────────────┘         └─────────────────┘                │
│            │                         │                          │
│            └────────┬────────────────┘                          │
│                     ▼                                           │
│           Reciprocal Rank Fusion                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Cross-Encoder Reranking                       │
│        Rerank top candidates for final selection                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Generation                             │
│  • Generate answer grounded in retrieved docs                   │
│  • Include citations for each claim                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Hallucination Detection                        │
│  • Check if claims are supported by sources                     │
│  • Flag unsupported statements                                  │
│  • Compute confidence score                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Response                                 │
│  • Answer with citations                                        │
│  • Confidence score                                             │
│  • Source documents                                             │
│  • Hallucination warnings (if any)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
threat-intel-rag/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and settings
│   ├── data_loader.py      # Load and process threat intel data
│   ├── chunker.py          # Document chunking strategies
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # ChromaDB vector store
│   ├── retriever.py        # Hybrid retrieval (dense + BM25)
│   ├── reranker.py         # Cross-encoder reranking
│   ├── generator.py        # LLM response generation
│   ├── hallucination.py    # Hallucination detection
│   ├── evaluator.py        # RAG evaluation metrics
│   └── pipeline.py         # End-to-end RAG pipeline
├── data/
│   ├── cve_sample.json     # Sample CVE data
│   └── mitre_attack.json   # Sample MITRE ATT&CK data
├── scripts/
│   ├── ingest_data.py      # Data ingestion script
│   └── evaluate.py         # Run evaluation
├── tests/
│   └── test_pipeline.py    # Unit tests
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/threat-intel-rag.git
cd threat-intel-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Ingest Data

```bash
# Ingest sample threat intelligence data
python scripts/ingest_data.py
```

### 4. Run the App

```bash
# Start Streamlit UI
streamlit run app.py
```

### 5. Use the API (Optional)

```python
from src.pipeline import ThreatIntelRAG

# Initialize pipeline
rag = ThreatIntelRAG()

# Ask a question
response = rag.query("What is CVE-2021-44228?")

print(response.answer)
print(f"Confidence: {response.confidence}")
print(f"Sources: {response.sources}")
```

## Configuration

Edit `src/config.py` or use environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `EMBEDDING_MODEL` | Embedding model to use | `text-embedding-3-small` |
| `LLM_MODEL` | LLM for generation | `gpt-4o-mini` |
| `CHUNK_SIZE` | Document chunk size | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K_RETRIEVAL` | Number of docs to retrieve | `10` |
| `TOP_K_RERANK` | Number of docs after reranking | `5` |

## Evaluation

Run the evaluation suite to measure RAG performance:

```bash
python scripts/evaluate.py
```

Metrics computed:
- **Retrieval Precision**: % of retrieved docs that are relevant
- **Retrieval Recall**: % of relevant docs that were retrieved
- **Answer Relevance**: How well the answer addresses the question
- **Faithfulness**: % of claims supported by retrieved documents
- **Hallucination Rate**: % of unsupported claims

## Deployment

### Docker

```bash
# Build image
docker build -t threat-intel-rag .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key threat-intel-rag
```

## Adding Custom Data

1. Prepare your data as JSON with this structure:
```json
{
  "id": "unique-id",
  "title": "Document Title",
  "content": "Full text content...",
  "source": "Source name",
  "metadata": {
    "date": "2024-01-01",
    "category": "vulnerability"
  }
}
```

2. Run ingestion:
```bash
python scripts/ingest_data.py --input your_data.json
```

## License

MIT License - see LICENSE file for details.
