"""
Streamlit UI for the Threat Intelligence RAG System.
Run with: streamlit run app.py
"""

import streamlit as st
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ThreatIntelRAG
from src.config import settings


# Page config
st.set_page_config(
    page_title="CyberThreat LLM Analyzer",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline."""
    rag = ThreatIntelRAG(
        use_reranker=False,  # Disable for faster startup
        check_hallucinations=True
    )
    
    # Check if data is loaded
    stats = rag.get_stats()
    if stats["vector_store"]["total_chunks"] == 0:
        with st.spinner("Loading threat intelligence data..."):
            rag.ingest_documents()
    
    return rag


def format_confidence(confidence: float) -> str:
    """Format confidence score with color."""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.0%})</span>'
    elif confidence >= 0.5:
        return f'<span class="confidence-medium">Medium ({confidence:.0%})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.0%})</span>'


def render_query_page(rag):
    """Render the main query interface."""
    st.header("💬 Ask a Question")
    
    # Filters
    with st.expander("🔍 Advanced Filters (SQL)"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_severity = st.selectbox(
                "Severity",
                [None, "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                format_func=lambda x: "All" if x is None else x
            )
        with col2:
            filter_type = st.selectbox(
                "Document Type",
                [None, "cve", "mitre_attack"],
                format_func=lambda x: "All" if x is None else x
            )
        with col3:
            filter_cvss = st.number_input(
                "Min CVSS Score",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5
            )
            filter_cvss = filter_cvss if filter_cvss > 0 else None
        with col4:
            filter_date = st.date_input(
                "After Date",
                value=None
            )
            filter_date = str(filter_date) if filter_date else None
    
    # Question input
    default_question = st.session_state.get("question", "")
    question = st.text_input(
        "Enter your security question:",
        value=default_question,
        placeholder="e.g., What is CVE-2021-44228?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and question:
        with st.spinner("Searching knowledge base..."):
            try:
                response = rag.query(
                    question,
                    filter_severity=filter_severity,
                    filter_doc_type=filter_type,
                    filter_min_cvss=filter_cvss,
                    filter_after_date=filter_date
                )
                
                # Store for feedback
                st.session_state.last_query_id = response.query_id
                
                # Display answer
                st.header("📝 Answer")
                st.markdown(response.answer)
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"**Confidence:** {format_confidence(response.confidence)}",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(f"**Response Time:** {response.metadata.get('total_time_ms', 0):.0f}ms")
                with col3:
                    st.markdown(f"**Tokens Used:** {response.metadata.get('tokens_used', 0)}")
                
                # Hallucination warning
                if response.hallucination_warning:
                    st.warning(response.hallucination_warning)
                
                # Feedback
                st.subheader("Rate this response")
                col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,3])
                for i, col in enumerate([col1, col2, col3, col4, col5], 1):
                    with col:
                        if st.button(f"{'⭐' * i}", key=f"rating_{i}"):
                            if response.query_id:
                                rag.add_feedback(response.query_id, i)
                                st.success(f"Thanks for your {i}-star rating!")
                
                # Sources
                st.header("📄 Sources")
                if response.sources:
                    for i, source in enumerate(response.sources):
                        with st.expander(
                            f"Source {i+1}: {source.get('title', 'Unknown')[:60]}... "
                            f"(Score: {source.get('relevance_score', 0):.2f})"
                        ):
                            st.markdown(f"**Document ID:** {source.get('document_id', 'N/A')}")
                            st.markdown(f"**Source:** {source.get('source', 'N/A')}")
                            st.markdown("**Content:**")
                            st.text(source.get('content', 'No content available'))
                else:
                    st.info("No sources found for this query.")
                
                # Metadata
                with st.expander("🔧 Technical Details"):
                    st.json(response.metadata)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)
    
    elif search_button:
        st.warning("Please enter a question.")


def render_analytics_page(rag):
    """Render the SQL analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    # Time range selector
    days = st.selectbox("Time Range", [7, 30, 90], index=1, format_func=lambda x: f"Last {x} days")
    
    # Get analytics
    try:
        analytics = rag.get_analytics(days)
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        return
    
    # Summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", analytics["total_queries"])
    with col2:
        st.metric("Avg Confidence", f"{analytics['avg_confidence']:.0%}")
    with col3:
        st.metric("Avg Latency", f"{analytics['avg_latency_ms']:.0f}ms")
    with col4:
        st.metric("Avg Rating", f"{analytics['avg_rating']:.1f}/5" if analytics['avg_rating'] else "N/A")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Queries by Retrieval Method")
        if analytics["queries_by_method"]:
            import pandas as pd
            df = pd.DataFrame(
                list(analytics["queries_by_method"].items()),
                columns=["Method", "Count"]
            )
            st.bar_chart(df.set_index("Method"))
        else:
            st.info("No query data yet")
    
    with col2:
        st.subheader("Documents by Severity")
        if analytics["documents_by_severity"]:
            import pandas as pd
            df = pd.DataFrame(
                list(analytics["documents_by_severity"].items()),
                columns=["Severity", "Count"]
            )
            st.bar_chart(df.set_index("Severity"))
        else:
            st.info("No document data")
    
    # Top queried documents
    st.subheader("Most Queried Documents")
    if analytics["top_queried_documents"]:
        for doc in analytics["top_queried_documents"][:5]:
            st.markdown(f"- **{doc['id']}**: {doc['title'][:50]}... ({doc['count']} queries)")
    else:
        st.info("No query data yet")
    
    # Low-rated queries for review
    st.subheader("⚠️ Low-Rated Queries (Need Review)")
    if analytics["low_rated_queries"]:
        for item in analytics["low_rated_queries"]:
            with st.expander(f"Rating: {item['rating']}/5 - {item['query'][:50]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Response:** {item['response']}")
                if item['comment']:
                    st.markdown(f"**User Comment:** {item['comment']}")
    else:
        st.success("No low-rated queries!")


def render_browse_page(rag):
    """Render the document browser with SQL filters."""
    st.header("📚 Browse Documents")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        severity = st.selectbox(
            "Filter by Severity",
            [None, "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            format_func=lambda x: "All" if x is None else x,
            key="browse_severity"
        )
    with col2:
        doc_type = st.selectbox(
            "Filter by Type",
            [None, "cve", "mitre_attack"],
            format_func=lambda x: "All" if x is None else x.upper(),
            key="browse_type"
        )
    with col3:
        search = st.text_input("Search Title", key="browse_search")
    
    # Get documents
    docs = rag.filter_documents_sql(
        severity=severity,
        doc_type=doc_type,
        search_title=search if search else None
    )
    
    st.markdown(f"**Found {len(docs)} documents**")
    
    # Display documents
    for doc in docs:
        severity_color = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }.get(doc.get("severity"), "⚪")
        
        with st.expander(f"{severity_color} {doc['id']}: {doc['title'][:60]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Source:** {doc['source']}")
                st.markdown(f"**Severity:** {doc.get('severity', 'N/A')}")
            with col2:
                st.markdown(f"**CVSS Score:** {doc.get('cvss_score', 'N/A')}")
                st.markdown(f"**Published:** {doc.get('published_date', 'N/A')}")


def render_history_page(rag):
    """Render query history from SQL."""
    st.header("📜 Query History")
    
    queries = rag.get_recent_queries(limit=50)
    
    if not queries:
        st.info("No queries yet. Ask a question to get started!")
        return
    
    for q in queries:
        confidence_icon = "🟢" if q["confidence"] >= 0.8 else "🟡" if q["confidence"] >= 0.5 else "🔴"
        
        with st.expander(f"{confidence_icon} {q['query'][:60]}... ({q['created_at'][:10]})"):
            st.markdown(f"**Query:** {q['query']}")
            st.markdown(f"**Response:** {q['response']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Confidence:** {q['confidence']:.0%}")
            with col2:
                st.markdown(f"**Latency:** {q['latency_ms']}ms")
            with col3:
                st.markdown(f"**Time:** {q['created_at']}")


def main():
    st.title("🔒 CyberThreat LLM Analyzer")
    
    # Load pipeline
    try:
        rag = load_rag_pipeline()
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        st.info("Make sure you have set your OPENAI_API_KEY in the .env file")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["🔍 Query", "📊 Analytics", "📚 Browse", "📜 History"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Stats
        st.header("📊 System Stats")
        stats = rag.get_stats()
        st.metric("Total Chunks", stats["vector_store"]["total_chunks"])
        st.metric("Documents", stats["vector_store"]["unique_documents"])
        
        st.divider()
        
        # Example questions (only on query page)
        if "Query" in page:
            st.header("📚 Example Questions")
            example_questions = [
                "What is Log4Shell?",
                "How does Zerologon work?",
                "PowerShell attack techniques?",
                "Critical CVEs from 2024?",
            ]
            for q in example_questions:
                if st.button(q, key=q, use_container_width=True):
                    st.session_state.question = q
    
    # Render selected page
    if "Query" in page:
        render_query_page(rag)
    elif "Analytics" in page:
        render_analytics_page(rag)
    elif "Browse" in page:
        render_browse_page(rag)
    elif "History" in page:
        render_history_page(rag)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        CyberThreat LLM Analyzer | RAG + SQL Analytics | Built with OpenAI, ChromaDB, SQLite
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
