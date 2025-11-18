"""
Streamlit UI for StudyBuddy application.
Provides an interactive web interface for document upload, Q&A, summaries, and flashcards.
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from datetime import datetime

# Import custom modules
from ingest import DocumentIngestionPipeline
from embeddings import EmbeddingGenerator
from vectorstore import VectorStore
from rag import RAGPipeline
from utils import FlashcardExporter, format_file_size

# Page configuration
st.set_page_config(
    page_title="StudyBuddy - RAG Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'ingestion_pipeline' not in st.session_state:
        st.session_state.ingestion_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def initialize_components():
    """Initialize all components if not already initialized."""
    try:
        if st.session_state.embedder is None:
            with st.spinner("Initializing embedding model..."):
                # Use OpenRouter by default
                st.session_state.embedder = EmbeddingGenerator(use_openrouter=True, use_local=False)

        if st.session_state.vectorstore is None:
            with st.spinner("Initializing vector store..."):
                st.session_state.vectorstore = VectorStore()

        if st.session_state.ingestion_pipeline is None:
            chunk_size = int(os.getenv('CHUNK_SIZE', 500))
            chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
            st.session_state.ingestion_pipeline = DocumentIngestionPipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        if st.session_state.rag is None:
            st.session_state.rag = RAGPipeline(
                vectorstore=st.session_state.vectorstore,
                embedder=st.session_state.embedder
            )

        return True

    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.info("Make sure your .env file is configured with OPENROUTER_API_KEY")
        return False


def sidebar_upload():
    """Sidebar for file upload and document management."""
    st.sidebar.title("📚 Document Management")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload study materials to analyze"
    )

    # Ingest button
    if uploaded_file is not None:
        st.sidebar.write(f"**File:** {uploaded_file.name}")
        st.sidebar.write(f"**Size:** {format_file_size(uploaded_file.size)}")

        if st.sidebar.button("📥 Ingest Document", type="primary"):
            ingest_document(uploaded_file)

    st.sidebar.markdown("---")

    # Show uploaded documents
    st.sidebar.subheader("📄 Uploaded Documents")

    # Use safe access to session_state
    vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
    if vs:
        try:
            sources = vs.get_all_sources()
        except Exception:
            sources = []
        if sources:
            for source in sources:
                st.sidebar.write(f"• {source}")
        else:
            st.sidebar.info("No documents uploaded yet")

    # Collection stats
    if st.sidebar.button("📊 View Statistics"):
        show_statistics()

    # Clear all data
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
        vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
        if vs:
            try:
                vs.delete_collection()
            except Exception:
                pass
        # reset uploaded_files safely
        if hasattr(st, 'session_state'):
            st.session_state.setdefault('uploaded_files', [])
            st.session_state['uploaded_files'] = []
        st.sidebar.success("All data cleared!")
        # Attempt rerun only if running via Streamlit
        try:
            st.rerun()
        except Exception:
            pass


def ingest_document(uploaded_file):
    """Ingest an uploaded document."""
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Ingest document
            ingestion = st.session_state.get('ingestion_pipeline') if hasattr(st, 'session_state') else None
            if not ingestion:
                st.sidebar.error("Ingestion pipeline not initialized")
                return
            chunks = ingestion.ingest_document(tmp_path)

            if not chunks:
                st.sidebar.error("Failed to process document")
                return

            # Generate embeddings
            st.sidebar.info(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [chunk['text'] for chunk in chunks]
            embedder = st.session_state.get('embedder') if hasattr(st, 'session_state') else None
            if not embedder:
                st.sidebar.error("Embedding generator not initialized")
                return
            embeddings = embedder.embed_batch(texts)

            # Add to vector store
            st.sidebar.info("Storing in vector database...")
            vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
            if not vs:
                st.sidebar.error("Vector store not initialized")
                return
            success = vs.add_documents(chunks, embeddings)

            # Clean up
            os.unlink(tmp_path)

            if success:
                st.session_state.setdefault('uploaded_files', [])
                st.session_state['uploaded_files'].append(uploaded_file.name)
                st.sidebar.success(f"✅ Successfully ingested {uploaded_file.name}!")
                st.sidebar.info(f"Created {len(chunks)} chunks")
            else:
                st.sidebar.error("Failed to store document")

    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")


def show_statistics():
    """Show collection statistics in sidebar."""
    vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
    if vs:
        stats = vs.get_collection_stats()
        st.sidebar.markdown("### 📊 Collection Statistics")
        st.sidebar.write(f"**Total Documents:** {stats.get('total_documents', 0)}")
        st.sidebar.write(f"**Total Sources:** {stats.get('total_sources', 0)}")


def tab_qa():
    """Q&A tab interface."""
    st.header("💬 Ask Questions")
    st.write("Ask questions about your uploaded documents")

    # Check if documents are uploaded
    vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
    if not vs or not vs.get_all_sources():
        st.warning("⚠️ Please upload documents first using the sidebar")
        return

    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="What is the main topic discussed in the document?",
        key="qa_question"
    )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of sources to retrieve", 1, 10, 3)
        with col2:
            include_sources = st.checkbox("Show source chunks", value=True)

    # Ask button
    if st.button("🔍 Ask", type="primary"):
        if question:
            with st.spinner("Thinking..."):
                rag = st.session_state.get('rag') if hasattr(st, 'session_state') else None
                if not rag:
                    st.error("RAG pipeline not initialized")
                    return
                result = rag.ask(
                    question=question,
                    top_k=top_k,
                    include_sources=include_sources
                )

                # Display answer
                st.markdown("### 📝 Answer")
                st.write(result['answer'])

                # Display sources
                if include_sources and result.get('sources'):
                    st.markdown("### 📚 Sources")
                    for source in result['sources']:
                        with st.expander(f"Source {source['index']}: {source['source']}"):
                            st.write(source['text_preview'])

                # Add to chat history safely
                st.session_state.setdefault('chat_history', [])
                st.session_state['chat_history'].append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        else:
            st.warning("Please enter a question")

    # Show chat history
    if st.session_state.get('chat_history'):
        st.markdown("---")
        st.markdown("### 📜 Chat History")
        for i, chat in enumerate(reversed(st.session_state.get('chat_history', [])[-5:]), 1):
            with st.expander(f"{chat['timestamp']} - {chat['question'][:50]}..."):
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer']}")


def tab_summary():
    """Summary tab interface."""
    st.header("📝 Document Summaries")
    st.write("Generate concise summaries of your documents")

    # Check if documents are uploaded
    vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
    sources = vs.get_all_sources() if vs else []

    if not sources:
        st.warning("⚠️ Please upload documents first using the sidebar")
        return

    # Document selector
    selected_source = st.selectbox(
        "Select document to summarize:",
        options=sources,
        key="summary_source"
    )

    # Summary options
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Summary length (words)", 100, 500, 200)
    with col2:
        style = st.selectbox(
            "Summary style",
            options=["concise", "detailed", "bullet-points"]
        )

    # Generate button
    if st.button("✨ Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            rag = st.session_state.get('rag') if hasattr(st, 'session_state') else None
            if not rag:
                st.error("RAG pipeline not initialized")
                return
            summary = rag.summarize_document(
                source=selected_source,
                max_length=max_length,
                style=style
            )

            st.markdown("### 📄 Summary")
            st.write(summary)

            # Download button
            st.download_button(
                label="💾 Download Summary",
                data=summary,
                file_name=f"{selected_source}_summary.txt",
                mime="text/plain"
            )


def tab_flashcards():
    """Flashcards tab interface."""
    st.header("🎴 Flashcard Generator")
    st.write("Generate study flashcards from your documents")

    # Check if documents are uploaded
    vs = st.session_state.get('vectorstore') if hasattr(st, 'session_state') else None
    sources = vs.get_all_sources() if vs else []

    if not sources:
        st.warning("⚠️ Please upload documents first using the sidebar")
        return

    # Document selector
    selected_source = st.selectbox(
        "Select document for flashcards:",
        options=sources,
        key="flashcard_source"
    )

    # Flashcard options
    col1, col2 = st.columns(2)
    with col1:
        num_cards = st.slider("Number of flashcards", 5, 20, 10)
    with col2:
        difficulty = st.selectbox(
            "Difficulty level",
            options=["easy", "medium", "hard"]
        )

    # Generate button
    if st.button("🎴 Generate Flashcards", type="primary"):
        with st.spinner("Generating flashcards..."):
            rag = st.session_state.get('rag') if hasattr(st, 'session_state') else None
            if not rag:
                st.error("RAG pipeline not initialized")
                return
            flashcards = rag.generate_flashcards(
                source=selected_source,
                num_cards=num_cards,
                difficulty=difficulty
            )

            if flashcards:
                st.success(f"✅ Generated {len(flashcards)} flashcards!")

                # Display flashcards
                st.markdown("### 📇 Flashcards")
                for i, card in enumerate(flashcards, 1):
                    with st.expander(f"Card {i}: {card['question'][:50]}..."):
                        st.markdown(f"**Question:** {card['question']}")
                        st.markdown(f"**Answer:** {card['answer']}")

                # Export to CSV
                st.markdown("---")
                st.markdown("### 💾 Export Flashcards")

                exporter = FlashcardExporter()

                # Create temporary CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp_file:
                    tmp_path = tmp_file.name

                if exporter.export_to_csv(flashcards, tmp_path):
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        csv_data = f.read()

                    st.download_button(
                        label="📥 Download CSV for Anki",
                        data=csv_data,
                        file_name=f"{selected_source}_flashcards.csv",
                        mime="text/csv"
                    )

                    os.unlink(tmp_path)

            else:
                st.error("Failed to generate flashcards")


def main():
    """Main application."""
    # Initialize session state
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📚 StudyBuddy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your Personal RAG Learning Assistant</div>', unsafe_allow_html=True)

    # Initialize components
    if not initialize_components():
        st.error("Failed to initialize application. Please check your configuration.")
        # If not running under streamlit run, stop gracefully
        try:
            st.stop()
        except Exception:
            print("Please run this app with: streamlit run app.py")
            return

    # Sidebar
    sidebar_upload()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📝 Summaries", "🎴 Flashcards"])

    with tab1:
        tab_qa()

    with tab2:
        tab_summary()

    with tab3:
        tab_flashcards()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "StudyBuddy - Built with Streamlit, LangChain, and OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
