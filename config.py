"""
Loads and validates environment variables.
Configuration settings for StudyBuddy application.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file (for local development)
load_dotenv()

# Try to import streamlit for secrets access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_config_value(key: str, default: str = '') -> str:
    """Get configuration value from Streamlit secrets or environment variables."""
    if STREAMLIT_AVAILABLE:
        try:
            # Try to get from Streamlit secrets first
            if hasattr(st, 'secrets') and key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    # Fall back to environment variables
    return os.getenv(key, default)


class Config:
    """Application configuration."""
    
    # OpenAI / OpenRouter Settings
    OPENAI_API_KEY = get_config_value('OPENAI_API_KEY')
    # Support both OPENROUTER_API_KEY and OPENROUTER_KEY
    OPENROUTER_API_KEY = get_config_value('OPENROUTER_API_KEY') or get_config_value('OPENROUTER_KEY')
    # Support multiple env var names for base URL
    OPENROUTER_API_BASE = (
        get_config_value('OPENROUTER_API_BASE') or
        get_config_value('OPENROUTER_BASE_URL') or
        get_config_value('OPENROUTER_BASE') or
        'https://openrouter.ai/api/v1'
    )
    EMBEDDING_MODEL = get_config_value('EMBEDDING_MODEL', 'text-embedding-ada-002')
    LLM_MODEL = get_config_value('LLM_MODEL', 'openai/gpt-4o-mini')
    LLM_TEMPERATURE = float(get_config_value('LLM_TEMPERATURE', '0.1'))
    MAX_TOKENS = int(get_config_value('MAX_TOKENS', '500'))

    # Vector Store Settings (Qdrant)
    QDRANT_URL = get_config_value('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = get_config_value('QDRANT_API_KEY')
    QDRANT_COLLECTION = get_config_value('QDRANT_COLLECTION', 'studybuddy')
    VECTOR_DB_PATH = get_config_value('VECTOR_DB_PATH', './vector_db')

    # Chunking Settings
    CHUNK_SIZE = int(get_config_value('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(get_config_value('CHUNK_OVERLAP', '50'))

    # RAG Settings
    TOP_K_RESULTS = int(get_config_value('TOP_K_RESULTS', '3'))

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    TESTS_DIR = BASE_DIR / 'tests'

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        missing = False
        if not cls.OPENROUTER_API_KEY and not cls.OPENAI_API_KEY:
            print("⚠️  Warning: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. LLM/embeddings will fail without a key.")
            missing = True

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            print("⚠️  Error: CHUNK_OVERLAP must be less than CHUNK_SIZE")
            return False

        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TESTS_DIR.mkdir(exist_ok=True)
        Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

        return not missing

    @classmethod
    def display(cls):
        """Display current configuration."""
        print("=" * 50)
        print("StudyBuddy Configuration")
        print("=" * 50)
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"LLM Temperature: {cls.LLM_TEMPERATURE}")
        print(f"Max Tokens: {cls.MAX_TOKENS}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Top K Results: {cls.TOP_K_RESULTS}")
        print(f"Qdrant URL: {cls.QDRANT_URL}")
        print(f"Qdrant Collection: {cls.QDRANT_COLLECTION}")
        print(f"Vector DB Path: {cls.VECTOR_DB_PATH}")
        print(f"OpenRouter Base: {cls.OPENROUTER_API_BASE}")
        print("=" * 50)