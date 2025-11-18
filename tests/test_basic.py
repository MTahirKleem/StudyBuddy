"""
Test module for StudyBuddy application.
Run tests to verify core functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import FileValidator, TextCleaner, MetadataExtractor
from ingest import DocumentLoader, TextChunker, DocumentIngestionPipeline


def test_file_validator():
    """Test file validation."""
    print("Testing FileValidator...")

    # Test valid extensions
    assert '.pdf' in FileValidator.SUPPORTED_EXTENSIONS
    assert '.txt' in FileValidator.SUPPORTED_EXTENSIONS

    # Test extension check
    assert FileValidator.get_file_extension("test.pdf") == '.pdf'
    assert FileValidator.get_file_extension("test.txt") == '.txt'

    print("✅ FileValidator tests passed")


def test_text_cleaner():
    """Test text cleaning."""
    print("Testing TextCleaner...")

    cleaner = TextCleaner()

    # Test whitespace removal
    text = "This    has   too    many   spaces"
    cleaned = cleaner.clean_text(text)
    assert "  " not in cleaned

    # Test line break normalization
    text = "Line1\r\nLine2\rLine3"
    normalized = cleaner.normalize_line_breaks(text)
    assert '\r' not in normalized

    print("✅ TextCleaner tests passed")


def test_text_chunker():
    """Test text chunking."""
    print("Testing TextChunker...")

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    # Test chunking
    text = "This is a test. " * 50  # Create text longer than chunk size
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1, "Should create multiple chunks"
    assert all('chunk_id' in chunk for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)

    print(f"✅ TextChunker tests passed - Created {len(chunks)} chunks")


def test_document_loader():
    """Test document loading."""
    print("Testing DocumentLoader...")

    loader = DocumentLoader()

    # Test text file creation and loading
    test_file = "tests/test_sample.txt"
    test_content = "This is a test document for StudyBuddy.\n" * 5

    # Create test file
    os.makedirs("tests", exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # Load document
    result = loader.load_document(test_file)

    assert result is not None, "Should load document"
    assert 'text' in result
    assert 'metadata' in result
    assert len(result['text']) > 0

    # Clean up
    os.unlink(test_file)

    print("✅ DocumentLoader tests passed")


def test_ingestion_pipeline():
    """Test complete ingestion pipeline."""
    print("Testing DocumentIngestionPipeline...")

    pipeline = DocumentIngestionPipeline(chunk_size=100, chunk_overlap=20)

    # Create test file
    test_file = "tests/test_ingestion.txt"
    test_content = "This is test content for ingestion pipeline. " * 20

    os.makedirs("tests", exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # Ingest document
    chunks = pipeline.ingest_document(test_file)

    assert chunks is not None, "Should ingest document"
    assert len(chunks) > 0, "Should create chunks"
    assert all('text' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)

    # Clean up
    os.unlink(test_file)

    print(f"✅ DocumentIngestionPipeline tests passed - Created {len(chunks)} chunks")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running StudyBuddy Tests")
    print("=" * 50)
    print()

    try:
        test_file_validator()
        test_text_cleaner()
        test_text_chunker()
        test_document_loader()
        test_ingestion_pipeline()

        print()
        print("=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)

        return True

    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 50)
        return False

    except Exception as e:
        print()
        print("=" * 50)
        print(f"❌ ERROR: {str(e)}")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
