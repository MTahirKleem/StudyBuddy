"""
Document ingestion module for StudyBuddy.
Handles loading PDFs and text files, extracting content, and chunking text.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from utils import FileValidator, TextCleaner, MetadataExtractor

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from various file formats."""

    def __init__(self):
        self.validator = FileValidator()
        self.text_cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

    def load_document(self, file_path: str) -> Optional[Dict]:
        """Load document and return content with metadata."""
        try:
            # Validate file
            if not self.validator.is_valid_file(file_path):
                logger.error(f"Invalid file: {file_path}")
                return None

            # Get file extension
            extension = self.validator.get_file_extension(file_path)

            # Load based on file type
            if extension == '.pdf':
                text = self.load_pdf(file_path)
            elif extension == '.txt':
                text = self.load_text(file_path)
            else:
                logger.error(f"Unsupported file type: {extension}")
                return None

            if not text:
                logger.error(f"No text extracted from {file_path}")
                return None

            # Extract metadata
            metadata = self.extract_metadata(file_path)

            # Clean text
            cleaned_text = self.text_cleaner.clean_text(text)

            return {
                'text': cleaned_text,
                'metadata': metadata,
                'raw_text': text
            }

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return None

    def load_pdf(self, file_path: str) -> Optional[str]:
        """Load and extract text from PDF file using PyPDF2 or pypdf."""
        try:
            # Import PDF reader lazily to avoid import-time errors
            try:
                import pypdf
                PdfReader = pypdf.PdfReader
                logger.info("Using pypdf for PDF extraction")
            except Exception:
                try:
                    import PyPDF2
                    PdfReader = PyPDF2.PdfReader
                    logger.info("Using PyPDF2 for PDF extraction")
                except Exception:
                    logger.error("No PDF library available. Install pypdf or PyPDF2.")
                    return None

            text_content = []

            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                num_pages = len(pdf_reader.pages)

                logger.info(f"Loading PDF with {num_pages} pages")

                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        # Some readers use extract_text(), others may use .extract_text
                        text = None
                        if hasattr(page, 'extract_text'):
                            text = page.extract_text()
                        elif hasattr(page, 'extractText'):
                            text = page.extractText()

                        if text:
                            text_content.append(text)

                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue

            full_text = '\n'.join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from PDF")

            return full_text if full_text else None

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return None

    def load_text(self, file_path: str) -> Optional[str]:
        """Load text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            logger.info(f"Loaded {len(text)} characters from text file")
            return text

        except UnicodeDecodeError:
            # Try different encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                logger.info(f"Loaded text file with latin-1 encoding")
                return text
            except Exception as e:
                logger.error(f"Error loading text file: {e}")
                return None

        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            return None

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract metadata from file."""
        return self.metadata_extractor.extract_file_metadata(file_path)


class TextChunker:
    """Chunks text into smaller pieces with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize TextChunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            metadata: Optional metadata to add to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []

        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # Get chunk text
            chunk_text = text[start:end]

            # Try to break at sentence or word boundary if possible
            if end < text_length:
                # Look for sentence end
                last_period = chunk_text.rfind('.')
                last_question = chunk_text.rfind('?')
                last_exclamation = chunk_text.rfind('!')

                sentence_end = max(last_period, last_question, last_exclamation)

                if sentence_end > self.chunk_size * 0.7:  # At least 70% of chunk size
                    end = start + sentence_end + 1
                    chunk_text = text[start:end]
                else:
                    # Try to break at word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.chunk_size * 0.7:
                        end = start + last_space
                        chunk_text = text[start:end]

            # Create chunk dictionary
            chunk = {
                'chunk_id': f"chunk_{chunk_index}",
                'text': chunk_text.strip(),
                'chunk_index': chunk_index,
                'start_char': start,
                'end_char': end,
                'chunk_size': len(chunk_text.strip())
            }

            # Add metadata if provided
            if metadata:
                chunk['metadata'] = metadata.copy()
                chunk['source'] = metadata.get('filename', 'unknown')

            chunks.append(chunk)

            # Move start position (with overlap)
            start = end - self.chunk_overlap
            chunk_index += 1

        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")

        return chunks

    def add_metadata(self, chunks: List[Dict], metadata: Dict) -> List[Dict]:
        """Add or update metadata for all chunks."""
        for chunk in chunks:
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata'].update(metadata)
            chunk['source'] = metadata.get('filename', 'unknown')

        return chunks


class DocumentIngestionPipeline:
    """Complete pipeline for ingesting documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_document(self, file_path: str) -> Optional[List[Dict]]:
        """
        Complete ingestion pipeline: load document and create chunks.

        Args:
            file_path: Path to document file

        Returns:
            List of chunks with metadata, or None if failed
        """
        try:
            logger.info(f"Starting ingestion for: {file_path}")

            # Load document
            doc_data = self.loader.load_document(file_path)
            if not doc_data:
                return None

            # Extract text and metadata
            text = doc_data['text']
            metadata = doc_data['metadata']

            # Chunk text
            chunks = self.chunker.chunk_text(text, metadata)

            if not chunks:
                logger.error("No chunks created")
                return None

            # Add total chunks to metadata
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)

            logger.info(f"Successfully ingested document: {len(chunks)} chunks created")

            return chunks

        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}")
            return None

    def ingest_multiple_documents(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file paths to their chunks
        """
        results = {}

        for file_path in file_paths:
            chunks = self.ingest_document(file_path)
            if chunks:
                results[file_path] = chunks

        logger.info(f"Ingested {len(results)} out of {len(file_paths)} documents")

        return results


# Example usage
if __name__ == "__main__":
    # Test ingestion
    pipeline = DocumentIngestionPipeline(chunk_size=500, chunk_overlap=50)

    # Test with a sample file
    test_file = "data/sample.pdf"
    if Path(test_file).exists():
        chunks = pipeline.ingest_document(test_file)
        if chunks:
            print(f"Created {len(chunks)} chunks")
            print(f"First chunk: {chunks[0]['text'][:100]}...")
    else:
        print(f"Test file not found: {test_file}")
