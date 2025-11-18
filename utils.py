"""
Utility functions for StudyBuddy application.
Handles file validation, text cleaning, metadata extraction, and export operations.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileValidator:
    """Validates uploaded files."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    @classmethod
    def is_valid_file(cls, file_path: str) -> bool:
        """Check if file is valid for processing."""
        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False

            # Check file extension
            if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
                logger.error(f"Unsupported file type: {path.suffix}")
                return False

            # Check file size
            if path.stat().st_size > cls.MAX_FILE_SIZE:
                logger.error(f"File too large: {path.stat().st_size} bytes")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False

    @classmethod
    def get_file_extension(cls, file_path: str) -> str:
        """Get file extension."""
        return Path(file_path).suffix.lower()


class TextCleaner:
    """Cleans and normalizes text."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Remove extra spaces from text."""
        return ' '.join(text.split())

    @staticmethod
    def normalize_line_breaks(text: str) -> str:
        """Normalize line breaks."""
        return text.replace('\r\n', '\n').replace('\r', '\n')


class MetadataExtractor:
    """Extracts metadata from files."""

    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict:
        """Extract metadata from file."""
        try:
            path = Path(file_path)
            stats = path.stat()

            metadata = {
                'filename': path.name,
                'file_path': str(path.absolute()),
                'file_size': stats.st_size,
                'file_extension': path.suffix.lower(),
                'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'processed_at': datetime.now().isoformat()
            }

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}


class FlashcardExporter:
    """Exports flashcards to various formats."""

    @staticmethod
    def export_to_csv(flashcards: List[Dict], output_path: str) -> bool:
        """Export flashcards to CSV format for Anki."""
        try:
            if not flashcards:
                logger.warning("No flashcards to export")
                return False

            # Create DataFrame
            df = pd.DataFrame(flashcards)

            # Ensure required columns exist
            if 'question' not in df.columns or 'answer' not in df.columns:
                logger.error("Flashcards missing required fields")
                return False

            # Add tags column if not exists
            if 'tags' not in df.columns:
                df['tags'] = ''

            # Reorder columns for Anki format
            df = df[['question', 'answer', 'tags']]

            # Export to CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Exported {len(flashcards)} flashcards to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error exporting flashcards: {e}")
            return False

    @staticmethod
    def format_for_anki(flashcards: List[Dict]) -> str:
        """Format flashcards for Anki import."""
        lines = []
        for card in flashcards:
            question = card.get('question', '').replace('\n', '<br>')
            answer = card.get('answer', '').replace('\n', '<br>')
            tags = card.get('tags', '')
            lines.append(f"{question}\t{answer}\t{tags}")

        return '\n'.join(lines)


class PathManager:
    """Manages file paths and directories."""

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Ensure directory exists, create if not."""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return False

    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Get safe filename by removing invalid characters."""
        # Remove invalid characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        safe_name = safe_name.strip('. ')
        return safe_name

    @staticmethod
    def get_unique_filename(directory: str, filename: str) -> str:
        """Get unique filename by adding number if exists."""
        path = Path(directory) / filename
        if not path.exists():
            return str(path)

        stem = path.stem
        suffix = path.suffix
        counter = 1

        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = Path(directory) / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text (rough approximation)."""
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def chunk_overlap_percentage(chunk_size: int, overlap: int) -> float:
    """Calculate overlap percentage."""
    if chunk_size == 0:
        return 0.0
    return (overlap / chunk_size) * 100
