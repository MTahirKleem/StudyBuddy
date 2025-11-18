"""
Command-line interface for StudyBuddy.
Provides CLI access to ingestion and RAG operations.
"""

import argparse
import sys
from pathlib import Path

from config import Config
from ingest import DocumentIngestionPipeline
from embeddings import EmbeddingGenerator
from vectorstore import VectorStore
from rag import RAGPipeline
from utils import FlashcardExporter


def ingest_command(args):
    """Handle document ingestion."""
    print(f"Ingesting document: {args.file}")

    try:
        # Initialize components
        pipeline = DocumentIngestionPipeline(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        embedder = EmbeddingGenerator(use_openrouter=True, use_local=False)
        vectorstore = VectorStore()

        # Ingest document
        chunks = pipeline.ingest_document(args.file)

        if not chunks:
            print("❌ Failed to ingest document")
            return 1

        print(f"✅ Created {len(chunks)} chunks")

        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_batch(texts)

        # Store in vector database
        print("Storing in vector database...")
        success = vectorstore.add_documents(chunks, embeddings)

        if success:
            print(f"✅ Successfully ingested {args.file}")
            return 0
        else:
            print("❌ Failed to store in vector database")
            return 1

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def query_command(args):
    """Handle Q&A queries."""
    print(f"Question: {args.question}")

    try:
        # Initialize components
        embedder = EmbeddingGenerator(use_openrouter=True, use_local=False)
        vectorstore = VectorStore()
        rag = RAGPipeline(vectorstore, embedder)

        # Ask question
        result = rag.ask(args.question, top_k=args.top_k)

        print("\n" + "=" * 50)
        print("ANSWER:")
        print("=" * 50)
        print(result['answer'])
        print()

        if result.get('sources'):
            print("=" * 50)
            print("SOURCES:")
            print("=" * 50)
            for source in result['sources']:
                print(f"\n[{source['index']}] {source['source']}")
                print(f"    {source['text_preview']}")

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def summarize_command(args):
    """Handle document summarization."""
    print(f"Summarizing: {args.source}")

    try:
        # Initialize components
        embedder = EmbeddingGenerator(use_openrouter=True, use_local=False)
        vectorstore = VectorStore()
        rag = RAGPipeline(vectorstore, embedder)

        # Generate summary
        summary = rag.summarize_document(
            source=args.source,
            max_length=args.length,
            style=args.style
        )

        print("\n" + "=" * 50)
        print("SUMMARY:")
        print("=" * 50)
        print(summary)
        print()

        # Save if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✅ Summary saved to {args.output}")

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def flashcards_command(args):
    """Handle flashcard generation."""
    print(f"Generating flashcards from: {args.source}")

    try:
        # Initialize components
        embedder = EmbeddingGenerator(use_openrouter=True, use_local=False)
        vectorstore = VectorStore()
        rag = RAGPipeline(vectorstore, embedder)

        # Generate flashcards
        flashcards = rag.generate_flashcards(
            source=args.source,
            num_cards=args.number,
            difficulty=args.difficulty
        )

        if not flashcards:
            print("❌ Failed to generate flashcards")
            return 1

        print(f"\n✅ Generated {len(flashcards)} flashcards")

        # Display flashcards
        print("\n" + "=" * 50)
        print("FLASHCARDS:")
        print("=" * 50)
        for i, card in enumerate(flashcards, 1):
            print(f"\nCard {i}:")
            print(f"Q: {card['question']}")
            print(f"A: {card['answer']}")

        # Save if output file specified
        if args.output:
            exporter = FlashcardExporter()
            if exporter.export_to_csv(flashcards, args.output):
                print(f"\n✅ Flashcards saved to {args.output}")
            else:
                print(f"\n❌ Failed to save flashcards")

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def stats_command(args):
    """Show collection statistics."""
    try:
        vectorstore = VectorStore()
        stats = vectorstore.get_collection_stats()

        print("\n" + "=" * 50)
        print("COLLECTION STATISTICS")
        print("=" * 50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Sources: {stats['total_sources']}")
        print(f"\nSources:")
        for source in stats['sources']:
            print(f"  • {source}")
        print()

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StudyBuddy - RAG Learning Assistant CLI"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a document')
    ingest_parser.add_argument('file', help='Path to document file')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a question')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--top-k', type=int, default=3, help='Number of sources to retrieve')
    
    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize a document')
    summarize_parser.add_argument('source', help='Source document name')
    summarize_parser.add_argument('--length', type=int, default=200, help='Summary length in words')
    summarize_parser.add_argument('--style', choices=['concise', 'detailed', 'bullet-points'], default='concise')
    summarize_parser.add_argument('--output', help='Output file path')
    
    # Flashcards command
    flashcards_parser = subparsers.add_parser('flashcards', help='Generate flashcards')
    flashcards_parser.add_argument('source', help='Source document name')
    flashcards_parser.add_argument('--number', type=int, default=10, help='Number of flashcards')
    flashcards_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium')
    flashcards_parser.add_argument('--output', help='Output CSV file path')
    
    # Stats command
    subparsers.add_parser('stats', help='Show collection statistics')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate configuration
    if not Config.validate():
        print("❌ Invalid configuration. Please check your .env file.")
        return 1
    
    # Execute command
    if args.command == 'ingest':
        return ingest_command(args)
    elif args.command == 'query':
        return query_command(args)
    elif args.command == 'summarize':
        return summarize_command(args)
    elif args.command == 'flashcards':
        return flashcards_command(args)
    elif args.command == 'stats':
        return stats_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
