"""
Quick start script for StudyBuddy.
Run this to start the Streamlit application.
"""

import subprocess
import sys
from pathlib import Path
from config import Config

def main():
    """Start the StudyBuddy application."""
    print("=" * 60)
    print(" " * 15 + "StudyBuddy - RAG Learning Assistant")
    print("=" * 60)
    print()

    # Validate configuration
    print("Checking configuration...")
    if not Config.validate():
        print()
        print("❌ Configuration error!")
        print()
        print("Please make sure:")
        print("1. You have created a .env file from .env.example")
        print("2. You have added your OPENROUTER_API_KEY (or OPENAI_API_KEY) to the .env file")
        print()
        print("Example .env file:")
        print("  OPENROUTER_API_KEY=your_api_key_here")
        print()
        return 1

    print("✅ Configuration valid")
    print()

    # Display configuration
    Config.display()
    print()

    # Start Streamlit
    print("Starting Streamlit application...")
    print()
    print("The application will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    print()

    try:
        # Run streamlit
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down StudyBuddy...")
        print("Goodbye! 👋")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
