# StudyBuddy — Personal RAG Learning Assistant 📚🤖

A Retrieval-Augmented Generation (RAG) application that helps students learn from their own study materials. Upload PDFs or text files, ask questions, generate summaries, and create flashcards—all based on YOUR documents, not the internet.

---

## 🎯 What is StudyBuddy?

StudyBuddy is a smart study assistant that:
- 📄 **Ingests** your PDFs and text files
- 🧠 **Understands** your content using embeddings
- 💬 **Answers** questions based only on your materials
- 📝 **Summarizes** documents for quick review
- 🎴 **Creates** flashcards for active recall practice

---

## ✨ Features

### Core Features
- ✅ **Document Upload & Ingestion**: Upload PDFs and text files
- ✅ **Smart Chunking**: Splits documents into optimal pieces with metadata
- ✅ **Vector Storage**: Stores embeddings for fast retrieval
- ✅ **RAG Q&A**: Ask questions and get context-based answers
- ✅ **Summaries**: Generate concise document summaries
- ✅ **Flashcard Generation**: Create Q&A pairs for studying
- ✅ **CSV Export**: Export flashcards for Anki or other tools

### Bonus Features
- 🔍 **Citations**: See which document chunks were used
- 🎨 **Interactive UI**: Built with Streamlit
- 🔎 **Search Filters**: Filter by document or topic
- 📊 **Metadata Tracking**: Page numbers, chunk indexes, and more

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (or use local models)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd StudyBuddy
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example file
copy .env .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open your browser**
Navigate to `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload Documents
- Click "Upload PDF or TXT file" in the sidebar
- Select your study materials
- Click "Ingest Document" to process

### 2. Ask Questions
- Type your question in the text box
- Click "Ask" to get an answer based on your documents
- View citations to see which parts were used

### 3. Generate Summaries
- Select a document from the dropdown
- Click "Generate Summary"
- Get a concise overview of the content

### 4. Create Flashcards
- Choose number of flashcards (5-20)
- Click "Generate Flashcards"
- Download as CSV for import into Anki

---

## 🏗️ Project Structure

```
studybuddy/
├── README.md                 # This file
├── PROJECT.md               # Project specification
├── IMPLEMENTATION_GUIDE.md  # Detailed step-by-step guide
├── CHECKLIST.md             # Quick reference checklist
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── .env.example             # Example environment file
├── data/                    # Sample PDFs and documents
├── ingest.py                # Document loading and chunking
├── embeddings.py            # Embedding generation
├── vectorstore.py           # Vector database operations
├── rag.py                   # RAG pipeline implementation
├── app.py                   # Streamlit UI
├── utils.py                 # Helper functions
└── tests/                   # Unit tests
```

---

## 🛠️ Technology Stack

- **Document Processing**: PyPDF2, pypdf
- **Embeddings**: OpenAI API or Sentence Transformers
- **Vector Database**: ChromaDB or FAISS
- **LLM**: OpenAI GPT-3.5/4 or local models
- **Framework**: LangChain
- **UI**: Streamlit
- **Export**: Pandas (CSV)

---

## 📚 Documentation

- **[PROJECT.md](PROJECT.md)**: Full project specification and requirements
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Detailed step-by-step implementation guide
- **[CHECKLIST.md](CHECKLIST.md)**: Quick reference checklist for development

---

## 🎓 Learning Objectives

By building this project, you will learn:
- How Retrieval-Augmented Generation (RAG) works
- Vector embeddings and similarity search
- Document chunking strategies
- Prompt engineering for LLMs
- Building end-to-end ML applications
- Best practices for AI application development

---

## 🔧 Configuration

Key settings in `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.1

# Vector Store Configuration
VECTOR_DB_PATH=./vector_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3
```

---

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

Manual testing checklist available in [CHECKLIST.md](CHECKLIST.md).

---

## 🎯 Roadmap

### Phase 1: Core (60%) ✅
- [x] Document ingestion
- [x] Chunking and embeddings
- [x] Vector storage
- [x] Q&A functionality
- [x] Summaries
- [x] Flashcards

### Phase 2: Quality (20%) 🔄
- [ ] Code documentation
- [ ] Error handling
- [ ] Unit tests
- [ ] Performance optimization

### Phase 3: Bonus (20%) 🚀
- [ ] Citations and sources
- [ ] Advanced filters
- [ ] Multiple LLM support
- [ ] Chat history
- [ ] Enhanced UI

---

## 🤝 Contributing

This is a learning project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is for educational purposes. Feel free to use and modify for learning.

---

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- OpenAI for embeddings and LLM APIs
- ChromaDB team for the vector database
- Streamlit for the amazing UI framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🎉 Get Started!

Ready to build your personal study assistant?

1. Read [PROJECT.md](PROJECT.md) for the full specification
2. Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) step by step
3. Use [CHECKLIST.md](CHECKLIST.md) to track your progress

**Happy coding! 🚀📚**

