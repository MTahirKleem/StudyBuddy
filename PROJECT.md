# StudyBuddy — Student RAG App (Project Specification)
- Reflection page
- Working demo
- GitHub repo (or ZIP)
## 12. Final Submission

1–2 weeks total learning pace (1–2 hours per day)
## 11. Estimated Time

- Test retrieval separately before adding LLM
- Keep LLM temperature low (0–0.2)
- Always store metadata
- Use small chunk sizes
- Start with one PDF
## 10. Tips for Students

- Export options included
- Citations added
- UI implemented
### ✅ Bonus (20%)

- Readable README
- Clear comments
- Structured code
### ✅ Code Quality (20%)

- Summaries and flashcards function properly
- QA system returns context-based answers
- Chunking is correct
- Uploading + ingesting documents works
### ✅ Basic (60%)

## 9. Acceptance Criteria

- CSV export
- Flashcards
- Summaries
### Milestone 6 — Bonus Features

- Display results
- Input box for questions
- Upload interface
### Milestone 5 — UI or API

- Return response
- Send them to LLM
- Retrieve relevant chunks
- Embed user question
### Milestone 4 — Retrieval & Answer Generation

- Create retrieval interface
- Save embeddings in vector database
- Embed chunks
### Milestone 3 — Embeddings & Vector Store

- Store chunk metadata
- Split into chunks
- Extract text
- Load PDF
### Milestone 2 — Ingestion

- Add sample PDFs
- Add requirements + .env file
- Create project folder
### Milestone 1 — Setup

## 8. Implementation Plan (Milestones)

```
└── utils.py              # helpers
├── app.py                # UI or API
├── rag.py                # RAG pipeline
├── vectorstore.py        # store/search embeddings
├── embeddings.py         # embedding logic
├── ingest.py             # file ingestion + chunking
├── data/                 # sample PDFs
├── .env.example
├── requirements.txt
├── PROJECT.md
├── README.md
studybuddy/
```
## 7. Recommended Folder Structure

  - What you would improve
  - What was challenging
  - What you learned
- **Reflection Document** (1 page):
  - Summary and flashcards generation
  - Asking questions and receiving answers
  - Ingestion process
  - File upload
- **Demo** (screenshots or a short video) showing:
  - Sample files in data/
  - Source code files
  - requirements.txt
  - README.md
  - PROJECT.md
- **Project structure**, including:
Student must deliver:
## 6. Deliverables

- Export summaries and flashcards to files
- Add search filters (file name, topic, chapter)
- Implement a simple UI (Streamlit or web interface)
- Show citations under each answer (chunk indexes or page numbers)
## 5. Optional (Bonus) Features

- Allow exporting cards as CSV for Anki or other study apps
- Create a set of flashcards
- Convert document content into Q/A pairs
### 5) Flashcard Generation

- Summary should be concise and based only on the uploaded content
- Ability to summarize a full document or any selected file
### 4) Document Summary

- LLM generates an answer strictly based on retrieved context
- Passes chunks + question to an LLM
- Retrieves top-K most relevant chunks
- System embeds the question
- User enters a question
### 3) Retrieval + Question Answering (RAG Flow)

- Build the index so retrieval becomes fast
- Store embeddings + metadata in a local vector store
- Convert chunks into embeddings using an embedding model
### 2) Embeddings + Vector Storage

- Saves chunk metadata (page number, chunk index)
- Splits text into small overlapping chunks
- Application extracts text
- User can upload PDF or text files
### 1) File Upload / Ingestion

## 4. Required Features

This ensures the AI answers from your material, not the internet.

- Generate summaries and flashcards for study help
- Use the stored information to answer questions
- Store embeddings in a vector database
- Convert chunks into embeddings
- Extract and split the content into chunks
- Upload study materials (PDFs or text files)
StudyBuddy is a small application that allows a student to:
## 3. Short Description

The goal of this project is to build a simple RAG-based study assistant that can read the student's uploaded notes or PDFs and answer questions based only on that material. This project helps students learn how retrieval, embeddings, and LLM-based responses work together.
## 2. Objective

**StudyBuddy** — Personal Retrieval-Augmented Generation (RAG) Learning Assistant
## 1. Project Title
**StudyBuddy**

