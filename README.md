# PDF RAG Agent

A Python project to build a PDF-based Retrieval-Augmented Generation (RAG) chatbot using Google's Agent Development Kit (ADK).

## Project Goals

- Load PDF documents and extract their text.
- Split extracted text into chunks for retrieval.
- Generate embeddings using an open-source embedding model.
- Store and retrieve relevant chunks using a vector database.
- Use a language model to answer questions grounded in PDF content.
- Integrate the RAG workflow with Google ADK.

## Technology Stack

- Python 3.12+
- Google ADK
- pypdf
- ChromaDB (planned)
- pytest
- ReportLab for test PDF generation

## Current Progress

- [x] PDF text extraction
- [x] Validation for missing, invalid, and malformed PDF inputs
- [x] Multi-page PDF extraction tests
- [ ] Text chunking
- [ ] Embeddings and vector storage
- [ ] Retrieval and answer generation
- [ ] ADK agent integration

## Run Tests

Activate the project's virtual environment, then run:

```powershell
pytest -v
```

## Project Status

This project is under development. Future milestones will add document chunking, retrieval, and question-answering functionality.