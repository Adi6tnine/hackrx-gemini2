# HackRx 6.0 Document Q&A Service

A production-ready FastAPI service that processes documents (PDF/DOCX) and answers questions using LLM + retrieval augmented generation (RAG). Built for the HackRx 6.0 challenge with emphasis on accuracy, token efficiency, low latency, and explainability.

## Features

- **Document Processing**: Downloads and parses PDF/DOCX files from URLs
- **Smart Chunking**: Intelligent text segmentation with sentence boundary awareness (400-600 tokens per chunk)
- **Vector Search**: Google Gemini embeddings with FAISS similarity search
- **Clause Matching**: Semantic matching with regex heuristics for policy-specific terms
- **LLM Integration**: Google Gemini Pro for answer generation with structured responses
- **Explainability**: Detailed explanations showing clause matches, similarity scores, and reasoning logic
- **Token Optimization**: Efficient context management to minimize API costs
- **Authentication**: Bearer token security
- **Production Ready**: Comprehensive error handling, logging, and testing

## Technology Stack

- **Backend**: FastAPI + Uvicorn
- **Document Processing**: PyMuPDF (PDF), python-docx (DOCX)
- **Text Processing**: LangChain text splitter
- **Embeddings**: Google Gemini embedding-001
- **Vector Search**: FAISS (in-memory)
- **LLM**: Google Gemini Pro
- **Testing**: pytest + httpx

## Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt
