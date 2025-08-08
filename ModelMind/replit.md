# Overview

This is a production-ready FastAPI service built for the HackRx 6.0 challenge that processes documents (PDF/DOCX) and provides intelligent question-answering capabilities using LLM and retrieval augmented generation (RAG). The service downloads documents from URLs, extracts and chunks text intelligently, creates embeddings for semantic search, and uses GPT-4o to generate accurate answers with detailed explainability features.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture
- **Framework**: FastAPI with Uvicorn server for high-performance async operations
- **Authentication**: Bearer token security with environment-based token validation
- **Error Handling**: Comprehensive exception handling with detailed logging throughout all components
- **Modular Design**: Clean separation of concerns across core modules (ingest, embeddings, retriever, llm)

## Document Processing Pipeline
- **Document Ingestion**: Multi-format support (PDF via PyMuPDF, DOCX via python-docx) with URL-based downloading
- **Text Chunking**: LangChain RecursiveCharacterTextSplitter with smart boundary detection (400-600 tokens per chunk, 50 token overlap)
- **Metadata Management**: Maintains document metadata including chunk IDs, page ranges, and text excerpts for traceability

## Vector Search and Retrieval
- **Embeddings**: Google Gemini embedding-001 model for semantic vector generation
- **Vector Store**: FAISS in-memory index for fast similarity search
- **Clause Matching**: Hybrid approach combining semantic similarity with regex pattern matching for domain-specific terms
- **Retrieval Strategy**: Top-K similarity search with confidence scoring and clause-level matching

## LLM Integration
- **Primary Model**: Google Gemini 1.5 Flash for text generation
- **Token Optimization**: Context length management (2500 token limit) to minimize API costs
- **Prompt Engineering**: Structured prompts for consistent JSON responses with answer, confidence, and reasoning
- **Response Structure**: Standardized output format including answers, sources, and explainability traces

## Data Flow
1. Documents downloaded and parsed into clean text
2. Text split into semantically meaningful chunks with metadata
3. Chunks embedded using Google Gemini and stored in FAISS index
4. Questions processed through similarity search and clause matching
5. Relevant chunks passed to LLM with optimized context
6. Structured responses generated with source attribution and reasoning

# External Dependencies

## Core Services
- **Google Gemini API**: Primary dependency for embeddings (embedding-001) and language model (gemini-1.5-flash)
- **Document Sources**: Supports public PDF/DOCX URLs including Azure Blob Storage endpoints

## Python Libraries
- **FastAPI**: Web framework with automatic API documentation
- **PyMuPDF (fitz)**: PDF text extraction and parsing
- **python-docx**: DOCX document processing
- **FAISS**: Facebook AI Similarity Search for vector operations
- **LangChain**: Text splitting utilities and pipeline components
- **Google Generative AI**: Official Google Gemini API integration
- **Requests**: HTTP client for document downloading

## Testing and Development
- **pytest**: Test framework with async support
- **httpx**: Async HTTP client for API testing
- **unittest.mock**: Mocking capabilities for isolated testing

## Environment Configuration
- **GEMINI_API_KEY**: Required for all LLM and embedding operations
- **TEAM_TOKEN**: Bearer token for API authentication (defaults to provided challenge token)