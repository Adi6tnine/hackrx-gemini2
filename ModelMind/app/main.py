"""
FastAPI application for HackRx 6.0 document Q&A service.
Provides endpoint for processing documents and answering questions using LLM + retrieval.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware

from app.core.ingest import DocumentIngestor
from app.core.embeddings import EmbeddingManager
from app.core.retriever import DocumentRetriever
from app.core.llm import LLMProcessor
from app.schemas import ProcessRequest, ProcessResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
ingestor = DocumentIngestor()
embedding_manager = EmbeddingManager()
retriever = DocumentRetriever()
llm_processor = LLMProcessor()

# Security
security = HTTPBearer()

# Environment variables
TEAM_TOKEN = os.getenv("TEAM_TOKEN", "954e06e1c53324def16260167cb6a51f3a144221af5afb0faa6ca9bd2c836641")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting HackRx 6.0 Document Q&A Service")
    yield
    logger.info("Shutting down service")

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 Document Q&A Service",
    description="LLM-powered document processing and question answering service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authentication."""
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "HackRx 6.0 Document Q&A Service"}

@app.post("/api/v1/hackrx/run", response_model=ProcessResponse)
async def process_documents_and_questions(
    request: ProcessRequest,
    token: str = Depends(verify_token)
):
    """
    Process documents and answer questions using LLM + retrieval.
    
    Args:
        request: Request containing documents and questions
        token: Bearer token for authentication
        
    Returns:
        ProcessResponse: Structured answers with sources and explainability
    """
    try:
        logger.info(f"Processing request with {len(request.documents)} documents and {len(request.questions)} questions")
        
        # Step 1: Ingest documents
        logger.info("Step 1: Ingesting documents...")
        documents = []
        for doc_url in request.documents:
            try:
                doc_content = await ingestor.process_document(doc_url)
                documents.extend(doc_content)
                logger.info(f"Successfully processed document: {doc_url}")
            except Exception as e:
                logger.error(f"Failed to process document {doc_url}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process document {doc_url}: {str(e)}"
                )
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents could be processed"
            )
        
        # Step 2: Create embeddings and build FAISS index
        logger.info("Step 2: Creating embeddings and building index...")
        try:
            embedding_manager.build_index(documents)
            logger.info(f"Built FAISS index with {len(documents)} chunks")
        except Exception as e:
            logger.error(f"Failed to build embeddings index: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to build embeddings index: {str(e)}"
            )
        
        # Step 3: Process each question
        logger.info("Step 3: Processing questions...")
        answers = []
        
        for question in request.questions:
            try:
                logger.info(f"Processing question: {question}")
                
                # Retrieve relevant chunks
                relevant_chunks = retriever.retrieve_chunks(
                    question, embedding_manager, top_k=5
                )
                
                if not relevant_chunks:
                    # No relevant chunks found
                    answers.append({
                        "question": question,
                        "answer": "Information not found in the provided documents.",
                        "sources": [],
                        "explainability": {
                            "clause_matches": [],
                            "similarity_scores": [],
                            "logic": "No relevant chunks found for this question"
                        }
                    })
                    continue
                
                # Generate answer using LLM
                answer_data = await llm_processor.generate_answer(
                    question, relevant_chunks
                )
                
                answers.append(answer_data)
                logger.info(f"Generated answer for question: {question}")
                
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {str(e)}")
                # Add error answer instead of failing the entire request
                answers.append({
                    "question": question,
                    "answer": "Error processing this question.",
                    "sources": [],
                    "explainability": {
                        "clause_matches": [],
                        "similarity_scores": [],
                        "logic": f"Error: {str(e)}"
                    }
                })
        
        logger.info(f"Successfully processed all questions. Generated {len(answers)} answers.")
        return ProcessResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_documents_and_questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
