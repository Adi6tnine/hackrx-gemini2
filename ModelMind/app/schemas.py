"""
Pydantic models for request and response validation.
Defines the API contract for the HackRx 6.0 document Q&A service.
"""

from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, validator

class ProcessRequest(BaseModel):
    """Request model for document processing and question answering."""
    
    documents: Union[str, List[str]] = Field(
        ..., 
        description="Single document URL or list of document URLs (PDF/DOCX)"
    )
    questions: List[str] = Field(
        ..., 
        description="List of natural language questions to answer"
    )
    
    @validator('documents')
    def normalize_documents(cls, v):
        """Normalize documents to always be a list."""
        if isinstance(v, str):
            return [v]
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list."""
        if not v:
            raise ValueError("At least one question is required")
        
        # Filter out empty questions
        non_empty_questions = [q.strip() for q in v if q.strip()]
        if not non_empty_questions:
            raise ValueError("At least one non-empty question is required")
        
        return non_empty_questions
    
    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
                ],
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?"
                ]
            }
        }

class SourceInfo(BaseModel):
    """Information about the source document chunk."""
    
    doc: str = Field(..., description="Document name/identifier")
    chunk_id: int = Field(..., description="Chunk identifier within the document")
    excerpt: str = Field(..., description="Short excerpt from the chunk")

class ClauseMatch(BaseModel):
    """Information about a matched clause."""
    
    clause: str = Field(..., description="Type of clause matched")
    score: float = Field(..., ge=0.0, le=1.0, description="Matching score")

class Explainability(BaseModel):
    """Explainability information for the answer."""
    
    clause_matches: List[ClauseMatch] = Field(
        default_factory=list,
        description="List of clause matches found"
    )
    similarity_scores: List[float] = Field(
        default_factory=list,
        description="Similarity scores for retrieved chunks"
    )
    logic: str = Field(..., description="Logic explanation for the answer")

class Answer(BaseModel):
    """Individual answer with sources and explainability."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Source chunks used for the answer"
    )
    explainability: Explainability = Field(
        ...,
        description="Explainability information"
    )

class ProcessResponse(BaseModel):
    """Response model for the document processing endpoint."""
    
    answers: List[Answer] = Field(..., description="List of answers for the questions")
    
    @validator('answers')
    def validate_answers(cls, v):
        """Validate answers list."""
        if not v:
            raise ValueError("At least one answer is required")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "answers": [
                    {
                        "question": "What is the grace period for premium payment?",
                        "answer": "The grace period for premium payment is 30 days from the due date.",
                        "sources": [
                            {
                                "doc": "policy.pdf",
                                "chunk_id": 5,
                                "excerpt": "Grace period of 30 days is allowed for premium payment..."
                            }
                        ],
                        "explainability": {
                            "clause_matches": [
                                {
                                    "clause": "grace period",
                                    "score": 0.95
                                }
                            ],
                            "similarity_scores": [0.89],
                            "logic": "Found relevant clauses: grace period → high semantic similarity"
                        }
                    }
                ]
            }
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error message")
    error_code: str = Field(default="GENERAL_ERROR", description="Error code")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid authentication token",
                "error_code": "AUTH_ERROR"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "service": "HackRx 6.0 Document Q&A Service"
            }
        }
