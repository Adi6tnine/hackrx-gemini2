"""
Unit tests for the HackRx 6.0 Document Q&A API.
Tests the main endpoint with sample data from the problem statement.
"""

import pytest
import os
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

from app.main import app

# Test configuration
TEST_TOKEN = "954e06e1c53324def16260167cb6a51f3a144221af5afb0faa6ca9bd2c836641"
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
SAMPLE_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

class TestDocumentQAAPI:
    """Test cases for the Document Q&A API."""
    
    @pytest.fixture
    def mock_gemini_responses(self):
        """Mock Gemini API responses to avoid real API calls during testing."""
        
        # Mock embedding response - Gemini embedding-001 has 768 dimensions
        mock_embedding_response = {'embedding': [0.1] * 768}
        
        # Mock text generation response
        mock_generation_response = MagicMock()
        mock_generation_response.text = '{"answer": "Test answer from mocked Gemini", "confidence": "medium"}'
        
        return mock_embedding_response, mock_generation_response
    
    @pytest.fixture
    def mock_document_content(self):
        """Mock document content to avoid downloading real documents."""
        return """
        National Parivar Mediclaim Plus Policy
        
        Grace Period: The grace period for premium payment is 30 days from the due date.
        
        Waiting Period for Pre-existing Diseases: Pre-existing diseases are covered after a waiting period of 36 months.
        
        Maternity Coverage: Maternity expenses are covered after a waiting period of 24 months from the policy commencement date.
        
        Cataract Surgery: Waiting period for cataract surgery is 12 months from the policy start date.
        
        Organ Donor Coverage: Medical expenses for organ donors are covered up to the sum insured.
        
        No Claim Discount: A No Claim Discount of 10% is offered for each claim-free year.
        
        Health Check-up: Annual preventive health check-up is covered up to ₹5,000.
        
        Hospital Definition: Hospital means any institution established for in-patient care and day care treatment.
        
        AYUSH Treatment: Coverage for AYUSH treatments is available up to 25% of the sum insured.
        
        Room Rent Limits: For Plan A, room rent is limited to 2% of sum insured per day and ICU charges to 5% of sum insured per day.
        """

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
            
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Document Q&A Service" in data["service"]

    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Test that authentication is required for the main endpoint."""
        request_data = {
            "documents": [SAMPLE_DOCUMENT_URL],
            "questions": ["What is the grace period?"]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/hackrx/run", json=request_data)
        
        assert response.status_code == 403  # FastAPI returns 403 for missing auth header

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        """Test authentication with invalid token."""
        request_data = {
            "documents": [SAMPLE_DOCUMENT_URL],
            "questions": ["What is the grace period?"]
        }
        
        headers = {"Authorization": "Bearer invalid_token"}
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 401
        assert "Invalid authentication token" in response.json()["detail"]

    @pytest.mark.asyncio 
    async def test_invalid_request_format(self):
        """Test validation of request format."""
        # Missing documents
        request_data = {
            "questions": ["What is the grace period?"]
        }
        
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_empty_questions_list(self):
        """Test validation with empty questions list."""
        request_data = {
            "documents": [SAMPLE_DOCUMENT_URL],
            "questions": []
        }
        
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_document_qa_endpoint_success(self, mock_gemini_responses, mock_document_content):
        """Test successful document Q&A processing with mocked responses."""
        mock_embedding_response, mock_generation_response = mock_gemini_responses
        
        # Mock the document ingestor to return mock content
        with patch('app.core.ingest.DocumentIngestor.process_document') as mock_process:
            mock_process.return_value = [
                {
                    "doc_id": "policy.pdf",
                    "chunk_id": 0,
                    "text": mock_document_content,
                    "excerpt": mock_document_content[:100] + "...",
                    "char_count": len(mock_document_content),
                    "page_range": "1-1"
                }
            ]
            
            # Mock Gemini API calls
            with patch('google.generativeai.embed_content') as mock_embed:
                mock_embed.return_value = mock_embedding_response
                
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    mock_model = MagicMock()
                    mock_model.generate_content.return_value = mock_generation_response
                    mock_model_class.return_value = mock_model
                    
                    request_data = {
                        "documents": [SAMPLE_DOCUMENT_URL],
                        "questions": SAMPLE_QUESTIONS[:3]  # Test with first 3 questions
                    }
                    
                    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
                    
                    async with AsyncClient(app=app, base_url="http://test") as client:
                        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Validate response structure
                    assert "answers" in data
                    assert len(data["answers"]) == len(request_data["questions"])
                    
                    # Validate each answer has required fields
                    for answer in data["answers"]:
                        assert "question" in answer
                        assert "answer" in answer
                        assert "sources" in answer
                        assert "explainability" in answer
                        
                        # Validate explainability structure
                        explainability = answer["explainability"]
                        assert "clause_matches" in explainability
                        assert "similarity_scores" in explainability
                        assert "logic" in explainability
                        
                        # Validate sources structure if present
                        for source in answer["sources"]:
                            assert "doc" in source
                            assert "chunk_id" in source
                            assert "excerpt" in source

    @pytest.mark.asyncio
    async def test_document_qa_with_single_document_string(self, mock_gemini_responses, mock_document_content):
        """Test the endpoint with a single document URL as string (not list)."""
        mock_embedding_response, mock_generation_response = mock_gemini_responses
        
        with patch('app.core.ingest.DocumentIngestor.process_document') as mock_process:
            mock_process.return_value = [
                {
                    "doc_id": "policy.pdf",
                    "chunk_id": 0,
                    "text": mock_document_content,
                    "excerpt": mock_document_content[:100] + "...",
                    "char_count": len(mock_document_content),
                    "page_range": "1-1"
                }
            ]
            
            with patch('google.generativeai.embed_content') as mock_embed:
                mock_embed.return_value = mock_embedding_response
                
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    mock_model = MagicMock()
                    mock_model.generate_content.return_value = mock_generation_response
                    mock_model_class.return_value = mock_model
                    
                    request_data = {
                        "documents": SAMPLE_DOCUMENT_URL,  # Single string, not list
                        "questions": ["What is the grace period?"]
                    }
                    
                    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
                    
                    async with AsyncClient(app=app, base_url="http://test") as client:
                        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert len(data["answers"]) == 1

    @pytest.mark.asyncio
    async def test_response_quality_requirements(self, mock_gemini_responses, mock_document_content):
        """Test that responses meet the quality requirements from the problem statement."""
        mock_embedding_response, mock_generation_response = mock_gemini_responses
        
        with patch('app.core.ingest.DocumentIngestor.process_document') as mock_process:
            mock_process.return_value = [
                {
                    "doc_id": "policy.pdf",
                    "chunk_id": 0,
                    "text": mock_document_content,
                    "excerpt": mock_document_content[:100] + "...",
                    "char_count": len(mock_document_content),
                    "page_range": "1-1"
                }
            ]
            
            with patch('google.generativeai.embed_content') as mock_embed:
                mock_embed.return_value = mock_embedding_response
                
                with patch('google.generativeai.GenerativeModel') as mock_model_class:
                    mock_model = MagicMock()
                    mock_model.generate_content.return_value = mock_generation_response
                    mock_model_class.return_value = mock_model
                    
                    request_data = {
                        "documents": [SAMPLE_DOCUMENT_URL],
                        "questions": SAMPLE_QUESTIONS  # All questions
                    }
                    
                    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
                    
                    async with AsyncClient(app=app, base_url="http://test") as client:
                        response = await client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Test requirement: answers array has same length as questions
                    assert len(data["answers"]) == len(SAMPLE_QUESTIONS)
                    
                    # Test requirement: at least 50% of answers are non-empty
                    non_empty_answers = [
                        answer for answer in data["answers"] 
                        if answer["answer"] and answer["answer"].strip() != ""
                    ]
                    
                    success_rate = len(non_empty_answers) / len(data["answers"])
                    assert success_rate >= 0.5, f"Success rate {success_rate:.2%} is below 50%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
