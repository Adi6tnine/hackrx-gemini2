"""
Document ingestion module for downloading and parsing PDF/DOCX files.
Handles text extraction, cleaning, and chunking with smart boundary detection.
"""

import logging
import re
import tempfile
from typing import List, Dict, Any
from urllib.parse import urlparse
import os

import fitz  # PyMuPDF
import requests
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Handles document downloading, parsing, and chunking."""
    
    def __init__(self):
        """Initialize the document ingestor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Target chunk size in characters (roughly 400-600 tokens)
            chunk_overlap=50,  # Overlap to maintain context
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            is_separator_regex=False,
        )
    
    async def process_document(self, document_url: str) -> List[Dict[str, Any]]:
        """
        Download and process a document from URL.
        
        Args:
            document_url: URL to the document
            
        Returns:
            List of document chunks with metadata
        """
        try:
            # Download document
            logger.info(f"Downloading document from: {document_url}")
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL or content-type
            content_type = response.headers.get('content-type', '').lower()
            url_path = urlparse(document_url).path.lower()
            
            is_pdf = (
                'application/pdf' in content_type or 
                url_path.endswith('.pdf') or
                response.content[:4] == b'%PDF'
            )
            
            is_docx = (
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type or
                url_path.endswith('.docx') or
                response.content[:2] == b'PK'  # ZIP signature (DOCX is ZIP-based)
            )
            
            # Extract document name
            doc_name = os.path.basename(urlparse(document_url).path) or "document"
            if '?' in doc_name:
                doc_name = doc_name.split('?')[0]
            if not doc_name or doc_name == '/':
                doc_name = "document.pdf" if is_pdf else "document.docx"
            
            # Process based on file type
            if is_pdf:
                text_content = self._extract_pdf_text(response.content)
            elif is_docx:
                text_content = self._extract_docx_text(response.content)
            else:
                logger.warning(f"Unknown file type for {document_url}, attempting PDF extraction")
                text_content = self._extract_pdf_text(response.content)
            
            if not text_content.strip():
                raise ValueError(f"No text content extracted from document: {document_url}")
            
            # Clean and chunk the text
            cleaned_text = self._clean_text(text_content)
            chunks = self._create_chunks(cleaned_text, doc_name)
            
            logger.info(f"Successfully processed document {doc_name}: {len(chunks)} chunks created")
            return chunks
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download document {document_url}: {str(e)}")
            raise ValueError(f"Failed to download document: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process document {document_url}: {str(e)}")
            raise ValueError(f"Failed to process document: {str(e)}")
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF content using PyMuPDF."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                text_content = ""
                
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        text_content += f"\n--- Page {page_num + 1} ---\n{text}"
                
                doc.close()
                os.unlink(temp_file.name)
                
                return text_content
                
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {str(e)}")
            raise ValueError(f"Failed to extract PDF text: {str(e)}")
    
    def _extract_docx_text(self, docx_content: bytes) -> str:
        """Extract text from DOCX content using python-docx."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(docx_content)
                temp_file.flush()
                
                doc = DocxDocument(temp_file.name)
                text_content = ""
                
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_content += para.text + "\n"
                
                os.unlink(temp_file.name)
                
                return text_content
                
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {str(e)}")
            raise ValueError(f"Failed to extract DOCX text: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers but keep some structure
        text = re.sub(r'--- Page \d+ ---', '\n\n', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _create_chunks(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """
        Create text chunks with metadata.
        
        Args:
            text: Cleaned text content
            doc_name: Name of the source document
            
        Returns:
            List of chunks with metadata
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        chunk_list = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            # Create excerpt for display (first 100 characters)
            excerpt = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            
            chunk_data = {
                "doc_id": doc_name,
                "chunk_id": i,
                "text": chunk_text.strip(),
                "excerpt": excerpt.strip(),
                "char_count": len(chunk_text),
                "page_range": "N/A"  # Could be enhanced to track actual page ranges
            }
            
            chunk_list.append(chunk_data)
        
        return chunk_list
