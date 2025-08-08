"""
Embedding management module using Google Gemini embeddings and FAISS for vector similarity search.
Handles embedding generation, index building, and persistence.
"""

import logging
import os
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import google.generativeai as genai

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings generation and FAISS index operations."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required")
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Google Gemini.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process texts individually as Gemini embedding API doesn't support batch processing
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            # Fallback: return zero vectors to prevent complete failure
            logger.warning("Using zero vectors as embedding fallback")
            return [[0.0] * 768 for _ in texts]  # Gemini embedding-001 has 768 dimensions
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document chunks.
        
        Args:
            documents: List of document chunks with text and metadata
        """
        try:
            if not documents:
                raise ValueError("No documents provided for index building")
            
            # Extract text from documents
            texts = [doc["text"] for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)
            
            # Store metadata
            self.chunk_metadata = documents.copy()
            
            logger.info(f"Built FAISS index with {len(documents)} vectors (dimension: {dimension})")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}")
            raise ValueError(f"Failed to build FAISS index: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        try:
            if self.index is None:
                raise ValueError("FAISS index not built. Call build_index first.")
            
            # Generate embedding for query using retrieval_query task type
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunk_metadata):
                    chunk = self.chunk_metadata[idx].copy()
                    
                    # Convert L2 distance to cosine similarity approximation
                    # For normalized vectors: cosine_similarity ≈ 1 - (L2_distance^2 / 2)
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))
                    
                    chunk["similarity_score"] = float(similarity_score)
                    chunk["rank"] = i + 1
                    
                    results.append(chunk)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            return []
    
    def save_index(self, filepath: str) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            filepath: Path to save the index (without extension)
        """
        try:
            if self.index is None:
                raise ValueError("No index to save")
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.metadata", "wb") as f:
                pickle.dump(self.chunk_metadata, f)
            
            logger.info(f"Saved FAISS index and metadata to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def load_index(self, filepath: str) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            filepath: Path to load the index from (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if files exist
            index_file = f"{filepath}.faiss"
            metadata_file = f"{filepath}.metadata"
            
            if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
                logger.info("Index files not found, will build new index")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, "rb") as f:
                self.chunk_metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "no_index", "total_vectors": 0}
        
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "chunks_count": len(self.chunk_metadata)
        }
