"""
Document retrieval module with clause-level matching and similarity scoring.
Handles semantic retrieval, clause detection, and explainability features.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document retrieval with clause matching and explainability."""
    
    def __init__(self):
        """Initialize the document retriever."""
        # Define clause patterns for detection
        self.clause_patterns = {
            "waiting_period": [
                r"waiting\s+period",
                r"wait\s+for\s+(\d+)",
                r"(\d+)\s+(months?|years?|days?)\s+waiting",
                r"after\s+(\d+)\s+(months?|years?|days?)"
            ],
            "grace_period": [
                r"grace\s+period",
                r"grace\s+of\s+(\d+)",
                r"(\d+)\s+(days?|months?)\s+grace"
            ],
            "maternity": [
                r"maternity",
                r"pregnancy",
                r"childbirth",
                r"delivery",
                r"maternal"
            ],
            "pre_existing": [
                r"pre[\s-]existing",
                r"PED",
                r"existing\s+condition",
                r"prior\s+condition"
            ],
            "coverage_limit": [
                r"coverage\s+limit",
                r"maximum\s+coverage",
                r"up\s+to\s+₹?(\d+)",
                r"limit\s+of\s+₹?(\d+)"
            ],
            "percentage": [
                r"(\d+)%",
                r"(\d+)\s+percent"
            ],
            "time_period": [
                r"(\d+)\s+(days?|months?|years?)",
                r"within\s+(\d+)",
                r"after\s+(\d+)"
            ],
            "medical_procedure": [
                r"surgery",
                r"treatment",
                r"procedure",
                r"operation",
                r"consultation"
            ]
        }
    
    def retrieve_chunks(
        self, 
        query: str, 
        embedding_manager, 
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The question/query to search for
            embedding_manager: EmbeddingManager instance
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant chunks with enhanced metadata
        """
        try:
            # Get similar chunks from embedding manager
            similar_chunks = embedding_manager.search_similar(query, top_k)
            
            if not similar_chunks:
                logger.warning(f"No similar chunks found for query: {query}")
                return []
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in similar_chunks 
                if chunk.get("similarity_score", 0) >= similarity_threshold
            ]
            
            if not filtered_chunks:
                logger.info(f"No chunks above similarity threshold ({similarity_threshold}) for query: {query}")
                # Return top chunk even if below threshold, but with warning
                if similar_chunks:
                    filtered_chunks = [similar_chunks[0]]
            
            # Enhance chunks with clause matching
            enhanced_chunks = []
            for chunk in filtered_chunks:
                enhanced_chunk = self._enhance_chunk_with_clauses(chunk, query)
                enhanced_chunks.append(enhanced_chunk)
            
            # Sort by combined score (similarity + clause matching)
            enhanced_chunks.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            logger.info(f"Retrieved {len(enhanced_chunks)} relevant chunks for query")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {str(e)}")
            return []
    
    def _enhance_chunk_with_clauses(self, chunk: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Enhance chunk with clause matching information.
        
        Args:
            chunk: Document chunk with text and metadata
            query: Original query for context
            
        Returns:
            Enhanced chunk with clause matching data
        """
        enhanced_chunk = chunk.copy()
        text = chunk.get("text", "").lower()
        query_lower = query.lower()
        
        # Find clause matches
        clause_matches = []
        clause_score_bonus = 0.0
        
        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Calculate relevance to query
                    query_relevance = self._calculate_query_relevance(clause_type, query_lower)
                    
                    clause_match = {
                        "clause": clause_type,
                        "matched_text": match.group(0),
                        "score": min(0.95, chunk.get("similarity_score", 0) + query_relevance),
                        "position": match.start()
                    }
                    
                    clause_matches.append(clause_match)
                    clause_score_bonus += query_relevance * 0.1  # Small bonus for relevant clauses
        
        # Remove duplicate clause matches
        clause_matches = self._deduplicate_clause_matches(clause_matches)
        
        # Calculate combined score
        base_similarity = chunk.get("similarity_score", 0)
        combined_score = min(1.0, base_similarity + clause_score_bonus)
        
        # Add enhancement data
        enhanced_chunk.update({
            "clause_matches": clause_matches,
            "combined_score": combined_score,
            "clause_bonus": clause_score_bonus,
            "explainability": self._generate_explainability(
                chunk, clause_matches, base_similarity, query
            )
        })
        
        return enhanced_chunk
    
    def _calculate_query_relevance(self, clause_type: str, query: str) -> float:
        """Calculate how relevant a clause type is to the query."""
        relevance_keywords = {
            "waiting_period": ["waiting", "wait", "period", "months", "years"],
            "grace_period": ["grace", "premium", "payment", "due"],
            "maternity": ["maternity", "pregnancy", "childbirth", "delivery"],
            "pre_existing": ["pre-existing", "existing", "PED", "disease"],
            "coverage_limit": ["limit", "coverage", "maximum", "amount"],
            "percentage": ["discount", "percent", "%", "rate"],
            "time_period": ["period", "months", "years", "days", "time"],
            "medical_procedure": ["surgery", "treatment", "procedure", "cataract"]
        }
        
        keywords = relevance_keywords.get(clause_type, [])
        relevance_score = 0.0
        
        for keyword in keywords:
            if keyword in query:
                relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _deduplicate_clause_matches(self, clause_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate clause matches based on text and position."""
        seen = set()
        deduplicated = []
        
        for match in clause_matches:
            key = (match["clause"], match["matched_text"], match["position"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(match)
        
        return deduplicated
    
    def _generate_explainability(
        self, 
        chunk: Dict[str, Any], 
        clause_matches: List[Dict[str, Any]], 
        base_similarity: float,
        query: str
    ) -> Dict[str, Any]:
        """Generate explainability information for the chunk."""
        # Create logic explanation
        logic_parts = []
        
        if clause_matches:
            clause_types = list(set(match["clause"] for match in clause_matches))
            logic_parts.append(f"Found clauses: {', '.join(clause_types)}")
        
        if base_similarity > 0.7:
            logic_parts.append("High semantic similarity")
        elif base_similarity > 0.5:
            logic_parts.append("Medium semantic similarity")
        else:
            logic_parts.append("Low semantic similarity")
        
        logic_explanation = " + ".join(logic_parts) if logic_parts else "Basic text matching"
        
        return {
            "clause_matches": [
                {
                    "clause": match["clause"].replace("_", " "),
                    "score": round(match["score"], 3)
                }
                for match in clause_matches
            ],
            "similarity_scores": [round(base_similarity, 3)],
            "logic": logic_explanation
        }
    
    def calculate_token_efficiency_score(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate a score representing token efficiency of the chunk selection.
        
        Args:
            chunks: List of selected chunks
            
        Returns:
            Efficiency score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
        avg_similarity = sum(chunk.get("similarity_score", 0) for chunk in chunks) / len(chunks)
        clause_coverage = len(set(
            match["clause"] 
            for chunk in chunks 
            for match in chunk.get("clause_matches", [])
        ))
        
        # Balance between content quality and quantity
        char_efficiency = min(1.0, 2500 / max(total_chars, 1))  # Target ~2500 chars
        quality_score = avg_similarity
        diversity_score = min(1.0, clause_coverage / 5.0)  # Up to 5 different clause types
        
        efficiency_score = (char_efficiency * 0.4 + quality_score * 0.4 + diversity_score * 0.2)
        
        return round(efficiency_score, 3)
