"""
LLM processing module for generating answers using Google Gemini models.
Handles prompt construction, token optimization, and response parsing.
"""

import json
import logging
import os
from typing import List, Dict, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Handles LLM-based answer generation with token optimization."""
    
    def __init__(self):
        """Initialize the LLM processor."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-1.5-flash"
        self.max_context_tokens = 2500  # Conservative limit for context
    
    async def generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an answer for a question using relevant document chunks.
        
        Args:
            question: The question to answer
            relevant_chunks: List of relevant document chunks
            
        Returns:
            Dictionary containing answer, sources, and explainability
        """
        try:
            # Optimize chunks for token efficiency
            optimized_chunks = self._optimize_chunks_for_tokens(relevant_chunks)
            
            if not optimized_chunks:
                return self._create_no_info_response(question)
            
            # Construct prompt
            context_text = self._build_context_text(optimized_chunks)
            prompt = self._build_prompt(question, context_text)
            
            # Generate answer using Gemini
            try:
                model = genai.GenerativeModel(self.model)
                
                # Construct the prompt with system instruction
                system_instruction = (
                    "You are an insurance policy expert assistant. Answer questions using only "
                    "the provided document context. If information is not available in the context, "
                    "clearly state that. Provide concise, accurate answers with specific details "
                    "when available. Format your response as JSON with 'answer' and 'confidence' fields."
                )
                
                full_prompt = f"{system_instruction}\n\n{prompt}\n\nPlease respond in JSON format with 'answer' and 'confidence' fields."
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistency
                        max_output_tokens=500,
                    )
                )
                
                llm_response = response.text
                
            except Exception as e:
                logger.error(f"Gemini model failed: {str(e)}")
                # Create a fallback response
                llm_response = '{"answer": "Error processing with Gemini API", "confidence": "low"}'
            
            # Parse LLM response
            parsed_response = self._parse_llm_response(llm_response)
            
            # Build final response with sources and explainability
            final_response = self._build_final_response(
                question, parsed_response, optimized_chunks
            )
            
            logger.info(f"Generated answer for question: {question}")
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to generate answer for question '{question}': {str(e)}")
            return self._create_error_response(question, str(e))
    
    def _optimize_chunks_for_tokens(
        self, 
        chunks: List[Dict[str, Any]], 
        max_chunks: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Optimize chunks for token efficiency while maintaining relevance.
        
        Args:
            chunks: List of relevant chunks
            max_chunks: Maximum number of chunks to use
            
        Returns:
            Optimized list of chunks
        """
        if not chunks:
            return []
        
        # Sort by combined score (similarity + clause matching)
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.get("combined_score", x.get("similarity_score", 0)), 
            reverse=True
        )
        
        # Select top chunks within token budget
        selected_chunks = []
        total_chars = 0
        
        for chunk in sorted_chunks[:max_chunks]:
            chunk_text = chunk.get("text", "")
            # Rough token estimation (1 token ≈ 4 characters)
            chunk_tokens = len(chunk_text) / 4
            
            if total_chars + len(chunk_text) <= self.max_context_tokens * 4:
                selected_chunks.append(chunk)
                total_chars += len(chunk_text)
            else:
                # Truncate chunk if it's the first one and still too long
                if not selected_chunks:
                    max_chars = self.max_context_tokens * 4
                    truncated_chunk = chunk.copy()
                    truncated_chunk["text"] = chunk_text[:max_chars] + "..."
                    selected_chunks.append(truncated_chunk)
                break
        
        logger.info(f"Optimized from {len(chunks)} to {len(selected_chunks)} chunks ({total_chars} chars)")
        return selected_chunks
    
    def _build_context_text(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context text from chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("doc_id", "Unknown")
            chunk_id = chunk.get("chunk_id", i)
            text = chunk.get("text", "")
            
            context_parts.append(f"[Context {i} - {doc_id} Chunk {chunk_id}]:\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""Based on the following document context, answer this question: "{question}"

Context:
{context}

Instructions:
1. Use only information from the provided context
2. If the answer is not in the context, respond with "Information not found in the provided documents"
3. Be specific and include relevant details (numbers, periods, conditions)
4. Provide your confidence level (low/medium/high)
5. Format response as JSON with "answer" and "confidence" fields

Question: {question}
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response safely."""
        try:
            parsed = json.loads(response)
            return {
                "answer": parsed.get("answer", "Unable to parse response"),
                "confidence": parsed.get("confidence", "low")
            }
        except json.JSONDecodeError:
            # Try to extract answer from plain text
            logger.warning("Failed to parse JSON response, using fallback")
            return {
                "answer": response[:500] + "..." if len(response) > 500 else response,
                "confidence": "low"
            }
    
    def _build_final_response(
        self, 
        question: str, 
        parsed_response: Dict[str, Any], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the final structured response."""
        # Build sources
        sources = []
        for chunk in chunks:
            sources.append({
                "doc": chunk.get("doc_id", "Unknown"),
                "chunk_id": chunk.get("chunk_id", 0),
                "excerpt": chunk.get("excerpt", "")[:100] + "..." if len(chunk.get("excerpt", "")) > 100 else chunk.get("excerpt", "")
            })
        
        # Aggregate explainability from all chunks
        all_clause_matches = []
        all_similarity_scores = []
        
        for chunk in chunks:
            explainability = chunk.get("explainability", {})
            all_clause_matches.extend(explainability.get("clause_matches", []))
            all_similarity_scores.extend(explainability.get("similarity_scores", []))
        
        # Remove duplicate clause matches
        unique_clause_matches = []
        seen_clauses = set()
        for match in all_clause_matches:
            clause_key = match.get("clause", "")
            if clause_key not in seen_clauses:
                seen_clauses.add(clause_key)
                unique_clause_matches.append(match)
        
        # Build logic explanation
        logic_parts = []
        if unique_clause_matches:
            clause_names = [match.get("clause", "") for match in unique_clause_matches]
            logic_parts.append(f"Found relevant clauses: {', '.join(clause_names)}")
        
        if all_similarity_scores:
            avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores)
            if avg_similarity > 0.7:
                logic_parts.append("high semantic similarity")
            elif avg_similarity > 0.5:
                logic_parts.append("medium semantic similarity")
            else:
                logic_parts.append("low semantic similarity")
        
        logic = " → ".join(logic_parts) if logic_parts else "text matching applied"
        
        return {
            "question": question,
            "answer": parsed_response["answer"],
            "sources": sources,
            "explainability": {
                "clause_matches": unique_clause_matches,
                "similarity_scores": all_similarity_scores,
                "logic": logic
            }
        }
    
    def _create_no_info_response(self, question: str) -> Dict[str, Any]:
        """Create response when no relevant information is found."""
        return {
            "question": question,
            "answer": "Information not found in the provided documents.",
            "sources": [],
            "explainability": {
                "clause_matches": [],
                "similarity_scores": [],
                "logic": "No relevant chunks found for this question"
            }
        }
    
    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Create response when an error occurs."""
        return {
            "question": question,
            "answer": "Error processing this question.",
            "sources": [],
            "explainability": {
                "clause_matches": [],
                "similarity_scores": [],
                "logic": f"Error occurred: {error}"
            }
        }
