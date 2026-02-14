"""
Main RAG pipeline orchestration.

Coordinates retrieval and generation for question answering.
"""
import logging
import requests
from typing import Dict, Any, Optional, List

from src.retriever import DocumentRetriever
from src.prompts import create_rag_prompt, create_no_context_prompt, format_response_with_sources
from src.config import settings

# HuggingFace router — OpenAI-compatible chat completions endpoint
_HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the RAG pipeline: retrieve → generate → format.
    
    Features:
    - Smart retrieval with filtering
    - LLM generation via HuggingFace Inference API
    - Source attribution
    - Error handling with graceful degradation
    """
    
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_model: Optional[str] = None,
        min_similarity_score: float = 0.5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Document retriever instance
            llm_model: Optional LLM model name override
            min_similarity_score: Minimum score for relevant results
        """
        self.retriever = retriever
        self.llm_model = llm_model or settings.llm_model
        self.min_similarity_score = min_similarity_score
        
        self._api_url = _HF_API_URL
        self._headers = {
            "Authorization": f"Bearer {settings.hf_token}",
            "Content-Type": "application/json",
        }
        logger.info(f"LLM endpoint: {self._api_url} model={self.llm_model}")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Step 1: Retrieve relevant context
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Log raw scores for diagnostics
            scores = [round(c["score"], 4) for c in retrieved_chunks]
            logger.info(f"Raw chunk scores: {scores}")

            # Filter by minimum similarity score
            relevant_chunks = [
                chunk for chunk in retrieved_chunks
                if chunk["score"] >= self.min_similarity_score
            ]

            logger.info(f"Found {len(relevant_chunks)} relevant chunks (threshold: {self.min_similarity_score})")
            
            # Step 2: Generate answer
            if not relevant_chunks:
                answer = f"I couldn't find relevant information in the {settings.docs_name} documentation to answer this question. Could you rephrase or ask about a different topic?"
                return {
                    "answer": answer,
                    "sources": [],
                    "source_count": 0,
                    "confidence": "low",
                    "chunks_retrieved": 0
                }
            
            # Create prompt
            prompt = create_rag_prompt(question, relevant_chunks)
            
            # Generate answer
            answer = self._generate_answer(prompt)
            
            # Step 3: Format response
            response = format_response_with_sources(answer, relevant_chunks)
            
            # Add metadata
            response["confidence"] = self._estimate_confidence(relevant_chunks)
            response["chunks_retrieved"] = len(relevant_chunks)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "source_count": 0,
                "confidence": "error",
                "chunks_retrieved": 0
            }
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer text
        """
        try:
            # Use OpenAI-compatible chat completions endpoint
            payload = {
                "model": f"{self.llm_model}:fastest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "top_p": 0.9,
            }
            response = requests.post(
                self._api_url,
                headers=self._headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"Generated answer ({len(answer)} chars)")

            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Estimate confidence based on retrieval scores.
        
        Args:
            chunks: Retrieved chunks with scores
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not chunks:
            return "low"
        
        avg_score = sum(chunk["score"] for chunk in chunks) / len(chunks)
        
        if avg_score >= 0.75:
            return "high"
        elif avg_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "llm_model": self.llm_model,
            "min_similarity_score": self.min_similarity_score,
            **self.retriever.get_collection_stats()
        }


def create_rag_pipeline(
    retriever: Optional[DocumentRetriever] = None
) -> RAGPipeline:
    """
    Factory function to create RAG pipeline.
    
    Args:
        retriever: Optional retriever override
        
    Returns:
        RAGPipeline instance
    """
    from src.retriever import create_retriever
    
    if retriever is None:
        retriever = create_retriever()
    
    return RAGPipeline(
        retriever=retriever,
        min_similarity_score=settings.min_similarity_score
    )
