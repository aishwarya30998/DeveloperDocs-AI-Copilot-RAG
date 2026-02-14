"""
Embedding generation for RAG system.

Handles text-to-vector conversion using sentence-transformers.
"""
from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers.
    
    Features:
    - Batch processing for efficiency
    - Caching of model
    - Normalized embeddings for cosine similarity
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(text, str):
            text = [text]
        
        if not text:
            raise ValueError("No text provided for embedding")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                text,
                normalize_embeddings=True,  # For cosine similarity
                show_progress_bar=len(text) > 10,
                batch_size=32
            )
            
            logger.debug(f"Generated embeddings for {len(text)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        embedding = self.embed_text(query)
        return embedding[0]  # Return single embedding
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            2D numpy array of embeddings
        """
        return self.embed_text(documents)


def create_embedding_generator(model_name: str = None) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        EmbeddingGenerator instance
    """
    from src.config import settings
    
    model = model_name or settings.embedding_model
    return EmbeddingGenerator(model_name=model)
