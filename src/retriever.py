"""
Vector retrieval system using ChromaDB.

Handles document storage, indexing, and semantic search.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.embeddings import EmbeddingGenerator
from src.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Manages document storage and retrieval using ChromaDB.
    
    Features:
    - Persistent vector storage
    - Semantic similarity search
    - Metadata filtering
    - Source attribution
    """
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize retriever.
        
        Args:
            persist_directory: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_generator: Embedding generator instance
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection (cosine distance for proper similarity scores)
        self.collection = self._get_or_create_collection()
        coll_meta = self.collection.metadata or {}
        self._use_cosine = coll_meta.get("hnsw:space") == "cosine"
        logger.info(f"Collection '{collection_name}' ready. Count: {self.collection.count()}. Distance: {'cosine' if self._use_cosine else 'l2'}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection with cosine distance so scores stay in [0, 1]
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "Developer documentation chunks"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        return collection
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to collection")
        
        # Prepare data for ChromaDB
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.embed_documents(documents)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            logger.debug(f"Added batch {i//batch_size + 1}")
        
        logger.info(f"Successfully added {len(chunks)} chunks. Total: {self.collection.count()}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with content, metadata, and scores
        """
        logger.debug(f"Retrieving top {top_k} results for query: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                d = results["distances"][0][i]
                score = max(0.0, 1 - d) if self._use_cosine else max(0.0, 1 - d ** 2 / 2)
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                    "id": results["ids"][0][i] if "ids" in results else None
                })
        
        logger.info(f"Retrieved {len(formatted_results)} results")
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Sample a document to get metadata fields
        sample = self.collection.peek(limit=1)
        metadata_fields = list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "metadata_fields": metadata_fields,
            "embedding_dimension": self.embedding_generator.embedding_dim
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
    
    def reset_collection(self) -> None:
        """Reset collection (delete and recreate)."""
        logger.warning("Resetting collection")
        try:
            self.delete_collection()
        except Exception:
            pass
        self.collection = self._get_or_create_collection()


def create_retriever(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_generator: Optional[EmbeddingGenerator] = None
) -> DocumentRetriever:
    """
    Factory function to create retriever.
    
    Args:
        persist_directory: Optional directory override
        collection_name: Optional collection name override
        embedding_generator: Optional embedding generator override
        
    Returns:
        DocumentRetriever instance
    """
    from src.config import settings
    from src.embeddings import create_embedding_generator
    
    persist_dir = persist_directory or settings.chroma_persist_dir
    coll_name = collection_name or settings.collection_name
    emb_gen = embedding_generator or create_embedding_generator()
    
    return DocumentRetriever(
        persist_directory=persist_dir,
        collection_name=coll_name,
        embedding_generator=emb_gen
    )
