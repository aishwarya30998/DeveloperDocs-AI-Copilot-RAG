#!/usr/bin/env python3
"""
Test retrieval quality independently.

Useful for debugging and tuning retrieval parameters.
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


TEST_QUERIES = [
    "How do I create a router in FastAPI?",
    "What are dependencies?",
    "How do I handle errors?",
    "Show me authentication examples",
    "How do I validate request bodies?",
]


def test_retrieval():
    """Test retrieval with various queries."""
    
    logger.info("=" * 60)
    logger.info("Retrieval Quality Test")
    logger.info("=" * 60)
    
    # Initialize retriever
    retriever = create_retriever()
    stats = retriever.get_collection_stats()
    
    logger.info(f"\nVector Database Stats:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Collection: {stats['collection_name']}")
    logger.info(f"  Embedding dim: {stats['embedding_dimension']}")
    
    # Test each query
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"Test {i}/{len(TEST_QUERIES)}")
        logger.info("=" * 60)
        logger.info(f"Query: {query}")
        
        # Retrieve
        results = retriever.retrieve(query, top_k=3)
        
        logger.info(f"\nFound {len(results)} results:")
        
        for j, result in enumerate(results, 1):
            logger.info(f"\n--- Result {j} ---")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Source: {result['metadata'].get('title', 'Unknown')}")
            logger.info(f"Section: {result['metadata'].get('section', 'Unknown')}")
            logger.info(f"Content preview:")
            logger.info(f"{result['content'][:200]}...")
        
        # Quality check
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        logger.info(f"\nAverage relevance score: {avg_score:.4f}")
        
        if avg_score >= 0.75:
            logger.info("✓ High quality results")
        elif avg_score >= 0.6:
            logger.info("⚠ Medium quality results")
        else:
            logger.info("✗ Low quality results - consider tuning")
    
    logger.info("\n" + "=" * 60)
    logger.info("Retrieval test complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_retrieval()
