"""
Tests for document chunking functionality.
"""
import pytest
from src.chunking import SemanticChunker, DocumentChunk


@pytest.fixture
def chunker():
    """Create a chunker instance for testing."""
    return SemanticChunker(chunk_size=200, chunk_overlap=50)


def test_basic_chunking(chunker):
    """Test basic document chunking."""
    text = """
    FastAPI is a modern, fast (high-performance) web framework.
    
    It is based on standard Python type hints.
    
    The key features are:
    - Fast: Very high performance
    - Fast to code: Increase development speed
    - Fewer bugs: Reduce human errors
    """
    
    chunks = chunker.chunk_document(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.content for chunk in chunks)


def test_chunk_metadata(chunker):
    """Test that metadata is properly attached."""
    text = "FastAPI is awesome."
    metadata = {
        "source": "test.md",
        "title": "Test Document",
        "url": "https://example.com"
    }
    
    chunks = chunker.chunk_document(text, metadata=metadata)
    
    assert len(chunks) > 0
    chunk = chunks[0]
    
    assert chunk.metadata["source"] == "test.md"
    assert chunk.metadata["title"] == "Test Document"
    assert chunk.metadata["url"] == "https://example.com"
    assert "chunk_index" in chunk.metadata


def test_code_block_preservation(chunker):
    """Test that code blocks are preserved."""
    text = """
    Here's an example:
    
    ```python
    from fastapi import FastAPI
    app = FastAPI()
    ```
    
    This creates an app.
    """
    
    chunks = chunker.chunk_document(text)
    
    # Code block should be preserved
    combined_content = " ".join(chunk.content for chunk in chunks)
    assert "```python" in combined_content
    assert "FastAPI" in combined_content


def test_empty_text(chunker):
    """Test handling of empty text."""
    chunks = chunker.chunk_document("")
    assert chunks == []
    
    chunks = chunker.chunk_document("   ")
    assert chunks == []


def test_to_dict(chunker):
    """Test DocumentChunk serialization."""
    text = "Test content"
    metadata = {"source": "test"}
    
    chunks = chunker.chunk_document(text, metadata=metadata)
    chunk = chunks[0]
    
    chunk_dict = chunk.to_dict()
    
    assert "content" in chunk_dict
    assert "metadata" in chunk_dict
    assert "chunk_id" in chunk_dict
    assert chunk_dict["content"] == chunk.content
