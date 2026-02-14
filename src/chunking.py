"""
Document chunking strategies for RAG.

Implements semantic chunking with overlap, metadata enrichment,
and configurable strategies for different content types.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }


class SemanticChunker:
    """
    Smart chunking that preserves semantic meaning.
    
    Features:
    - Splits on natural boundaries (paragraphs, sentences)
    - Maintains context with overlap
    - Preserves code blocks intact
    - Enriches chunks with metadata
    """
    
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        preserve_code_blocks: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_code_blocks = preserve_code_blocks
    
    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split document into semantically meaningful chunks.
        
        Args:
            text: Document text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        metadata = metadata or {}
        
        # Extract and preserve code blocks
        code_blocks = []
        if self.preserve_code_blocks:
            text, code_blocks = self._extract_code_blocks(text)
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            para_tokens = self._estimate_tokens(para)
            
            # If single paragraph exceeds chunk size, split by sentences
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        metadata,
                        len(chunks)
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split long paragraph
                sentence_chunks = self._split_long_paragraph(para, metadata, len(chunks))
                chunks.extend(sentence_chunks)
                continue
            
            # Add paragraph to current chunk
            if current_size + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_size += para_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        metadata,
                        len(chunks)
                    ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_size = self._estimate_tokens(overlap_text) + para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                metadata,
                len(chunks)
            ))
        
        # Reinsert code blocks
        if code_blocks:
            chunks = self._reinsert_code_blocks(chunks, code_blocks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _extract_code_blocks(self, text: str) -> tuple[str, List[Dict[str, str]]]:
        """Extract code blocks to preserve them intact."""
        code_pattern = r'```[\s\S]*?```|`[^`]+`'
        code_blocks = []
        
        def replace_code(match):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append({
                "placeholder": placeholder,
                "content": match.group(0)
            })
            return placeholder
        
        text_without_code = re.sub(code_pattern, replace_code, text)
        return text_without_code, code_blocks
    
    def _reinsert_code_blocks(
        self,
        chunks: List[DocumentChunk],
        code_blocks: List[Dict[str, str]]
    ) -> List[DocumentChunk]:
        """Reinsert code blocks into chunks."""
        for chunk in chunks:
            for code_block in code_blocks:
                chunk.content = chunk.content.replace(
                    code_block["placeholder"],
                    code_block["content"]
                )
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_long_paragraph(
        self,
        paragraph: str,
        metadata: Dict[str, Any],
        start_idx: int
    ) -> List[DocumentChunk]:
        """Split a long paragraph by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_size + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        metadata,
                        start_idx + len(chunks)
                    ))
                current_chunk = [sentence]
                current_size = sentence_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                metadata,
                start_idx + len(chunks)
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        text_segments: List[str],
        metadata: Dict[str, Any],
        chunk_idx: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from text segments."""
        content = "\n\n".join(text_segments)
        
        # Enrich metadata
        enriched_metadata = {
            **metadata,
            "chunk_index": chunk_idx,
            "chunk_size": len(content),
            "has_code": "```" in content or "`" in content,
        }
        
        chunk_id = f"{metadata.get('source', 'unknown')}_{chunk_idx}"
        
        return DocumentChunk(
            content=content,
            metadata=enriched_metadata,
            chunk_id=chunk_id
        )
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """Get overlap text from previous chunks."""
        if not chunks:
            return ""
        
        combined = " ".join(chunks[-2:])
        tokens = self._estimate_tokens(combined)
        
        if tokens <= self.chunk_overlap:
            return combined
        
        # Truncate to overlap size
        words = combined.split()
        overlap_words = words[-(self.chunk_overlap // 4):]
        return " ".join(overlap_words)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4


def create_chunker(chunk_size: int = 600, chunk_overlap: int = 100) -> SemanticChunker:
    """Factory function to create a chunker instance."""
    return SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_code_blocks=True
    )
