"""
Prompt templates for the RAG system.
"""
from typing import List, Dict, Any

from src.config import settings

_DOCS_NAME = settings.docs_name


def _build_system_prompt(docs_name: str) -> str:
    return f"""You are a helpful assistant specialized in {docs_name} documentation.

Your role is to answer questions ONLY using the provided context from the official {docs_name} documentation.

Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in the documentation to answer that"
3. Preserve code formatting and indentation
4. Include code examples when available in the context
5. Cite sources by mentioning the section (e.g., "According to the Routing section...")
6. Be concise but complete
7. Use technical language appropriate for developers

If you're unsure, it's better to admit it than to make up information."""


SYSTEM_PROMPT = _build_system_prompt(_DOCS_NAME)


def create_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Create the full RAG prompt with context and query.
    
    Args:
        query: User's question
        context_chunks: Retrieved document chunks with metadata
        
    Returns:
        Formatted prompt string
    """
    # Build context section
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        section = chunk["metadata"].get("section", "")
        
        context_header = f"[Context {i}"
        if section:
            context_header += f" - {section}"
        context_header += f" from {source}]"
        
        context_parts.append(f"{context_header}\n{chunk['content']}\n")
    
    context_text = "\n".join(context_parts)
    
    # Create full prompt
    prompt = f"""{SYSTEM_PROMPT}

---

CONTEXT FROM DOCUMENTATION:

{context_text}

---

USER QUESTION: {query}

ANSWER (based only on the context above):"""
    
    return prompt


def create_no_context_prompt(query: str) -> str:
    """
    Create prompt when no relevant context is found.

    Args:
        query: User's question

    Returns:
        Formatted prompt string
    """
    prompt = f"""{SYSTEM_PROMPT}

USER QUESTION: {query}

Unfortunately, I couldn't find relevant information in the {_DOCS_NAME} documentation to answer this question.

This could mean:
1. The question is about a topic not covered in the documentation I have access to
2. The question might need to be rephrased
3. The topic might be covered in a different section

Can you rephrase your question or provide more context?"""

    return prompt


def format_response_with_sources(
    answer: str,
    sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format the final response with sources.
    
    Args:
        answer: Generated answer
        sources: Retrieved source chunks
        
    Returns:
        Formatted response dictionary
    """
    # Extract unique sources
    unique_sources = {}
    for source in sources:
        metadata = source["metadata"]
        source_key = metadata.get("url", metadata.get("source", "Unknown"))
        
        if source_key not in unique_sources:
            unique_sources[source_key] = {
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "section": metadata.get("section", ""),
                "score": source.get("score", 0.0)
            }
    
    # Sort by relevance score
    sorted_sources = sorted(
        unique_sources.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return {
        "answer": answer,
        "sources": sorted_sources,
        "source_count": len(sorted_sources)
    }
