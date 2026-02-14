"""
Configuration management for Developer Docs AI Copilot.
"""
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Model Configuration
    llm_model: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        alias="LLM_MODEL"
    )
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )

    # Vector Database
    chroma_persist_dir: str = Field(
        default="./data/vectordb",
        alias="CHROMA_PERSIST_DIR"
    )
    collection_name: str = Field(
        default="developer_docs",
        alias="COLLECTION_NAME"
    )

    # Chunking Configuration
    chunk_size: int = Field(default=600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Retrieval Configuration
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    min_similarity_score: float = Field(
        default=0.2,
        alias="MIN_SIMILARITY_SCORE"
    )

    # Application Settings
    app_port: int = Field(default=7860, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Documentation Source
    docs_url: str = Field(
        default="https://fastapi.tiangolo.com",
        alias="DOCS_URL"
    )
    # Human-readable name for the docs. it is auto-derived from URL if not set
    docs_name: str = Field(default="", alias="DOCS_NAME")
    
    docs_url_patterns: str = Field(default="", alias="DOCS_URL_PATTERNS")

    @model_validator(mode="after")
    def set_docs_name(self) -> "Settings":
        if not self.docs_name:
            hostname = urlparse(self.docs_url).hostname or ""
            name = hostname.split(".")[0].replace("-", " ").title()
            self.docs_name = name
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Directory paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"
EVALS_DIR = PROJECT_ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORDB_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
