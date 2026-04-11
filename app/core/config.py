# config.py
# This file contains the settings for the RAG Agent API.
# It reads environment variables from the .env file, define app settings and Mistral API settings.

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "RAG Agent API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Mistral
    mistral_api_key: str = "CF2DvjIoshzasO0mtBkPj44fo2nXDwPk"
    mistral_model: str = "mistral-small-latest"
    mistral_embed_model: str = "mistral-embed"

    # Chunking
    chunk_size: int = 512 # temporary chunk size
    chunk_overlap: int = 64 # temporary chunk overlap

    # Retrieval
    top_k: int = 5 # temporary top k results
    semantic_weight: float = 0.7   # temporary weight for semantic score in hybrid search
    keyword_weight: float = 0.3    # temporaryweight for BM25 score in hybrid search

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
