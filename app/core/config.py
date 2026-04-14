# config.py
# This file contains the settings for the RAG Agent API.
# It reads environment variables from the .env file, define app settings and Mistral API settings.

from typing import Literal
from functools import lru_cache
from pydantic_settings import BaseSettings


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

    # Embedding tuning
    embed_batch_size: int = 8           # chunks per Mistral embedding call (lower = easier to debug)
    embed_timeout_seconds: int = 30     # max seconds to wait for one embedding batch

    # Embedding mode:
    #   "real" — call the Mistral API and store actual vectors (production)
    #   "skip" — store 1024-dim zero-vectors, skip the API (isolates extraction/chunking bugs)
    # Set EMBED_MODE=skip in .env to enable debug mode; change to real when done.
    embed_mode: Literal["real", "skip"] = "real"

    # Legacy alias kept for backward compatibility; overridden by embed_mode when set.
    debug_skip_embeddings: bool = False

    # ── Scalability / safety limits ──────────────────────────────────────────
    # Hard cap on files accepted per /ingest request; guards against memory exhaustion.
    max_files_per_request: int = 20
    # Maximum PDF size accepted (bytes = mb * 1024^2); enforced after reading bytes.
    max_file_size_mb: int = 20
    # Maximum candidates returned by hybrid search before re-ranking.
    # Keeps the reranker LLM prompt from exceeding the context window.
    max_retrieve_k: int = 50
    # Maximum chunks forwarded to the LLM reranker (subset of retrieve_k).
    max_rerank_chunks: int = 25

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
