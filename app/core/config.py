"""
Application settings.

All values are read from environment variables (or a .env file) via
pydantic-settings.  Each field has a sensible default so the app starts
without any configuration, but production deployments should supply at
minimum MISTRAL_API_KEY and set EMBED_MODE=real.

The module-level get_settings() is memoised with @lru_cache so the Settings
object is constructed exactly once per process lifetime; restart the server
after changing .env values.
"""

from typing import Literal
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RAG Agent API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Mistral
    mistral_api_key: str = "wSFV43kzPBYKyr3YpDvRcFuka0MutNE2"
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
    max_files_per_request: int = 50
    # Maximum PDF size accepted (bytes = mb * 1024^2); enforced after reading bytes.
    max_file_size_mb: int = 50
    # Maximum total chunks allowed across the entire session workspace.
    # Prevents unbounded memory growth; ~5 000 chunks ≈ several hundred pages of text.
    # Users see a friendly capacity percentage before hitting this ceiling.
    max_workspace_chunks: int = 6000
    # Maximum candidates returned by hybrid search before re-ranking.
    # Keeps the reranker LLM prompt from exceeding the context window.
    max_retrieve_k: int = 50
    # Maximum chunks forwarded to the LLM reranker (subset of retrieve_k).
    max_rerank_chunks: int = 25

    # ── Embedding retry / rate-limit handling ─────────────────────────────────
    # Number of times to retry a single embedding batch after a 429 response.
    # Set to 0 to disable retries (fail fast).
    embed_max_retries: int = 6
    # Base delay (seconds) for the first retry; doubles each attempt (exponential
    # backoff) plus a small random jitter to avoid thundering-herd on 429s.
    embed_retry_base_delay: float = 5.0

    # ── Evidence threshold ────────────────────────────────────────────────────
    # Minimum hybrid score (0.0–1.0) that at least one retrieved chunk must reach.
    # The hybrid score = semantic_weight * cosine_similarity + keyword_weight * bm25.
    # If the best chunk falls below this value the pipeline refuses to generate an
    # answer and returns "insufficient evidence" instead.
    # Set to 0.0 via env (MIN_SIMILARITY_THRESHOLD=0) to disable the guard.
    min_similarity_threshold: float = 0.25

    # ── Hallucination filter ──────────────────────────────────────────────────
    # When True, a post-hoc evidence check verifies each factual claim in the
    # generated answer against the retrieved chunks and removes unsupported ones.
    # Disable via EVIDENCE_CHECK_ENABLED=false if latency is a concern.
    evidence_check_enabled: bool = True
    # Maximum number of claims to verify per answer (avoids token overflow).
    evidence_check_max_claims: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
