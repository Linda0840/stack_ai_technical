from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Ingestion ---

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int
    chunk_index: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class IngestResponse(BaseModel):
    document_ids: list[str]
    total_chunks: int
    filenames: list[str]
    message: str


# --- Query ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    transformed_query: Optional[str] = None
    intent_triggered_search: bool
    # None  → search was not triggered (chitchat / greeting)
    # False → search triggered but no chunk met the similarity threshold
    # True  → search triggered and at least one chunk passed the threshold
    evidence_sufficient: Optional[bool] = None
    # Shape of the generated answer; drives prompt-template selection.
    # One of: 'list', 'table', 'comparison', 'definition', 'instruction', 'factual'
    # None when no answer was generated (empty KB, chitchat, insufficient evidence).
    query_shape: Optional[str] = None
    answer: str
    sources: list[RetrievedChunk]
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
