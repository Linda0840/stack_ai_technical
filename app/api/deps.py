"""
FastAPI dependency providers.

The vector_store and bm25_index are plain dicts held in app.state so they
survive the lifetime of the process without requiring an external database
during development.
"""

from fastapi import Request

from app.services.ingestion import IngestionService
from app.services.query import QueryService


def get_vector_store(request: Request) -> dict:
    return request.app.state.vector_store


def get_bm25_index(request: Request) -> dict:
    return request.app.state.bm25_index


def get_document_registry(request: Request) -> list:
    return request.app.state.document_registry


def get_workspace_stats(request: Request) -> dict:
    return request.app.state.workspace_stats


def get_ingestion_service(request: Request) -> IngestionService:
    return IngestionService(
        vector_store=request.app.state.vector_store,
        bm25_index=request.app.state.bm25_index,
        workspace_stats=request.app.state.workspace_stats,
    )


def get_query_service(request: Request) -> QueryService:
    return QueryService(
        vector_store=request.app.state.vector_store,
        bm25_index=request.app.state.bm25_index,
    )
