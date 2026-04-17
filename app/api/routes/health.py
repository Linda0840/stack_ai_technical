"""
Health route: GET /health.

Returns the application version and the number of documents currently indexed
in the in-memory vector store.
"""

from fastapi import APIRouter, Depends, status

from app.models.schemas import HealthResponse
from app.core.config import get_settings
from app.api.deps import get_vector_store

router = APIRouter(tags=["Health"])


@router.get(
   "/health",
   response_model=HealthResponse,
   status_code=status.HTTP_200_OK,
   summary="Health check",
)
async def health(
   vector_store: dict = Depends(get_vector_store),
) -> HealthResponse:
   settings = get_settings()
   doc_ids = {v["chunk"].document_id for v in vector_store.values() if "chunk" in v}
   return HealthResponse(
       status="ok",
       version=settings.app_version,
       documents_indexed=len(doc_ids),
   )
