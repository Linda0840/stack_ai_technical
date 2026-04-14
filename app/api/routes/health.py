# APIRouter → group routes cleanly
# Depends → FastAPI dependency injection
# status → HTTP status codes




from fastapi import APIRouter, Depends, status


from app.models.schemas import HealthResponse
from app.core.config import get_settings # config access
from app.api.deps import get_vector_store # pull shared in-memory store


router = APIRouter(tags=["Health"]) # Groups this endpoint under “Health”


@router.get(
   "/health",
   response_model=HealthResponse,
   status_code=status.HTTP_200_OK,
   summary="Health check",
)




async def health(
   vector_store: dict = Depends(get_vector_store), # FastAPI injects the shared store to ensure cleaner design
) -> HealthResponse:
   settings = get_settings()
   # Count unique document IDs currently indexed
   doc_ids = {v["chunk"].document_id for v in vector_store.values() if "chunk" in v}
   return HealthResponse(
       status="ok",
       version=settings.app_version,
       documents_indexed=len(doc_ids),
   )
