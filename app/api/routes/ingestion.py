# This endpoint:


# creates a POST route at /ingest
# accepts one or more uploaded files
# validates that files are present
# validates that each file is a PDF
# injects an IngestionService
# passes the files to the service for processing
# returns a structured IngestResponse




from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status


from app.core.config import get_settings
from app.models.schemas import IngestResponse # response is typed with IngestResponse
from app.services.ingestion import IngestionService # injection logic
from app.api.deps import get_ingestion_service, get_vector_store, get_bm25_index


router = APIRouter(prefix="/ingest", tags=["Ingestion"])
_settings = get_settings()

ALLOWED_CONTENT_TYPES = {"application/pdf"}




@router.post(
   "",
   response_model=IngestResponse,
   status_code=status.HTTP_201_CREATED,
   summary="Upload and ingest one or more PDF files",
)
async def ingest_documents(
   files: list[UploadFile] = File(..., description="One or more PDF files to ingest"),
   service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
   """
   Accepts one or more PDF files, extracts and chunks their text, generates
   embeddings, and adds them to the search index.
   """
   if not files:
       raise HTTPException(
           status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
           detail="At least one file must be provided.",
       )

   # Guard: reject oversized batches before reading any bytes.
   if len(files) > _settings.max_files_per_request:
       raise HTTPException(
           status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
           detail=(
               f"At most {_settings.max_files_per_request} files may be submitted "
               "per request. Please split your upload into smaller batches."
           ),
       )

   for f in files:
       if f.content_type not in ALLOWED_CONTENT_TYPES:
           raise HTTPException(
               status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
               detail=f"'{f.filename}' is not a PDF (got content-type: {f.content_type}).",
           )

   try:
       document_ids, total_chunks = await service.ingest_files(files)
   except ValueError as exc:
       # Raised by the service when an individual file exceeds the size limit.
       raise HTTPException(
           status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
           detail=str(exc),
       ) from exc

   if total_chunks == 0:
       raise HTTPException(
           status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
           detail=(
               "No text could be extracted from the uploaded file(s). "
               "This usually means the PDF is a scanned image without a text layer. "
               "Please use a text-based PDF and try again."
           ),
       )

   return IngestResponse(
       document_ids=document_ids,
       total_chunks=total_chunks,
       filenames=[f.filename for f in files],
       message=f"Successfully ingested {len(files)} file(s) into {total_chunks} chunk(s).",
   )


@router.delete(
   "",
   status_code=status.HTTP_200_OK,
   summary="Clear all indexed documents from the knowledge base",
)
async def clear_documents(
   vector_store: dict = Depends(get_vector_store),
   bm25_index: dict  = Depends(get_bm25_index),
) -> dict:
   """Remove every indexed chunk from the in-memory vector store and BM25 index."""
   doc_ids = {v["chunk"].document_id for v in vector_store.values() if "chunk" in v}
   removed = len(doc_ids)
   vector_store.clear()
   bm25_index.clear()
   return {
       "message": f"Knowledge base cleared. {removed} document(s) removed.",
       "documents_removed": removed,
   }
