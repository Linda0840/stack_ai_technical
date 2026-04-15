# This endpoint:


# creates a POST route at /ingest
# accepts one or more uploaded files
# validates that files are present
# validates that each file is a PDF
# injects an IngestionService
# passes the files to the service for processing
# returns a structured IngestResponse




from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.schemas import DocumentInfo, DocumentListResponse, IngestResponse, WorkspaceUsage
from app.services.ingestion import IngestionService
from app.api.deps import (
    get_document_registry,
    get_ingestion_service,
    get_vector_store,
    get_bm25_index,
    get_workspace_stats,
)


router = APIRouter(prefix="/ingest", tags=["Ingestion"])
_settings = get_settings()

ALLOWED_CONTENT_TYPES = {"application/pdf"}




@router.get(
   "",
   response_model=DocumentListResponse,
   status_code=status.HTTP_200_OK,
   summary="List all documents indexed in the current session",
)
async def list_documents(
   registry: list = Depends(get_document_registry),
   vector_store: dict = Depends(get_vector_store),
   workspace_stats: dict = Depends(get_workspace_stats),
) -> DocumentListResponse:
   """
   Returns every file that has been uploaded and indexed since the server
   started (or since the knowledge base was last cleared).  Multiple upload
   batches in the same session are accumulated here.
   """
   # Count chunks per document_id directly from the live vector store so the
   # numbers are always accurate even if a future operation removes chunks.
   chunk_counts: dict[str, int] = {}
   for entry in vector_store.values():
       if "chunk" in entry:
           doc_id = entry["chunk"].document_id
           chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1

   documents = [
       DocumentInfo(
           document_id=rec["document_id"],
           filename=rec["filename"],
           chunk_count=chunk_counts.get(rec["document_id"], 0),
           uploaded_at=rec["uploaded_at"],
       )
       for rec in registry
   ]
   cap = _settings.max_workspace_chunks
   used = workspace_stats["total_chunks"]
   usage = WorkspaceUsage(
       total_chunks=used,
       total_chars=workspace_stats["total_chars"],
       max_chunks=cap,
       percent_used=round(used / cap * 100, 1),
   )
   return DocumentListResponse(
       documents=documents,
       total_documents=len(documents),
       total_chunks=sum(d.chunk_count for d in documents),
       workspace_usage=usage,
       max_file_size_mb=_settings.max_file_size_mb,
   )


@router.post(
   "",
   response_model=IngestResponse,
   status_code=status.HTTP_201_CREATED,
   summary="Upload and ingest one or more PDF files",
)
async def ingest_documents(
   files: list[UploadFile] = File(..., description="One or more PDF files to ingest"),
   service: IngestionService = Depends(get_ingestion_service),
   registry: list = Depends(get_document_registry),
   workspace_stats: dict = Depends(get_workspace_stats),
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
       msg = str(exc)
       # Capacity exceeded → 400; oversized file → 413.
       if "capacity exceeded" in msg.lower():
           raise HTTPException(
               status_code=status.HTTP_400_BAD_REQUEST,
               detail=msg,
           ) from exc
       raise HTTPException(
           status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
           detail=msg,
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

   # Record each file in the session registry.
   # document_ids are returned in the same order as files by ingest_files.
   uploaded_at = datetime.utcnow()
   for doc_id, f in zip(document_ids, files):
       registry.append({
           "document_id": doc_id,
           "filename": f.filename,
           "uploaded_at": uploaded_at,
       })

   cap = _settings.max_workspace_chunks
   used = workspace_stats["total_chunks"]
   usage = WorkspaceUsage(
       total_chunks=used,
       total_chars=workspace_stats["total_chars"],
       max_chunks=cap,
       percent_used=round(used / cap * 100, 1),
   )
   return IngestResponse(
       document_ids=document_ids,
       total_chunks=total_chunks,
       filenames=[f.filename for f in files],
       message=(
           f"Successfully ingested {len(files)} file(s) into {total_chunks} chunk(s). "
           f"Workspace usage: {used}/{cap} chunks ({usage.percent_used}%)."
       ),
       workspace_usage=usage,
   )


@router.delete(
   "",
   status_code=status.HTTP_200_OK,
   summary="Clear all indexed documents from the knowledge base",
)
async def clear_documents(
   vector_store: dict = Depends(get_vector_store),
   bm25_index: dict  = Depends(get_bm25_index),
   registry: list    = Depends(get_document_registry),
   workspace_stats: dict = Depends(get_workspace_stats),
) -> dict:
   """Remove every indexed chunk from the in-memory vector store, BM25 index,
   and session document registry."""
   removed = len(registry)
   vector_store.clear()
   bm25_index.clear()
   registry.clear()
   workspace_stats["total_chunks"] = 0
   workspace_stats["total_chars"] = 0
   return {
       "message": f"Knowledge base cleared. {removed} document(s) removed.",
       "documents_removed": removed,
   }
