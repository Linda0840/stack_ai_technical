# This endpoint:


# creates a POST route at /ingest
# accepts one or more uploaded files
# validates that files are present
# validates that each file is a PDF
# injects an IngestionService
# passes the files to the service for processing
# returns a structured IngestResponse




from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status


from app.models.schemas import IngestResponse # response is typed with IngestResponse
from app.services.ingestion import IngestionService # injection logic
from app.api.deps import get_ingestion_service  # FastAPI injects the service through get_ingestion_service


router = APIRouter(prefix="/ingest", tags=["Ingestion"])


ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE_MB = 20




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


   for f in files:
       if f.content_type not in ALLOWED_CONTENT_TYPES:
           raise HTTPException(
               status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
               detail=f"'{f.filename}' is not a PDF (got content-type: {f.content_type}).",
           )


   document_ids, total_chunks = await service.ingest_files(files)


   return IngestResponse(
       document_ids=document_ids,
       total_chunks=total_chunks,
       filenames=[f.filename for f in files],
       message=f"Successfully ingested {len(files)} file(s) into {total_chunks} chunk(s).",
   )
