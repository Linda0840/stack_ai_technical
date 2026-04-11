from fastapi import APIRouter, File, UploadFile, HTTPException, status

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Upload PDF files")
async def ingest_documents(
    files: list[UploadFile] = File(..., description="One or more PDF files to ingest"),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided.",
        )

    for f in files:
        if f.content_type != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"'{f.filename}' is not a PDF.",
            )

    return {
        "message": f"Received {len(files)} PDF file(s).",
        "filenames": [f.filename for f in files],
    }
