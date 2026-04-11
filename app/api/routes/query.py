from fastapi import APIRouter, status
from app.models.schemas import QueryRequest

router = APIRouter(prefix="/query", tags=["Query"])

@router.post("", status_code=status.HTTP_200_OK, summary="Query the knowledge base")
async def query_knowledge_base(request: QueryRequest):
    return {
        "question": request.question,
        "message": "query pipeline not implemented yet"
    }
