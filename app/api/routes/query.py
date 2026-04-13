# At a high level, this endpoint:


# defines a POST route at /query
# expects a structured request body of type QueryRequest
# asks FastAPI to inject a QueryService
# forwards the request to service.handle_query(request)
# returns a structured QueryResponse


from fastapi import APIRouter, Depends, status


from app.models.schemas import QueryRequest, QueryResponse
from app.services.query import QueryService
from app.api.deps import get_query_service


router = APIRouter(prefix="/query", tags=["Query"])




@router.post(
   "",
   response_model=QueryResponse,
   status_code=status.HTTP_200_OK,
   summary="Query the knowledge base",
)




async def query_knowledge_base(
   request: QueryRequest,
   service: QueryService = Depends(get_query_service),
) -> QueryResponse:
   """
   Processes a user query through the full RAG pipeline:
   intent detection → query transformation → hybrid search →
   re-ranking → LLM generation.
   """
   return await service.handle_query(request)

