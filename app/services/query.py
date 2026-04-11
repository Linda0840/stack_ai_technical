from app.models.schemas import QueryRequest, QueryResponse

NO_SEARCH_ANSWER = "Hello! How can I help you today?"


class QueryService:
    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        query = request.query.strip().lower()
        greetings = {"hello", "hi", "hey", "thanks", "bye", "goodbye", "ok", "okay"}

        if query in greetings:
            return QueryResponse(
                query=request.query,
                answer=NO_SEARCH_ANSWER,
                sources=[],
            )

        return QueryResponse(
            query=request.query,
            answer="Knowledge-base search not implemented yet.",
            sources=[],
        )
