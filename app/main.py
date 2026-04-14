# main.py
# This file initializes the FastAPI application, sets up global configuration, 
# prepares in-memory storage for retrieval, enables CORS, and registers API routes.

from contextlib import asynccontextmanager  # used to define startup/shutdown logic
import logging

from fastapi import FastAPI # backend app
from fastapi.middleware.cors import CORSMiddleware     # allows frontend (like UI) to call API
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import get_settings    # loads config (env variables, etc.)
from app.api.routes import ingestion, query, health # API route modules


# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pull information from config like app name, version, mistral API key, etc.
settings = get_settings()


# define startup/shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise in-memory stores on startup."""
    # semantic
    app.state.vector_store: dict = {}  # chunk_id -> {chunk, embedding}

    # keyword
    app.state.bm25_index: dict = {}    # chunk_id -> token list

    # lightweight upload log: list of {document_id, filename, uploaded_at}
    # accumulates across multiple upload batches in the same session
    app.state.document_registry: list = []

    logger.info("In-memory stores initialised.")
    yield
    logger.info("Shutting down.")


# create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "RAG pipeline API: upload PDF files and query them with natural language. "
        "Powered by Mistral AI."
    ),
    lifespan=lifespan,
)

# allow backend to be called from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router) # health check endpoint
app.include_router(ingestion.router, prefix="/api/v1") # ingestion endpoint
app.include_router(query.router, prefix="/api/v1") # query endpoint

# serve the frontend
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join(_static_dir, "index.html"))
