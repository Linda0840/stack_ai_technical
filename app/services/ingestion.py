# This file intends to implement the ingestion pipeline:
# - read uploaded PDF files
# - extract text page by page
# - split text into chunks
# - generate embeddings
# - store chunks and keyword tokens in memory


"""
Ingestion service: PDF extraction → chunking → embedding → index storage.


Considerations
--------------
- Chunking strategy: fixed-size with overlap to avoid splitting mid-sentence.
 Overlap size (chunk_overlap) ensures context isn't lost at chunk boundaries.
- Page-aware splitting: chunks are tagged with their source page number so
 citations stay accurate.
- Embeddings are generated via Mistral's embedding API and stored in an
 in-memory vector store (dict keyed by chunk_id), since we cannot use third-party
 vector databases for this project.
"""


import asyncio
import logging
import re
import uuid
from io import BytesIO

from pypdf import PdfReader
from mistralai.client import Mistral

from app.core.config import get_settings
from app.models.schemas import DocumentChunk


logger = logging.getLogger(__name__)
settings = get_settings()

_EMBED_BATCH_SIZE = 64


def _tokenize(s: str) -> list[str]:
   return re.findall(r"[a-z0-9]+", s.lower())


class IngestionService:
   def __init__(self, vector_store: dict, bm25_index: dict):
       # Injected shared stores (populated at startup via lifespan)
       self._vector_store = vector_store   # chunk_id -> {chunk, embedding}
       self._bm25_index = bm25_index       # chunk_id -> tokenised text (for BM25)


   # ------------------------------------------------------------------
   # Public
   # ------------------------------------------------------------------

   # main orchestration function

   async def ingest_files(self, files: list) -> tuple[list[str], int]:
       """
       Accept a list of UploadFile objects, extract text, chunk, embed and index.
       Returns (document_ids, total_chunks).
       """
       document_ids: list[str] = []
       total_chunks = 0

       for upload_file in files:
           doc_id = str(uuid.uuid4())
           raw_bytes = await upload_file.read()
           pages = self._extract_text(raw_bytes, upload_file.filename)
           chunks = self._chunk_pages(pages, doc_id, upload_file.filename)
           await self._embed_and_store(chunks)
           document_ids.append(doc_id)
           total_chunks += len(chunks)
           logger.info("Ingested %s → %d chunks (doc_id=%s)", upload_file.filename, len(chunks), doc_id)


       return document_ids, total_chunks


   # ------------------------------------------------------------------
   # Private helpers
   # ------------------------------------------------------------------


   def _extract_text(self, raw_bytes: bytes, filename: str) -> list[tuple[int, str]]:
       """
       Extract text from PDF bytes.
       Returns list of (page_number, page_text).
       """
       logger.debug("_extract_text called for %s (%d bytes)", filename, len(raw_bytes))
       if not raw_bytes:
           return [(1, "")]

       try:
           reader = PdfReader(BytesIO(raw_bytes))
       except Exception as exc:
           logger.warning("PDF read failed for %s: %s", filename, exc)
           return [(1, "")]

       pages: list[tuple[int, str]] = []
       for i, page in enumerate(reader.pages, start=1):
           try:
               text = page.extract_text() or ""
           except Exception as exc:
               logger.debug("extract_text failed page %d %s: %s", i, filename, exc)
               text = ""
           pages.append((i, text))

       if not pages:
           return [(1, "")]
       return pages


   def _chunk_end_with_sentence_preference(
       self, text: str, start: int, chunk_size: int, min_chunk: int
   ) -> int:
       """Pick end index: prefer the last sentence-like break in the tail of the window."""
       n = len(text)
       if start >= n:
           return n
       max_end = min(start + chunk_size, n)
       if max_end >= n:
           return n

        # search window logic:
        # at least 80 characters, at most 200 characters
        # usually ~1/3 of the chunk size
       lookback = min(200, max(80, chunk_size // 3))

       # start + min_chunk: make sure chunk is at least min_chunk
       # max_end - lookback: search within the last lookback characters
       region_start = max(start + min_chunk, max_end - lookback)
       region = text[region_start:max_end]

       best = max_end

       # sentence breaks
       # e.g. "\n\n" is paragraph break
       for needle in ("\n\n", "\n", ". ", ".\n", "! ", "?\n", ".", "!", "?"):
           idx = region.rfind(needle) # find the last occurrence of the needle
           if idx == -1:
               continue
           cand = region_start + idx + len(needle)

           # ensure chunk is not tiny and not exceed target size
           if cand > start + min_chunk // 2 and cand <= max_end:
               best = cand
               break
       return best


   def _chunk_pages(
       self,
       pages: list[tuple[int, str]],
       doc_id: str,
       filename: str,
   ) -> list[DocumentChunk]:
       """
       Split page text into fixed-size chunks with overlap.
       Uses a sliding window with overlap; ends chunks at sentence boundaries when
       a suitable break appears in the trailing portion of the window.
       """
       chunks: list[DocumentChunk] = []
       chunk_index = 0
       min_chunk = max(48, settings.chunk_size // 8) # prevent tiny chunks

       for page_num, text in pages:
           if not text:
               continue
           start = 0
           while start < len(text):
               end = self._chunk_end_with_sentence_preference(
                   text, start, settings.chunk_size, min_chunk
               )
               chunk_text = text[start:end].strip()
               if not chunk_text:
                   start = end if end > start else start + 1
                   continue
               chunks.append(
                   DocumentChunk(
                       chunk_id=str(uuid.uuid4()),
                       document_id=doc_id,
                       filename=filename,
                       text=chunk_text,
                       page_number=page_num,
                       chunk_index=chunk_index,
                   )
               )
               chunk_index += 1
               if end >= len(text):
                   break
               next_start = end - settings.chunk_overlap
               start = max(start + 1, next_start)

       return chunks


   async def _embed_and_store(self, chunks: list[DocumentChunk]) -> None:
       """
       Call Mistral embedding API, store vectors + BM25 tokens.
       Batches inputs to reduce round-trips and respect typical payload limits.
       """
       if not chunks:
           return

       client = Mistral(api_key=settings.mistral_api_key)


        # send batches of chunks to the embedding API
       for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
           batch = chunks[i : i + _EMBED_BATCH_SIZE]
           texts = [c.text for c in batch]

           def _call_embed() -> object:
               return client.embeddings.create(
                   model=settings.mistral_embed_model,
                   inputs=texts,
               )

        # response.data = [
        #     { index: 0, embedding: [embedding vector for batch 1] },
        #     { index: 1, embedding: [embedding vector for batch 2] },
        #     ...
        # ]

           try:
               response = await asyncio.to_thread(_call_embed)
           except Exception as exc:
               logger.exception("Embedding API failed for batch starting %d: %s", i, exc)
               raise

           by_index: dict[int, list[float]] = {}
           for row in response.data:
               idx = row.index if row.index is not None else 0
               if row.embedding:
                   by_index[idx] = list(row.embedding)

           for j, chunk in enumerate(batch):
               embedding = by_index.get(j, [])
               # semantic store: chunk_id -> {chunk, embedding}
               self._vector_store[chunk.chunk_id] = {
                   "chunk": chunk,
                   "embedding": embedding,
               }
               # keyword store: chunk_id -> token list
               self._bm25_index[chunk.chunk_id] = _tokenize(chunk.text)

       logger.debug("Stored %d chunks in vector/BM25 stores", len(chunks))
