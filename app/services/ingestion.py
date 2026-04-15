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
import time
import uuid
from io import BytesIO

import pdfplumber
from pypdf import PdfReader
from mistralai.client import Mistral

from app.core.config import get_settings
from app.models.schemas import DocumentChunk


logger = logging.getLogger(__name__)
settings = get_settings()

# Dimension of the Mistral embed model; used for placeholder vectors in debug mode.
_MISTRAL_EMBED_DIM = 1024
# Maximum loop iterations per page during chunking; guards against infinite loops.
_MAX_CHUNKS_PER_PAGE = 2000


def _tokenize(s: str) -> list[str]:
   return re.findall(r"[a-z0-9]+", s.lower())


def _tokenize_filename(filename: str) -> list[str]:
   """
   Produce BM25 tokens from a filename with letter-digit boundary splitting.

   Without this, 'Lecture1' in a filename is a single token that never matches
   the query tokens ['lecture', '1'] from 'can you summarize lecture 1?'.

   Examples
   --------
   '2026_C51_Lecture1_overview_hardcopy.pdf'
       → ['2026', 'c', '51', 'lecture', '1', 'overview', 'hardcopy']
   '15.C51-2026-lecture02.pdf'
       → ['15', 'c', '51', '2026', 'lecture', '02']
   """
   stem = filename.rsplit(".", 1)[0]                   # drop extension
   stem = re.sub(r"[_\-\.]+", " ", stem)              # separators → space
   stem = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)  # Lecture1  → Lecture 1
   stem = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", stem)  # 2Intro    → 2 Intro
   return re.findall(r"[a-z0-9]+", stem.lower())


def _filename_label(filename: str) -> str:
   """Human-readable label used as a context prefix when embedding chunks."""
   stem = filename.rsplit(".", 1)[0]
   stem = re.sub(r"[_\-\.]+", " ", stem)
   stem = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)
   stem = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", stem)
   return stem.strip()


class IngestionService:
   def __init__(self, vector_store: dict, bm25_index: dict, workspace_stats: dict):
       # Injected shared stores (populated at startup via lifespan)
       self._vector_store = vector_store   # chunk_id -> {chunk, embedding}
       self._bm25_index = bm25_index       # chunk_id -> tokenised text (for BM25)
       self._workspace_stats = workspace_stats  # {"total_chunks": int, "total_chars": int}


   # ------------------------------------------------------------------
   # Public
   # ------------------------------------------------------------------

   # main orchestration function

   async def ingest_files(self, files: list) -> tuple[list[str], int]:
       """
       Accept a list of UploadFile objects, extract text, chunk, embed and index.
       Returns (document_ids, total_chunks).

       Scalability strategy
       --------------------
       Phase 1 (read → extract → chunk) runs concurrently for all files via
       asyncio.gather.  PDF parsing is CPU-bound so each file's _extract_text call
       is offloaded to a thread (asyncio.to_thread) so it never blocks the event
       loop.

       Phase 2 (embed → store) is kept sequential per-batch to respect Mistral
       rate limits; all chunks are submitted in a single _embed_and_store call so
       the batch size setting still applies globally across all files.
       """
       pipeline_start = time.monotonic()
       max_bytes = settings.max_file_size_mb * 1024 * 1024

       async def _prepare(upload_file) -> tuple[str, str, list]:
           """Read, validate size, extract text, and chunk one file."""
           fname = upload_file.filename
           doc_id = str(uuid.uuid4())
           logger.info("[ingest] ── START %s (doc_id=%s)", fname, doc_id)

           # ── Step 1: read ─────────────────────────────────────────────
           t0 = time.monotonic()
           raw_bytes = await upload_file.read()
           logger.info("[ingest] step=read   file=%s  bytes=%d  elapsed=%.3fs",
                       fname, len(raw_bytes), time.monotonic() - t0)

           # ── Size guard ───────────────────────────────────────────────
           if len(raw_bytes) > max_bytes:
               raise ValueError(
                   f"'{fname}' is {len(raw_bytes) // (1024*1024)} MB, "
                   f"which exceeds the {settings.max_file_size_mb} MB per-file limit."
               )

           # ── Step 2: extract (offloaded to thread pool) ───────────────
           t0 = time.monotonic()
           pages = await asyncio.to_thread(self._extract_text, raw_bytes, fname)
           total_chars = sum(len(t) for _, t in pages)
           logger.info("[ingest] step=extract file=%s  pages=%d  chars=%d  elapsed=%.3fs",
                       fname, len(pages), total_chars, time.monotonic() - t0)

           # ── Step 3: chunk ────────────────────────────────────────────
           t0 = time.monotonic()
           chunks = self._chunk_pages(pages, doc_id, fname)
           logger.info("[ingest] step=chunk  file=%s  chunks=%d  elapsed=%.3fs",
                       fname, len(chunks), time.monotonic() - t0)

           return doc_id, fname, chunks

       # ── Phase 1: all files in parallel ───────────────────────────────────
       results = await asyncio.gather(
           *[_prepare(f) for f in files],
           return_exceptions=True,
       )

       # Propagate the first error (e.g. size violation) as a plain exception so
       # the route layer can convert it to an appropriate HTTP response.
       for result in results:
           if isinstance(result, Exception):
               raise result

       # ── Phase 2: collect all chunks, embed in one sequential pass ─────────
       document_ids: list[str] = []
       all_chunks: list = []
       for doc_id, fname, chunks in results:  # type: ignore[misc]
           document_ids.append(doc_id)
           all_chunks.extend(chunks)
           logger.info("[ingest] queued %d chunks from %s", len(chunks), fname)

       # ── Workspace capacity guard ──────────────────────────────────────────
       current = self._workspace_stats["total_chunks"]
       incoming = len(all_chunks)
       cap = settings.max_workspace_chunks
       if current + incoming > cap:
           pct = int(current / cap * 100)
           raise ValueError(
               f"Session capacity exceeded: adding {incoming} chunk(s) would bring "
               f"the workspace to {current + incoming}/{cap} chunks "
               f"(currently at {pct}%). Please clear the workspace first."
           )

       t0 = time.monotonic()
       await self._embed_and_store(all_chunks)
       logger.info(
           "[ingest] embed+store: %d total chunks  elapsed=%.3fs  total_pipeline=%.3fs",
           len(all_chunks), time.monotonic() - t0, time.monotonic() - pipeline_start,
       )

       # ── Update session-level usage counters ───────────────────────────────
       self._workspace_stats["total_chunks"] += incoming
       self._workspace_stats["total_chars"] += sum(len(c.text) for c in all_chunks)
       logger.info(
           "[ingest] workspace usage: %d/%d chunks (%.1f%%)",
           self._workspace_stats["total_chunks"], cap,
           self._workspace_stats["total_chunks"] / cap * 100,
       )

       return document_ids, len(all_chunks)


   # ------------------------------------------------------------------
   # Private helpers
   # ------------------------------------------------------------------


   def _extract_text(self, raw_bytes: bytes, filename: str) -> list[tuple[int, str]]:
       """
       Extract text from PDF bytes.
       Returns list of (page_number, page_text).

       Strategy: try pdfplumber first (handles more font/layout types); fall back
       to pypdf if pdfplumber yields no usable text across all pages.
       """
       logger.debug("_extract_text called for %s (%d bytes)", filename, len(raw_bytes))
       if not raw_bytes:
           return [(1, "")]

       pages = self._extract_with_pdfplumber(raw_bytes, filename)
       total_chars = sum(len(t) for _, t in pages)

       if total_chars < 50:
           logger.info(
               "pdfplumber yielded only %d chars for %s; falling back to pypdf.",
               total_chars, filename,
           )
           pages = self._extract_with_pypdf(raw_bytes, filename)
           total_chars = sum(len(t) for _, t in pages)

       logger.info("_extract_text: %d pages, %d total chars from %s", len(pages), total_chars, filename)
       return pages if pages else [(1, "")]


   def _extract_with_pdfplumber(self, raw_bytes: bytes, filename: str) -> list[tuple[int, str]]:
       """Use pdfplumber for text extraction (robust with most font/layout types)."""
       pages: list[tuple[int, str]] = []
       try:
           with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
               for i, page in enumerate(pdf.pages, start=1):
                   try:
                       text = page.extract_text() or ""
                   except Exception as exc:
                       logger.debug("pdfplumber page %d failed for %s: %s", i, filename, exc)
                       text = ""
                   pages.append((i, text))
       except Exception as exc:
           logger.warning("pdfplumber open failed for %s: %s", filename, exc)
       return pages


   def _extract_with_pypdf(self, raw_bytes: bytes, filename: str) -> list[tuple[int, str]]:
       """Use pypdf for text extraction (fallback)."""
       pages: list[tuple[int, str]] = []
       try:
           reader = PdfReader(BytesIO(raw_bytes))
       except Exception as exc:
           logger.warning("pypdf read failed for %s: %s", filename, exc)
           return [(1, "")]
       for i, page in enumerate(reader.pages, start=1):
           try:
               text = page.extract_text() or ""
           except Exception as exc:
               logger.debug("pypdf page %d failed for %s: %s", i, filename, exc)
               text = ""
           pages.append((i, text))
       return pages if pages else [(1, "")]


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
           iters = 0
           while start < len(text):
               iters += 1
               if iters > _MAX_CHUNKS_PER_PAGE:
                   logger.error(
                       "[chunk] safety-break after %d iterations on page %d of %s "
                       "(start=%d, len=%d) — possible infinite loop, skipping rest of page",
                       iters, page_num, filename, start, len(text),
                   )
                   break

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

           logger.debug("[chunk] page=%d  iters=%d  chunks_so_far=%d", page_num, iters, len(chunks))

       return chunks


   async def _embed_and_store(self, chunks: list[DocumentChunk]) -> None:
       """
       Call Mistral embedding API, store vectors + BM25 tokens.

       Each batch is wrapped in asyncio.wait_for so a hanging API call never
       blocks the server indefinitely.  Set DEBUG_SKIP_EMBEDDINGS=true in .env
       to store zero-vectors and skip the API entirely (useful to isolate whether
       slowness comes from extraction/chunking or from Mistral).
       """
       if not chunks:
           return

       batch_size = settings.embed_batch_size
       timeout    = settings.embed_timeout_seconds
       # embed_mode is the primary flag; debug_skip_embeddings is the legacy fallback.
       skip_api   = settings.embed_mode == "skip" or settings.debug_skip_embeddings

       logger.info(
           "[embed] mode=%s  chunks=%d  batch_size=%d  timeout=%ds",
           "skip" if skip_api else "real", len(chunks), batch_size, timeout,
       )

       if skip_api:
           logger.warning(
               "[embed] SKIP mode — storing %d zero-vectors, Mistral API will NOT be called",
               len(chunks),
           )
           placeholder = [0.0] * _MISTRAL_EMBED_DIM
           for chunk in chunks:
               self._vector_store[chunk.chunk_id] = {"chunk": chunk, "embedding": placeholder}
               self._bm25_index[chunk.chunk_id]   = (
                   _tokenize(chunk.text) + _tokenize_filename(chunk.filename)
               )
           logger.info("[embed] stored %d placeholder chunks — to use real embeddings set EMBED_MODE=real", len(chunks))
           return

       client = Mistral(api_key=settings.mistral_api_key)
       logger.info(
           "[embed] REAL mode — starting Mistral API calls: %d chunks, batch_size=%d, timeout=%ds",
           len(chunks), batch_size, timeout,
       )

       for i in range(0, len(chunks), batch_size):
           batch = chunks[i : i + batch_size]
           # Prepend a filename context line so the embedding captures which
           # document the chunk belongs to.  This helps semantic search
           # distinguish e.g. "Lecture 1" from "Lecture 2" when the body text
           # of a file never explicitly mentions its own name.
           batch_texts = [
               f"[Document: {_filename_label(c.filename)}]\n{c.text}"
               for c in batch
           ]

           def _call_embed(texts=batch_texts) -> object:
               return client.embeddings.create(
                   model=settings.mistral_embed_model,
                   inputs=texts,
               )

           batch_num = i // batch_size + 1
           total_batches = (len(chunks) + batch_size - 1) // batch_size
           logger.info(
               "[embed] batch %d/%d — %d chunks (chars: %d)",
               batch_num, total_batches, len(batch),
               sum(len(t) for t in batch_texts),
           )

           t0 = time.monotonic()
           try:
               response = await asyncio.wait_for(
                   asyncio.to_thread(_call_embed),
                   timeout=timeout,
               )
           except asyncio.TimeoutError:
               logger.error(
                   "[embed] batch %d/%d TIMED OUT after %ds — "
                   "Mistral API did not respond. Check network/API key.",
                   batch_num, total_batches, timeout,
               )
               raise RuntimeError(
                   f"Embedding API timed out after {timeout}s on batch {batch_num}/{total_batches}. "
                   "Check your network connection and Mistral API key."
               )
           except Exception as exc:
               logger.exception(
                   "[embed] batch %d/%d FAILED after %.3fs: %s",
                   batch_num, total_batches, time.monotonic() - t0, exc,
               )
               raise

           elapsed = time.monotonic() - t0
           logger.info(
               "[embed] batch %d/%d OK — %.3fs — received %d embeddings",
               batch_num, total_batches, elapsed, len(response.data),
           )

           # The API returns embeddings in the same order as inputs; zip directly
           # instead of relying on row.index (which is Optional and may be None).
           embeddings = [
               list(row.embedding) if row.embedding else []
               for row in response.data
           ]
           for chunk, embedding in zip(batch, embeddings):
               self._vector_store[chunk.chunk_id] = {"chunk": chunk, "embedding": embedding}
               self._bm25_index[chunk.chunk_id]   = (
                   _tokenize(chunk.text) + _tokenize_filename(chunk.filename)
               )

       logger.info("[embed] done — stored %d chunks in vector/BM25 stores", len(chunks))
