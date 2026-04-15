"""
Query service: intent detection → query transformation → hybrid search →
re-ranking → LLM generation.
"""


import asyncio
import json
import logging
import math
import re
from typing import Any

import numpy as np

from mistralai.client import Mistral
from mistralai.client.models.assistantmessage import AssistantMessage
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.usermessage import UserMessage

from app.core.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk


logger = logging.getLogger(__name__)
settings = get_settings()


NO_SEARCH_ANSWER = "Hello! How can I help you today?"
EMPTY_KB_ANSWER = (
    "The current knowledge base is empty. "
    "Please upload at least one PDF file first."
)
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find sufficiently relevant information in the indexed documents "
    "to answer your question confidently. Please try rephrasing, or upload "
    "documents that cover this topic."
)


# ── Answer-shaping: intent → format template ──────────────────────────────────
#
# Patterns are checked in order; the first match wins.
# "table" is before "comparison" so "compare in a table" → table, not comparison.
# "list" is before "definition" so "list what X is" → list, not definition.
_SHAPE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(table|tabulate|tabular|spreadsheet)\b",                               "table"),
    (r"\b(compar|contrast|difference|similarities|vs\.?|versus)\b",             "comparison"),
    (r"\b(list|enumerate|bullet|itemize|what are|give me all|name all)\b",      "list"),
    (r"\b(how\s+to|how\s+do\s+i|steps?\s+(to|for)|guide\s+(to|for)|"
     r"walk\s+me\s+through|instructions?\s+(for|to))\b",                        "instruction"),
    (r"\b(what\s+is|what\s+are|define|definition\s+of|"
     r"meaning\s+of|explain\s+what)\b",                                         "definition"),
]

_PROMPT_TEMPLATES: dict[str, str] = {
    "list": (
        "You are a careful assistant. Answer using ONLY the provided context.\n"
        "Format your answer as a markdown bulleted list (- item).\n"
        "Every bullet must end with a citation in the form (filename, p.N).\n"
        "If the context is insufficient, say so briefly instead of guessing."
    ),
    "table": (
        "You are a careful assistant. Answer using ONLY the provided context.\n"
        "Format your answer as a markdown table with a clear header row.\n"
        "Include a 'Source' column that cites filename and page number for each row.\n"
        "If the context is insufficient, say so briefly instead of guessing."
    ),
    "comparison": (
        "You are a careful assistant. Answer using ONLY the provided context.\n"
        "Format your answer as a markdown comparison table: items to compare as "
        "columns, attributes as rows.\n"
        "Follow the table with a one-paragraph prose summary.\n"
        "Cite sources by filename and page. "
        "If the context is insufficient, say so briefly instead of guessing."
    ),
    "definition": (
        "You are a careful assistant. Answer using ONLY the provided context.\n"
        "Give a concise definition in one or two sentences, then add a short "
        "elaborating paragraph if the context supports it.\n"
        "Cite sources by filename and page. "
        "If the context is insufficient, say so briefly instead of guessing."
    ),
    "instruction": (
        "You are a careful assistant. Answer using ONLY the provided context.\n"
        "Format your answer as a numbered step-by-step list. "
        "Each step must be a single, clear action sentence.\n"
        "Cite sources by filename and page at the end of the list.\n"
        "If the context is insufficient, say so briefly instead of guessing."
    ),
    "factual": (
        "You are a careful assistant. Answer using only the provided context. "
        "If the context is insufficient, say so briefly. "
        "Cite sources by filename and page when you use them."
    ),
}


def _extract_claims(text: str) -> list[tuple[int, str]]:
    """
    Walk the answer line-by-line and return (line_index, plain_claim) pairs
    for lines that contain verifiable factual claims.

    Skipped automatically
    ---------------------
    - Blank lines
    - Markdown headers  (``### Heading``)
    - Horizontal rules  (``---``, ``***``)
    - Table-separator rows  (``| --- | --- |``)
    - Lines whose plain text is shorter than 15 chars (fragments, labels)

    Stripped before sending to the LLM
    -----------------------------------
    - List markers  ``- ``, ``* ``, ``1. ``
    - Table pipe borders
    - Bold/italic/strikethrough markers  ``**``, ``_``, ``~~``
    """
    claims: list[tuple[int, str]] = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip structural markdown — these are not factual claims.
        if (re.match(r"^#{1,6}\s", stripped)
                or re.match(r"^[-*_]{3,}$", stripped)
                or re.match(r"^\|[-:\s|]+\|$", stripped)):
            continue
        # Strip formatting to expose the plain claim text for the LLM.
        plain = re.sub(r"^[-*+]\s+|\d+\.\s+", "", stripped)   # list bullets
        plain = re.sub(r"^\|(.+)\|$", r"\1", plain)            # table cells
        plain = re.sub(r"\*{1,3}|_{1,3}|~~", "", plain).strip()
        if len(plain) >= 15:
            claims.append((idx, plain))
    return claims


def _classify_query_shape(query: str) -> str:
    """
    Classify a query into a formatting shape using regex heuristics.

    Returns one of: 'list', 'table', 'comparison', 'definition',
    'instruction', 'factual' (default).

    No LLM call is made — this runs entirely locally so it adds zero latency
    to the pipeline.  The patterns are ordered from most-specific to least so
    an earlier match always takes precedence.
    """
    q = query.lower()
    for pattern, shape in _SHAPE_PATTERNS:
        if re.search(pattern, q):
            return shape
    return "factual"


def _tokenize(s: str) -> list[str]:
   return re.findall(r"[a-z0-9]+", s.lower())


def _cosine_dense(a: list[float], b: list[float]) -> float:
   if not a or not b or len(a) != len(b):
       return 0.0
   dot = sum(x * y for x, y in zip(a, b))
   na = math.sqrt(sum(x * x for x in a))
   nb = math.sqrt(sum(y * y for y in b))
   if na == 0 or nb == 0:
       return 0.0
   return dot / (na * nb)


def _assistant_text(message: AssistantMessage | None) -> str:
   if message is None:
       return ""
   content: Any = message.content
   if isinstance(content, str):
       return content.strip()
   if isinstance(content, list):
       parts: list[str] = []
       for chunk in content:
           t = getattr(chunk, "text", None)
           if t:
               parts.append(str(t))
       return "".join(parts).strip()
   return ""


def _bm25_scores_for_query(
   query_tokens: list[str],
   bm25_index: dict[str, list[str]],
   k1: float = 1.5,
   b: float = 0.75,
) -> dict[str, float]:
   """BM25 score per chunk_id for the given query tokens."""
   if not query_tokens or not bm25_index:
       return {}

   doc_tokens: dict[str, list[str]] = {
       cid: toks for cid, toks in bm25_index.items() if toks
   }
   if not doc_tokens:
       return {}

   # N: number of chunks in _bm25_index
   N = len(doc_tokens)

   # lengths: document length
   lengths = {cid: len(toks) for cid, toks in doc_tokens.items()}

   # avgdl: average chunk length for this document
   avgdl = sum(lengths.values()) / N if N else 1.0
   if avgdl == 0:
       avgdl = 1.0

   # query
   # df: {"machine": count of how many _bm25_index chunks contain it,
   #      "learning": count of how many _bm25_index chunks contain it,
   #      "is": count of how many _bm25_index chunks contain it,
   #      "useful": count of how many _bm25_index chunks contain it,
   #      ...}
   
   df: dict[str, int] = {}
   for term in set(query_tokens):
       df[term] = sum(1 for toks in doc_tokens.values() if term in toks)

   scores: dict[str, float] = {}

   # _bm25_index
   # doc_tokens: {0: ["machine", "learning"],
   #              1: ["data", "science"],
   #              ...}
   for cid, toks in doc_tokens.items():

      # dl is the chunk length
       dl = lengths[cid]

       # _bm25_index
       # tf = {"machine": 1, "learning": 1, "data": 1, "science": 1}
       tf: dict[str, int] = {}
       for t in toks:
           tf[t] = tf.get(t, 0) + 1
       s = 0.0

       # example: query_tokens = ["machine", "learning", "is", "useful"]
       for term in query_tokens:
           f = tf.get(term, 0)
           if f == 0:
               continue
           n_df = df.get(term, 0)  # if query term is rare, n_df is small
           # log: rare words -> higher score
           idf = math.log((N - n_df + 0.5) / (n_df + 0.5) + 1.0)

           # dl / avgdl: longer chunks are penalized
           # k1: control TF importance, b: control length normalization
           denom = f + k1 * (1.0 - b + b * (dl / avgdl))
           s += idf * (f * (k1 + 1.0)) / denom if denom else 0.0
       scores[cid] = s
   return scores


def _min_max_norm(values: dict[str, float]) -> dict[str, float]:
   if not values:
       return {}
   xs = list(values.values())
   lo, hi = min(xs), max(xs)
   if hi - lo < 1e-12:
       return {k: 1.0 for k in values}
   return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class QueryService:
   def __init__(self, vector_store: dict, bm25_index: dict):
       self._vector_store = vector_store
       self._bm25_index = bm25_index


   def _client(self) -> Mistral:
       return Mistral(api_key=settings.mistral_api_key)


   # ------------------------------------------------------------------
   # Public entry point
   # ------------------------------------------------------------------


   async def handle_query(self, request: QueryRequest) -> QueryResponse:
       # Short-circuit before any LLM call when nothing has been ingested yet.
       if not self._vector_store:
           return QueryResponse(
               query=request.query,
               intent_triggered_search=False,
               evidence_sufficient=False,
               answer=EMPTY_KB_ANSWER,
               sources=[],
           )

       triggered = await self._detect_intent(request.query)

       if not triggered:
           return QueryResponse(
               query=request.query,
               intent_triggered_search=False,
               answer=NO_SEARCH_ANSWER,
               sources=[],
           )

       transformed = await self._transform_query(request.query)
       chunks = await self._hybrid_search(transformed, request.top_k)

       # ── Evidence threshold guard ─────────────────────────────────────────
       # Refuse to generate an answer when the knowledge base contains nothing
       # relevant enough.  Checking the max hybrid score here (before reranking)
       # avoids two unnecessary LLM calls when the index simply doesn't cover
       # the topic.  The hybrid score ∈ [0, 1] combines cosine similarity and
       # normalised BM25, so a low ceiling means both signals are weak.
       best_score = max((c.score for c in chunks), default=0.0)
       if best_score < settings.min_similarity_threshold:
           logger.info(
               "Insufficient evidence: best_score=%.4f < threshold=%.4f  query=%r",
               best_score, settings.min_similarity_threshold, request.query,
           )
           return QueryResponse(
               query=request.query,
               transformed_query=transformed,
               intent_triggered_search=True,
               evidence_sufficient=False,
               answer=INSUFFICIENT_EVIDENCE_ANSWER,
               sources=[],
           )

       # Classify the output format from the *original* query (before transformation
       # rewrites it for retrieval) so structural cues like "list" or "compare" are
       # still present when the shape is determined.
       shape = _classify_query_shape(request.query)
       logger.debug("Query shape: %s  query=%r", shape, request.query)

       reranked = await self._rerank(request.query, chunks, request.top_k)
       answer   = await self._generate(request.query, reranked, shape)

       # Post-hoc hallucination filter: verify each claim in the answer
       # against the retrieved chunks and remove unsupported ones.
       if settings.evidence_check_enabled:
           answer, hallucination_warning = await self._evidence_check(answer, reranked)
       else:
           hallucination_warning = False

       return QueryResponse(
           query=request.query,
           transformed_query=transformed,
           intent_triggered_search=True,
           evidence_sufficient=True,
           query_shape=shape,
           hallucination_warning=hallucination_warning,
           answer=answer,
           sources=reranked,
       )


   # ------------------------------------------------------------------
   # Pipeline steps
   # ------------------------------------------------------------------


   async def _detect_intent(self, query: str) -> bool:
       """
       Determine whether the query requires a knowledge-base search.
       Uses Mistral to classify as chitchat vs knowledge; falls back to a small
       greeting heuristic if the model call fails.
       """
       greetings = {"hello", "hi", "hey", "thanks", "bye", "goodbye", "ok", "okay"}
       q = query.strip()
       if q.lower() in greetings:
           return False

       def _call() -> str:
           client = self._client()
           resp = client.chat.complete(
               model=settings.mistral_model,
               temperature=0.0,
               max_tokens=8,
               messages=[
                   SystemMessage(
                       content=(
                           "Classify the user message as SEARCH or CHITCHAT.\n"
                           "CHITCHAT: greetings, farewells, thanks, or pure social "
                           "pleasantries with no informational intent "
                           "(e.g. 'hello', 'thanks', 'how are you').\n"
                           "SEARCH: every other message — any question, request for "
                           "information, or topic-related statement, regardless of "
                           "whether the answer exists in a knowledge base.\n"
                           "Reply with exactly one word: SEARCH or CHITCHAT."
                       )
                   ),
                   UserMessage(content=f"User message:\n{query}"),
               ],
           )
           choice = resp.choices[0] if resp.choices else None
           return _assistant_text(choice.message if choice else None).upper()

       try:
           label = await asyncio.to_thread(_call)
       except Exception as exc:
           logger.warning("Intent detection failed, defaulting to SEARCH: %s", exc)
           return True

       if "CHITCHAT" in label and "SEARCH" not in label:
           return False
       if "SEARCH" in label:
           return True
       return True


   async def _transform_query(self, query: str) -> str:
       """
       Rewrite the query to improve retrieval (concise search-oriented phrasing).
       """
       def _call() -> str:
           client = self._client()
           resp = client.chat.complete(
               model=settings.mistral_model,
               temperature=0.2,
               max_tokens=200,
               messages=[
                   SystemMessage(
                       content=(
                           "Rewrite the user's message into a short standalone search query "
                           "optimized for lexical and semantic retrieval. Output only the "
                           "rewritten query text, no quotes or explanation."
                       )
                   ),
                   UserMessage(content=query),
               ],
           )
           choice = resp.choices[0] if resp.choices else None
           text = _assistant_text(choice.message if choice else None)
           return text or query

       try:
           return (await asyncio.to_thread(_call)).strip() or query
       except Exception as exc:
           logger.warning("Query transform failed, using original: %s", exc)
           return query


   async def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
       """
       Combine semantic (cosine similarity) and keyword (BM25) results.

       Hybrid score = semantic_weight * semantic_score
                    + keyword_weight  * bm25_score

       Scalability notes
       -----------------
       - Cosine similarity is computed via a single numpy matrix-vector multiply
         (one BLAS call) instead of n pure-Python loops, giving ~100× speedup for
         large indexes.
       - retrieve_k is hard-capped at settings.max_retrieve_k so the downstream
         reranker prompt never exceeds the LLM context window regardless of top_k.
       - A production system with millions of chunks would replace this linear scan
         with an ANN index (e.g. FAISS / ScaNN), but for thousands of chunks this
         numpy approach is fast enough without an extra dependency.
       """
       logger.debug("_hybrid_search: query=%r top_k=%d store_size=%d", query, top_k, len(self._vector_store))
       if not self._vector_store:
           return []

       # Hard cap: never return more candidates than max_retrieve_k, regardless of top_k.
       retrieve_k = min(
           len(self._vector_store),
           min(settings.max_retrieve_k, max(top_k * 4, 15)),
       )

       query_tokens = _tokenize(query)
       bm25_raw = _bm25_scores_for_query(query_tokens, self._bm25_index)
       bm25_norm = _min_max_norm(bm25_raw)

       def _embed_query() -> list[float]:
           client = self._client()
           r = client.embeddings.create(
               model=settings.mistral_embed_model,
               inputs=[query],
           )
           row = r.data[0] if r.data else None
           return list(row.embedding) if row and row.embedding else []

       try:
           q_emb = await asyncio.to_thread(_embed_query)
       except Exception as exc:
           logger.warning("Query embedding failed, keyword-only hybrid: %s", exc)
           q_emb = []

       # Snapshot the store keys once so iteration order is consistent.
       chunk_ids = list(self._vector_store.keys())
       entries = [self._vector_store[cid] for cid in chunk_ids]
       n = len(chunk_ids)

       # ── Vectorised semantic scores ────────────────────────────────────────
       # Build a (n × dim) float32 matrix then do one matrix-vector multiply.
       # This replaces n individual _cosine_dense() Python calls.
       sem_scores = np.zeros(n, dtype=np.float32)
       if q_emb:
           q_vec = np.array(q_emb, dtype=np.float32)
           q_norm = float(np.linalg.norm(q_vec))
           if q_norm > 0:
               dim = len(q_emb)
               emb_matrix = np.zeros((n, dim), dtype=np.float32)
               for i, entry in enumerate(entries):
                   emb = entry.get("embedding") or []
                   if len(emb) == dim:
                       emb_matrix[i] = emb
               row_norms = np.linalg.norm(emb_matrix, axis=1)
               valid = row_norms > 0
               if valid.any():
                   sem_scores[valid] = (emb_matrix[valid] @ q_vec) / (row_norms[valid] * q_norm)

       # ── BM25 scores ───────────────────────────────────────────────────────
       bm25_arr = np.array([bm25_norm.get(cid, 0.0) for cid in chunk_ids], dtype=np.float32)

       # ── Hybrid fusion ─────────────────────────────────────────────────────
       hybrid_scores = settings.semantic_weight * sem_scores + settings.keyword_weight * bm25_arr

       # argpartition is O(n) vs O(n log n) for a full sort — only pay log-k cost.
       if retrieve_k >= n:
           top_idx = np.argsort(hybrid_scores)[::-1]
       else:
           part = np.argpartition(hybrid_scores, -retrieve_k)[-retrieve_k:]
           top_idx = part[np.argsort(hybrid_scores[part])[::-1]]

       results: list[RetrievedChunk] = []
       for i in top_idx:
           chunk = entries[i]["chunk"]
           score = float(hybrid_scores[i])
           results.append(
               RetrievedChunk(
                   chunk_id=chunk.chunk_id,
                   document_id=chunk.document_id,
                   filename=chunk.filename,
                   text=chunk.text,
                   page_number=chunk.page_number,
                   score=score,
               )
           )
       return results


   async def _rerank(
       self, query: str, chunks: list[RetrievedChunk], top_k: int
   ) -> list[RetrievedChunk]:
       """
       Re-rank retrieved chunks with a Mistral relevance prompt (scores 0–10 per chunk).
       """
       if not chunks:
           return []
       if len(chunks) == 1:
           return chunks[:top_k]

       # Hard cap: clamp the candidate list before building the LLM prompt so we
       # never exceed the model's context window regardless of retrieve_k.
       # With max_body=450 chars/chunk and max_rerank_chunks=25 the block is at
       # most ~11 250 chars (~2 800 tokens) — well within Mistral Small's limit.
       candidates = chunks[: settings.max_rerank_chunks]

        # for each chunk, it removes newlines, truncates to 450 characters,
        # labels with [0], [1], ...

        # [0] (file=report.pdf, page=1)
        # Machine learning is a field of artificial intelligence that...

        # [1] (file=report.pdf, page=2)
        # Deep learning methods use layered neural networks...

        # finally, joins all passages with [0], [1], ... into a single string

       max_body = 450
       lines: list[str] = []
       for i, c in enumerate(candidates):
           snippet = c.text.replace("\n", " ")[:max_body]
           lines.append(f"[{i}] (file={c.filename}, page={c.page_number})\n{snippet}")

       block = "\n\n".join(lines)

       def _call() -> str:
           client = self._client()
           resp = client.chat.complete(
               model=settings.mistral_model,
               temperature=0.0, # make output deterministic
               max_tokens=500,
               messages=[
                   SystemMessage(
                       content=(
                           "You score how relevant each passage is to answer the user's question. "
                           "Reply with ONLY a JSON array of numbers, one score 0.0–10.0 per passage "
                           "in the same order as the [0], [1], ... blocks. No other text."
                       )
                   ),
                   UserMessage(content=f"Question:\n{query}\n\nPassages:\n{block}"),
               ],
           )
           choice = resp.choices[0] if resp.choices else None
           return _assistant_text(choice.message if choice else None)

       try:
           raw = await asyncio.to_thread(_call)
       except Exception as exc:
           logger.warning("Re-rank failed, keeping hybrid order: %s", exc)
           return chunks[:top_k]

       scores: list[float] = []
       try:
           cleaned = raw.strip()
           if cleaned.startswith("```"):
               cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
               cleaned = re.sub(r"\s*```$", "", cleaned)
           start = cleaned.find("[")
           end = cleaned.rfind("]") + 1
           if start >= 0 and end > start:
               scores = [float(x) for x in json.loads(cleaned[start:end])]
       except (json.JSONDecodeError, ValueError, TypeError) as exc:
           logger.debug("Could not parse rerank JSON: %s raw=%r", exc, raw[:200])

       # The LLM scores exactly the candidates slice, not the full chunks list.
       if len(scores) != len(candidates):
           return chunks[:top_k]

       ordered = sorted(
           zip(candidates, scores),
           key=lambda x: x[1],
           reverse=True,
       )
       return [
           c.model_copy(update={"score": float(s)})
           for c, s in ordered[:top_k]
       ]

    # Mistral chat with system + user context; handles empty chunks and API failure with clear messages
   async def _generate(
       self, query: str, chunks: list[RetrievedChunk], shape: str = "factual"
   ) -> str:
       """
       Build a prompt from retrieved context and call Mistral to generate an answer.

       The ``shape`` parameter selects a prompt template that steers the model
       toward the output format most appropriate for the query type:
         list        → markdown bulleted list with per-bullet citations
         table       → markdown table with a Source column
         comparison  → side-by-side markdown table + prose summary
         definition  → concise definition paragraph
         instruction → numbered step-by-step list
         factual     → free-form prose (default)
       """
       if not chunks:
           return "I could not find relevant information in the indexed documents."

       # Truncate each chunk's text so a high top_k cannot produce an unbounded prompt.
       max_ctx_chars = 800
       context = "\n\n".join(
           f"[Source: {c.filename}, p.{c.page_number}]\n{c.text[:max_ctx_chars]}" for c in chunks
       )
       system = _PROMPT_TEMPLATES.get(shape, _PROMPT_TEMPLATES["factual"])
       user = f"Context:\n{context}\n\nQuestion: {query}"

       def _call() -> str:
           client = self._client()
           resp = client.chat.complete(
               model=settings.mistral_model,
               temperature=0.2,
               max_tokens=1024,
               messages=[
                   SystemMessage(content=system),
                   UserMessage(content=user),
               ],
           )
           choice = resp.choices[0] if resp.choices else None
           return _assistant_text(choice.message if choice else None)

       try:
           text = await asyncio.to_thread(_call)
       except Exception as exc:
           logger.exception("Generation failed: %s", exc)
           return "Sorry, the answer could not be generated right now."

       return text.strip() or "Sorry, I had no answer text from the model."


   async def _evidence_check(
       self, answer: str, chunks: list[RetrievedChunk]
   ) -> tuple[str, bool]:
       """
       Post-hoc hallucination filter.

       Extracts verifiable claims from the answer (skipping markdown headers,
       rules, and table separators), asks Mistral to classify each as
       SUPPORTED or UNSUPPORTED against the retrieved source chunks, then
       rebuilds the answer with unsupported lines removed.

       Returns
       -------
       (filtered_answer, had_unsupported)
           filtered_answer   – answer with unverifiable claims removed
           had_unsupported   – True if at least one claim was removed
       """
       claim_pairs = _extract_claims(answer)
       if not claim_pairs:
           return answer, False

       # Cap to avoid token overflow in the LLM call.
       cap = settings.evidence_check_max_claims
       claim_pairs = claim_pairs[:cap]
       line_indices = [idx for idx, _ in claim_pairs]
       claims      = [cl  for _, cl  in claim_pairs]

       # Compact context — shorter than _generate so the prompt fits comfortably.
       max_ctx_chars = 400
       context = "\n\n".join(
           f"[{c.filename}, p.{c.page_number}]\n{c.text[:max_ctx_chars]}"
           for c in chunks
       )
       numbered = "\n".join(f"[{i}] {cl}" for i, cl in enumerate(claims))

       def _call() -> str:
           client = self._client()
           resp = client.chat.complete(
               model=settings.mistral_model,
               temperature=0.0,
               max_tokens=300,
               messages=[
                   SystemMessage(
                       content=(
                           "You are a strict fact-checker. "
                           "Given source passages and numbered claims from an AI-generated answer, "
                           "classify each claim as SUPPORTED (directly verifiable from the sources) "
                           "or UNSUPPORTED (not verifiable, goes beyond, or contradicts the sources). "
                           "Reply with ONLY a JSON array of strings in the same order as the claims. "
                           'Example for 3 claims: ["SUPPORTED","UNSUPPORTED","SUPPORTED"]. '
                           "No other text."
                       )
                   ),
                   UserMessage(
                       content=f"Sources:\n{context}\n\nClaims to verify:\n{numbered}"
                   ),
               ],
           )
           choice = resp.choices[0] if resp.choices else None
           return _assistant_text(choice.message if choice else None)

       try:
           raw = await asyncio.to_thread(_call)
       except Exception as exc:
           logger.warning("Evidence check failed, returning original answer: %s", exc)
           return answer, False

       verdicts: list[str] = []
       try:
           cleaned = raw.strip()
           if cleaned.startswith("```"):
               cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
               cleaned = re.sub(r"\s*```$", "", cleaned)
           start = cleaned.find("[")
           end   = cleaned.rfind("]") + 1
           if start >= 0 and end > start:
               verdicts = [str(v).upper() for v in json.loads(cleaned[start:end])]
       except (json.JSONDecodeError, ValueError, TypeError) as exc:
           logger.debug("Could not parse evidence-check JSON: %s  raw=%r", exc, raw[:200])
           return answer, False

       if len(verdicts) != len(claims):
           logger.debug(
               "Evidence-check verdict count mismatch: got %d for %d claims",
               len(verdicts), len(claims),
           )
           return answer, False

       unsupported_indices = {
           line_indices[i]
           for i, v in enumerate(verdicts)
           if v != "SUPPORTED"
       }
       removed = len(unsupported_indices)

       if removed == 0:
           logger.info("Evidence check: all %d claims supported.", len(claims))
           return answer, False

       logger.info(
           "Evidence check: %d/%d claim(s) unsupported — removed from answer.",
           removed, len(claims),
       )

       # Rebuild answer, dropping unsupported lines.
       original_lines = answer.splitlines()
       filtered_lines = [
           line for idx, line in enumerate(original_lines)
           if idx not in unsupported_indices
       ]
       filtered = "\n".join(filtered_lines).strip()

       if not filtered:
           filtered = (
               "_No claims in the generated answer could be verified "
               "against the retrieved source documents._"
           )
       else:
           s = "s" if removed > 1 else ""
           filtered += (
               f"\n\n> **Evidence check:** {removed} claim{s} removed — "
               "not directly supported by the retrieved sources."
           )

       return filtered, True
