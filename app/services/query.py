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

from mistralai.client import Mistral
from mistralai.client.models.assistantmessage import AssistantMessage
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.usermessage import UserMessage

from app.core.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk


logger = logging.getLogger(__name__)
settings = get_settings()


NO_SEARCH_ANSWER = "Hello! How can I help you today?"


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
       reranked = await self._rerank(request.query, chunks, request.top_k)
       answer = await self._generate(request.query, reranked)


       return QueryResponse(
           query=request.query,
           transformed_query=transformed,
           intent_triggered_search=True,
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
                           "Decide if the user wants document-grounded search or only "
                           "casual chat. Reply with exactly one token: SEARCH or CHITCHAT."
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
       """
       logger.debug("_hybrid_search: query=%r top_k=%d store_size=%d", query, top_k, len(self._vector_store))
       if not self._vector_store:
           return []


        # min of len(self._vector_score) ensures not to retrieve more chunks than available
       retrieve_k = min(len(self._vector_store), max(top_k * 4, max(15, top_k)))

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

       hybrid: list[tuple[str, float, RetrievedChunk]] = []
       for chunk_id, entry in self._vector_store.items():
           chunk = entry["chunk"]
           emb = entry.get("embedding") or []

           # compare query embedding with every vector embedding
           sem = _cosine_dense(q_emb, emb) if q_emb else 0.0
           kw = bm25_norm.get(chunk_id, 0.0)
           score = settings.semantic_weight * sem + settings.keyword_weight * kw
           hybrid.append(
               (
                   chunk_id,
                   score,
                   RetrievedChunk(
                       chunk_id=chunk.chunk_id,
                       document_id=chunk.document_id,
                       filename=chunk.filename,
                       text=chunk.text,
                       page_number=chunk.page_number,
                       score=score,
                   ),
               )
           )

       hybrid.sort(key=lambda x: x[1], reverse=True)
       return [h[2] for h in hybrid[:retrieve_k]]


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


        # for each chunk, it removes newlines, truncates to 450 characters,
        # labels with [0], [1], ...

        # [0] (file=report.pdf, page=1)
        # Machine learning is a field of artificial intelligence that...

        # [1] (file=report.pdf, page=2)
        # Deep learning methods use layered neural networks...

        # finally, joins all passages with [0], [1], ... into a single string

       max_body = 450
       lines: list[str] = []
       for i, c in enumerate(chunks):
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

        # check if the model returns one score per chunk
       if len(scores) != len(chunks):
           return chunks[:top_k]

       ordered = sorted(
           zip(chunks, scores),
           key=lambda x: x[1],
           reverse=True,
       )
       return [
           c.model_copy(update={"score": float(s)})
           for c, s in ordered[:top_k]
       ]

    # Mistral chat with system + user context; handles empty chunks and API failure with clear messages
   async def _generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
       """
       Build a prompt from retrieved context and call Mistral to generate an answer.
       """
       if not chunks:
           return "I could not find relevant information in the indexed documents."

       context = "\n\n".join(
           f"[Source: {c.filename}, p.{c.page_number}]\n{c.text}" for c in chunks
       )
       system = (
           "You are a careful assistant. Answer using only the provided context. "
           "If the context is insufficient, say so briefly. Cite sources by filename "
           "and page when you use them."
       )
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
