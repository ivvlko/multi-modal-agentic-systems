from __future__ import annotations

import logging
import time

import asyncpg
import httpx
from pydantic import BaseModel

from ..models.contracts import QueryIntent, TextChunk

logger = logging.getLogger(__name__)

_HYBRID_SQL = """
WITH vector_search AS (
    SELECT id, doc_id, source, chunk_index, content,
           1 - (embedding <=> $1::vector) AS vector_score,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS vector_rank
    FROM text_chunks
    LIMIT $2
),
bm25_search AS (
    SELECT id, doc_id, source, chunk_index, content,
           ts_rank_cd(tsv, plainto_tsquery('english', $3)) AS bm25_score,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $3)) DESC) AS bm25_rank
    FROM text_chunks
    WHERE tsv @@ plainto_tsquery('english', $3)
    LIMIT $2
),
combined AS (
    SELECT
        COALESCE(v.doc_id, b.doc_id) AS doc_id,
        COALESCE(v.source, b.source) AS source,
        COALESCE(v.chunk_index, b.chunk_index) AS chunk_index,
        COALESCE(v.content, b.content) AS content,
        COALESCE(v.vector_score, 0.0) AS vector_score,
        COALESCE(b.bm25_score, 0.0) AS bm25_score,
        (1.0 / (60 + COALESCE(v.vector_rank, 1000)) + 1.0 / (60 + COALESCE(b.bm25_rank, 1000))) AS rrf_score
    FROM vector_search v
    FULL OUTER JOIN bm25_search b ON v.id = b.id
)
SELECT * FROM combined ORDER BY rrf_score DESC LIMIT $2
"""

_DENSE_SQL = """
SELECT doc_id, source, chunk_index, content,
       1 - (embedding <=> $1::vector) AS vector_score,
       0.0 AS bm25_score,
       1 - (embedding <=> $1::vector) AS rrf_score
FROM text_chunks
ORDER BY embedding <=> $1::vector
LIMIT $2
"""


class TextRetrieverInput(BaseModel):
    query_intent: QueryIntent
    top_k: int = 20
    use_hybrid: bool = True


class TextRetrieverOutput(BaseModel):
    chunks: list[TextChunk]
    retrieval_mode: str
    latency_ms: float


class TextRetrievalError(Exception):
    pass


def _build_query_text(query_intent: QueryIntent) -> str:
    terms = [query_intent.normalised_query] + query_intent.expanded_terms
    return " ".join(terms)


def _row_to_text_chunk(row: asyncpg.Record) -> TextChunk:
    return TextChunk(
        doc_id=row["doc_id"],
        source=row["source"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        vector_score=float(row["vector_score"]),
        bm25_score=float(row["bm25_score"]),
        rrf_score=float(row["rrf_score"]),
    )


class TextRetrieverAgent:
    def __init__(self, db_pool: asyncpg.Pool, embedder_client: httpx.AsyncClient) -> None:
        self._db_pool = db_pool
        self._embedder_client = embedder_client

    async def run(self, agent_input: TextRetrieverInput) -> TextRetrieverOutput:
        start_ms = time.monotonic() * 1000
        embedding = await self._get_query_embedding(agent_input.query_intent.normalised_query)

        if agent_input.use_hybrid:
            query_text = _build_query_text(agent_input.query_intent)
            text_chunks = await self._hybrid_search(embedding, query_text, agent_input.top_k)
            retrieval_mode = "hybrid"
        else:
            text_chunks = await self._dense_search(embedding, agent_input.top_k)
            retrieval_mode = "dense"

        latency_ms = time.monotonic() * 1000 - start_ms

        if len(text_chunks) < 3:
            logger.warning(
                {
                    "event": "low_retrieval_count",
                    "count": len(text_chunks),
                    "query": agent_input.query_intent.normalised_query,
                    "filters": [f.model_dump() for f in agent_input.query_intent.filters],
                }
            )

        logger.info(
            {
                "event": "text_retrieval_complete",
                "retrieval_mode": retrieval_mode,
                "chunk_count": len(text_chunks),
                "latency_ms": round(latency_ms, 2),
            }
        )

        return TextRetrieverOutput(
            chunks=text_chunks,
            retrieval_mode=retrieval_mode,
            latency_ms=round(latency_ms, 2),
        )

    async def _get_query_embedding(self, text: str) -> list[float]:
        try:
            response = await self._embedder_client.post("/embed/text", json={"texts": [text]})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TextRetrievalError(f"Embedder service unreachable: {exc}") from exc
        return response.json()["embeddings"][0]

    async def _hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int
    ) -> list[TextChunk]:
        try:
            async with self._db_pool.acquire() as connection:
                rows = await connection.fetch(_HYBRID_SQL, str(embedding), top_k, query_text)
        except (asyncpg.PostgresError, OSError) as exc:
            raise TextRetrievalError(f"Database unreachable during hybrid search: {exc}") from exc
        return [_row_to_text_chunk(row) for row in rows]

    async def _dense_search(self, embedding: list[float], top_k: int) -> list[TextChunk]:
        try:
            async with self._db_pool.acquire() as connection:
                rows = await connection.fetch(_DENSE_SQL, str(embedding), top_k)
        except (asyncpg.PostgresError, OSError) as exc:
            raise TextRetrievalError(f"Database unreachable during dense search: {exc}") from exc
        return [_row_to_text_chunk(row) for row in rows]
