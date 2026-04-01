from __future__ import annotations

import logging
import time

import asyncpg
import httpx
import openai
from pydantic import BaseModel

from ..models.contracts import ImageResult, QueryIntent

logger = logging.getLogger(__name__)

_VECTOR_SEARCH_SQL = """
SELECT doc_id, source, image_url, caption,
       1 - (embedding <=> $1::vector) AS clip_score
FROM image_assets
ORDER BY embedding <=> $1::vector
LIMIT $2
"""

_UPDATE_CAPTION_SQL = """
UPDATE image_assets SET caption = $1 WHERE doc_id = $2
"""

_CAPTION_PROMPT = (
    "Describe this image in 1-2 sentences for a fashion/trend search system."
)

_MIN_CLIP_SCORE_FOR_ENRICHMENT = 0.25


class ImageRetrieverInput(BaseModel):
    query_intent: QueryIntent
    top_k: int = 10
    reference_image_url: str | None = None
    enrich_with_captions: bool = True


class ImageRetrieverOutput(BaseModel):
    images: list[ImageResult]
    search_mode: str
    latency_ms: float


class ImageRetrievalError(Exception):
    pass


def _row_to_image_result(row: asyncpg.Record) -> ImageResult:
    return ImageResult(
        doc_id=row["doc_id"],
        source=row["source"],
        image_url=row["image_url"],
        caption=row["caption"],
        clip_score=float(row["clip_score"]),
    )


class ImageRetrieverAgent:
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embedder_client: httpx.AsyncClient,
        openai_client: openai.AsyncOpenAI,
    ) -> None:
        self._db_pool = db_pool
        self._embedder_client = embedder_client
        self._openai_client = openai_client

    async def run(self, agent_input: ImageRetrieverInput) -> ImageRetrieverOutput:
        start_ms = time.monotonic() * 1000

        if agent_input.reference_image_url is not None:
            embedding = await self._get_image_clip_embedding(
                agent_input.reference_image_url
            )
            search_mode = "image_to_image"
        else:
            embedding = await self._get_text_clip_embedding(
                agent_input.query_intent.normalised_query
            )
            search_mode = "text_to_image"

        images = await self._vector_search(embedding, agent_input.top_k)

        if agent_input.enrich_with_captions:
            images = await self._enrich_results_with_captions(images)

        latency_ms = time.monotonic() * 1000 - start_ms

        logger.info(
            {
                "event": "image_retrieval_complete",
                "search_mode": search_mode,
                "image_count": len(images),
                "latency_ms": round(latency_ms, 2),
            }
        )

        return ImageRetrieverOutput(
            images=images,
            search_mode=search_mode,
            latency_ms=round(latency_ms, 2),
        )

    async def _get_text_clip_embedding(self, query: str) -> list[float]:
        try:
            response = await self._embedder_client.post(
                "/embed/query-clip", json={"texts": [query]}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageRetrievalError(
                f"Embedder service unreachable for text-to-image encoding: {exc}"
            ) from exc
        return response.json()["embeddings"][0]

    async def _get_image_clip_embedding(self, image_url: str) -> list[float]:
        try:
            response = await self._embedder_client.post(
                "/embed/image", json={"image_urls": [image_url]}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageRetrievalError(
                f"Embedder service unreachable for image encoding: {exc}"
            ) from exc
        return response.json()["embeddings"][0]

    async def _vector_search(
        self, embedding: list[float], top_k: int
    ) -> list[ImageResult]:
        try:
            async with self._db_pool.acquire() as connection:
                rows = await connection.fetch(_VECTOR_SEARCH_SQL, str(embedding), top_k)
                return [_row_to_image_result(row) for row in rows]
        except (asyncpg.PostgresError, OSError) as exc:
            raise ImageRetrievalError(
                f"Database unreachable during image vector search: {exc}"
            ) from exc

    async def _enrich_results_with_captions(
        self, images: list[ImageResult]
    ) -> list[ImageResult]:
        enriched = []
        for image in images:
            if image.caption is None and image.clip_score >= _MIN_CLIP_SCORE_FOR_ENRICHMENT:
                generated_caption = await self._enrich_caption(image.image_url)
                if generated_caption is not None:
                    await self._update_caption_cache(image.doc_id, generated_caption)
                    image = image.model_copy(update={"caption": generated_caption})
            enriched.append(image)
        return enriched

    async def _enrich_caption(self, image_url: str) -> str | None:
        try:
            completion = await self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": _CAPTION_PROMPT},
                        ],
                    }
                ],
                max_tokens=100,
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            logger.warning(
                {
                    "event": "caption_enrichment_failed",
                    "image_url": image_url,
                    "error": str(exc),
                }
            )
            return None

    async def _update_caption_cache(self, doc_id: str, caption: str) -> None:
        try:
            async with self._db_pool.acquire() as connection:
                await connection.execute(_UPDATE_CAPTION_SQL, caption, doc_id)
        except (asyncpg.PostgresError, OSError) as exc:
            logger.warning(
                {
                    "event": "caption_cache_write_failed",
                    "doc_id": doc_id,
                    "error": str(exc),
                }
            )
