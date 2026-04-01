from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ..models.contracts import ImageResult, ModalityType, QueryIntent, RankedItem, TextChunk

logger = logging.getLogger(__name__)


class RankerInput(BaseModel):
    text_chunks: list[TextChunk]
    image_results: list[ImageResult]
    query_intent: QueryIntent
    max_context_items: int = 10


class RankerOutput(BaseModel):
    ranked_items: list[RankedItem]
    text_count: int
    image_count: int
    fusion_method: str = "NSF"
    latency_ms: float


def _compute_weights(query_intent: QueryIntent) -> tuple[float, float]:
    if query_intent.fallback_to_text_only:
        return 1.0, 0.0
    if query_intent.modality == ModalityType.text:
        return 1.0, 0.0
    if query_intent.modality == ModalityType.image:
        return 0.2, 0.8
    return 0.5, 0.5


def _build_ranked_items(
    text_chunks: list[TextChunk],
    image_results: list[ImageResult],
    text_weight: float,
    image_weight: float,
) -> list[RankedItem]:
    text_items = [
        RankedItem(
            modality="text",
            doc_id=chunk.doc_id,
            source=chunk.source,
            fused_score=text_weight * chunk.rrf_score,
            text_chunk=chunk,
        )
        for chunk in text_chunks
    ]
    image_items = [
        RankedItem(
            modality="image",
            doc_id=result.doc_id,
            source=result.source,
            fused_score=image_weight * result.clip_score,
            image_result=result,
        )
        for result in image_results
    ]
    return sorted(text_items + image_items, key=lambda item: item.fused_score, reverse=True)


def _apply_diversity_cap(items: list[RankedItem], max_items: int) -> list[RankedItem]:
    diversity_threshold = 0.6
    cap = int(max_items * diversity_threshold)

    text_seen = 0
    image_seen = 0
    demoted: list[RankedItem] = []

    for item in items:
        if item.modality == "text":
            if text_seen >= cap:
                demoted.append(item.model_copy(update={"fused_score": item.fused_score - 0.1}))
                continue
            text_seen += 1
        else:
            if image_seen >= cap:
                demoted.append(item.model_copy(update={"fused_score": item.fused_score - 0.1}))
                continue
            image_seen += 1
        demoted.append(item)

    return sorted(demoted, key=lambda item: item.fused_score, reverse=True)


class CrossModalRanker:
    def run(self, agent_input: RankerInput) -> RankerOutput:
        started_at = time.monotonic()

        text_weight, image_weight = _compute_weights(agent_input.query_intent)

        if not agent_input.text_chunks:
            logger.warning("text retrieval returned 0 results for query=%r", agent_input.query_intent.normalised_query)
        if not agent_input.image_results:
            logger.warning("image retrieval returned 0 results for query=%r", agent_input.query_intent.normalised_query)

        all_items = _build_ranked_items(
            agent_input.text_chunks,
            agent_input.image_results,
            text_weight,
            image_weight,
        )

        top_items = all_items[: agent_input.max_context_items]
        diversity_applied = _apply_diversity_cap(top_items, agent_input.max_context_items)
        final_items = diversity_applied[: agent_input.max_context_items]

        text_count = sum(1 for item in final_items if item.modality == "text")
        image_count = sum(1 for item in final_items if item.modality == "image")
        latency_ms = (time.monotonic() - started_at) * 1000

        logger.info(
            "ranker complete text_count=%d image_count=%d latency_ms=%.2f",
            text_count,
            image_count,
            latency_ms,
        )

        return RankerOutput(
            ranked_items=final_items,
            text_count=text_count,
            image_count=image_count,
            latency_ms=latency_ms,
        )
