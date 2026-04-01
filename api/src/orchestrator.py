from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncGenerator

import asyncpg
import httpx
import openai

from .agents.image_retriever import ImageRetrieverAgent, ImageRetrieverInput
from .agents.query_intent import QueryIntentAgent, QueryIntentInput
from .agents.ranker import CrossModalRanker, RankerInput
from .agents.synthesizer import AnswerSynthesizer, SynthesizerInput
from .agents.text_retriever import TextRetrieverAgent, TextRetrieverInput
from .models.contracts import (
    ImageResult,
    ModalityType,
    QueryIntent,
    SynthesizerOutput,
    TextChunk,
)
from .observability import PipelineTrace, emit_trace


async def _empty_text() -> list[TextChunk]:
    return []


async def _empty_images() -> list[ImageResult]:
    return []


class Orchestrator:
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embedder_client: httpx.AsyncClient,
        openai_client: openai.AsyncOpenAI,
    ) -> None:
        self._intent_agent = QueryIntentAgent(openai_client)
        self._text_retriever = TextRetrieverAgent(db_pool, embedder_client)
        self._image_retriever = ImageRetrieverAgent(db_pool, embedder_client, openai_client)
        self._ranker = CrossModalRanker()
        self._synthesizer = AnswerSynthesizer(openai_client)

    async def run(self, raw_query: str) -> tuple[SynthesizerOutput, PipelineTrace]:
        trace = PipelineTrace(trace_id=str(uuid.uuid4()), query=raw_query)
        pipeline_start = time.monotonic()

        intent = await self._intent_agent.run(QueryIntentInput(raw_query=raw_query))
        trace.intent_modality = intent.modality.value
        trace.intent_confidence = intent.confidence

        text_output, image_output = await self._run_retrievers(intent)
        _populate_retrieval_trace(trace, text_output, image_output)

        ranker_output = self._ranker.run(RankerInput(
            text_chunks=text_output.chunks if text_output else [],
            image_results=image_output.images if image_output else [],
            query_intent=intent,
        ))
        trace.ranked_text_count = ranker_output.text_count
        trace.ranked_image_count = ranker_output.image_count
        trace.fusion_method = ranker_output.fusion_method

        synthesis_output = await self._synthesizer.run(SynthesizerInput(
            ranked_items=ranker_output.ranked_items,
            original_query=raw_query,
            query_intent=intent,
        ))
        trace.is_grounded = synthesis_output.grounding_passed
        trace.citation_count = len(synthesis_output.citations)
        trace.synthesis_latency_ms = synthesis_output.latency_ms
        trace.unsupported_claims = synthesis_output.unsupported_claims
        trace.total_latency_ms = (time.monotonic() - pipeline_start) * 1000

        emit_trace(trace)
        return synthesis_output, trace

    async def stream(self, raw_query: str) -> AsyncGenerator[str, None]:
        intent = await self._intent_agent.run(QueryIntentInput(raw_query=raw_query))
        text_output, image_output = await self._run_retrievers(intent)

        ranker_output = self._ranker.run(RankerInput(
            text_chunks=text_output.chunks if text_output else [],
            image_results=image_output.images if image_output else [],
            query_intent=intent,
        ))

        synthesizer_input = SynthesizerInput(
            ranked_items=ranker_output.ranked_items,
            original_query=raw_query,
            query_intent=intent,
        )
        async for token in self._synthesizer.stream_tokens(synthesizer_input):
            yield token

    async def _run_retrievers(self, intent: QueryIntent):
        needs_text = intent.modality in (ModalityType.text, ModalityType.both) or intent.fallback_to_text_only
        needs_image = intent.modality in (ModalityType.image, ModalityType.both) and not intent.fallback_to_text_only

        text_coro = (
            self._text_retriever.run(TextRetrieverInput(query_intent=intent))
            if needs_text else _empty_text()
        )
        image_coro = (
            self._image_retriever.run(ImageRetrieverInput(query_intent=intent))
            if needs_image else _empty_images()
        )
        return await asyncio.gather(text_coro, image_coro)


def _populate_retrieval_trace(trace, text_output, image_output) -> None:
    from .agents.text_retriever import TextRetrieverOutput
    from .agents.image_retriever import ImageRetrieverOutput

    if isinstance(text_output, TextRetrieverOutput):
        trace.text_count = len(text_output.chunks)
        trace.text_latency_ms = text_output.latency_ms
        trace.text_mode = text_output.retrieval_mode

    if isinstance(image_output, ImageRetrieverOutput):
        trace.image_count = len(image_output.images)
        trace.image_latency_ms = image_output.latency_ms
        trace.image_mode = image_output.search_mode
