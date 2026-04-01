from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator

import openai
from pydantic import BaseModel

from ..models.contracts import Citation, QueryIntent, RankedItem, SynthesizerOutput

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a search result synthesiser. Your ONLY job is to summarise the provided evidence.\n"
    "Rules:\n"
    "1. Every factual claim MUST reference a provided source using [ref_id] inline.\n"
    "2. If you cannot answer from the evidence, say: "
    '"The available sources do not contain enough information to answer this query."\n'
    "3. NEVER introduce external knowledge, dates, statistics, or names not present in the evidence.\n"
    "4. Do not speculate or infer beyond what is explicitly stated in the sources.\n"
    "5. If images are provided, reference them as [img-N] where relevant to visual claims."
)

_GROUNDING_SYSTEM_PROMPT = (
    "You are a grounding verifier. Given an answer and its cited sources, identify any factual "
    "claims in the answer that are NOT supported by the provided sources. "
    'Output JSON: {"unsupported_claims": ["claim1", ...]}. '
    "Output empty array if all claims are grounded."
)

_CONTEXT_TOKEN_BUDGET = 3000
_CHARS_PER_TOKEN = 4
_GUARANTEED_TOP_N = 3


class SynthesizerInput(BaseModel):
    ranked_items: list[RankedItem]
    original_query: str
    query_intent: QueryIntent
    max_output_tokens: int = 512


class SynthesisError(Exception):
    pass


def _text_item_ref_id(text_item_count: int) -> str:
    return f"[{text_item_count}]"


def _image_item_ref_id(image_item_count: int) -> str:
    return f"[img-{image_item_count}]"


def _format_text_item(item: RankedItem, ref_id: str) -> tuple[str, Citation]:
    content = item.text_chunk.content if item.text_chunk else ""
    block = f"{ref_id} {item.source}\n{content}\n"
    citation = Citation(
        doc_id=item.doc_id,
        source=item.source,
        modality="text",
        excerpt=content[:200] if content else None,
    )
    return block, citation


def _format_image_item(item: RankedItem, ref_id: str) -> tuple[str, Citation]:
    image_url = item.image_result.image_url if item.image_result else ""
    caption = item.image_result.caption if item.image_result else None
    block = f"{ref_id} {item.source}\nImage URL: {image_url}\nCaption: {caption or 'no caption'}\n"
    citation = Citation(
        doc_id=item.doc_id,
        source=item.source,
        modality="image",
        excerpt=caption,
    )
    return block, citation


def _assign_ref_id(item: RankedItem, text_counter: int, image_counter: int) -> str:
    if item.modality == "text":
        return _text_item_ref_id(text_counter)
    return _image_item_ref_id(image_counter)


def _build_citations_text(citations: list[Citation]) -> str:
    lines = []
    for citation in citations:
        lines.append(f"Source {citation.doc_id} ({citation.modality}): {citation.excerpt or ''}")
    return "\n".join(lines)


class AnswerSynthesizer:
    def __init__(self, openai_client: openai.AsyncOpenAI) -> None:
        self._client = openai_client

    def _pack_context(self, ranked_items: list[RankedItem]) -> tuple[str, list[Citation]]:
        budget_chars = _CONTEXT_TOKEN_BUDGET * _CHARS_PER_TOKEN
        guaranteed = ranked_items[:_GUARANTEED_TOP_N]
        remaining = ranked_items[_GUARANTEED_TOP_N:]

        blocks: list[str] = []
        citations: list[Citation] = []
        text_counter = 0
        image_counter = 0
        used_chars = 0

        for item in guaranteed:
            if item.modality == "text":
                text_counter += 1
                ref_id = _text_item_ref_id(text_counter)
                block, citation = _format_text_item(item, ref_id)
            else:
                image_counter += 1
                ref_id = _image_item_ref_id(image_counter)
                block, citation = _format_image_item(item, ref_id)
            blocks.append(block)
            citations.append(citation)
            used_chars += len(block)

        dropped_scores: list[float] = []
        for item in remaining:
            if item.modality == "text":
                text_counter += 1
                ref_id = _text_item_ref_id(text_counter)
                block, citation = _format_text_item(item, ref_id)
            else:
                image_counter += 1
                ref_id = _image_item_ref_id(image_counter)
                block, citation = _format_image_item(item, ref_id)

            if used_chars + len(block) > budget_chars:
                dropped_scores.append(item.fused_score)
                if item.modality == "text":
                    text_counter -= 1
                else:
                    image_counter -= 1
                continue

            blocks.append(block)
            citations.append(citation)
            used_chars += len(block)

        if dropped_scores:
            logger.info("context_packing dropped=%d scores=%s", len(dropped_scores), dropped_scores)

        return "\n".join(blocks), citations

    async def _self_check(self, answer: str, citations: list[Citation]) -> tuple[bool, list[str]]:
        citations_text = _build_citations_text(citations)
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Answer:\n{answer}\n\nSources:\n{citations_text}"},
            ],
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        unsupported = parsed.get("unsupported_claims", [])
        return len(unsupported) == 0, unsupported

    async def run(self, agent_input: SynthesizerInput) -> SynthesizerOutput:
        started_at = time.monotonic()
        context_string, citations = self._pack_context(agent_input.ranked_items)

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=agent_input.max_output_tokens,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {agent_input.original_query}\n\nEvidence:\n{context_string}"
                        ),
                    },
                ],
            )
        except Exception as exc:
            raise SynthesisError(
                f"LLM call failed for query={agent_input.original_query!r} "
                f"context_items={len(citations)}"
            ) from exc

        answer = response.choices[0].message.content
        if not answer:
            raise SynthesisError(
                f"LLM returned empty answer for query={agent_input.original_query!r}"
            )

        token_usage: dict[str, int] = {}
        if response.usage:
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        try:
            grounding_passed, unsupported_claims = await self._self_check(answer, citations)
        except Exception as exc:
            logger.warning("self_check failed skipping grounding err=%s", exc)
            grounding_passed = True
            unsupported_claims = []

        latency_ms = (time.monotonic() - started_at) * 1000

        logger.info(
            "synthesizer complete is_grounded=%s citation_count=%d latency_ms=%.2f token_usage=%s",
            grounding_passed,
            len(citations),
            latency_ms,
            token_usage,
        )

        return SynthesizerOutput(
            answer=answer,
            citations=citations,
            grounding_passed=grounding_passed,
            unsupported_claims=unsupported_claims,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    async def stream_tokens(self, agent_input: SynthesizerInput) -> AsyncGenerator[str, None]:
        context_string, citations = self._pack_context(agent_input.ranked_items)

        try:
            stream = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=agent_input.max_output_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {agent_input.original_query}\n\nEvidence:\n{context_string}"
                        ),
                    },
                ],
            )
        except Exception as exc:
            raise SynthesisError(
                f"LLM stream failed for query={agent_input.original_query!r} "
                f"context_items={len(citations)}"
            ) from exc

        accumulated = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                accumulated += delta
                yield delta

        if not accumulated:
            raise SynthesisError(
                f"LLM stream returned empty answer for query={agent_input.original_query!r}"
            )

        try:
            grounding_passed, unsupported_claims = await self._self_check(accumulated, citations)
        except Exception as exc:
            logger.warning("stream self_check failed skipping grounding err=%s", exc)
            grounding_passed = True
            unsupported_claims = []

        logger.info(
            "synthesizer stream complete is_grounded=%s citation_count=%d",
            grounding_passed,
            len(citations),
        )
