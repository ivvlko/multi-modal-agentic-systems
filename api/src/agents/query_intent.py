from __future__ import annotations

import json
import logging
import time

import openai
from pydantic import BaseModel, ValidationError

from ..models.contracts import ModalityType, QueryIntent, SearchFilter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a search query parser for a multi-modal fashion/trend content archive.\n"
    "Output a JSON object with these exact fields:\n"
    "- normalised_query: cleaned query text (string)\n"
    "- modality: one of \"text\", \"image\", \"both\"\n"
    "- filters: array of {\"field\": string, \"value\": string} metadata filter objects\n"
    "- expanded_terms: array of related terms/synonyms for recall expansion\n"
    "- confidence: float 0-1 for modality classification certainty\n"
    "\n"
    "Modality rules:\n"
    "- \"image\" when query contains visual terms: photo, look, style, colour, aesthetic, visual, show me\n"
    "- \"text\" for conceptual/factual queries: trends, analysis, what, why, how, explain\n"
    "- \"both\" when ambiguous\n"
    "Only include filters you are highly confident about; omit uncertain ones."
)


class QueryIntentInput(BaseModel):
    raw_query: str
    session_id: str | None = None


class IntentParseError(Exception):
    pass


def _parse_llm_json(raw_json: str, raw_query: str) -> QueryIntent:
    parsed = json.loads(raw_json)
    filters = [
        SearchFilter(field=f["field"], value=f["value"])
        for f in parsed.get("filters", [])
    ]
    return QueryIntent(
        raw_query=raw_query,
        normalised_query=parsed["normalised_query"],
        modality=ModalityType(parsed["modality"]),
        filters=filters,
        expanded_terms=parsed.get("expanded_terms", []),
        confidence=float(parsed.get("confidence", 1.0)),
        fallback_to_text_only=False,
    )


def _text_only_fallback(raw_query: str) -> QueryIntent:
    return QueryIntent(
        raw_query=raw_query,
        normalised_query=raw_query,
        modality=ModalityType.text,
        filters=[],
        expanded_terms=[],
        confidence=0.0,
        fallback_to_text_only=True,
    )


class QueryIntentAgent:
    def __init__(self, openai_client: openai.AsyncOpenAI) -> None:
        self._client = openai_client

    async def run(self, agent_input: QueryIntentInput) -> QueryIntent:
        start_ms = time.monotonic() * 1000
        try:
            intent = await self._call_llm(agent_input.raw_query)
        except Exception:
            intent = _text_only_fallback(agent_input.raw_query)
        latency_ms = time.monotonic() * 1000 - start_ms
        logger.info(
            {
                "event": "query_intent_resolved",
                "raw_query": agent_input.raw_query,
                "normalised_query": intent.normalised_query,
                "modality": intent.modality,
                "confidence": intent.confidence,
                "latency_ms": round(latency_ms, 2),
            }
        )
        return intent

    async def _call_llm(self, raw_query: str) -> QueryIntent:
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
        )
        raw_json = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        logger.info({"event": "llm_call_complete", "input_tokens": input_tokens})
        try:
            return _parse_llm_json(raw_json, raw_query)
        except (ValidationError, KeyError, ValueError, json.JSONDecodeError):
            pass
        retry_response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
        )
        retry_json = retry_response.choices[0].message.content
        try:
            return _parse_llm_json(retry_json, raw_query)
        except (ValidationError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise IntentParseError(f"LLM returned unparseable JSON after retry: {exc}") from exc
