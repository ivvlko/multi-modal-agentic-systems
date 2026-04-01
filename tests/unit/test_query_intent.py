from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.src.agents.query_intent import (
    IntentParseError,
    QueryIntentAgent,
    QueryIntentInput,
    _parse_llm_json,
    _text_only_fallback,
)
from api.src.models.contracts import ModalityType


def _make_llm_response(payload: dict) -> MagicMock:
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = 42
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_openai_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _valid_llm_payload(
    modality: str = "text",
    confidence: float = 0.9,
    filters: list | None = None,
    expanded_terms: list | None = None,
) -> dict:
    return {
        "normalised_query": "test query",
        "modality": modality,
        "filters": filters or [],
        "expanded_terms": expanded_terms or ["related"],
        "confidence": confidence,
    }


class TestTextOnlyFallback:
    def test_returns_text_modality(self) -> None:
        result = _text_only_fallback("some query")
        assert result.modality == ModalityType.text

    def test_returns_zero_confidence(self) -> None:
        result = _text_only_fallback("some query")
        assert result.confidence == 0.0

    def test_sets_fallback_flag(self) -> None:
        result = _text_only_fallback("some query")
        assert result.fallback_to_text_only is True

    def test_preserves_raw_query(self) -> None:
        result = _text_only_fallback("my query")
        assert result.raw_query == "my query"
        assert result.normalised_query == "my query"

    def test_empty_filters_and_terms(self) -> None:
        result = _text_only_fallback("q")
        assert result.filters == []
        assert result.expanded_terms == []


class TestParseLlmJson:
    def test_parses_text_modality(self) -> None:
        raw = json.dumps(_valid_llm_payload(modality="text"))
        result = _parse_llm_json(raw, "test query")
        assert result.modality == ModalityType.text

    def test_parses_image_modality(self) -> None:
        raw = json.dumps(_valid_llm_payload(modality="image"))
        result = _parse_llm_json(raw, "test query")
        assert result.modality == ModalityType.image

    def test_parses_both_modality(self) -> None:
        raw = json.dumps(_valid_llm_payload(modality="both"))
        result = _parse_llm_json(raw, "test query")
        assert result.modality == ModalityType.both

    def test_parses_filters(self) -> None:
        payload = _valid_llm_payload(filters=[{"field": "category", "value": "womenswear"}])
        result = _parse_llm_json(json.dumps(payload), "test query")
        assert len(result.filters) == 1
        assert result.filters[0].field == "category"
        assert result.filters[0].value == "womenswear"

    def test_parses_expanded_terms(self) -> None:
        payload = _valid_llm_payload(expanded_terms=["fashion", "apparel"])
        result = _parse_llm_json(json.dumps(payload), "test query")
        assert "fashion" in result.expanded_terms

    def test_sets_fallback_to_false(self) -> None:
        raw = json.dumps(_valid_llm_payload())
        result = _parse_llm_json(raw, "test query")
        assert result.fallback_to_text_only is False

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("not json", "test query")

    def test_raises_on_missing_required_field(self) -> None:
        payload = {"modality": "text", "confidence": 0.9}
        with pytest.raises(KeyError):
            _parse_llm_json(json.dumps(payload), "test query")

    def test_raises_on_invalid_modality(self) -> None:
        payload = _valid_llm_payload()
        payload["modality"] = "unknown"
        with pytest.raises(ValueError):
            _parse_llm_json(json.dumps(payload), "test query")


class TestQueryIntentAgent:
    @pytest.mark.asyncio
    async def test_successful_run_returns_intent(self) -> None:
        response = _make_llm_response(_valid_llm_payload(modality="text", confidence=0.95))
        client = _make_openai_client(response)
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(QueryIntentInput(raw_query="what are the latest trends"))
        assert result.modality == ModalityType.text
        assert result.confidence == 0.95
        assert result.fallback_to_text_only is False

    @pytest.mark.asyncio
    async def test_image_query_returns_image_modality(self) -> None:
        response = _make_llm_response(_valid_llm_payload(modality="image", confidence=0.88))
        client = _make_openai_client(response)
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(QueryIntentInput(raw_query="show me a photo of street style"))
        assert result.modality == ModalityType.image

    @pytest.mark.asyncio
    async def test_llm_failure_triggers_fallback(self) -> None:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network error"))
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(QueryIntentInput(raw_query="summer aesthetics"))
        assert result.fallback_to_text_only is True
        assert result.confidence == 0.0
        assert result.modality == ModalityType.text

    @pytest.mark.asyncio
    async def test_invalid_json_on_first_attempt_retries(self) -> None:
        bad_message = MagicMock()
        bad_message.content = "not json"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_usage = MagicMock()
        bad_usage.prompt_tokens = 10
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]
        bad_response.usage = bad_usage

        good_response = _make_llm_response(_valid_llm_payload(modality="both", confidence=0.6))
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[bad_response, good_response])
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(QueryIntentInput(raw_query="autumn vibes"))
        assert result.modality == ModalityType.both
        assert client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_json_on_both_attempts_raises_intent_parse_error(self) -> None:
        bad_message = MagicMock()
        bad_message.content = "not json"
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_usage = MagicMock()
        bad_usage.prompt_tokens = 10
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]
        bad_response.usage = bad_usage

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=bad_response)
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(QueryIntentInput(raw_query="winter collection"))
        assert result.fallback_to_text_only is True

    @pytest.mark.asyncio
    async def test_run_logs_correct_fields(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        response = _make_llm_response(_valid_llm_payload(modality="text", confidence=0.85))
        client = _make_openai_client(response)
        agent = QueryIntentAgent(openai_client=client)
        with caplog.at_level(logging.INFO):
            await agent.run(QueryIntentInput(raw_query="trend report 2024"))
        log_messages = [r.getMessage() for r in caplog.records]
        assert any("query_intent_resolved" in str(m) for m in log_messages)

    @pytest.mark.asyncio
    async def test_session_id_is_accepted(self) -> None:
        response = _make_llm_response(_valid_llm_payload())
        client = _make_openai_client(response)
        agent = QueryIntentAgent(openai_client=client)
        result = await agent.run(
            QueryIntentInput(raw_query="colour palette spring", session_id="sess-123")
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_model_and_params(self) -> None:
        response = _make_llm_response(_valid_llm_payload())
        client = _make_openai_client(response)
        agent = QueryIntentAgent(openai_client=client)
        await agent.run(QueryIntentInput(raw_query="minimalist aesthetic"))
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 300
        assert call_kwargs["response_format"] == {"type": "json_object"}
