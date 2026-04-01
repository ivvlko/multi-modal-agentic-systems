from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.agents.synthesizer import (
    AnswerSynthesizer,
    SynthesisError,
    SynthesizerInput,
    _CHARS_PER_TOKEN,
    _CONTEXT_TOKEN_BUDGET,
    _GUARANTEED_TOP_N,
    _build_citations_text,
)
from api.src.models.contracts import (
    Citation,
    ImageResult,
    ModalityType,
    QueryIntent,
    RankedItem,
    TextChunk,
)


def _make_query_intent(modality: ModalityType = ModalityType.text) -> QueryIntent:
    return QueryIntent(
        raw_query="test query",
        normalised_query="test query",
        modality=modality,
    )


def _make_text_chunk(doc_id: str = "doc-1", content: str = "some content") -> TextChunk:
    return TextChunk(
        doc_id=doc_id,
        source=f"source-{doc_id}",
        chunk_index=0,
        content=content,
        vector_score=0.9,
        bm25_score=0.8,
        rrf_score=0.85,
    )


def _make_image_result(doc_id: str = "img-doc-1", caption: str | None = "a caption") -> ImageResult:
    return ImageResult(
        doc_id=doc_id,
        source=f"source-{doc_id}",
        image_url=f"https://example.com/{doc_id}.jpg",
        caption=caption,
        clip_score=0.7,
    )


def _make_text_ranked_item(doc_id: str = "doc-1", score: float = 0.9) -> RankedItem:
    return RankedItem(
        modality="text",
        doc_id=doc_id,
        source=f"source-{doc_id}",
        fused_score=score,
        text_chunk=_make_text_chunk(doc_id),
    )


def _make_image_ranked_item(doc_id: str = "img-1", score: float = 0.7) -> RankedItem:
    return RankedItem(
        modality="image",
        doc_id=doc_id,
        source=f"source-{doc_id}",
        fused_score=score,
        image_result=_make_image_result(doc_id),
    )


def _make_synthesizer_input(
    ranked_items: list[RankedItem] | None = None,
    query: str = "what is trending",
) -> SynthesizerInput:
    return SynthesizerInput(
        ranked_items=ranked_items or [_make_text_ranked_item()],
        original_query=query,
        query_intent=_make_query_intent(),
    )


def _make_llm_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 100) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_grounding_response(unsupported_claims: list[str]) -> MagicMock:
    return _make_llm_response(json.dumps({"unsupported_claims": unsupported_claims}))


def _make_openai_client(
    synthesis_response: MagicMock,
    grounding_response: MagicMock | None = None,
) -> MagicMock:
    if grounding_response is None:
        grounding_response = _make_grounding_response([])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[synthesis_response, grounding_response]
    )
    return client


class TestPackContext:
    def test_single_text_item_produces_ref_id_1(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_text_ranked_item("doc-1")]
        context, citations = synthesizer._pack_context(items)
        assert "[1]" in context
        assert len(citations) == 1

    def test_image_item_produces_img_ref_id(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_image_ranked_item("img-1")]
        context, citations = synthesizer._pack_context(items)
        assert "[img-1]" in context
        assert len(citations) == 1

    def test_mixed_items_produce_separate_counters(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [
            _make_text_ranked_item("doc-1"),
            _make_image_ranked_item("img-1"),
            _make_text_ranked_item("doc-2"),
        ]
        context, citations = synthesizer._pack_context(items)
        assert "[1]" in context
        assert "[img-1]" in context
        assert "[2]" in context
        assert len(citations) == 3

    def test_always_includes_top_three_items(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_text_ranked_item(f"doc-{i}", score=float(10 - i)) for i in range(6)]
        _, citations = synthesizer._pack_context(items)
        included_doc_ids = {c.doc_id for c in citations}
        for i in range(_GUARANTEED_TOP_N):
            assert f"doc-{i}" in included_doc_ids

    def test_image_caption_none_renders_no_caption(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        item = RankedItem(
            modality="image",
            doc_id="img-no-cap",
            source="src",
            fused_score=0.5,
            image_result=_make_image_result("img-no-cap", caption=None),
        )
        context, _ = synthesizer._pack_context([item])
        assert "no caption" in context

    def test_context_contains_source_name(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_text_ranked_item("doc-xyz")]
        context, _ = synthesizer._pack_context(items)
        assert "source-doc-xyz" in context

    def test_drops_items_exceeding_budget_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        large_content = "x" * (_CONTEXT_TOKEN_BUDGET * _CHARS_PER_TOKEN)
        large_item = RankedItem(
            modality="text",
            doc_id="large-doc",
            source="large-source",
            fused_score=0.5,
            text_chunk=TextChunk(
                doc_id="large-doc",
                source="large-source",
                chunk_index=0,
                content=large_content,
                vector_score=0.5,
                bm25_score=0.5,
                rrf_score=0.5,
            ),
        )
        guaranteed = [_make_text_ranked_item(f"doc-{i}") for i in range(_GUARANTEED_TOP_N)]
        items = guaranteed + [large_item]
        with caplog.at_level(logging.INFO):
            _, citations = synthesizer._pack_context(items)
        included_ids = {c.doc_id for c in citations}
        assert "large-doc" not in included_ids
        assert any("dropped" in r.getMessage() for r in caplog.records)

    def test_text_citation_has_excerpt(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_text_ranked_item("doc-1")]
        _, citations = synthesizer._pack_context(items)
        assert citations[0].excerpt is not None
        assert citations[0].modality == "text"

    def test_image_citation_excerpt_is_caption(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        items = [_make_image_ranked_item("img-1")]
        _, citations = synthesizer._pack_context(items)
        assert citations[0].excerpt == "a caption"
        assert citations[0].modality == "image"

    def test_empty_items_returns_empty_context_and_citations(self) -> None:
        synthesizer = AnswerSynthesizer(openai_client=MagicMock())
        context, citations = synthesizer._pack_context([])
        assert context == ""
        assert citations == []


class TestSelfCheck:
    @pytest.mark.asyncio
    async def test_all_grounded_returns_true_empty_list(self) -> None:
        grounding_response = _make_grounding_response([])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=grounding_response)
        synthesizer = AnswerSynthesizer(openai_client=client)
        citations = [Citation(doc_id="doc-1", source="src", modality="text", excerpt="some fact")]
        grounded, claims = await synthesizer._self_check("some fact [1]", citations)
        assert grounded is True
        assert claims == []

    @pytest.mark.asyncio
    async def test_unsupported_claims_returns_false_with_claims(self) -> None:
        grounding_response = _make_grounding_response(["claim A is unsupported"])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=grounding_response)
        synthesizer = AnswerSynthesizer(openai_client=client)
        citations = [Citation(doc_id="doc-1", source="src", modality="text", excerpt="other fact")]
        grounded, claims = await synthesizer._self_check("claim A is unsupported", citations)
        assert grounded is False
        assert "claim A is unsupported" in claims

    @pytest.mark.asyncio
    async def test_self_check_uses_correct_model_and_params(self) -> None:
        grounding_response = _make_grounding_response([])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=grounding_response)
        synthesizer = AnswerSynthesizer(openai_client=client)
        citations = [Citation(doc_id="doc-1", source="src", modality="text", excerpt="fact")]
        await synthesizer._self_check("answer text", citations)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestAnswerSynthesizerRun:
    @pytest.mark.asyncio
    async def test_successful_run_returns_output(self) -> None:
        synthesis_resp = _make_llm_response("Trends show bold colours [1].")
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        result = await synthesizer.run(_make_synthesizer_input())
        assert result.answer == "Trends show bold colours [1]."
        assert result.grounding_passed is True
        assert result.unsupported_claims == []

    @pytest.mark.asyncio
    async def test_token_usage_populated_from_response(self) -> None:
        synthesis_resp = _make_llm_response("answer", prompt_tokens=60, completion_tokens=80)
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        result = await synthesizer.run(_make_synthesizer_input())
        assert result.token_usage["prompt_tokens"] == 60
        assert result.token_usage["completion_tokens"] == 80
        assert result.token_usage["total_tokens"] == 140

    @pytest.mark.asyncio
    async def test_latency_ms_is_positive(self) -> None:
        synthesis_resp = _make_llm_response("some answer")
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        result = await synthesizer.run(_make_synthesizer_input())
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_llm_failure_raises_synthesis_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network error"))
        synthesizer = AnswerSynthesizer(openai_client=client)
        with pytest.raises(SynthesisError):
            await synthesizer.run(_make_synthesizer_input())

    @pytest.mark.asyncio
    async def test_empty_answer_raises_synthesis_error(self) -> None:
        synthesis_resp = _make_llm_response("")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=synthesis_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        with pytest.raises(SynthesisError):
            await synthesizer.run(_make_synthesizer_input())

    @pytest.mark.asyncio
    async def test_ungrounded_answer_returns_false_grounding_with_claims(self) -> None:
        synthesis_resp = _make_llm_response("fabricated claim about something")
        grounding_resp = _make_grounding_response(["fabricated claim about something"])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        result = await synthesizer.run(_make_synthesizer_input())
        assert result.grounding_passed is False
        assert len(result.unsupported_claims) == 1

    @pytest.mark.asyncio
    async def test_self_check_failure_does_not_raise(self) -> None:
        synthesis_resp = _make_llm_response("valid answer [1]")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[synthesis_resp, Exception("grounding service down")]
        )
        synthesizer = AnswerSynthesizer(openai_client=client)
        result = await synthesizer.run(_make_synthesizer_input())
        assert result.answer == "valid answer [1]"
        assert result.grounding_passed is True

    @pytest.mark.asyncio
    async def test_citations_count_matches_ranked_items(self) -> None:
        synthesis_resp = _make_llm_response("answer with refs [1] [img-1]")
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        items = [_make_text_ranked_item("doc-1"), _make_image_ranked_item("img-1")]
        result = await synthesizer.run(_make_synthesizer_input(ranked_items=items))
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_run_uses_gpt4o_mini(self) -> None:
        synthesis_resp = _make_llm_response("answer")
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        await synthesizer.run(_make_synthesizer_input())
        first_call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        assert first_call_kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_run_logs_synthesis_completion(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        synthesis_resp = _make_llm_response("logged answer [1]")
        grounding_resp = _make_grounding_response([])
        client = _make_openai_client(synthesis_resp, grounding_resp)
        synthesizer = AnswerSynthesizer(openai_client=client)
        with caplog.at_level(logging.INFO):
            await synthesizer.run(_make_synthesizer_input())
        log_messages = [r.getMessage() for r in caplog.records]
        assert any("synthesizer complete" in m for m in log_messages)


class TestStreamTokens:
    def _make_stream_chunk(self, content: str | None) -> MagicMock:
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        return chunk

    async def _make_async_iter(self, items: list) -> AsyncMock:
        async def _gen():
            for item in items:
                yield item
        return _gen()

    @pytest.mark.asyncio
    async def test_yields_tokens_from_stream(self) -> None:
        chunks = [
            self._make_stream_chunk("Hello"),
            self._make_stream_chunk(" world"),
            self._make_stream_chunk(None),
        ]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        stream_response = async_chunks()
        grounding_resp = _make_grounding_response([])

        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[stream_response, grounding_resp]
        )
        synthesizer = AnswerSynthesizer(openai_client=client)

        collected: list[str] = []
        async for token in synthesizer.stream_tokens(_make_synthesizer_input()):
            collected.append(token)

        assert collected == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_llm_failure_raises_synthesis_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("stream error"))
        synthesizer = AnswerSynthesizer(openai_client=client)
        with pytest.raises(SynthesisError):
            async for _ in synthesizer.stream_tokens(_make_synthesizer_input()):
                pass

    @pytest.mark.asyncio
    async def test_stream_empty_response_raises_synthesis_error(self) -> None:
        chunks = [self._make_stream_chunk(None)]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        stream_response = async_chunks()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream_response)
        synthesizer = AnswerSynthesizer(openai_client=client)
        with pytest.raises(SynthesisError):
            async for _ in synthesizer.stream_tokens(_make_synthesizer_input()):
                pass

    @pytest.mark.asyncio
    async def test_stream_performs_self_check_after_generation(self) -> None:
        chunks = [self._make_stream_chunk("streamed answer [1]")]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        stream_response = async_chunks()
        grounding_resp = _make_grounding_response(["unsupported streamed claim"])

        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[stream_response, grounding_resp]
        )
        synthesizer = AnswerSynthesizer(openai_client=client)

        async for _ in synthesizer.stream_tokens(_make_synthesizer_input()):
            pass

        assert client.chat.completions.create.call_count == 2


class TestBuildCitationsText:
    def test_formats_text_citation(self) -> None:
        citation = Citation(doc_id="doc-1", source="src", modality="text", excerpt="fact here")
        result = _build_citations_text([citation])
        assert "doc-1" in result
        assert "fact here" in result
        assert "text" in result

    def test_formats_image_citation_with_none_excerpt(self) -> None:
        citation = Citation(doc_id="img-1", source="src", modality="image", excerpt=None)
        result = _build_citations_text([citation])
        assert "img-1" in result
        assert "image" in result

    def test_multiple_citations_all_present(self) -> None:
        citations = [
            Citation(doc_id="doc-1", source="s1", modality="text", excerpt="e1"),
            Citation(doc_id="doc-2", source="s2", modality="image", excerpt="e2"),
        ]
        result = _build_citations_text(citations)
        assert "doc-1" in result
        assert "doc-2" in result
