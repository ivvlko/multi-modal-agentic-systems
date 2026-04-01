from __future__ import annotations

import pytest

from api.src.agents.ranker import CrossModalRanker, RankerInput
from api.src.models.contracts import ImageResult, ModalityType, QueryIntent, TextChunk


def _make_query_intent(modality: ModalityType, fallback: bool = False) -> QueryIntent:
    return QueryIntent(
        raw_query="test query",
        normalised_query="test query",
        modality=modality,
        fallback_to_text_only=fallback,
    )


def _make_text_chunk(doc_id: str, rrf_score: float) -> TextChunk:
    return TextChunk(
        doc_id=doc_id,
        source=f"source_{doc_id}",
        chunk_index=0,
        content="sample content",
        vector_score=rrf_score,
        bm25_score=rrf_score,
        rrf_score=rrf_score,
    )


def _make_image_result(doc_id: str, clip_score: float) -> ImageResult:
    return ImageResult(
        doc_id=doc_id,
        source=f"source_{doc_id}",
        image_url=f"https://example.com/{doc_id}.jpg",
        caption=None,
        clip_score=clip_score,
    )


@pytest.fixture()
def ranker() -> CrossModalRanker:
    return CrossModalRanker()


class TestComputeWeights:
    def test_text_only_modality_gives_full_text_weight(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.9)],
            image_results=[_make_image_result("i1", 0.9)],
            query_intent=_make_query_intent(ModalityType.text),
        )
        output = ranker.run(agent_input)
        text_item = next(item for item in output.ranked_items if item.modality == "text")
        image_item = next(item for item in output.ranked_items if item.modality == "image")
        assert text_item.fused_score == pytest.approx(0.9)
        assert image_item.fused_score == pytest.approx(0.0)

    def test_image_only_modality_applies_0_8_weight(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.5)],
            image_results=[_make_image_result("i1", 0.5)],
            query_intent=_make_query_intent(ModalityType.image),
        )
        output = ranker.run(agent_input)
        text_item = next(item for item in output.ranked_items if item.modality == "text")
        image_item = next(item for item in output.ranked_items if item.modality == "image")
        assert text_item.fused_score == pytest.approx(0.1)
        assert image_item.fused_score == pytest.approx(0.4)

    def test_both_modality_applies_equal_weights(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.8)],
            image_results=[_make_image_result("i1", 0.6)],
            query_intent=_make_query_intent(ModalityType.both),
        )
        output = ranker.run(agent_input)
        text_item = next(item for item in output.ranked_items if item.modality == "text")
        image_item = next(item for item in output.ranked_items if item.modality == "image")
        assert text_item.fused_score == pytest.approx(0.4)
        assert image_item.fused_score == pytest.approx(0.3)

    def test_fallback_to_text_only_overrides_image_modality(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.7)],
            image_results=[_make_image_result("i1", 0.9)],
            query_intent=_make_query_intent(ModalityType.image, fallback=True),
        )
        output = ranker.run(agent_input)
        image_item = next(item for item in output.ranked_items if item.modality == "image")
        assert image_item.fused_score == pytest.approx(0.0)


class TestRanking:
    def test_items_are_sorted_by_fused_score_descending(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[
                _make_text_chunk("t1", 0.9),
                _make_text_chunk("t2", 0.3),
                _make_text_chunk("t3", 0.6),
            ],
            image_results=[],
            query_intent=_make_query_intent(ModalityType.text),
        )
        output = ranker.run(agent_input)
        scores = [item.fused_score for item in output.ranked_items]
        assert scores == sorted(scores, reverse=True)

    def test_max_context_items_is_respected(self, ranker: CrossModalRanker) -> None:
        text_chunks = [_make_text_chunk(f"t{i}", float(i) / 10) for i in range(20)]
        agent_input = RankerInput(
            text_chunks=text_chunks,
            image_results=[],
            query_intent=_make_query_intent(ModalityType.text),
            max_context_items=5,
        )
        output = ranker.run(agent_input)
        assert len(output.ranked_items) <= 5

    def test_counts_reflect_final_ranked_items(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.8), _make_text_chunk("t2", 0.7)],
            image_results=[_make_image_result("i1", 0.9)],
            query_intent=_make_query_intent(ModalityType.both),
            max_context_items=3,
        )
        output = ranker.run(agent_input)
        assert output.text_count + output.image_count == len(output.ranked_items)

    def test_ranked_items_carry_original_item_references(self, ranker: CrossModalRanker) -> None:
        chunk = _make_text_chunk("t1", 0.8)
        image = _make_image_result("i1", 0.6)
        agent_input = RankerInput(
            text_chunks=[chunk],
            image_results=[image],
            query_intent=_make_query_intent(ModalityType.both),
        )
        output = ranker.run(agent_input)
        text_item = next(item for item in output.ranked_items if item.modality == "text")
        image_item = next(item for item in output.ranked_items if item.modality == "image")
        assert text_item.text_chunk == chunk
        assert image_item.image_result == image


class TestDiversityCap:
    def test_demotes_excess_same_modality_items(self, ranker: CrossModalRanker) -> None:
        text_chunks = [_make_text_chunk(f"t{i}", 0.9 - i * 0.05) for i in range(8)]
        image_results = [_make_image_result("i1", 0.5)]
        agent_input = RankerInput(
            text_chunks=text_chunks,
            image_results=image_results,
            query_intent=_make_query_intent(ModalityType.both),
            max_context_items=10,
        )
        output = ranker.run(agent_input)
        assert output.image_count >= 1

    def test_no_demotion_when_under_diversity_threshold(self, ranker: CrossModalRanker) -> None:
        text_chunks = [_make_text_chunk(f"t{i}", 0.8) for i in range(3)]
        image_results = [_make_image_result(f"i{i}", 0.7) for i in range(3)]
        agent_input = RankerInput(
            text_chunks=text_chunks,
            image_results=image_results,
            query_intent=_make_query_intent(ModalityType.both),
            max_context_items=10,
        )
        output = ranker.run(agent_input)
        text_scores = [item.fused_score for item in output.ranked_items if item.modality == "text"]
        assert all(score >= 0.4 for score in text_scores)


class TestFallbackBehaviour:
    def test_empty_image_results_proceeds_with_text_only(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.8)],
            image_results=[],
            query_intent=_make_query_intent(ModalityType.both),
        )
        output = ranker.run(agent_input)
        assert output.text_count == 1
        assert output.image_count == 0

    def test_empty_text_chunks_proceeds_with_image_only(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[],
            image_results=[_make_image_result("i1", 0.8)],
            query_intent=_make_query_intent(ModalityType.both),
        )
        output = ranker.run(agent_input)
        assert output.image_count == 1
        assert output.text_count == 0

    def test_both_empty_returns_empty_output(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[],
            image_results=[],
            query_intent=_make_query_intent(ModalityType.both),
        )
        output = ranker.run(agent_input)
        assert output.ranked_items == []
        assert output.text_count == 0
        assert output.image_count == 0


class TestOutputContract:
    def test_fusion_method_is_nsf(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.5)],
            image_results=[],
            query_intent=_make_query_intent(ModalityType.text),
        )
        output = ranker.run(agent_input)
        assert output.fusion_method == "NSF"

    def test_latency_ms_is_positive(self, ranker: CrossModalRanker) -> None:
        agent_input = RankerInput(
            text_chunks=[_make_text_chunk("t1", 0.5)],
            image_results=[],
            query_intent=_make_query_intent(ModalityType.text),
        )
        output = ranker.run(agent_input)
        assert output.latency_ms > 0
