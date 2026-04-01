from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.src.agents.image_retriever import (
    ImageRetrievalError,
    ImageRetrieverAgent,
    ImageRetrieverInput,
    ImageRetrieverOutput,
)
from api.src.models.contracts import ModalityType, QueryIntent


def _make_query_intent(query: str = "floral summer dress") -> QueryIntent:
    return QueryIntent(
        raw_query=query,
        normalised_query=query,
        modality=ModalityType.image,
        confidence=0.9,
    )


def _make_db_row(
    doc_id: str = "img-1",
    caption: str | None = None,
    clip_score: float = 0.8,
) -> dict:
    return {
        "doc_id": doc_id,
        "source": "s3://bucket/images/img-1.jpg",
        "image_url": "https://cdn.example.com/img-1.jpg",
        "caption": caption,
        "clip_score": clip_score,
    }


def _make_agent(
    db_pool: MagicMock | None = None,
    embedder_client: MagicMock | None = None,
    openai_client: MagicMock | None = None,
) -> ImageRetrieverAgent:
    return ImageRetrieverAgent(
        db_pool=db_pool or MagicMock(),
        embedder_client=embedder_client or MagicMock(),
        openai_client=openai_client or MagicMock(),
    )


def _make_embedder_mock(fake_embedding: list[float]) -> MagicMock:
    embedder = MagicMock()
    embedder.post = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=lambda: {"embeddings": [fake_embedding]},
            raise_for_status=lambda: None,
        )
    )
    return embedder


def _make_db_pool_mock(rows: list, execute_mock: AsyncMock | None = None) -> MagicMock:
    mock_connection = AsyncMock()
    mock_connection.fetch = AsyncMock(return_value=rows)
    if execute_mock is not None:
        mock_connection.execute = execute_mock
    db_pool = MagicMock()
    db_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_connection),
            __aexit__=AsyncMock(),
        )
    )
    return db_pool


@pytest.fixture()
def fake_embedding() -> list[float]:
    return [0.1] * 768


@pytest.mark.asyncio
async def test_text_to_image_mode_returns_output(fake_embedding: list[float]) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([_make_db_row(caption="a floral dress")])

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(),
        top_k=5,
        enrich_with_captions=False,
    )

    result = await agent.run(agent_input)

    assert isinstance(result, ImageRetrieverOutput)
    assert result.search_mode == "text_to_image"
    assert len(result.images) == 1
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_image_to_image_mode_uses_reference_url(fake_embedding: list[float]) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([])

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(),
        reference_image_url="https://cdn.example.com/ref.jpg",
        enrich_with_captions=False,
    )

    result = await agent.run(agent_input)

    assert result.search_mode == "image_to_image"
    embedder.post.assert_called_once()
    call_kwargs = embedder.post.call_args
    assert call_kwargs[1]["json"]["image_urls"] == ["https://cdn.example.com/ref.jpg"]


@pytest.mark.asyncio
async def test_embedder_http_error_raises_image_retrieval_error() -> None:
    import httpx

    embedder = MagicMock()
    embedder.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    agent = _make_agent(embedder_client=embedder)
    agent_input = ImageRetrieverInput(query_intent=_make_query_intent())

    with pytest.raises(ImageRetrievalError, match="Embedder service unreachable"):
        await agent.run(agent_input)


@pytest.mark.asyncio
async def test_db_error_raises_image_retrieval_error(fake_embedding: list[float]) -> None:
    import asyncpg

    embedder = _make_embedder_mock(fake_embedding)

    mock_connection = AsyncMock()
    mock_connection.fetch = AsyncMock(
        side_effect=asyncpg.PostgresConnectionError("connection lost")
    )
    db_pool = MagicMock()
    db_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_connection),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=False
    )

    with pytest.raises(ImageRetrievalError, match="Database unreachable"):
        await agent.run(agent_input)


@pytest.mark.asyncio
async def test_caption_enrichment_skips_when_clip_score_below_threshold(
    fake_embedding: list[float],
) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([_make_db_row(caption=None, clip_score=0.2)])

    openai_client = MagicMock()
    openai_client.chat.completions.create = AsyncMock()

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder, openai_client=openai_client)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=True
    )

    result = await agent.run(agent_input)

    openai_client.chat.completions.create.assert_not_called()
    assert result.images[0].caption is None


@pytest.mark.asyncio
async def test_caption_enrichment_sets_caption_and_writes_cache(
    fake_embedding: list[float],
) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    execute_mock = AsyncMock()
    db_pool = _make_db_pool_mock(
        [_make_db_row(caption=None, clip_score=0.85)], execute_mock=execute_mock
    )

    generated_caption = "A vibrant floral summer dress on a white background."
    mock_choice = MagicMock()
    mock_choice.message.content = generated_caption
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    openai_client = MagicMock()
    openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder, openai_client=openai_client)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=True
    )

    result = await agent.run(agent_input)

    assert result.images[0].caption == generated_caption
    execute_mock.assert_called_once()


@pytest.mark.asyncio
async def test_caption_enrichment_failure_returns_result_without_caption(
    fake_embedding: list[float],
) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([_make_db_row(caption=None, clip_score=0.9)])

    openai_client = MagicMock()
    openai_client.chat.completions.create = AsyncMock(
        side_effect=Exception("OpenAI rate limit exceeded")
    )

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder, openai_client=openai_client)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=True
    )

    result = await agent.run(agent_input)

    assert result.images[0].caption is None
    assert len(result.images) == 1


@pytest.mark.asyncio
async def test_existing_caption_not_re_enriched(fake_embedding: list[float]) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([_make_db_row(caption="existing caption", clip_score=0.9)])

    openai_client = MagicMock()
    openai_client.chat.completions.create = AsyncMock()

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder, openai_client=openai_client)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=True
    )

    result = await agent.run(agent_input)

    openai_client.chat.completions.create.assert_not_called()
    assert result.images[0].caption == "existing caption"


@pytest.mark.asyncio
async def test_enrich_with_captions_false_skips_enrichment(
    fake_embedding: list[float],
) -> None:
    embedder = _make_embedder_mock(fake_embedding)
    db_pool = _make_db_pool_mock([_make_db_row(caption=None, clip_score=0.9)])

    openai_client = MagicMock()
    openai_client.chat.completions.create = AsyncMock()

    agent = _make_agent(db_pool=db_pool, embedder_client=embedder, openai_client=openai_client)
    agent_input = ImageRetrieverInput(
        query_intent=_make_query_intent(), enrich_with_captions=False
    )

    result = await agent.run(agent_input)

    openai_client.chat.completions.create.assert_not_called()
    assert result.images[0].caption is None
