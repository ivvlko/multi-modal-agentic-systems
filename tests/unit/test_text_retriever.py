from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from api.src.agents.text_retriever import (
    TextRetrievalError,
    TextRetrieverAgent,
    TextRetrieverInput,
    TextRetrieverOutput,
    _build_query_text,
)
from api.src.models.contracts import ModalityType, QueryIntent, SearchFilter, TextChunk


def _make_query_intent(
    normalised_query: str = "sustainable fashion trends",
    expanded_terms: list[str] | None = None,
    filters: list[SearchFilter] | None = None,
) -> QueryIntent:
    return QueryIntent(
        raw_query=normalised_query,
        normalised_query=normalised_query,
        modality=ModalityType.text,
        filters=filters or [],
        expanded_terms=expanded_terms or [],
        confidence=0.9,
        fallback_to_text_only=False,
    )


def _make_db_row(
    doc_id: str = "doc1",
    source: str = "report.pdf",
    chunk_index: int = 0,
    content: str = "Fashion trend content",
    vector_score: float = 0.85,
    bm25_score: float = 0.6,
    rrf_score: float = 0.03,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "doc_id": doc_id,
        "source": source,
        "chunk_index": chunk_index,
        "content": content,
        "vector_score": vector_score,
        "bm25_score": bm25_score,
        "rrf_score": rrf_score,
    }[key]
    return row


def _make_db_pool(rows: list[MagicMock]) -> MagicMock:
    connection = AsyncMock()
    connection.fetch = AsyncMock(return_value=rows)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=connection), __aexit__=AsyncMock(return_value=None)))
    return pool


def _make_embedder_client(embedding: list[float] | None = None) -> AsyncMock:
    embedding = embedding or [0.1] * 1536
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"embeddings": [embedding]})
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_hybrid_search_returns_chunks_ordered_by_rrf():
    rows = [
        _make_db_row(doc_id="doc1", rrf_score=0.04),
        _make_db_row(doc_id="doc2", rrf_score=0.03),
    ]
    db_pool = _make_db_pool(rows)
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=embedder_client)

    agent_input = TextRetrieverInput(
        query_intent=_make_query_intent(),
        top_k=20,
        use_hybrid=True,
    )
    result = await agent.run(agent_input)

    assert isinstance(result, TextRetrieverOutput)
    assert result.retrieval_mode == "hybrid"
    assert len(result.chunks) == 2
    assert result.chunks[0].doc_id == "doc1"
    assert result.chunks[1].doc_id == "doc2"
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_dense_search_used_when_use_hybrid_false():
    rows = [_make_db_row(doc_id="doc_dense", bm25_score=0.0)]
    db_pool = _make_db_pool(rows)
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=embedder_client)

    agent_input = TextRetrieverInput(
        query_intent=_make_query_intent(),
        top_k=10,
        use_hybrid=False,
    )
    result = await agent.run(agent_input)

    assert result.retrieval_mode == "dense"
    assert len(result.chunks) == 1
    assert result.chunks[0].doc_id == "doc_dense"
    assert result.chunks[0].bm25_score == 0.0


@pytest.mark.asyncio
async def test_raises_text_retrieval_error_when_embedder_unreachable():
    db_pool = _make_db_pool([])
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=client)

    agent_input = TextRetrieverInput(query_intent=_make_query_intent())

    with pytest.raises(TextRetrievalError, match="Embedder service unreachable"):
        await agent.run(agent_input)


@pytest.mark.asyncio
async def test_raises_text_retrieval_error_when_db_unreachable():
    import asyncpg

    connection = AsyncMock()
    connection.fetch = AsyncMock(side_effect=asyncpg.PostgresConnectionError("connection lost"))
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=connection),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=pool, embedder_client=embedder_client)

    agent_input = TextRetrieverInput(query_intent=_make_query_intent())

    with pytest.raises(TextRetrievalError, match="Database unreachable"):
        await agent.run(agent_input)


@pytest.mark.asyncio
async def test_warns_when_fewer_than_three_chunks_returned(caplog):
    import logging

    rows = [_make_db_row(doc_id="only_one")]
    db_pool = _make_db_pool(rows)
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=embedder_client)

    agent_input = TextRetrieverInput(query_intent=_make_query_intent())

    with caplog.at_level(logging.WARNING, logger="api.src.agents.text_retriever"):
        result = await agent.run(agent_input)

    assert len(result.chunks) == 1
    assert any("low_retrieval_count" in str(record.message) for record in caplog.records)


@pytest.mark.asyncio
async def test_chunk_fields_mapped_correctly_from_db_row():
    rows = [
        _make_db_row(
            doc_id="doc42",
            source="style_guide.pdf",
            chunk_index=3,
            content="Oversized silhouettes dominate spring.",
            vector_score=0.92,
            bm25_score=0.55,
            rrf_score=0.038,
        )
    ]
    db_pool = _make_db_pool(rows)
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=embedder_client)

    result = await agent.run(TextRetrieverInput(query_intent=_make_query_intent(), top_k=5))

    chunk = result.chunks[0]
    assert chunk.doc_id == "doc42"
    assert chunk.source == "style_guide.pdf"
    assert chunk.chunk_index == 3
    assert chunk.content == "Oversized silhouettes dominate spring."
    assert chunk.vector_score == pytest.approx(0.92)
    assert chunk.bm25_score == pytest.approx(0.55)
    assert chunk.rrf_score == pytest.approx(0.038)


@pytest.mark.asyncio
async def test_embedder_called_with_normalised_query():
    rows = [_make_db_row()]
    db_pool = _make_db_pool(rows)
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=db_pool, embedder_client=embedder_client)

    query_intent = _make_query_intent(normalised_query="minimalist wardrobe")
    await agent.run(TextRetrieverInput(query_intent=query_intent))

    embedder_client.post.assert_called_once_with(
        "/embed/text", json={"texts": ["minimalist wardrobe"]}
    )


def test_build_query_text_joins_normalised_and_expanded_terms():
    query_intent = _make_query_intent(
        normalised_query="capsule wardrobe",
        expanded_terms=["minimalism", "essential clothing"],
    )
    result = _build_query_text(query_intent)
    assert result == "capsule wardrobe minimalism essential clothing"


def test_build_query_text_with_no_expanded_terms():
    query_intent = _make_query_intent(normalised_query="denim trends", expanded_terms=[])
    result = _build_query_text(query_intent)
    assert result == "denim trends"


@pytest.mark.asyncio
async def test_top_k_passed_to_db_query():
    import asyncpg as _asyncpg

    connection = AsyncMock()
    connection.fetch = AsyncMock(return_value=[])
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=connection),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    embedder_client = _make_embedder_client()
    agent = TextRetrieverAgent(db_pool=pool, embedder_client=embedder_client)

    await agent.run(TextRetrieverInput(query_intent=_make_query_intent(), top_k=7))

    call_args = connection.fetch.call_args
    assert 7 in call_args.args
