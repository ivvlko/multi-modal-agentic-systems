from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import asyncpg
import httpx
import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel

from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

search_requests_total = Counter(
    "wgsn_search_requests_total",
    "Total search requests",
    ["endpoint", "status"],
)
search_latency_seconds = Histogram(
    "wgsn_search_latency_seconds",
    "End-to-end search latency",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
grounded_total = Counter("wgsn_synthesis_grounded_total", "Grounded synthesis responses")
ungrounded_total = Counter("wgsn_synthesis_ungrounded_total", "Ungrounded synthesis responses")
retrieved_items = Histogram(
    "wgsn_retrieved_items_count",
    "Retrieved item count per request",
    ["modality"],
    buckets=[0, 1, 5, 10, 20, 50],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await asyncpg.create_pool(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DB"],
        min_size=2,
        max_size=10,
    )
    app.state.embedder_client = httpx.AsyncClient(
        base_url=os.environ.get("EMBEDDER_URL", "http://embedder:8001"),
        timeout=30.0,
    )
    app.state.openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    yield
    await app.state.db_pool.close()
    await app.state.embedder_client.aclose()


app = FastAPI(title="WGSN Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_orchestrator(app_state) -> Orchestrator:
    return Orchestrator(
        db_pool=app_state.db_pool,
        embedder_client=app_state.embedder_client,
        openai_client=app_state.openai_client,
    )


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
async def search(request: SearchRequest):
    start = time.monotonic()
    status = "success"
    try:
        orchestrator = _make_orchestrator(app.state)
        output, trace = await orchestrator.run(request.query)
        retrieved_items.labels(modality="text").observe(trace.text_count)
        retrieved_items.labels(modality="image").observe(trace.image_count)
        if output.grounding_passed:
            grounded_total.inc()
        else:
            ungrounded_total.inc()
        return {"output": output.model_dump(), "trace_id": trace.trace_id}
    except Exception:
        status = "error"
        raise
    finally:
        search_latency_seconds.labels(endpoint="rest").observe(time.monotonic() - start)
        search_requests_total.labels(endpoint="rest", status=status).inc()


@app.websocket("/ws/search")
async def ws_search(websocket: WebSocket):
    await websocket.accept()
    start = time.monotonic()
    status = "success"
    try:
        data = await websocket.receive_json()
        raw_query = data.get("query", "")
        orchestrator = _make_orchestrator(app.state)
        async for token in orchestrator.stream(raw_query):
            await websocket.send_text(json.dumps({"type": "token", "content": token}))
        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as exc:
        status = "error"
        logger.exception("websocket_error error=%s", exc)
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        search_latency_seconds.labels(endpoint="ws").observe(time.monotonic() - start)
        search_requests_total.labels(endpoint="ws", status=status).inc()
        await websocket.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
