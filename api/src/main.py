from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import httpx
import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)


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
    orchestrator = _make_orchestrator(app.state)
    output, trace = await orchestrator.run(request.query)
    return {"output": output.model_dump(), "trace_id": trace.trace_id}


@app.websocket("/ws/search")
async def ws_search(websocket: WebSocket):
    await websocket.accept()
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
        logger.exception("websocket_error error=%s", exc)
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        await websocket.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return {"status": "metrics_not_yet_wired"}
