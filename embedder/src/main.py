import logging
import os
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_clip_model: SentenceTransformer | None = None


def clip_model() -> SentenceTransformer:
    assert _clip_model is not None, "CLIP model not initialised"
    return _clip_model


openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _clip_model
    logger.info("loading CLIP model clip-ViT-L-14")
    _clip_model = SentenceTransformer("clip-ViT-L-14")
    logger.info("CLIP model ready")
    yield


app = FastAPI(title="WGSN Embedder", lifespan=lifespan)


class TextEmbedRequest(BaseModel):
    texts: list[str]


class TextEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


class ImageEmbedRequest(BaseModel):
    image_urls: list[str]


class ImageEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


class QueryClipRequest(BaseModel):
    texts: list[str]


@app.post("/embed/text", response_model=TextEmbedResponse)
async def embed_text(request: TextEmbedRequest) -> TextEmbedResponse:
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=request.texts,
    )
    embeddings = [item.embedding for item in response.data]
    logger.info("embedded %d text chunks", len(embeddings))
    return TextEmbedResponse(
        embeddings=embeddings,
        model="text-embedding-3-small",
        dimensions=len(embeddings[0]),
    )


@app.post("/embed/image", response_model=ImageEmbedResponse)
async def embed_images(request: ImageEmbedRequest) -> ImageEmbedResponse:
    if not request.image_urls:
        raise HTTPException(status_code=422, detail="image_urls must not be empty")

    embeddings = clip_model().encode(request.image_urls, convert_to_numpy=True).tolist()
    logger.info("embedded %d images", len(embeddings))
    return ImageEmbedResponse(
        embeddings=embeddings,
        model="clip-ViT-L-14",
        dimensions=len(embeddings[0]),
    )


@app.post("/embed/query-clip", response_model=ImageEmbedResponse)
async def embed_query_clip(request: QueryClipRequest) -> ImageEmbedResponse:
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    embeddings = clip_model().encode(request.texts, convert_to_numpy=True).tolist()
    logger.info("clip-encoded %d query texts", len(embeddings))
    return ImageEmbedResponse(
        embeddings=embeddings,
        model="clip-ViT-L-14",
        dimensions=len(embeddings[0]),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
