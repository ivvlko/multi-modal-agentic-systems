import logging
import os

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WGSN Embedder")

_clip_model: SentenceTransformer | None = None


def clip_model() -> SentenceTransformer:
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer("clip-ViT-L-14")
    return _clip_model


openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


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

    model = clip_model()
    embeddings = model.encode(request.image_urls, convert_to_numpy=True).tolist()
    logger.info("embedded %d images", len(embeddings))
    return ImageEmbedResponse(
        embeddings=embeddings,
        model="clip-ViT-L-14",
        dimensions=len(embeddings[0]),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
