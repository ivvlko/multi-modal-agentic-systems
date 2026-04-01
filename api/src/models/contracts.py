from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    text = "text"
    image = "image"
    both = "both"


class SearchFilter(BaseModel):
    field: str
    value: str


class QueryIntent(BaseModel):
    raw_query: str
    normalised_query: str
    modality: ModalityType
    filters: list[SearchFilter] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    fallback_to_text_only: bool = False


class TextChunk(BaseModel):
    doc_id: str
    source: str
    chunk_index: int
    content: str
    vector_score: float
    bm25_score: float
    rrf_score: float


class ImageResult(BaseModel):
    doc_id: str
    source: str
    image_url: str
    caption: str | None
    clip_score: float


class RankedItem(BaseModel):
    modality: Literal["text", "image"]
    doc_id: str
    source: str
    fused_score: float
    text_chunk: TextChunk | None = None
    image_result: ImageResult | None = None


class Citation(BaseModel):
    doc_id: str
    source: str
    modality: Literal["text", "image"]
    excerpt: str | None = None


class SynthesizerOutput(BaseModel):
    answer: str
    citations: list[Citation]
    grounding_passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
