from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    trace_id: str
    query: str
    intent_modality: str = ""
    intent_confidence: float = 0.0
    text_count: int = 0
    text_latency_ms: float = 0.0
    text_mode: str = ""
    image_count: int = 0
    image_latency_ms: float = 0.0
    image_mode: str = ""
    ranked_text_count: int = 0
    ranked_image_count: int = 0
    fusion_method: str = "NSF"
    is_grounded: bool = True
    citation_count: int = 0
    synthesis_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)


def emit_trace(trace: PipelineTrace) -> None:
    logger.info(
        json.dumps({
            "trace_id": trace.trace_id,
            "query": trace.query,
            "intent": {
                "modality": trace.intent_modality,
                "confidence": trace.intent_confidence,
            },
            "text_retrieval": {
                "count": trace.text_count,
                "latency_ms": trace.text_latency_ms,
                "mode": trace.text_mode,
            },
            "image_retrieval": {
                "count": trace.image_count,
                "latency_ms": trace.image_latency_ms,
                "mode": trace.image_mode,
            },
            "ranking": {
                "text_count": trace.ranked_text_count,
                "image_count": trace.ranked_image_count,
                "fusion_method": trace.fusion_method,
            },
            "synthesis": {
                "is_grounded": trace.is_grounded,
                "citation_count": trace.citation_count,
                "latency_ms": trace.synthesis_latency_ms,
                "unsupported_claims": trace.unsupported_claims,
            },
            "total_latency_ms": trace.total_latency_ms,
        })
    )
