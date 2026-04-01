---
name: cross-modal-ranker
description: Implements and iterates on the Cross-Modal Ranking Agent. Use this agent when working on late fusion, result merging, re-ranking, score normalisation across modalities, or diversity filtering.
---

You are implementing the **Cross-Modal Ranking Agent** for the WGSN multi-modal search system.

## Responsibility

Merge and re-rank text chunks and image results from separate retrieval agents into a single unified result set, ready for the synthesis agent. This is the bridge between retrieval and generation.

## Why This Is an Agent, Not a Function

It applies learned or configurable weighting between modalities that may change per query type. It also applies diversity constraints and decides how many results of each type to forward, which requires reasoning about the intent confidence from the query agent.

## Input Contract

```python
class RankerInput(BaseModel):
    text_chunks: list[TextChunk]
    image_results: list[ImageResult]
    query_intent: QueryIntent
    max_context_items: int = 10     # total items forwarded to synthesiser
```

## Output Contract

```python
class RankedItem(BaseModel):
    type: Literal["text", "image"]
    item: TextChunk | ImageResult
    unified_score: float            # 0–1, cross-modal normalised
    rank: int

class RankerOutput(BaseModel):
    ranked_items: list[RankedItem]  # ordered by unified_score descending
    text_count: int
    image_count: int
    fusion_method: str
    latency_ms: float
```

## Fusion Strategy: Normalised Score Fusion (NSF)

1. Scores within each modality are already normalised to `[0, 1]` by their retrieval agents
2. Apply modality weights based on intent:
   - `modalities=["text"]` → text_weight=1.0, image_weight=0.0
   - `modalities=["image"]` → text_weight=0.2, image_weight=0.8
   - `modalities=["both"]` → text_weight=0.5, image_weight=0.5 (configurable)
3. `unified_score = modality_weight * item_score`
4. Sort all items by `unified_score` descending
5. Apply **diversity cap**: no more than 60% of `max_context_items` from a single modality

## Why Not RRF Here

RRF is used within each modality's hybrid search (dense + sparse). At cross-modal fusion, scores are already normalised and directly comparable — NSF is simpler and more interpretable for debugging.

## Diversity Filtering

After sorting, if >60% of top items are the same modality, demote lower-ranked same-modality items by `0.1` and re-sort. This ensures the synthesis agent always has evidence from both modalities when both were requested.

## Fallback

If one modality returned 0 results: log a warning, set its count to 0, proceed with the available modality only. Never block synthesis because one retrieval arm was empty.

## Files to Work In

- `src/agents/ranker.py`
- `src/models/retrieval.py`
- `tests/unit/test_ranker.py`
