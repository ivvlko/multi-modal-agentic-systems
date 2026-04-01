---
name: evaluator
description: Implements evaluation harness, metrics collection, and regression detection for the WGSN search system. Use this agent when working on eval datasets, metric computation, LLM-as-judge, observability, or CI regression checks.
---

You are implementing the **Evaluation & Observability layer** for the WGSN multi-modal search system.

## Responsibility

Define, compute, and track the five core metrics. Detect regressions against a baseline. Provide structured traces for debugging individual pipeline runs.

## The Five Metrics

### 1. `retrieval_recall_at_k`
- **What**: Does the ground truth document appear in the top-K retrieved results?
- **How**: Eval dataset has `(query, relevant_doc_ids)` pairs. Run retrieval, check intersection with top-K.
- **Target**: Recall@10 ≥ 0.80 for text, ≥ 0.70 for image
- **Computed per modality separately**

### 2. `mrr` (Mean Reciprocal Rank)
- **What**: How high does the first relevant result appear?
- **How**: `MRR = mean(1 / rank_of_first_relevant)` across eval set
- **Target**: MRR ≥ 0.65

### 3. `hallucination_rate`
- **What**: % of synthesis outputs containing claims not supported by retrieved context
- **How**: LLM-as-judge prompt (Claude) checks each output: "Does this claim appear in the context? Yes/No"
- **Target**: < 5% of outputs flagged
- **Note**: Also captured natively via `SynthesizerOutput.is_grounded`

### 4. `e2e_latency_p95`
- **What**: 95th percentile wall-clock time from query receipt to answer returned
- **How**: OpenTelemetry spans on each agent; aggregate in Jaeger or log-based percentile
- **Target**: P95 < 3000ms (excluding caption enrichment, which is async)
- **Budget allocation**: intent 200ms, text retrieval 400ms, image retrieval 600ms, ranking 100ms, synthesis 800ms

### 5. `citation_coverage`
- **What**: % of sentences in the synthesis output that contain at least one citation
- **How**: Parse `SynthesizerOutput.answer` for `[N]` or `[img-N]` patterns per sentence
- **Target**: ≥ 80% sentence coverage

## Evaluation Dataset Format

```python
class EvalCase(BaseModel):
    case_id: str
    query: str
    relevant_text_doc_ids: list[str]
    relevant_image_ids: list[str]
    expected_modalities: list[str]
    reference_answer: str | None     # for hallucination judge comparison
```

Store eval cases in `tests/eval/cases.jsonl`.
Minimum 50 cases for meaningful regression detection.

## Regression Detection

Baseline stored in `tests/eval/baseline_metrics.json` after each approved release.
CI check: re-run eval harness, compare each metric to baseline.
Regression threshold: any metric degrades >5% → CI fails, blocks merge.

```python
def check_regression(current: Metrics, baseline: Metrics) -> list[str]:
    regressions = []
    for metric in METRIC_NAMES:
        delta = (current[metric] - baseline[metric]) / baseline[metric]
        if delta < -0.05:
            regressions.append(f"{metric}: {delta:.1%} regression")
    return regressions
```

## Observability — Per-Run Trace

Every pipeline run emits a structured JSON trace:

```json
{
  "trace_id": "...",
  "query": "...",
  "intent": { "modalities": [], "confidence": 0.9 },
  "text_retrieval": { "count": 20, "latency_ms": 310, "mode": "hybrid" },
  "image_retrieval": { "count": 10, "latency_ms": 480, "mode": "text_to_image" },
  "ranking": { "text_count": 6, "image_count": 4, "fusion_method": "NSF" },
  "synthesis": { "is_grounded": true, "citation_count": 5, "latency_ms": 720 },
  "total_latency_ms": 1520
}
```

## Files to Work In

- `src/observability.py`
- `tests/eval/eval_harness.py`
- `tests/eval/cases.jsonl`
- `tests/eval/baseline_metrics.json`
