from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "cases.jsonl"
BASELINE_PATH = Path(__file__).parent / "baseline_metrics.json"

METRIC_NAMES = [
    "recall_at_10_text",
    "recall_at_10_image",
    "mrr",
    "hallucination_rate",
    "citation_coverage",
    "e2e_latency_p95_ms",
]


class EvalCase(BaseModel):
    case_id: str
    query: str
    relevant_text_doc_ids: list[str]
    relevant_image_ids: list[str]
    expected_modalities: list[str]
    reference_answer: str | None


@dataclass
class Metrics:
    recall_at_10_text: float
    recall_at_10_image: float
    mrr: float
    hallucination_rate: float
    citation_coverage: float
    e2e_latency_p95_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "recall_at_10_text": self.recall_at_10_text,
            "recall_at_10_image": self.recall_at_10_image,
            "mrr": self.mrr,
            "hallucination_rate": self.hallucination_rate,
            "citation_coverage": self.citation_coverage,
            "e2e_latency_p95_ms": self.e2e_latency_p95_ms,
        }


def compute_recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = top_k.intersection(relevant_ids)
    return len(hits) / len(relevant_ids)


def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_citation_coverage(answer: str) -> float:
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    citation_pattern = re.compile(r"\[(?:img-)?\d+\]")
    cited_count = sum(1 for s in sentences if citation_pattern.search(s))
    return cited_count / len(sentences)


def check_regression(current: dict, baseline: dict) -> list[str]:
    regressions = []
    for metric in METRIC_NAMES:
        if metric not in baseline or metric not in current:
            continue
        baseline_value = baseline[metric]
        if baseline_value == 0:
            continue
        delta = (current[metric] - baseline_value) / baseline_value
        if delta < -0.05:
            regressions.append(f"{metric}: {delta:.1%} regression")
    return regressions


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def _compute_metrics_from_results(
    results: list[dict], cases: list[EvalCase]
) -> Metrics:
    case_map = {c.case_id: c for c in cases}
    text_recalls: list[float] = []
    image_recalls: list[float] = []
    mrr_scores: list[float] = []
    citation_scores: list[float] = []
    hallucination_flags: list[float] = []
    latencies: list[float] = []

    for result in results:
        case = case_map.get(result["case_id"])
        if case is None:
            logger.warning('no eval case found for case_id="%s"', result["case_id"])
            continue

        text_recalls.append(
            compute_recall_at_k(result["retrieved_text_ids"], case.relevant_text_doc_ids, 10)
        )
        image_recalls.append(
            compute_recall_at_k(result["retrieved_image_ids"], case.relevant_image_ids, 10)
        )
        combined_retrieved = result["retrieved_text_ids"] + result["retrieved_image_ids"]
        combined_relevant = case.relevant_text_doc_ids + case.relevant_image_ids
        mrr_scores.append(compute_mrr(combined_retrieved, combined_relevant))
        citation_scores.append(compute_citation_coverage(result["answer"]))
        hallucination_flags.append(0.0 if result.get("is_grounded", True) else 1.0)
        latencies.append(result["latency_ms"])

    return Metrics(
        recall_at_10_text=float(np.mean(text_recalls)) if text_recalls else 0.0,
        recall_at_10_image=float(np.mean(image_recalls)) if image_recalls else 0.0,
        mrr=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        hallucination_rate=float(np.mean(hallucination_flags)) if hallucination_flags else 0.0,
        citation_coverage=float(np.mean(citation_scores)) if citation_scores else 0.0,
        e2e_latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
    )


def _print_metrics_report(metrics: Metrics) -> None:
    print("\n=== Eval Metrics Report ===")
    print(f"  recall@10 text:       {metrics.recall_at_10_text:.4f}  (target >= 0.80)")
    print(f"  recall@10 image:      {metrics.recall_at_10_image:.4f}  (target >= 0.70)")
    print(f"  MRR:                  {metrics.mrr:.4f}  (target >= 0.65)")
    print(f"  hallucination rate:   {metrics.hallucination_rate:.4f}  (target < 0.05)")
    print(f"  citation coverage:    {metrics.citation_coverage:.4f}  (target >= 0.80)")
    print(f"  e2e latency p95 ms:   {metrics.e2e_latency_p95_ms:.1f}  (target < 3000)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="WGSN eval harness")
    parser.add_argument("--results-file", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(CASES_PATH)
    logger.info('loaded %d eval cases from "%s"', len(cases), CASES_PATH)

    baseline: dict | None = None
    if BASELINE_PATH.exists():
        with BASELINE_PATH.open() as fh:
            baseline = json.load(fh)
        logger.info('loaded baseline metrics from "%s"', BASELINE_PATH)

    if args.results_file is None:
        print("Pipeline call stubbed — provide --results-file to evaluate")
        print(f"Loaded {len(cases)} eval cases.")
        sys.exit(0)

    with args.results_file.open() as fh:
        results: list[dict] = json.load(fh)

    logger.info('loaded %d result records from "%s"', len(results), args.results_file)

    metrics = _compute_metrics_from_results(results, cases)
    _print_metrics_report(metrics)

    if baseline is not None:
        regressions = check_regression(metrics.as_dict(), baseline)
        if regressions:
            print("REGRESSIONS DETECTED:")
            for regression in regressions:
                print(f"  - {regression}")
            print()
            sys.exit(1)
        else:
            print("No regressions detected vs baseline.")

    sys.exit(0)


if __name__ == "__main__":
    main()
