"""Exact-match scoring and record-level aggregates.

Scoring is intentionally trivial so the behavior contract lives in
normalization (see :mod:`src.evaluation.normalization`). This module
owns only two concerns:

1. Compare a normalized prediction to the gold string. Blank
   predictions never score as correct.
2. Reduce a stream of :class:`EvalRecord` rows into a dict summary that
   can be serialized to ``summary.json`` and reconstructed on a clean
   checkout from the JSONL records alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from src.contracts import EvalRecord


def score_exact_match(normalized_prediction: str, gold: str) -> bool:
    """Return ``True`` iff ``normalized_prediction == gold`` and non-empty.

    A blank normalized prediction is treated as "no answer" and scores
    as incorrect regardless of ``gold``.
    """
    if not isinstance(normalized_prediction, str):
        raise TypeError(
            "score_exact_match: normalized_prediction must be str, got "
            f"{type(normalized_prediction).__name__}"
        )
    if not isinstance(gold, str):
        raise TypeError(
            "score_exact_match: gold must be str, got "
            f"{type(gold).__name__}"
        )
    if normalized_prediction == "":
        return False
    return normalized_prediction == gold


def _percentile_linear(values: Sequence[float], p: float) -> float:
    """Return the ``p``-th percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = k - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _single_run_value(
    records: Sequence[EvalRecord],
    field_name: str,
) -> str | int | None:
    """Return the unique value for ``field_name`` across ``records``.

    Aggregate summaries represent one eval run. If a caller hands us a
    mixed-attribution record batch, fail closed instead of producing a
    misleading summary that hides drift behind list-valued metadata.
    """
    if not records:
        return None
    values = {getattr(record, field_name) for record in records}
    if len(values) != 1:
        raise ValueError(
            "score_records: records must describe a single run; "
            f"mixed {field_name} values: {sorted(values)!r}"
        )
    return next(iter(values))


def score_records(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Aggregate ``records`` into a serializable summary dict.

    The returned dict is intentionally derivable from the record stream
    alone — no extra state, no external config — so a clean checkout
    can rebuild it from the committed ``eval_records.jsonl``.
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, EvalRecord):
            raise TypeError(
                f"score_records: records[{i}] must be EvalRecord, got "
                f"{type(rec).__name__}"
            )

    total = len(records)
    correct = sum(1 for r in records if r.correct)
    accuracy = (correct / total) if total else 0.0

    per_cat: dict[str, dict[str, Any]] = {}
    for rec in records:
        bucket = per_cat.setdefault(
            rec.category, {"total": 0, "correct": 0, "accuracy": 0.0}
        )
        bucket["total"] += 1
        if rec.correct:
            bucket["correct"] += 1
    for stats in per_cat.values():
        stats["accuracy"] = (
            stats["correct"] / stats["total"] if stats["total"] else 0.0
        )

    latencies = [r.latency_ms for r in records]
    if latencies:
        latency_summary = {
            "mean": float(mean(latencies)),
            "p50": float(median(latencies)),
            "p95": _percentile_linear(latencies, 95.0),
        }
    else:
        latency_summary = {"mean": 0.0, "p50": 0.0, "p95": 0.0}

    run_id = _single_run_value(records, "run_id")
    model_id = _single_run_value(records, "model_id")
    prompt_template_id = _single_run_value(records, "prompt_template_id")
    normalizer_id = _single_run_value(records, "normalizer_id")
    seed = _single_run_value(records, "seed")
    split = _single_run_value(records, "split")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_category_accuracy": {
            cat: dict(per_cat[cat]) for cat in sorted(per_cat)
        },
        "latency_ms": latency_summary,
        "tokens_in_total": sum(r.tokens_in for r in records),
        "tokens_out_total": sum(r.tokens_out for r in records),
        "run_id": run_id,
        "model_id": model_id,
        "prompt_template_id": prompt_template_id,
        "normalizer_id": normalizer_id,
        "seed": seed,
        "split": split,
        "normalizer_ids": sorted({r.normalizer_id for r in records}),
        "run_ids": sorted({r.run_id for r in records}),
    }


# Kept as an alias so callers that prefer "summarize_records" naming
# can use it without importing from a different module.
summarize_records = score_records


__all__ = [
    "score_exact_match",
    "score_records",
    "summarize_records",
]
