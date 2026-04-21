"""Lightweight EDA helpers for the canonical schema.

Everything here is stdlib-only so it runs inside constrained Kaggle
kernels without extra deps. If pandas is installed, callers can still
feed pandas rows in via ``DataFrame.to_dict(orient='records')``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from src.contracts import ReasoningExample


def summarize_examples(examples: Iterable[ReasoningExample]) -> dict[str, Any]:
    """Return a small dict describing a pile of :class:`ReasoningExample`.

    The summary is intended for quick notebook sanity checks, not for
    formal reporting. Keys:

    - ``total``: total count
    - ``by_category``: count per ``category``
    - ``by_split``: count per ``split``
    - ``by_source``: count per ``source``
    - ``sample_ids``: up to three ids, in input order
    """
    total = 0
    by_category: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    sample_ids: list[str] = []

    for ex in examples:
        total += 1
        by_category[ex.category] += 1
        by_split[ex.split] += 1
        by_source[ex.source] += 1
        if len(sample_ids) < 3:
            sample_ids.append(ex.id)

    return {
        "total": total,
        "by_category": dict(by_category),
        "by_split": dict(by_split),
        "by_source": dict(by_source),
        "sample_ids": sample_ids,
    }


def detect_duplicates_by_id(
    examples: Iterable[ReasoningExample],
) -> list[str]:
    """Return ids that appear more than once, preserving first-seen order."""
    counts: Counter[str] = Counter()
    order: list[str] = []
    seen: set[str] = set()
    for ex in examples:
        if ex.id not in seen:
            order.append(ex.id)
            seen.add(ex.id)
        counts[ex.id] += 1
    return [ex_id for ex_id in order if counts[ex_id] > 1]


def detect_missing_fields(
    rows: Iterable[Mapping[str, Any]],
    required: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return per-row reports of missing required fields.

    Args:
        rows: Raw rows (pre-normalization). Each row is a mapping.
        required: Tuple of field names that must be present and not None.

    Returns:
        A list of ``{"row_index": i, "missing": [...]}`` dicts. Rows
        with no missing fields are omitted.
    """
    reports: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        missing = [
            f for f in required if f not in row or row[f] is None
        ]
        if missing:
            reports.append({"row_index": i, "missing": missing})
    return reports


__all__ = [
    "summarize_examples",
    "detect_duplicates_by_id",
    "detect_missing_fields",
]
