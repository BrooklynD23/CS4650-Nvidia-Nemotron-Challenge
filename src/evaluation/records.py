"""Construct and (de)serialize :class:`EvalRecord` objects.

JSONL is the on-disk format so sweeps can be appended to and tail-read
incrementally. One record per line; no trailing comma, no array
wrapper.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.contracts import EvalRecord


def make_eval_record(
    *,
    run_id: str,
    example_id: str,
    model_id: str,
    prompt_template_id: str,
    normalizer_id: str,
    category: str,
    split: str,
    gold: str,
    prediction: str,
    normalized_prediction: str,
    correct: bool,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    seed: int,
    decode_config: dict[str, Any],
) -> EvalRecord:
    """Keyword-only convenience constructor for :class:`EvalRecord`.

    This exists so notebook callers do not have to remember the field
    order; the dataclass remains the single source of truth for
    validation.
    """
    return EvalRecord(
        run_id=run_id,
        example_id=example_id,
        model_id=model_id,
        prompt_template_id=prompt_template_id,
        normalizer_id=normalizer_id,
        category=category,
        split=split,
        gold=gold,
        prediction=prediction,
        normalized_prediction=normalized_prediction,
        correct=correct,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        seed=seed,
        decode_config=dict(decode_config),
    )


def write_eval_records_jsonl(
    records: Iterable[EvalRecord],
    path: str | Path,
) -> int:
    """Write ``records`` to ``path`` as JSONL. Returns the count written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            if not isinstance(record, EvalRecord):
                raise TypeError(
                    "write_eval_records_jsonl: expected EvalRecord, got "
                    f"{type(record).__name__}"
                )
            fh.write(json.dumps(asdict(record), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_eval_records_jsonl(path: str | Path) -> list[EvalRecord]:
    """Read a JSONL file of :class:`EvalRecord` into a list.

    Each non-empty line must parse to a JSON object with all required
    fields; validation happens through :class:`EvalRecord.__post_init__`.
    """
    in_path = Path(path)
    records: list[EvalRecord] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"read_eval_records_jsonl: invalid JSON at "
                    f"{in_path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"read_eval_records_jsonl: line {line_no} is not a "
                    f"JSON object (got {type(data).__name__})"
                )
            records.append(EvalRecord(**data))
    return records


__all__ = [
    "make_eval_record",
    "write_eval_records_jsonl",
    "read_eval_records_jsonl",
]
