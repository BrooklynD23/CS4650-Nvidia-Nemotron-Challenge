"""Record-level artifact I/O and run-summary writing.

Every eval run emits four artifacts under ``data/eval/runs/<run_id>/``:

1. ``predictions.jsonl`` — raw predictions (pre-normalization), one per
   example. Used for disputes and for re-scoring under a different
   ``normalizer_id`` without re-running the model.
2. ``eval_records.jsonl`` — :class:`~src.contracts.EvalRecord` rows,
   one per example. Normalized + scored, with full attribution.
3. ``run_config.json`` — snapshot of the :class:`EvalRunConfig` used.
4. ``summary.json`` — aggregate metrics derived only from the records.

Writing order matters: records are flushed before the summary so a
process crash during summary computation leaves the record stream
intact for manual re-summarization.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.contracts import EvalRecord
from src.evaluation.config import (
    EvalRunConfig,
    validate_records_against_config,
    write_run_config,
)
from src.evaluation.scoring import score_records


PREDICTIONS_FILENAME: str = "predictions.jsonl"
EVAL_RECORDS_FILENAME: str = "eval_records.jsonl"
RUN_CONFIG_FILENAME: str = "run_config.json"
SUMMARY_FILENAME: str = "summary.json"


@dataclass(slots=True, frozen=True)
class RawPrediction:
    """The raw model output for one example, before normalization.

    Serialized as a sibling of :class:`EvalRecord` so ``prediction``
    always has a canonical on-disk home even when normalization changes.
    """

    run_id: str
    example_id: str
    model_id: str
    prompt_template_id: str
    seed: int
    prediction: str
    latency_ms: float
    tokens_in: int
    tokens_out: int

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "example_id",
            "model_id",
            "prompt_template_id",
            "prediction",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(
                    f"RawPrediction.{field_name} must be str, got "
                    f"{type(value).__name__}"
                )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                "RawPrediction.seed must be int, got "
                f"{type(self.seed).__name__}"
            )
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise TypeError(
                "RawPrediction.latency_ms must be float, got "
                f"{type(self.latency_ms).__name__}"
            )
        if isinstance(self.tokens_in, bool) or not isinstance(
            self.tokens_in, int
        ):
            raise TypeError(
                "RawPrediction.tokens_in must be int, got "
                f"{type(self.tokens_in).__name__}"
            )
        if isinstance(self.tokens_out, bool) or not isinstance(
            self.tokens_out, int
        ):
            raise TypeError(
                "RawPrediction.tokens_out must be int, got "
                f"{type(self.tokens_out).__name__}"
            )


def write_predictions_jsonl(
    predictions: Iterable[RawPrediction],
    path: str | Path,
) -> int:
    """Write raw predictions to ``path`` as JSONL. Returns the count written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            if not isinstance(pred, RawPrediction):
                raise TypeError(
                    "write_predictions_jsonl: expected RawPrediction, got "
                    f"{type(pred).__name__}"
                )
            fh.write(json.dumps(asdict(pred), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_predictions_jsonl(path: str | Path) -> list[RawPrediction]:
    """Read raw predictions from a JSONL file written by
    :func:`write_predictions_jsonl`.
    """
    in_path = Path(path)
    rows: list[RawPrediction] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"read_predictions_jsonl: invalid JSON at "
                    f"{in_path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"read_predictions_jsonl: line {line_no} is not a "
                    f"JSON object (got {type(data).__name__})"
                )
            rows.append(RawPrediction(**data))
    return rows


def write_eval_records_jsonl(
    records: Iterable[EvalRecord],
    path: str | Path,
) -> int:
    """Write :class:`EvalRecord` rows to ``path`` as JSONL.

    Deliberately mirrors :func:`src.evaluation.records.write_eval_records_jsonl`
    so the reporting module can own the full artifact surface without
    importing circular concerns from records.py.
    """
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
    """Read :class:`EvalRecord` rows from a JSONL artifact."""
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


def write_summary_json(summary: dict[str, Any], path: str | Path) -> Path:
    """Persist an aggregate summary dict as pretty-printed JSON."""
    if not isinstance(summary, dict):
        raise TypeError(
            "write_summary_json: expected dict, got "
            f"{type(summary).__name__}"
        )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return out_path


def read_summary_json(path: str | Path) -> dict[str, Any]:
    """Load an aggregate summary dict from JSON."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"read_summary_json: {in_path} did not contain a JSON object"
        )
    return data


def write_run_artifacts(
    *,
    output_dir: str | Path,
    config: EvalRunConfig,
    raw_predictions: Sequence[RawPrediction],
    records: Sequence[EvalRecord],
) -> dict[str, Path]:
    """Write the full set of artifacts for a baseline eval run.

    Ordering (important for partial failures):
    1. Validate every record against ``config`` (fail closed).
    2. Write ``predictions.jsonl``.
    3. Write ``eval_records.jsonl``.
    4. Write ``run_config.json``.
    5. Compute and write ``summary.json``.

    Returns a dict of logical artifact name → Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_records_against_config(records, config)

    predictions_path = out_dir / PREDICTIONS_FILENAME
    records_path = out_dir / EVAL_RECORDS_FILENAME
    config_path = out_dir / RUN_CONFIG_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME

    write_predictions_jsonl(raw_predictions, predictions_path)
    write_eval_records_jsonl(records, records_path)
    write_run_config(config, config_path)

    summary = score_records(records)
    write_summary_json(summary, summary_path)

    return {
        "predictions": predictions_path,
        "eval_records": records_path,
        "run_config": config_path,
        "summary": summary_path,
    }


__all__ = [
    "PREDICTIONS_FILENAME",
    "EVAL_RECORDS_FILENAME",
    "RUN_CONFIG_FILENAME",
    "SUMMARY_FILENAME",
    "RawPrediction",
    "write_predictions_jsonl",
    "read_predictions_jsonl",
    "write_eval_records_jsonl",
    "read_eval_records_jsonl",
    "write_summary_json",
    "read_summary_json",
    "write_run_artifacts",
]
