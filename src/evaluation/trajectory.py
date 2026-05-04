"""Trajectory collection and error-slice analysis over EvalRecord streams.

Consumes :class:`~src.contracts.EvalRecord` rows produced by the baseline
eval pipeline and produces typed trajectory rows with error classification,
recoverability flags, and filtered retry-candidate sets for downstream
synthetic-data generation (#24) and solver design (#23).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.contracts import EvalRecord


_BOXED_MARKER = r"\boxed{"
_TRUNCATION_TOKEN_THRESHOLD = 7500


class ErrorType(str, Enum):
    CORRECT = "correct"
    FORMAT_MISS = "format_miss"
    ARITHMETIC_SLIP = "arithmetic_slip"
    HALLUCINATED_REASONING = "hallucinated_reasoning"
    REFUSAL = "refusal"
    TRUNCATION = "truncation"


@dataclass(slots=True, frozen=True)
class TrajectoryRow:
    record: EvalRecord
    prompt_template_id: str
    error_type: ErrorType
    recoverability: bool


def classify_error(record: EvalRecord) -> ErrorType:
    """Classify a single EvalRecord into an ErrorType using heuristics.

    Priority order prevents double-labelling: correct first, then refusal
    (empty prediction), truncation (token length near evaluator max),
    format miss (no boxed), then arithmetic slip vs. hallucination.
    Hallucination is the catch-all for wrong answers that have valid format.
    """
    if record.correct:
        return ErrorType.CORRECT
    if record.prediction.strip() == "":
        return ErrorType.REFUSAL
    if record.tokens_out >= _TRUNCATION_TOKEN_THRESHOLD:
        return ErrorType.TRUNCATION
    if _BOXED_MARKER not in record.prediction:
        return ErrorType.FORMAT_MISS
    if record.normalized_prediction.strip() == "":
        return ErrorType.FORMAT_MISS
    # Boxed present but wrong — distinguish numeric slip from hallucination
    # by checking whether the normalized prediction looks numeric.
    try:
        float(record.normalized_prediction.replace(",", ""))
        return ErrorType.ARITHMETIC_SLIP
    except ValueError:
        return ErrorType.HALLUCINATED_REASONING


def mark_recoverability(record: EvalRecord) -> bool:
    """Return True if the record is incorrect but has a boxed-format answer.

    Recoverable failures are those where the model attempted the right
    format but got the wrong value — targeted prompting or a solver can
    often fix these without full retraining.
    """
    return not record.correct and _BOXED_MARKER in record.prediction


def _make_row(record: EvalRecord) -> TrajectoryRow:
    return TrajectoryRow(
        record=record,
        prompt_template_id=record.prompt_template_id,
        error_type=classify_error(record),
        recoverability=mark_recoverability(record),
    )


def build_trajectory_rows(records: list[EvalRecord]) -> list[TrajectoryRow]:
    """Convert a list of EvalRecords into TrajectoryRows."""
    return [_make_row(r) for r in records]


def slice_by_category(rows: list[TrajectoryRow]) -> dict[str, list[TrajectoryRow]]:
    result: dict[str, list[TrajectoryRow]] = {}
    for row in rows:
        result.setdefault(row.record.category, []).append(row)
    return result


def slice_by_error_type(rows: list[TrajectoryRow]) -> dict[str, list[TrajectoryRow]]:
    result: dict[str, list[TrajectoryRow]] = {}
    for row in rows:
        result.setdefault(row.error_type.value, []).append(row)
    return result


def produce_retry_candidates(rows: list[TrajectoryRow]) -> list[TrajectoryRow]:
    """Return rows that are incorrect AND recoverable (boxed format present)."""
    return [r for r in rows if not r.record.correct and r.recoverability]


def _row_to_dict(row: TrajectoryRow) -> dict:
    d = dataclasses.asdict(row)
    # ErrorType serializes as its .value via asdict since it's str subclass
    return d


def write_trajectory_jsonl(rows: list[TrajectoryRow], path: Path) -> None:
    """Write TrajectoryRows to a JSONL file, one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_row_to_dict(row), ensure_ascii=False))
            fh.write("\n")


def write_retry_candidates(candidates: list[TrajectoryRow], path: Path) -> None:
    """Write retry-candidate TrajectoryRows to a JSONL file."""
    write_trajectory_jsonl(candidates, path)


__all__ = [
    "ErrorType",
    "TrajectoryRow",
    "classify_error",
    "mark_recoverability",
    "build_trajectory_rows",
    "slice_by_category",
    "slice_by_error_type",
    "produce_retry_candidates",
    "write_trajectory_jsonl",
    "write_retry_candidates",
]
