"""End-to-end baseline eval pipeline.

Shape (see ``docs/execution/plans/issue-19-...``):

    ingest → predict → normalize → score → report

The pipeline is deliberately stateless: it takes a callable predictor
(so real models, stubs, and fixtures all drop in without runtime
branching), runs every split row through it, and returns a
:class:`RunResult` carrying the raw predictions, the scored records,
and a summary. Optional ``output_dir`` persists the four standard
artifacts via :mod:`src.evaluation.reporting`.

This module intentionally does NOT know anything about concrete model
loading or prompt templates; those concerns live upstream in the
predictor that the caller supplies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.contracts import EvalRecord
from src.evaluation.config import EvalRunConfig
from src.evaluation.normalization import normalize
from src.evaluation.records import make_eval_record
from src.evaluation.reporting import (
    RawPrediction,
    write_run_artifacts,
)
from src.evaluation.scoring import score_exact_match, score_records
from src.evaluation.splits import SplitArtifactRow


@dataclass(slots=True, frozen=True)
class PredictRequest:
    """One row handed to a predictor.

    ``seed`` is propagated from the run config so deterministic
    predictors can thread it through without digging into run config
    state.
    """

    example_id: str
    category: str
    prompt: str
    seed: int


@dataclass(slots=True, frozen=True)
class PredictResponse:
    """One raw prediction returned by a predictor."""

    prediction: str
    latency_ms: float
    tokens_in: int
    tokens_out: int

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, str):
            raise TypeError(
                "PredictResponse.prediction must be str, got "
                f"{type(self.prediction).__name__}"
            )
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise TypeError(
                "PredictResponse.latency_ms must be float, got "
                f"{type(self.latency_ms).__name__}"
            )
        for name in ("tokens_in", "tokens_out"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"PredictResponse.{name} must be int, got "
                    f"{type(value).__name__}"
                )


Predictor = Callable[[PredictRequest], PredictResponse]


@dataclass(slots=True, frozen=True)
class RunResult:
    """Returned by :func:`run_baseline_eval`.

    Carries the in-memory equivalents of the on-disk artifacts so the
    caller can inspect them without re-reading the files.
    """

    records: list[EvalRecord]
    raw_predictions: list[RawPrediction]
    summary: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def _validate_split_rows(
    split_rows: Sequence[SplitArtifactRow],
    config: EvalRunConfig,
) -> None:
    if not split_rows:
        raise ValueError("run_baseline_eval: split_rows is empty")
    seen: set[str] = set()
    for row in split_rows:
        if not isinstance(row, SplitArtifactRow):
            raise TypeError(
                "run_baseline_eval: expected SplitArtifactRow, got "
                f"{type(row).__name__}"
            )
        if row.split != config.split:
            raise ValueError(
                "run_baseline_eval: row split mismatch for "
                f"{row.example_id!r}: row.split={row.split!r}, "
                f"config.split={config.split!r}"
            )
        if row.example_id in seen:
            raise ValueError(
                "run_baseline_eval: duplicate example_id "
                f"{row.example_id!r} in split_rows"
            )
        seen.add(row.example_id)


def _raw_from(
    *,
    config: EvalRunConfig,
    row: SplitArtifactRow,
    response: PredictResponse,
) -> RawPrediction:
    return RawPrediction(
        run_id=config.run_id,
        example_id=row.example_id,
        model_id=config.model_id,
        prompt_template_id=config.prompt_template_id,
        seed=config.seed,
        prediction=response.prediction,
        latency_ms=float(response.latency_ms),
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
    )


def _record_from(
    *,
    config: EvalRunConfig,
    row: SplitArtifactRow,
    response: PredictResponse,
    normalized: str,
    correct: bool,
) -> EvalRecord:
    return make_eval_record(
        run_id=config.run_id,
        example_id=row.example_id,
        model_id=config.model_id,
        prompt_template_id=config.prompt_template_id,
        normalizer_id=config.normalizer_id,
        category=row.category,
        split=row.split,
        gold=row.gold,
        prediction=response.prediction,
        normalized_prediction=normalized,
        correct=correct,
        latency_ms=float(response.latency_ms),
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        seed=config.seed,
        decode_config=dict(config.decode_config),
    )


def run_baseline_eval(
    *,
    split_rows: Sequence[SplitArtifactRow],
    config: EvalRunConfig,
    predictor: Predictor,
    output_dir: str | Path | None = None,
) -> RunResult:
    """Run the full ingest → predict → normalize → score → report pipeline.

    Args:
        split_rows: Frozen split artifact rows (validation or golden).
        config: Run configuration snapshot stamped on every record.
        predictor: Callable that turns a :class:`PredictRequest` into a
            :class:`PredictResponse`. Tests pass stubs here; real runs
            wrap a model.
        output_dir: Optional target directory. When set, the four
            artifacts (``predictions.jsonl``, ``eval_records.jsonl``,
            ``run_config.json``, ``summary.json``) are written under
            it.

    Returns:
        :class:`RunResult` carrying the in-memory records, raw
        predictions, summary dict, and any artifact paths.
    """
    if not isinstance(config, EvalRunConfig):
        raise TypeError(
            "run_baseline_eval: config must be EvalRunConfig, got "
            f"{type(config).__name__}"
        )
    if not callable(predictor):
        raise TypeError(
            "run_baseline_eval: predictor must be callable, got "
            f"{type(predictor).__name__}"
        )
    _validate_split_rows(split_rows, config)

    raw_predictions: list[RawPrediction] = []
    records: list[EvalRecord] = []

    for row in split_rows:
        request = PredictRequest(
            example_id=row.example_id,
            category=row.category,
            prompt=row.prompt,
            seed=config.seed,
        )
        response = predictor(request)
        if not isinstance(response, PredictResponse):
            raise TypeError(
                "run_baseline_eval: predictor must return PredictResponse, "
                f"got {type(response).__name__}"
            )

        normalized = normalize(
            response.prediction,
            normalizer_id=config.normalizer_id,
            category=row.category,
        )
        correct = score_exact_match(normalized, row.gold)

        raw_predictions.append(_raw_from(config=config, row=row, response=response))
        records.append(
            _record_from(
                config=config,
                row=row,
                response=response,
                normalized=normalized,
                correct=correct,
            )
        )

    summary = score_records(records)

    artifact_paths: dict[str, Path] = {}
    if output_dir is not None:
        artifact_paths = write_run_artifacts(
            output_dir=output_dir,
            config=config,
            raw_predictions=raw_predictions,
            records=records,
        )

    return RunResult(
        records=records,
        raw_predictions=raw_predictions,
        summary=summary,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "PredictRequest",
    "PredictResponse",
    "Predictor",
    "RunResult",
    "run_baseline_eval",
]
