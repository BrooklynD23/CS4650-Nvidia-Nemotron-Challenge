"""Tests for :mod:`src.evaluation.scoring`.

These tests cover the scoring primitives (exact-match + record-level
score attachment) and verify the summary helpers produce stable
aggregates derivable from record-level artifacts alone.
"""

from __future__ import annotations

import pytest

from src.contracts import EvalRecord
from src.evaluation.scoring import (
    score_exact_match,
    score_records,
    summarize_records,
)


def _record(
    *,
    example_id: str,
    gold: str,
    prediction: str,
    normalized_prediction: str | None = None,
    correct: bool | None = None,
    category: str = "binary",
    normalizer_id: str = "exact_v1",
    latency_ms: float = 10.0,
    tokens_in: int = 8,
    tokens_out: int = 2,
    run_id: str = "run-1",
) -> EvalRecord:
    norm = normalized_prediction if normalized_prediction is not None else prediction
    is_correct = correct if correct is not None else (norm == gold)
    return EvalRecord(
        run_id=run_id,
        example_id=example_id,
        model_id="nemotron-x",
        prompt_template_id="tpl-v1",
        normalizer_id=normalizer_id,
        category=category,
        split="val",
        gold=gold,
        prediction=prediction,
        normalized_prediction=norm,
        correct=is_correct,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        seed=42,
        decode_config={"temperature": 0.0},
    )


class TestExactMatch:
    def test_equal_strings_match(self) -> None:
        assert score_exact_match("A", "A") is True

    def test_unequal_strings_do_not_match(self) -> None:
        assert score_exact_match("A", "B") is False

    def test_blank_prediction_is_always_incorrect(self) -> None:
        # Even when gold is empty, a blank prediction must not score.
        assert score_exact_match("", "") is False
        assert score_exact_match("", "A") is False

    def test_case_matters(self) -> None:
        assert score_exact_match("a", "A") is False

    def test_whitespace_matters(self) -> None:
        assert score_exact_match("A ", "A") is False

    def test_rejects_non_string_inputs(self) -> None:
        with pytest.raises(TypeError):
            score_exact_match(None, "A")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            score_exact_match("A", 1)  # type: ignore[arg-type]


class TestScoreRecords:
    def test_summary_surfaces_single_run_attribution(self) -> None:
        records = [
            _record(example_id="e-1", gold="A", prediction="A"),
            _record(example_id="e-2", gold="B", prediction="B"),
        ]
        summary = score_records(records)
        assert summary["run_id"] == "run-1"
        assert summary["model_id"] == "nemotron-x"
        assert summary["prompt_template_id"] == "tpl-v1"
        assert summary["normalizer_id"] == "exact_v1"
        assert summary["seed"] == 42
        assert summary["split"] == "val"

    def test_all_correct_yields_accuracy_1(self) -> None:
        records = [
            _record(example_id="e-1", gold="A", prediction="A"),
            _record(example_id="e-2", gold="B", prediction="B"),
        ]
        summary = score_records(records)
        assert summary["total"] == 2
        assert summary["correct"] == 2
        assert summary["accuracy"] == 1.0

    def test_partial_correctness(self) -> None:
        records = [
            _record(example_id="e-1", gold="A", prediction="A"),
            _record(example_id="e-2", gold="B", prediction="x", correct=False),
        ]
        summary = score_records(records)
        assert summary["total"] == 2
        assert summary["correct"] == 1
        assert summary["accuracy"] == 0.5

    def test_per_category_accuracy(self) -> None:
        records = [
            _record(example_id="b-1", category="binary", gold="A", prediction="A"),
            _record(
                example_id="b-2",
                category="binary",
                gold="B",
                prediction="x",
                correct=False,
            ),
            _record(example_id="c-1", category="cipher", gold="A", prediction="A"),
        ]
        summary = score_records(records)
        per_cat = summary["per_category_accuracy"]
        assert per_cat["binary"]["total"] == 2
        assert per_cat["binary"]["correct"] == 1
        assert per_cat["binary"]["accuracy"] == 0.5
        assert per_cat["cipher"]["accuracy"] == 1.0

    def test_latency_aggregates_reported(self) -> None:
        records = [
            _record(example_id="e-1", gold="A", prediction="A", latency_ms=10.0),
            _record(example_id="e-2", gold="A", prediction="A", latency_ms=30.0),
        ]
        summary = score_records(records)
        lat = summary["latency_ms"]
        assert lat["mean"] == 20.0
        # With two samples and linear interpolation, p50 is the midpoint.
        assert lat["p50"] == 20.0
        # p95 must be bounded by the observed max.
        assert 20.0 <= lat["p95"] <= 30.0

    def test_tokens_totals(self) -> None:
        records = [
            _record(
                example_id="e-1",
                gold="A",
                prediction="A",
                tokens_in=10,
                tokens_out=1,
            ),
            _record(
                example_id="e-2",
                gold="A",
                prediction="A",
                tokens_in=20,
                tokens_out=3,
            ),
        ]
        summary = score_records(records)
        assert summary["tokens_in_total"] == 30
        assert summary["tokens_out_total"] == 4

    def test_normalizer_and_run_ids_surfaced(self) -> None:
        records = [
            _record(
                example_id="e-1",
                gold="A",
                prediction="A",
                normalizer_id="exact_v1",
                run_id="run-1",
            ),
            _record(
                example_id="e-2",
                gold="A",
                prediction="A",
                normalizer_id="exact_v1",
                run_id="run-1",
            ),
        ]
        summary = score_records(records)
        assert summary["normalizer_ids"] == ["exact_v1"]
        assert summary["run_ids"] == ["run-1"]

    def test_rejects_mixed_run_attribution(self) -> None:
        records = [
            _record(example_id="e-1", gold="A", prediction="A", run_id="run-1"),
            _record(example_id="e-2", gold="A", prediction="A", run_id="run-2"),
        ]
        with pytest.raises(ValueError, match="single run"):
            score_records(records)

    def test_empty_records_yields_zero_accuracy(self) -> None:
        summary = score_records([])
        assert summary == {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "per_category_accuracy": {},
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
            "tokens_in_total": 0,
            "tokens_out_total": 0,
            "run_id": None,
            "model_id": None,
            "prompt_template_id": None,
            "normalizer_id": None,
            "seed": None,
            "split": None,
            "normalizer_ids": [],
            "run_ids": [],
        }

    def test_summarize_is_an_alias(self) -> None:
        # Kept around as a convenience import; must match score_records.
        records = [_record(example_id="e-1", gold="A", prediction="A")]
        assert summarize_records(records) == score_records(records)
