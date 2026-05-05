"""Tests for src/evaluation/trajectory.py (#22)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contracts import EvalRecord
from src.evaluation.trajectory import (
    ErrorType,
    TrajectoryRow,
    build_trajectory_rows,
    classify_error,
    mark_recoverability,
    produce_retry_candidates,
    slice_by_category,
    slice_by_error_type,
    write_retry_candidates,
    write_trajectory_jsonl,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    example_id: str = "ex1",
    category: str = "math",
    correct: bool = False,
    prediction: str = r"\boxed{42}",
    normalized_prediction: str = "42",
    tokens_out: int = 100,
    gold: str = "99",
) -> EvalRecord:
    return EvalRecord(
        run_id="run-test",
        example_id=example_id,
        model_id="model-test",
        prompt_template_id="tmpl-a",
        normalizer_id="norm-v1",
        category=category,
        split="val",
        gold=gold,
        prediction=prediction,
        normalized_prediction=normalized_prediction,
        correct=correct,
        latency_ms=100.0,
        tokens_in=50,
        tokens_out=tokens_out,
        seed=42,
    )


@pytest.fixture()
def mixed_records() -> list[EvalRecord]:
    """Ten records covering all ErrorType branches."""
    return [
        # correct
        _make_record(example_id="c1", correct=True, prediction=r"\boxed{7}", normalized_prediction="7", gold="7"),
        _make_record(example_id="c2", correct=True, prediction=r"\boxed{3}", normalized_prediction="3", gold="3"),
        # format_miss — no boxed marker
        _make_record(example_id="f1", prediction="42", normalized_prediction="42"),
        _make_record(example_id="f2", prediction="the answer is 5", normalized_prediction=""),
        # arithmetic_slip — boxed present, numeric normalized prediction, wrong
        _make_record(example_id="a1", prediction=r"\boxed{41}", normalized_prediction="41"),
        # hallucinated_reasoning — boxed present, non-numeric normalized
        _make_record(example_id="h1", prediction=r"\boxed{some text}", normalized_prediction="some text"),
        # refusal — empty prediction
        _make_record(example_id="r1", prediction="", normalized_prediction=""),
        # truncation — high token count
        _make_record(example_id="t1", prediction="partial", normalized_prediction="partial", tokens_out=7600),
        # different category
        _make_record(example_id="s1", category="science", prediction=r"\boxed{12}", normalized_prediction="12"),
        _make_record(example_id="s2", category="science", correct=True, prediction=r"\boxed{5}", normalized_prediction="5", gold="5"),
    ]


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

def test_classify_error_correct():
    r = _make_record(correct=True, prediction=r"\boxed{7}", normalized_prediction="7", gold="7")
    assert classify_error(r) == ErrorType.CORRECT


def test_classify_error_refusal():
    r = _make_record(prediction="", normalized_prediction="")
    assert classify_error(r) == ErrorType.REFUSAL


def test_classify_error_truncation():
    r = _make_record(prediction="partial answer", normalized_prediction="", tokens_out=7600)
    assert classify_error(r) == ErrorType.TRUNCATION


def test_classify_error_format_miss_no_boxed():
    r = _make_record(prediction="the answer is 5", normalized_prediction="5")
    assert classify_error(r) == ErrorType.FORMAT_MISS


def test_classify_error_format_miss_empty_normalized():
    r = _make_record(prediction=r"\boxed{}", normalized_prediction="")
    assert classify_error(r) == ErrorType.FORMAT_MISS


def test_classify_error_arithmetic_slip():
    r = _make_record(prediction=r"\boxed{41}", normalized_prediction="41")
    assert classify_error(r) == ErrorType.ARITHMETIC_SLIP


def test_classify_error_hallucinated():
    r = _make_record(prediction=r"\boxed{some nonsense}", normalized_prediction="some nonsense")
    assert classify_error(r) == ErrorType.HALLUCINATED_REASONING


# ---------------------------------------------------------------------------
# mark_recoverability
# ---------------------------------------------------------------------------

def test_recoverability_true_when_boxed_and_incorrect():
    r = _make_record(correct=False, prediction=r"\boxed{41}", normalized_prediction="41")
    assert mark_recoverability(r) is True


def test_recoverability_false_when_correct():
    r = _make_record(correct=True, prediction=r"\boxed{7}", normalized_prediction="7", gold="7")
    assert mark_recoverability(r) is False


def test_recoverability_false_when_no_boxed():
    r = _make_record(correct=False, prediction="42", normalized_prediction="42")
    assert mark_recoverability(r) is False


# ---------------------------------------------------------------------------
# slice_by_category
# ---------------------------------------------------------------------------

def test_slice_by_category(mixed_records):
    rows = build_trajectory_rows(mixed_records)
    slices = slice_by_category(rows)
    assert set(slices.keys()) == {"math", "science"}
    assert len(slices["math"]) == 8
    assert len(slices["science"]) == 2


# ---------------------------------------------------------------------------
# slice_by_error_type
# ---------------------------------------------------------------------------

def test_slice_by_error_type(mixed_records):
    rows = build_trajectory_rows(mixed_records)
    slices = slice_by_error_type(rows)
    assert ErrorType.CORRECT.value in slices
    assert len(slices[ErrorType.CORRECT.value]) == 3  # c1, c2, s2
    assert ErrorType.FORMAT_MISS.value in slices
    assert ErrorType.REFUSAL.value in slices


# ---------------------------------------------------------------------------
# produce_retry_candidates
# ---------------------------------------------------------------------------

def test_produce_retry_candidates_only_recoverable(mixed_records):
    rows = build_trajectory_rows(mixed_records)
    candidates = produce_retry_candidates(rows)
    # All must be incorrect + recoverable
    for c in candidates:
        assert not c.record.correct
        assert c.recoverability
    # Refusal and truncation and format_miss (no boxed) should not appear
    for c in candidates:
        assert c.error_type not in (ErrorType.REFUSAL, ErrorType.TRUNCATION)
        assert c.error_type != ErrorType.FORMAT_MISS or r"\boxed{" in c.record.prediction


def test_produce_retry_candidates_empty_when_all_correct():
    records = [
        _make_record(example_id=f"c{i}", correct=True, prediction=r"\boxed{1}", normalized_prediction="1", gold="1")
        for i in range(3)
    ]
    rows = build_trajectory_rows(records)
    assert produce_retry_candidates(rows) == []


# ---------------------------------------------------------------------------
# write_trajectory_jsonl / write_retry_candidates
# ---------------------------------------------------------------------------

def test_write_trajectory_jsonl(tmp_path, mixed_records):
    rows = build_trajectory_rows(mixed_records)
    out = tmp_path / "trajectories.jsonl"
    write_trajectory_jsonl(rows, out)
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == len(rows)
    for line in lines:
        data = json.loads(line)
        assert "record" in data
        assert "error_type" in data
        assert "recoverability" in data


def test_write_trajectory_jsonl_creates_parent_dirs(tmp_path, mixed_records):
    rows = build_trajectory_rows(mixed_records[:2])
    out = tmp_path / "nested" / "dir" / "out.jsonl"
    write_trajectory_jsonl(rows, out)
    assert out.exists()


def test_write_retry_candidates(tmp_path, mixed_records):
    rows = build_trajectory_rows(mixed_records)
    candidates = produce_retry_candidates(rows)
    out = tmp_path / "retry_candidates.jsonl"
    write_retry_candidates(candidates, out)
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == len(candidates)
    for line in lines:
        data = json.loads(line)
        assert data["recoverability"] is True
