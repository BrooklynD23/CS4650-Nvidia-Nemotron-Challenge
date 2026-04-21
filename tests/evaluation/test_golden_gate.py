"""Tests for :mod:`src.evaluation.golden_gate`."""

from __future__ import annotations

from src.contracts import EvalRecord
from src.evaluation.golden_gate import (
    GoldenGateResult,
    evaluate_golden_gate,
    summarize_gate,
)
from src.evaluation.splits import SplitArtifactRow


def _golden_row(example_id: str, gold: str = "G", category: str = "binary") -> SplitArtifactRow:
    return SplitArtifactRow(
        example_id=example_id,
        category=category,
        prompt=f"prompt-{example_id}",
        gold=gold,
        source="test",
        split="golden",
        dataset_version="v0.1",
        selection_seed=42,
        selection_rule="stratified-by-category",
        selection_reason="golden:stratified-by-category; seed=42",
    )


def _record(
    example_id: str,
    *,
    gold: str = "G",
    prediction: str = "G",
    normalized_prediction: str | None = None,
    correct: bool = True,
    category: str = "binary",
) -> EvalRecord:
    return EvalRecord(
        run_id="run-1",
        example_id=example_id,
        model_id="nemotron-mini",
        prompt_template_id="tpl-v1",
        normalizer_id="norm-v1",
        category=category,
        split="golden",
        gold=gold,
        prediction=prediction,
        normalized_prediction=(
            normalized_prediction if normalized_prediction is not None else prediction
        ),
        correct=correct,
        latency_ms=10.0,
        tokens_in=16,
        tokens_out=2,
        seed=42,
        decode_config={"temperature": 0.0},
    )


def test_all_correct_records_pass() -> None:
    golden = [_golden_row("g-1"), _golden_row("g-2"), _golden_row("g-3")]
    records = [
        _record("g-1"),
        _record("g-2"),
        _record("g-3"),
    ]
    result = evaluate_golden_gate(records, golden)
    assert isinstance(result, GoldenGateResult)
    assert result.passed is True
    assert result.misses == []
    assert result.total == 3


def test_single_miss_fails_gate() -> None:
    golden = [_golden_row("g-1"), _golden_row("g-2")]
    records = [
        _record("g-1"),
        _record("g-2", prediction="wrong", correct=False),
    ]
    result = evaluate_golden_gate(records, golden)
    assert result.passed is False
    assert len(result.misses) == 1
    assert result.misses[0]["example_id"] == "g-2"
    assert result.misses[0]["reason"] == "incorrect"


def test_validation_gains_do_not_override_golden_miss() -> None:
    """Gate stays red even if overall validation accuracy improves.

    We simulate "high validation accuracy" by including lots of
    non-golden correct records alongside one failing golden record.
    The gate only cares about the golden rows.
    """
    golden = [_golden_row("g-1"), _golden_row("g-2"), _golden_row("g-3")]

    # 50 "validation" successes (unrelated example_ids) + all-but-one
    # golden correct.
    val_hits = [
        _record(f"val-{i}", category="binary", correct=True) for i in range(50)
    ]
    golden_records = [
        _record("g-1"),
        _record("g-2"),
        _record("g-3", prediction="nope", correct=False),
    ]
    result = evaluate_golden_gate(val_hits + golden_records, golden)
    assert result.passed is False
    assert len(result.misses) == 1
    assert result.misses[0]["example_id"] == "g-3"


def test_no_record_for_golden_example_is_a_miss() -> None:
    golden = [_golden_row("g-1"), _golden_row("g-missing")]
    records = [_record("g-1")]  # Nothing for "g-missing"
    result = evaluate_golden_gate(records, golden)
    assert result.passed is False
    assert len(result.misses) == 1
    assert result.misses[0]["example_id"] == "g-missing"
    assert result.misses[0]["reason"] == "missing"


def test_nondeterministic_records_flagged_as_miss() -> None:
    golden = [_golden_row("g-1")]
    # Two records, same id, same correct flag but different normalized outputs.
    r1 = _record("g-1", prediction="A", normalized_prediction="A")
    r2 = _record("g-1", prediction="A ", normalized_prediction="B")
    result = evaluate_golden_gate([r1, r2], golden)
    assert result.passed is False
    assert len(result.misses) == 1
    assert result.misses[0]["reason"] == "nondeterministic"
    # Normalized predictions are surfaced for debugging.
    assert "A" in result.misses[0]["normalized_prediction"]
    assert "B" in result.misses[0]["normalized_prediction"]


def test_summarize_gate_pass() -> None:
    result = GoldenGateResult(passed=True, total=3, misses=[])
    summary = summarize_gate(result)
    assert "PASS" in summary
    assert "3/3" in summary


def test_summarize_gate_fail_lists_misses() -> None:
    result = GoldenGateResult(
        passed=False,
        total=2,
        misses=[
            {
                "example_id": "g-1",
                "category": "binary",
                "gold": "10101",
                "normalized_prediction": "10100",
                "reason": "incorrect",
            }
        ],
    )
    summary = summarize_gate(result)
    assert "FAIL" in summary
    assert "g-1" in summary
    assert "incorrect" in summary
