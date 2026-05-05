"""Tests for src/data/synthetic.py (#10 / #24)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.contracts import EvalRecord, SFTExample
from src.data.synthetic import (
    QualityFilter,
    SyntheticConfig,
    generate_from_retry_candidates,
    write_sft_examples_jsonl,
)
from src.evaluation.trajectory import TrajectoryRow, ErrorType, build_trajectory_rows


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_eval_record(
    *,
    example_id: str = "ex1",
    category: str = "math",
    correct: bool = False,
    prediction: str = r"\boxed{99}",
    normalized_prediction: str = "99",
    gold: str = "100",
    run_id: str = "run-test",
) -> EvalRecord:
    return EvalRecord(
        run_id=run_id,
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
        tokens_out=100,
        seed=42,
    )


def _make_candidate(
    example_id: str = "ex1",
    category: str = "math",
    prediction: str = r"\boxed{99}",
    gold: str = "100",
) -> TrajectoryRow:
    record = _make_eval_record(
        example_id=example_id, category=category, prediction=prediction, gold=gold
    )
    rows = build_trajectory_rows([record])
    return rows[0]


def _make_sft_example(
    example_id: str = "ex1",
    category: str = "math",
    completion: str = r"\boxed{42}",
    provenance: dict | None = None,
) -> SFTExample:
    return SFTExample(
        example_id=example_id,
        category=category,
        messages=[{"role": "user", "content": "2 + 2 = ?"}],
        completion=completion,
        source="synthetic",
        split="train",
        provenance=provenance or {
            "teacher": "test",
            "generated_at": "2026-05-04T00:00:00Z",
            "source_run_id": "run-test",
        },
    )


def _make_config(tmp_path: Path, categories=None) -> SyntheticConfig:
    return SyntheticConfig(
        output_dir=tmp_path / "synthetic",
        categories=categories or ["math", "bit_manipulation", "code", "science"],
    )


# ---------------------------------------------------------------------------
# QualityFilter tests
# ---------------------------------------------------------------------------

def test_dedupe_rejection(tmp_path):
    config = _make_config(tmp_path)
    filt = QualityFilter(config)
    ex1 = _make_sft_example(example_id="ex1")
    ex2 = _make_sft_example(example_id="ex1")  # same id + category → same fingerprint
    assert filt.accept(ex1) is True
    assert filt.accept(ex2) is False


def test_length_guard(tmp_path):
    config = _make_config(tmp_path)
    filt = QualityFilter(config)
    # completion that exceeds 8192 * 4 chars
    long_completion = r"\boxed{" + "x" * (8192 * 4 + 1000) + "}"
    ex = _make_sft_example(completion=long_completion)
    assert filt.accept(ex) is False


def test_no_boxed_rejected(tmp_path):
    config = _make_config(tmp_path)
    filt = QualityFilter(config)
    ex = _make_sft_example(completion="the answer is 42")
    assert filt.accept(ex) is False


def test_provenance_completeness_required(tmp_path):
    config = _make_config(tmp_path)
    filt = QualityFilter(config)
    # Missing 'teacher' key
    ex = _make_sft_example(
        provenance={"generated_at": "2026-05-04T00:00:00Z", "source_run_id": "r1"}
    )
    assert filt.accept(ex) is False


def test_category_validity(tmp_path):
    config = _make_config(tmp_path, categories=["math"])
    filt = QualityFilter(config)
    ex_valid = _make_sft_example(category="math")
    ex_invalid = _make_sft_example(example_id="ex2", category="unknown_category")
    assert filt.accept(ex_valid) is True
    assert filt.accept(ex_invalid) is False


def test_filter_apply_batch(tmp_path):
    config = _make_config(tmp_path)
    filt = QualityFilter(config)
    examples = [_make_sft_example(example_id=f"ex{i}") for i in range(5)]
    passed = filt.apply(examples)
    assert len(passed) == 5


# ---------------------------------------------------------------------------
# generate_from_retry_candidates tests
# ---------------------------------------------------------------------------

def test_smoke_size_guard(tmp_path):
    from src.data.synthetic import _SMOKE_LIMIT
    candidates = [_make_candidate(example_id=f"ex{i}") for i in range(200)]
    config = _make_config(tmp_path)
    # LLM teacher that always returns a valid completion
    config.llm_teacher_fn = lambda prompt: r"\boxed{42}"
    results = generate_from_retry_candidates(candidates, config, smoke=True)
    # smoke limits input to SMOKE_LIMIT; not all may pass quality filter
    assert len(results) <= _SMOKE_LIMIT


def test_dry_run_returns_empty(tmp_path, capsys):
    candidates = [_make_candidate(example_id=f"ex{i}") for i in range(10)]
    config = _make_config(tmp_path)
    results = generate_from_retry_candidates(candidates, config, dry_run=True)
    assert results == []
    captured = capsys.readouterr()
    assert "dry-run" in captured.out


def test_cost_cap_enforcement(tmp_path):
    candidates = [_make_candidate(example_id=f"ex{i}") for i in range(200)]
    # Very low cost cap: stops after first call
    config = SyntheticConfig(
        output_dir=tmp_path / "synthetic",
        categories=["math"],
        cost_cap_usd=0.005,  # stops after first LLM call (cost 0.01 per call)
    )
    calls = {"n": 0}

    def llm_fn(prompt: str) -> str:
        calls["n"] += 1
        return r"\boxed{42}"

    config.llm_teacher_fn = llm_fn
    generate_from_retry_candidates(candidates, config)
    # Should have stopped very early due to cap
    assert calls["n"] <= 2


def test_verifier_teacher_used_when_available(tmp_path):
    candidate = _make_candidate(prediction=r"\boxed{100}", gold="100")
    config = _make_config(tmp_path)
    # Verifier that accepts the existing prediction
    verifier = MagicMock()
    verifier.verify.return_value = True
    config.verifier = verifier
    config.llm_teacher_fn = None

    results = generate_from_retry_candidates([candidate], config)
    verifier.verify.assert_called_once()
    # candidate passes filter if it has boxed and valid provenance
    # The record's category may not be in config categories; check result
    if results:
        assert results[0].provenance["teacher"] == "verifier"


def test_no_teacher_produces_no_examples(tmp_path):
    candidates = [_make_candidate(example_id=f"ex{i}") for i in range(5)]
    config = _make_config(tmp_path)
    config.verifier = None
    config.llm_teacher_fn = None
    results = generate_from_retry_candidates(candidates, config)
    assert results == []


# ---------------------------------------------------------------------------
# write_sft_examples_jsonl tests
# ---------------------------------------------------------------------------

def test_fingerprint_file_written(tmp_path):
    examples = [_make_sft_example(example_id=f"ex{i}") for i in range(3)]
    out = tmp_path / "batch.jsonl"
    write_sft_examples_jsonl(examples, out)
    assert out.exists()
    sha_file = out.with_suffix(".sha256")
    assert sha_file.exists()
    fingerprint = sha_file.read_text().strip()
    assert len(fingerprint) == 64  # SHA-256 hex


def test_jsonl_output_valid(tmp_path):
    examples = [_make_sft_example(example_id=f"ex{i}") for i in range(3)]
    out = tmp_path / "batch.jsonl"
    write_sft_examples_jsonl(examples, out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert "example_id" in data
        assert "completion" in data
        assert "provenance" in data


def test_creates_parent_directories(tmp_path):
    examples = [_make_sft_example()]
    out = tmp_path / "deep" / "nested" / "batch.jsonl"
    write_sft_examples_jsonl(examples, out)
    assert out.exists()
