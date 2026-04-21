"""Tests for :mod:`src.evaluation.records`."""

from __future__ import annotations

from pathlib import Path

from src.contracts import EvalRecord
from src.evaluation.records import (
    make_eval_record,
    read_eval_records_jsonl,
    write_eval_records_jsonl,
)


def _sample_record(example_id: str = "r1", correct: bool = True) -> EvalRecord:
    return make_eval_record(
        run_id="run-1",
        example_id=example_id,
        model_id="nemotron-mini",
        prompt_template_id="tpl-v1",
        normalizer_id="norm-v1",
        category="binary",
        split="val",
        gold="10101",
        prediction="10101",
        normalized_prediction="10101",
        correct=correct,
        latency_ms=100.0,
        tokens_in=32,
        tokens_out=8,
        seed=42,
        decode_config={"temperature": 0.0, "top_p": 1.0},
    )


def test_make_eval_record_returns_valid_dataclass() -> None:
    record = _sample_record()
    assert isinstance(record, EvalRecord)
    assert record.normalizer_id == "norm-v1"
    assert record.decode_config == {"temperature": 0.0, "top_p": 1.0}


def test_jsonl_roundtrip_preserves_records(tmp_path: Path) -> None:
    records = [
        _sample_record("r1", correct=True),
        _sample_record("r2", correct=False),
        _sample_record("r3", correct=True),
    ]
    path = tmp_path / "eval.jsonl"

    count = write_eval_records_jsonl(records, path)
    assert count == 3

    loaded = read_eval_records_jsonl(path)
    assert len(loaded) == 3
    assert loaded == records


def test_jsonl_preserves_normalizer_id(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    record = _sample_record()
    write_eval_records_jsonl([record], path)
    loaded = read_eval_records_jsonl(path)
    assert loaded[0].normalizer_id == "norm-v1"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "eval.jsonl"
    write_eval_records_jsonl([_sample_record()], path)
    assert path.exists()


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_eval_records_jsonl([_sample_record()], path)
    # Append a blank line and ensure reader tolerates it.
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    loaded = read_eval_records_jsonl(path)
    assert len(loaded) == 1
