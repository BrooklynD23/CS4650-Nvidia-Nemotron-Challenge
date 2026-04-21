"""Tests for :mod:`src.data.schema_eda`."""

from __future__ import annotations

from src.contracts import ReasoningExample
from src.data.schema_eda import (
    detect_duplicates_by_id,
    detect_missing_fields,
    summarize_examples,
)


def _make(
    id_: str,
    category: str = "binary",
    source: str = "kaggle:train.csv",
    split: str = "train",
) -> ReasoningExample:
    return ReasoningExample(
        id=id_,
        category=category,
        prompt="p",
        answer="a",
        source=source,
        split=split,
        metadata={},
    )


def test_summarize_examples_counts() -> None:
    examples = [
        _make("1", category="binary", split="train"),
        _make("2", category="binary", split="val"),
        _make("3", category="arith", split="train"),
    ]
    summary = summarize_examples(examples)
    assert summary["total"] == 3
    assert summary["by_category"] == {"binary": 2, "arith": 1}
    assert summary["by_split"] == {"train": 2, "val": 1}
    assert summary["by_source"] == {"kaggle:train.csv": 3}
    assert summary["sample_ids"] == ["1", "2", "3"]


def test_summarize_examples_empty() -> None:
    summary = summarize_examples([])
    assert summary["total"] == 0
    assert summary["by_category"] == {}
    assert summary["sample_ids"] == []


def test_summarize_examples_caps_sample_ids_at_three() -> None:
    examples = [_make(str(i)) for i in range(10)]
    summary = summarize_examples(examples)
    assert summary["sample_ids"] == ["0", "1", "2"]


def test_detect_duplicates_by_id() -> None:
    examples = [
        _make("a"),
        _make("b"),
        _make("a"),
        _make("c"),
        _make("b"),
    ]
    dupes = detect_duplicates_by_id(examples)
    # Preserve first-seen order.
    assert dupes == ["a", "b"]


def test_detect_duplicates_by_id_none() -> None:
    examples = [_make("a"), _make("b"), _make("c")]
    assert detect_duplicates_by_id(examples) == []


def test_detect_missing_fields_reports_missing() -> None:
    rows = [
        {"id": "1", "prompt": "p", "answer": "a"},
        {"id": "2", "prompt": "p"},  # missing answer
        {"prompt": "p", "answer": "a"},  # missing id
        {"id": "3", "prompt": None, "answer": "a"},  # prompt is None
    ]
    reports = detect_missing_fields(rows, required=("id", "prompt", "answer"))
    assert reports == [
        {"row_index": 1, "missing": ["answer"]},
        {"row_index": 2, "missing": ["id"]},
        {"row_index": 3, "missing": ["prompt"]},
    ]


def test_detect_missing_fields_returns_empty_when_all_present() -> None:
    rows = [{"id": "1", "prompt": "p", "answer": "a"}]
    assert detect_missing_fields(rows, required=("id", "prompt", "answer")) == []
