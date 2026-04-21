"""Tests for :mod:`src.data.schema_mapping`."""

from __future__ import annotations

import pytest

from src.contracts import ReasoningExample, SFTExample
from src.data.schema_mapping import (
    ALIAS_MAP,
    reasoning_example_from_row,
    sft_example_from_reasoning,
)


def test_alias_map_collapses_question_and_expected_answer() -> None:
    assert ALIAS_MAP["question"] == "prompt"
    assert ALIAS_MAP["expected_answer"] == "answer"


def test_reasoning_example_from_row_with_aliases() -> None:
    row = {
        "id": "42",
        "category": "binary",
        "question": "Reverse 10101",
        "expected_answer": "10101",
    }
    ex = reasoning_example_from_row(
        row, source="kaggle:train.csv", split="train"
    )
    assert isinstance(ex, ReasoningExample)
    assert ex.id == "42"
    assert ex.prompt == "Reverse 10101"
    assert ex.answer == "10101"
    assert ex.category == "binary"
    assert ex.source == "kaggle:train.csv"
    assert ex.split == "train"


def test_reasoning_example_from_row_preserves_unknown_columns_in_metadata() -> None:
    row = {
        "id": "7",
        "prompt": "p",
        "answer": "a",
        "category": "arith",
        "source_file_row": 12,  # unknown column
        "difficulty": "hard",  # unknown column
    }
    ex = reasoning_example_from_row(row, source="mirror:hf", split="val")
    assert ex.metadata["source_file_row"] == 12
    assert ex.metadata["difficulty"] == "hard"


def test_reasoning_example_from_row_missing_required_raises_value_error() -> None:
    row = {
        "id": "1",
        "question": "p",
        # missing expected_answer / answer
    }
    with pytest.raises(ValueError, match="missing required field"):
        reasoning_example_from_row(
            row, source="kaggle:train.csv", split="train"
        )


def test_reasoning_example_from_row_custom_alias_map() -> None:
    row = {"id": "1", "q": "the prompt", "ans": "the answer"}
    ex = reasoning_example_from_row(
        row,
        source="s",
        split="train",
        alias_map={"q": "prompt", "ans": "answer"},
    )
    assert ex.prompt == "the prompt"
    assert ex.answer == "the answer"


def test_reasoning_example_from_row_infers_category_when_missing() -> None:
    row = {"id": "1", "prompt": "p", "answer": "a"}
    ex = reasoning_example_from_row(
        row, source="kaggle:test.csv", split="test"
    )
    # Falls back to "unknown" when no category info is present.
    assert ex.category == "unknown"


def test_reasoning_example_from_row_rejects_non_mapping_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        reasoning_example_from_row(
            {"id": "1", "prompt": "p", "answer": "a", "metadata": 0},
            source="kaggle:train.csv",
            split="train",
        )


def test_reasoning_example_from_row_rejects_missing_sentinel_text() -> None:
    with pytest.raises(ValueError, match="missing/empty sentinel"):
        reasoning_example_from_row(
            {"id": "1", "prompt": "p", "answer": "nan"},
            source="kaggle:train.csv",
            split="train",
        )


def test_sft_example_from_reasoning_mapping_preserves_ids_and_provenance() -> None:
    base = ReasoningExample(
        id="src-1",
        category="binary",
        prompt="p",
        answer="a",
        source="kaggle:train.csv",
        split="train",
        metadata={},
    )
    sft = sft_example_from_reasoning(
        base,
        messages=[
            {"role": "system", "content": "solve"},
            {"role": "user", "content": "p"},
        ],
        completion="a",
        prompt_template_id="tpl-v1",
    )
    assert isinstance(sft, SFTExample)
    assert sft.example_id == base.id
    assert sft.category == base.category
    assert sft.split == base.split
    assert sft.source == base.source  # defaults from example
    assert sft.provenance == {
        "prompt_template_id": "tpl-v1",
        "source_example_id": "src-1",
    }


def test_sft_example_from_reasoning_allows_source_override() -> None:
    base = ReasoningExample(
        id="src-1",
        category="binary",
        prompt="p",
        answer="a",
        source="kaggle:train.csv",
        split="train",
        metadata={},
    )
    sft = sft_example_from_reasoning(
        base,
        messages=[{"role": "user", "content": "p"}],
        completion="a",
        prompt_template_id="tpl-v1",
        source="teacher:gpt4-distill",
    )
    assert sft.source == "teacher:gpt4-distill"
