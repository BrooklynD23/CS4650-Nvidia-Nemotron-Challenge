"""Tests for :mod:`src.evaluation.splits`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts import ReasoningExample
from src.evaluation.splits import (
    SplitArtifactRow,
    read_split_jsonl,
    reserve_splits,
    write_split_jsonl,
)


def _ex(
    example_id: str,
    *,
    category: str = "binary",
    prompt: str = "p",
    answer: str = "a",
    source: str = "test",
    split: str = "train",
) -> ReasoningExample:
    """Synthetic :class:`ReasoningExample` builder."""
    return ReasoningExample(
        id=example_id,
        category=category,
        prompt=prompt,
        answer=answer,
        source=source,
        split=split,
        metadata={},
    )


def _corpus(n_per_category: int = 5) -> list[ReasoningExample]:
    """Deterministic corpus with 3 categories, N rows each."""
    rows: list[ReasoningExample] = []
    for cat in ("binary", "cipher", "logic"):
        for i in range(n_per_category):
            rows.append(
                _ex(
                    f"{cat}-{i}",
                    category=cat,
                    prompt=f"{cat} prompt {i}",
                    answer=f"{cat}-ans-{i}",
                )
            )
    return rows


def test_reserve_splits_is_deterministic_for_fixed_seed() -> None:
    corpus = _corpus()
    val1, gold1 = reserve_splits(
        corpus,
        validation_size=6,
        golden_size=3,
        seed=42,
        dataset_version="v0.1",
    )
    val2, gold2 = reserve_splits(
        corpus,
        validation_size=6,
        golden_size=3,
        seed=42,
        dataset_version="v0.1",
    )
    assert [r.example_id for r in val1] == [r.example_id for r in val2]
    assert [r.example_id for r in gold1] == [r.example_id for r in gold2]
    # And full equality of rows, not just ids.
    assert val1 == val2
    assert gold1 == gold2


def test_reserve_splits_different_seeds_differ() -> None:
    corpus = _corpus()
    _, gold_a = reserve_splits(
        corpus,
        validation_size=3,
        golden_size=3,
        seed=1,
        dataset_version="v0.1",
    )
    _, gold_b = reserve_splits(
        corpus,
        validation_size=3,
        golden_size=3,
        seed=9999,
        dataset_version="v0.1",
    )
    # It is technically possible for two seeds to produce the same order,
    # but extremely unlikely on this synthetic corpus.
    assert [r.example_id for r in gold_a] != [r.example_id for r in gold_b]


def test_reserve_splits_rejects_duplicate_example_id() -> None:
    dupes = [_ex("dup-1"), _ex("dup-1", category="cipher")]
    with pytest.raises(ValueError, match="duplicate example_id"):
        reserve_splits(
            dupes,
            validation_size=1,
            golden_size=1,
            seed=1,
            dataset_version="v0.1",
        )


def test_reserve_splits_rejects_missing_category() -> None:
    corpus = _corpus()
    # Sneak in a row with empty category.
    corpus.append(
        ReasoningExample(
            id="empty-cat",
            category="",
            prompt="p",
            answer="a",
            source="test",
            split="train",
            metadata={},
        )
    )
    with pytest.raises(ValueError, match="missing 'category'"):
        reserve_splits(
            corpus,
            validation_size=2,
            golden_size=1,
            seed=1,
            dataset_version="v0.1",
        )


def test_reserve_splits_rejects_missing_prompt() -> None:
    corpus = _corpus()
    corpus.append(
        ReasoningExample(
            id="empty-prompt",
            category="binary",
            prompt="",
            answer="a",
            source="test",
            split="train",
            metadata={},
        )
    )
    with pytest.raises(ValueError, match="missing 'prompt'"):
        reserve_splits(
            corpus,
            validation_size=2,
            golden_size=1,
            seed=1,
            dataset_version="v0.1",
        )


def test_reserve_splits_val_and_golden_are_disjoint() -> None:
    corpus = _corpus()
    val, gold = reserve_splits(
        corpus,
        validation_size=6,
        golden_size=3,
        seed=7,
        dataset_version="v0.1",
    )
    val_ids = {r.example_id for r in val}
    gold_ids = {r.example_id for r in gold}
    assert val_ids.isdisjoint(gold_ids)

    # Remaining rows in the corpus (the implicit "train") must not
    # contain any reserved id either.
    train_ids = {
        ex.id for ex in corpus if ex.id not in val_ids and ex.id not in gold_ids
    }
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(gold_ids)


def test_reserve_splits_covers_all_categories_when_possible() -> None:
    corpus = _corpus(n_per_category=3)  # 3 categories x 3 rows = 9 rows total
    val, gold = reserve_splits(
        corpus,
        validation_size=3,
        golden_size=3,
        seed=11,
        dataset_version="v0.1",
    )
    val_cats = {r.category for r in val}
    gold_cats = {r.category for r in gold}
    assert val_cats == {"binary", "cipher", "logic"}
    assert gold_cats == {"binary", "cipher", "logic"}


def test_reserve_splits_validates_sizes() -> None:
    corpus = _corpus(n_per_category=1)  # 3 rows total
    with pytest.raises(ValueError, match="validation_size must be >= 1"):
        reserve_splits(
            corpus,
            validation_size=0,
            golden_size=1,
            seed=1,
            dataset_version="v0.1",
        )
    with pytest.raises(ValueError, match="golden_size must be >= 1"):
        reserve_splits(
            corpus,
            validation_size=1,
            golden_size=0,
            seed=1,
            dataset_version="v0.1",
        )
    with pytest.raises(ValueError, match="exceeds corpus size"):
        reserve_splits(
            corpus,
            validation_size=5,
            golden_size=5,
            seed=1,
            dataset_version="v0.1",
        )


def test_reserve_splits_stamps_provenance_fields() -> None:
    corpus = _corpus()
    val, gold = reserve_splits(
        corpus,
        validation_size=3,
        golden_size=3,
        seed=42,
        dataset_version="v0.2",
    )
    for row in val:
        assert row.dataset_version == "v0.2"
        assert row.selection_seed == 42
        assert row.selection_rule == "stratified-by-category"
        assert row.selection_reason == "val:stratified-by-category; seed=42"
        assert row.split == "val"
    for row in gold:
        assert row.split == "golden"
        assert row.selection_reason == "golden:stratified-by-category; seed=42"


def test_jsonl_roundtrip_preserves_rows(tmp_path: Path) -> None:
    corpus = _corpus()
    val, gold = reserve_splits(
        corpus,
        validation_size=4,
        golden_size=2,
        seed=3,
        dataset_version="v0.1",
    )
    val_path = tmp_path / "validation_4.jsonl"
    gold_path = tmp_path / "golden_v1.jsonl"

    assert write_split_jsonl(val, val_path) == len(val)
    assert write_split_jsonl(gold, gold_path) == len(gold)

    loaded_val = read_split_jsonl(val_path)
    loaded_gold = read_split_jsonl(gold_path)
    assert loaded_val == val
    assert loaded_gold == gold
    assert all(isinstance(r, SplitArtifactRow) for r in loaded_val)
