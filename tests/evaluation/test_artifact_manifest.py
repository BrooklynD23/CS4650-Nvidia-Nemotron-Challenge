"""Tests for :mod:`src.evaluation.artifacts`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.evaluation.artifacts import (
    is_immutable_golden_path,
    load_selection_manifest,
    write_selection_manifest,
)


def test_is_immutable_golden_path_accepts_versioned_names() -> None:
    assert is_immutable_golden_path(Path("data/eval/golden_v1.jsonl")) is True
    assert is_immutable_golden_path(Path("data/eval/golden_20.jsonl")) is True
    assert is_immutable_golden_path(Path("data/eval/golden_v12.jsonl")) is True


def test_is_immutable_golden_path_rejects_bare_name() -> None:
    assert is_immutable_golden_path(Path("data/eval/golden.jsonl")) is False


def test_is_immutable_golden_path_rejects_wrong_extension() -> None:
    assert is_immutable_golden_path(Path("data/eval/golden_v1.json")) is False
    assert is_immutable_golden_path(Path("data/eval/golden_v1.txt")) is False


def test_is_immutable_golden_path_rejects_wrong_stem_prefix() -> None:
    assert is_immutable_golden_path(Path("data/eval/val_v1.jsonl")) is False
    assert is_immutable_golden_path(Path("data/eval/golden.jsonl")) is False


def test_manifest_roundtrip_preserves_fields(tmp_path: Path) -> None:
    path = tmp_path / "golden_v1.manifest.json"
    write_selection_manifest(
        path=path,
        dataset_version="v0.1",
        selection_seed=42,
        selection_rule="stratified-by-category",
        row_count=20,
    )
    loaded = load_selection_manifest(path)
    assert loaded["dataset_version"] == "v0.1"
    assert loaded["selection_seed"] == 42
    assert loaded["selection_rule"] == "stratified-by-category"
    assert loaded["row_count"] == 20
    # Must be ISO-8601 parseable.
    parsed = datetime.fromisoformat(loaded["created_at"])
    assert parsed.tzinfo is not None  # UTC timezone present


def test_manifest_write_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "manifest.json"
    write_selection_manifest(
        path=path,
        dataset_version="v0.1",
        selection_seed=1,
        selection_rule="stratified-by-category",
        row_count=5,
    )
    assert path.exists()


def test_manifest_load_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"dataset_version": "v1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_selection_manifest(path)


def test_manifest_load_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        load_selection_manifest(path)


def test_manifest_write_rejects_bad_row_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="row_count must be >= 0"):
        write_selection_manifest(
            path=path,
            dataset_version="v0.1",
            selection_seed=1,
            selection_rule="stratified-by-category",
            row_count=-1,
        )
