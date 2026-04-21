"""Tests for the four canonical contracts in :mod:`src.contracts`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.contracts import (
    MANIFEST_VERSION,
    EvalRecord,
    PackageManifest,
    ReasoningExample,
    SFTExample,
)


# ---------------------------------------------------------------------------
# ReasoningExample
# ---------------------------------------------------------------------------


def test_reasoning_example_accepts_valid_input() -> None:
    ex = ReasoningExample(
        id="r1",
        category="binary",
        prompt="What is 10101 reversed?",
        answer="10101",
        source="kaggle:train.csv",
        split="train",
        metadata={"raw_row": 0},
    )
    assert ex.id == "r1"
    assert ex.metadata == {"raw_row": 0}


def test_reasoning_example_rejects_missing_required_field() -> None:
    with pytest.raises(TypeError):
        # Missing `answer`.
        ReasoningExample(  # type: ignore[call-arg]
            id="r1",
            category="binary",
            prompt="p",
            source="kaggle:train.csv",
            split="train",
            metadata={},
        )


def test_reasoning_example_rejects_wrong_types() -> None:
    with pytest.raises(TypeError):
        ReasoningExample(
            id=1,  # type: ignore[arg-type]
            category="binary",
            prompt="p",
            answer="a",
            source="kaggle:train.csv",
            split="train",
            metadata={},
        )


def test_reasoning_example_is_immutable() -> None:
    ex = ReasoningExample(
        id="r1",
        category="binary",
        prompt="p",
        answer="a",
        source="kaggle:train.csv",
        split="train",
        metadata={},
    )
    with pytest.raises(FrozenInstanceError):
        ex.id = "r2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SFTExample
# ---------------------------------------------------------------------------


def test_sft_example_accepts_valid_input() -> None:
    ex = SFTExample(
        example_id="r1",
        category="binary",
        messages=[{"role": "user", "content": "hi"}],
        completion="hello",
        source="kaggle:train.csv",
        split="train",
        provenance={"prompt_template_id": "tpl-v1", "source_example_id": "r1"},
    )
    assert ex.example_id == "r1"


def test_sft_example_rejects_missing_message_keys() -> None:
    with pytest.raises(ValueError):
        SFTExample(
            example_id="r1",
            category="binary",
            messages=[{"role": "user"}],  # missing "content"
            completion="hello",
            source="kaggle:train.csv",
            split="train",
            provenance={},
        )


def test_sft_example_rejects_non_string_message_values() -> None:
    with pytest.raises(TypeError, match="messages\\[0\\]\\['role'\\]"):
        SFTExample(
            example_id="r1",
            category="binary",
            messages=[{"role": 1, "content": "hi"}],  # type: ignore[list-item]
            completion="hello",
            source="kaggle:train.csv",
            split="train",
            provenance={},
        )


def test_sft_example_is_immutable() -> None:
    ex = SFTExample(
        example_id="r1",
        category="binary",
        messages=[{"role": "user", "content": "hi"}],
        completion="hello",
        source="kaggle:train.csv",
        split="train",
        provenance={},
    )
    with pytest.raises(FrozenInstanceError):
        ex.completion = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalRecord
# ---------------------------------------------------------------------------


def _valid_eval_record_kwargs() -> dict:
    return dict(
        run_id="run-1",
        example_id="r1",
        model_id="nemotron-mini",
        prompt_template_id="tpl-v1",
        normalizer_id="norm-v1",
        category="binary",
        split="val",
        gold="10101",
        prediction="10101",
        normalized_prediction="10101",
        correct=True,
        latency_ms=123.4,
        tokens_in=32,
        tokens_out=8,
        seed=42,
        decode_config={"temperature": 0.0, "top_p": 1.0},
    )


def test_eval_record_accepts_valid_input() -> None:
    record = EvalRecord(**_valid_eval_record_kwargs())
    assert record.normalizer_id == "norm-v1"
    assert record.correct is True


def test_eval_record_rejects_missing_required_field() -> None:
    kwargs = _valid_eval_record_kwargs()
    del kwargs["normalizer_id"]
    with pytest.raises(TypeError):
        EvalRecord(**kwargs)


def test_eval_record_rejects_wrong_type() -> None:
    kwargs = _valid_eval_record_kwargs()
    kwargs["correct"] = "yes"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        EvalRecord(**kwargs)


def test_eval_record_is_immutable() -> None:
    record = EvalRecord(**_valid_eval_record_kwargs())
    with pytest.raises(FrozenInstanceError):
        record.correct = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PackageManifest
# ---------------------------------------------------------------------------


def _valid_manifest_kwargs() -> dict:
    return dict(
        manifest_version=MANIFEST_VERSION,
        base_model_id="nemotron-mini",
        adapter_rank=16,
        dataset_version="v0.1",
        eval_sha="abc123",
        artifact_paths={"adapter": "adapters/v1/"},
        created_at="2026-04-20T00:00:00+00:00",
    )


def test_package_manifest_accepts_valid_input() -> None:
    manifest = PackageManifest(**_valid_manifest_kwargs())
    assert manifest.manifest_version == MANIFEST_VERSION


def test_package_manifest_rejects_bad_artifact_paths() -> None:
    kwargs = _valid_manifest_kwargs()
    kwargs["artifact_paths"] = {"adapter": 123}  # non-str value
    with pytest.raises(TypeError):
        PackageManifest(**kwargs)


def test_package_manifest_is_immutable() -> None:
    manifest = PackageManifest(**_valid_manifest_kwargs())
    with pytest.raises(FrozenInstanceError):
        manifest.adapter_rank = 32  # type: ignore[misc]
