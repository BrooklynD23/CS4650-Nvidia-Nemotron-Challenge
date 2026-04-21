"""Tests for :mod:`src.evaluation.reporting`.

Covers:
- Raw prediction + eval record JSONL roundtrip.
- Attribution validation (record fields must match the run config).
- ``write_run_artifacts`` writes all four artifacts and is reconstructable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contracts import EvalRecord
from src.evaluation.config import EvalRunConfig, make_run_config
from src.evaluation.reporting import (
    RawPrediction,
    read_eval_records_jsonl,
    read_predictions_jsonl,
    read_summary_json,
    write_eval_records_jsonl,
    write_predictions_jsonl,
    write_run_artifacts,
    write_summary_json,
)


def _config(**overrides: object) -> EvalRunConfig:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "model_id": "nemotron-x",
        "prompt_template_id": "tpl-v1",
        "normalizer_id": "exact_v1",
        "seed": 42,
        "split": "val",
        "dataset_version": "v0.1",
        "decode_config": {"temperature": 0.0},
    }
    defaults.update(overrides)
    return make_run_config(**defaults)  # type: ignore[arg-type]


def _record(
    *,
    example_id: str,
    gold: str = "A",
    prediction: str = "A",
    correct: bool | None = None,
    category: str = "binary",
    run_id: str = "run-1",
    normalizer_id: str = "exact_v1",
) -> EvalRecord:
    norm = prediction
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
        latency_ms=10.0,
        tokens_in=8,
        tokens_out=2,
        seed=42,
        decode_config={"temperature": 0.0},
    )


def _raw(example_id: str, prediction: str = "A") -> RawPrediction:
    return RawPrediction(
        run_id="run-1",
        example_id=example_id,
        model_id="nemotron-x",
        prompt_template_id="tpl-v1",
        seed=42,
        prediction=prediction,
        latency_ms=10.0,
        tokens_in=8,
        tokens_out=2,
    )


class TestRawPredictionRoundtrip:
    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        rows = [_raw("e-1"), _raw("e-2", prediction="B")]
        path = tmp_path / "predictions.jsonl"
        assert write_predictions_jsonl(rows, path) == 2
        loaded = read_predictions_jsonl(path)
        assert loaded == rows

    def test_rejects_non_rawprediction(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="RawPrediction"):
            write_predictions_jsonl([{"example_id": "x"}], tmp_path / "p.jsonl")  # type: ignore[list-item]


class TestEvalRecordRoundtrip:
    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        records = [_record(example_id="e-1"), _record(example_id="e-2", gold="B", prediction="B")]
        path = tmp_path / "eval_records.jsonl"
        assert write_eval_records_jsonl(records, path) == 2
        loaded = read_eval_records_jsonl(path)
        assert loaded == records


class TestAttributionInvariants:
    """Record attribution must match the run config snapshot."""

    def test_matching_attribution_passes(self) -> None:
        from src.evaluation.config import validate_records_against_config

        cfg = _config()
        validate_records_against_config([_record(example_id="e-1")], cfg)

    def test_record_missing_normalizer_id_rejected(self) -> None:
        with pytest.raises(TypeError, match="normalizer_id"):
            # normalizer_id is a required field on EvalRecord.
            EvalRecord(
                run_id="run-1",
                example_id="e-1",
                model_id="nemotron-x",
                prompt_template_id="tpl-v1",
                normalizer_id=None,  # type: ignore[arg-type]
                category="binary",
                split="val",
                gold="A",
                prediction="A",
                normalized_prediction="A",
                correct=True,
                latency_ms=10.0,
                tokens_in=8,
                tokens_out=2,
                seed=42,
                decode_config={},
            )

    def test_run_config_mismatch_rejected(self) -> None:
        from src.evaluation.config import validate_record_matches_config

        cfg = _config(run_id="run-A")
        rec = _record(example_id="e-1", run_id="run-B")
        with pytest.raises(ValueError, match="attribution does not match"):
            validate_record_matches_config(rec, cfg)

    def test_normalizer_id_mismatch_rejected(self) -> None:
        from src.evaluation.config import validate_record_matches_config

        cfg = _config(normalizer_id="exact_v1")
        rec = _record(example_id="e-1", normalizer_id="strip_v1")
        with pytest.raises(ValueError, match="normalizer_id"):
            validate_record_matches_config(rec, cfg)


class TestSummary:
    def test_summary_json_roundtrip(self, tmp_path: Path) -> None:
        summary = {"total": 2, "correct": 1, "accuracy": 0.5}
        path = tmp_path / "summary.json"
        write_summary_json(summary, path)
        loaded = read_summary_json(path)
        assert loaded == summary

    def test_summary_is_pretty_printed(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        write_summary_json({"a": 1, "b": 2}, path)
        text = path.read_text(encoding="utf-8")
        # pretty printing == multi-line output
        assert "\n" in text


class TestWriteRunArtifacts:
    def test_writes_all_four_files(self, tmp_path: Path) -> None:
        cfg = _config()
        records = [
            _record(example_id="e-1"),
            _record(example_id="e-2", gold="B", prediction="x", correct=False),
        ]
        raw_preds = [_raw("e-1"), _raw("e-2", prediction="x")]

        result = write_run_artifacts(
            output_dir=tmp_path,
            config=cfg,
            raw_predictions=raw_preds,
            records=records,
        )

        assert (tmp_path / "predictions.jsonl").exists()
        assert (tmp_path / "eval_records.jsonl").exists()
        assert (tmp_path / "run_config.json").exists()
        assert (tmp_path / "summary.json").exists()

        assert result["predictions"] == tmp_path / "predictions.jsonl"
        assert result["eval_records"] == tmp_path / "eval_records.jsonl"
        assert result["run_config"] == tmp_path / "run_config.json"
        assert result["summary"] == tmp_path / "summary.json"

    def test_record_level_and_summary_are_consistent(self, tmp_path: Path) -> None:
        cfg = _config()
        records = [
            _record(example_id="e-1"),
            _record(example_id="e-2", gold="B", prediction="x", correct=False),
        ]
        write_run_artifacts(
            output_dir=tmp_path,
            config=cfg,
            raw_predictions=[_raw("e-1"), _raw("e-2", prediction="x")],
            records=records,
        )
        summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
        assert summary["total"] == 2
        assert summary["correct"] == 1
        assert summary["accuracy"] == 0.5
        assert summary["run_id"] == cfg.run_id
        assert summary["model_id"] == cfg.model_id
        assert summary["prompt_template_id"] == cfg.prompt_template_id
        assert summary["normalizer_id"] == cfg.normalizer_id
        assert summary["seed"] == cfg.seed
        assert summary["split"] == cfg.split

    def test_summary_can_be_rebuilt_exactly_from_eval_records(
        self, tmp_path: Path
    ) -> None:
        from src.evaluation.scoring import score_records

        cfg = _config()
        records = [
            _record(example_id="e-1"),
            _record(example_id="e-2", gold="B", prediction="x", correct=False),
        ]
        write_run_artifacts(
            output_dir=tmp_path,
            config=cfg,
            raw_predictions=[_raw("e-1"), _raw("e-2", prediction="x")],
            records=records,
        )

        on_disk_summary = read_summary_json(tmp_path / "summary.json")
        rebuilt = score_records(
            read_eval_records_jsonl(tmp_path / "eval_records.jsonl")
        )
        assert on_disk_summary == rebuilt

    def test_rejects_records_with_mismatched_attribution(self, tmp_path: Path) -> None:
        cfg = _config(run_id="run-A")
        records = [_record(example_id="e-1", run_id="run-B")]
        with pytest.raises(ValueError, match="attribution does not match"):
            write_run_artifacts(
                output_dir=tmp_path,
                config=cfg,
                raw_predictions=[_raw("e-1")],
                records=records,
            )

    def test_records_written_before_summary(self, tmp_path: Path) -> None:
        """Partial failures must still leave records on disk.

        We simulate a summary-write failure by making summary.json a
        read-only directory before the call; the records file must
        still exist after the failure.
        """
        cfg = _config()
        records = [_record(example_id="e-1")]

        # Pre-create summary.json as a DIRECTORY so the file write fails.
        (tmp_path / "summary.json").mkdir()

        with pytest.raises((IsADirectoryError, PermissionError, OSError)):
            write_run_artifacts(
                output_dir=tmp_path,
                config=cfg,
                raw_predictions=[_raw("e-1")],
                records=records,
            )

        # Records + run config must already exist (ordering invariant).
        assert (tmp_path / "predictions.jsonl").exists()
        assert (tmp_path / "eval_records.jsonl").exists()
        assert (tmp_path / "run_config.json").exists()
