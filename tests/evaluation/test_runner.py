"""End-to-end pipeline integration tests for :mod:`src.evaluation.runner`.

The runner is exercised with stub predictors so tests can run without
any model or dataset dependencies. Fixtures are tiny frozen
:class:`SplitArtifactRow` lists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.config import make_run_config
from src.evaluation.reporting import read_eval_records_jsonl, read_summary_json
from src.evaluation.runner import (
    PredictRequest,
    PredictResponse,
    RunResult,
    run_baseline_eval,
)
from src.evaluation.splits import SplitArtifactRow


def _row(
    example_id: str,
    gold: str = "A",
    *,
    category: str = "binary",
    prompt: str = "prompt",
    split: str = "val",
) -> SplitArtifactRow:
    return SplitArtifactRow(
        example_id=example_id,
        category=category,
        prompt=prompt,
        gold=gold,
        source="test",
        split=split,
        dataset_version="v0.1",
        selection_seed=42,
        selection_rule="stratified-by-category",
        selection_reason=f"{split}:stratified-by-category; seed=42",
    )


def _config(normalizer_id: str = "exact_v1", split: str = "val"):
    return make_run_config(
        run_id="run-1",
        model_id="stub",
        prompt_template_id="tpl-v1",
        normalizer_id=normalizer_id,
        seed=42,
        split=split,
        dataset_version="v0.1",
        decode_config={"temperature": 0.0},
    )


def _echo_predictor(request: PredictRequest) -> PredictResponse:
    """Stub predictor that returns the category and gold concatenated.

    In tests we wire the gold answer into ``prompt`` so a gold-perfect
    predictor is easy to build without a real model.
    """
    # Convention: the test treats the row's prompt as "answer: <gold>"
    # so a perfect predictor echoes the suffix.
    prediction = request.prompt.removeprefix("answer:").strip()
    return PredictResponse(
        prediction=prediction,
        latency_ms=1.0,
        tokens_in=len(request.prompt),
        tokens_out=len(prediction),
    )


class TestGoldenPath:
    def test_perfect_predictor_yields_100pct(self, tmp_path: Path) -> None:
        rows = [
            _row("e-1", gold="A", prompt="answer:A"),
            _row("e-2", gold="B", prompt="answer:B", category="cipher"),
        ]
        result = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=_echo_predictor,
            output_dir=tmp_path,
        )
        assert isinstance(result, RunResult)
        assert result.summary["total"] == 2
        assert result.summary["correct"] == 2
        assert result.summary["accuracy"] == 1.0
        # Artifacts are on disk.
        assert (tmp_path / "predictions.jsonl").exists()
        assert (tmp_path / "eval_records.jsonl").exists()
        assert (tmp_path / "run_config.json").exists()
        assert (tmp_path / "summary.json").exists()

    def test_no_output_dir_skips_artifact_writing(self) -> None:
        rows = [_row("e-1", gold="A", prompt="answer:A")]
        result = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=_echo_predictor,
        )
        assert result.summary["accuracy"] == 1.0
        assert result.records[0].example_id == "e-1"
        assert result.raw_predictions[0].example_id == "e-1"


class TestDriftScenarios:
    def test_same_raw_prediction_scores_differently_across_normalizers(
        self, tmp_path: Path
    ) -> None:
        """Exercise: "A\\n" vs gold "A" — strict fails, strip passes."""

        def predictor_with_newline(req: PredictRequest) -> PredictResponse:
            return PredictResponse(
                prediction="A\n",
                latency_ms=1.0,
                tokens_in=1,
                tokens_out=2,
            )

        rows = [_row("e-1", gold="A", prompt="anything")]

        strict = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="exact_v1"),
            predictor=predictor_with_newline,
        )
        permissive = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=predictor_with_newline,
        )

        assert strict.summary["accuracy"] == 0.0
        assert permissive.summary["accuracy"] == 1.0
        # And the normalizer_id is surfaced on every record.
        assert strict.records[0].normalizer_id == "exact_v1"
        assert permissive.records[0].normalizer_id == "strip_v1"

    def test_reasoning_prefix_passes_only_under_last_line_version(self) -> None:
        def reasoning_predictor(req: PredictRequest) -> PredictResponse:
            return PredictResponse(
                prediction="reasoning...\nA",
                latency_ms=1.0,
                tokens_in=1,
                tokens_out=3,
            )

        rows = [_row("e-1", gold="A", prompt="anything")]

        strict = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="exact_v1"),
            predictor=reasoning_predictor,
        )
        stripped = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=reasoning_predictor,
        )
        tail = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="last_line_v1"),
            predictor=reasoning_predictor,
        )

        assert strict.summary["accuracy"] == 0.0
        assert stripped.summary["accuracy"] == 0.0
        assert tail.summary["accuracy"] == 1.0

    def test_blank_prediction_is_always_incorrect(self) -> None:
        def blank_predictor(req: PredictRequest) -> PredictResponse:
            return PredictResponse(
                prediction="", latency_ms=1.0, tokens_in=0, tokens_out=0
            )

        rows = [_row("e-1", gold="A", prompt="x")]
        for norm_id in ("exact_v1", "strip_v1", "last_line_v1", "collapse_ws_v1"):
            result = run_baseline_eval(
                split_rows=rows,
                config=_config(normalizer_id=norm_id),
                predictor=blank_predictor,
            )
            assert result.summary["accuracy"] == 0.0, norm_id


class TestAttribution:
    def test_record_attribution_matches_run_config(self) -> None:
        rows = [_row("e-1", gold="A", prompt="answer:A")]
        cfg = _config(normalizer_id="strip_v1")
        result = run_baseline_eval(
            split_rows=rows,
            config=cfg,
            predictor=_echo_predictor,
        )
        rec = result.records[0]
        assert rec.run_id == cfg.run_id
        assert rec.model_id == cfg.model_id
        assert rec.prompt_template_id == cfg.prompt_template_id
        assert rec.normalizer_id == cfg.normalizer_id
        assert rec.seed == cfg.seed
        assert rec.split == cfg.split
        # decode_config is copied verbatim into each record.
        assert rec.decode_config == cfg.decode_config

    def test_rejects_rows_from_other_split(self) -> None:
        rows = [_row("e-1", gold="A", prompt="answer:A", split="golden")]
        with pytest.raises(ValueError, match="split"):
            run_baseline_eval(
                split_rows=rows,
                config=_config(split="val"),
                predictor=_echo_predictor,
            )

    def test_rejects_duplicate_example_ids(self) -> None:
        rows = [
            _row("dup", gold="A", prompt="answer:A"),
            _row("dup", gold="A", prompt="answer:A"),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            run_baseline_eval(
                split_rows=rows,
                config=_config(normalizer_id="strip_v1"),
                predictor=_echo_predictor,
            )


class TestDeterminism:
    """Rerunning with the same seed + normalizer yields byte-stable outputs."""

    def test_stable_output_across_runs(self, tmp_path: Path) -> None:
        rows = [
            _row("e-1", gold="A", prompt="answer:A"),
            _row("e-2", gold="B", prompt="answer:B", category="cipher"),
        ]
        cfg = _config(normalizer_id="strip_v1")

        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        run_baseline_eval(
            split_rows=rows,
            config=cfg,
            predictor=_echo_predictor,
            output_dir=first_dir,
        )
        run_baseline_eval(
            split_rows=rows,
            config=cfg,
            predictor=_echo_predictor,
            output_dir=second_dir,
        )

        first_records = read_eval_records_jsonl(first_dir / "eval_records.jsonl")
        second_records = read_eval_records_jsonl(second_dir / "eval_records.jsonl")
        assert first_records == second_records

        first_summary = read_summary_json(first_dir / "summary.json")
        second_summary = read_summary_json(second_dir / "summary.json")
        assert first_summary == second_summary


class TestIntegration:
    """End-to-end acceptance: the pipeline separates model and normalizer."""

    def test_model_vs_normalizer_differences_are_distinguishable(self) -> None:
        """A normalizer change and a model change both move accuracy,
        but the record-level ``normalizer_id`` and ``prediction`` make
        the source obvious.
        """

        def sloppy_predictor(req: PredictRequest) -> PredictResponse:
            return PredictResponse(
                prediction=f"{req.prompt.removeprefix('answer:').strip()} ",
                latency_ms=1.0,
                tokens_in=1,
                tokens_out=1,
            )

        rows = [_row("e-1", gold="A", prompt="answer:A")]

        # 1. Normalizer change alone explains the delta (same predictions).
        a = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="exact_v1"),
            predictor=sloppy_predictor,
        )
        b = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=sloppy_predictor,
        )
        assert a.records[0].prediction == b.records[0].prediction
        assert a.records[0].normalizer_id != b.records[0].normalizer_id
        assert a.summary["accuracy"] < b.summary["accuracy"]

        # 2. Model change alone explains the delta (same normalizer_id).
        c = run_baseline_eval(
            split_rows=rows,
            config=_config(normalizer_id="strip_v1"),
            predictor=_echo_predictor,  # emits "A" rather than "A "
        )
        assert b.summary["accuracy"] == c.summary["accuracy"]  # both pass
        assert b.records[0].prediction != c.records[0].prediction
