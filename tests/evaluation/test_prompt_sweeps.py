"""Tests for :mod:`src.evaluation.prompt_sweeps`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.config import make_run_config, write_run_config
from src.evaluation.records import make_eval_record
from src.evaluation.prompt_sweeps import (
    aggregate_sweep_results,
    build_best_of_n_specs,
    build_run_id,
    build_sparse_sweep_specs,
    load_baseline_reference,
    majority_vote,
    render_findings_markdown,
    require_split_artifacts,
    stable_int_seed,
    validate_baseline_compatibility,
    write_aggregate_csv,
    write_findings_markdown,
)
from src.evaluation.reporting import write_eval_records_jsonl, write_summary_json
from src.evaluation.splits import SplitArtifactRow, write_split_jsonl


def _row(example_id: str, *, split: str = "val") -> SplitArtifactRow:
    return SplitArtifactRow(
        example_id=example_id,
        category="unit_conversion",
        prompt=f"prompt-{example_id}",
        gold="42",
        source="test",
        split=split,
        dataset_version="dataset-v1",
        selection_seed=42,
        selection_rule="stratified-by-category",
        selection_reason=f"{split}:stratified-by-category; seed=42",
    )


def _baseline_repo(tmp_path: Path, *, total: int = 2) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "eval" / "runs" / "baseline-v1").mkdir(parents=True)
    rows = [_row(f"e-{i + 1}") for i in range(total)]
    config = make_run_config(
        run_id="baseline-v1",
        model_id="stub/model",
        prompt_template_id="zero-shot-v1",
        normalizer_id="strip_v1",
        seed=42,
        split="val",
        dataset_version="dataset-v1",
        decode_config={"temperature": 0.0, "top_p": 1.0},
    )
    write_run_config(
        config,
        repo_root / "data" / "eval" / "runs" / "baseline-v1" / "run_config.json",
    )
    write_eval_records_jsonl(
        [
            make_eval_record(
                run_id=config.run_id,
                example_id=row.example_id,
                model_id=config.model_id,
                prompt_template_id=config.prompt_template_id,
                normalizer_id=config.normalizer_id,
                category=row.category,
                split=row.split,
                gold=row.gold,
                prediction=row.gold,
                normalized_prediction=row.gold,
                correct=True,
                latency_ms=1.0,
                tokens_in=1,
                tokens_out=1,
                seed=config.seed,
                decode_config=dict(config.decode_config),
            )
            for row in rows
        ],
        repo_root / "data" / "eval" / "runs" / "baseline-v1" / "eval_records.jsonl",
    )
    write_summary_json(
        {
            "accuracy": 0.5,
            "correct": total // 2,
            "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0},
            "model_id": "stub/model",
            "normalizer_id": "strip_v1",
            "normalizer_ids": ["strip_v1"],
            "per_category_accuracy": {},
            "prompt_template_id": "zero-shot-v1",
            "run_id": "baseline-v1",
            "run_ids": ["baseline-v1"],
            "seed": 42,
            "split": "val",
            "tokens_in_total": 10,
            "tokens_out_total": 2,
            "total": total,
        },
        repo_root / "data" / "eval" / "runs" / "baseline-v1" / "summary.json",
    )
    return repo_root


class TestSplitRequirements:
    def test_require_split_artifacts_fails_without_real_files(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        (repo_root / "data" / "eval").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Synthetic fallback is intentionally disabled"):
            require_split_artifacts(repo_root)


class TestRunSpecExpansion:
    def test_sparse_grid_expands_to_24_unique_runs(self) -> None:
        specs = build_sparse_sweep_specs(date_stamp="2026-04-21")
        assert len(specs) == 24
        assert len({spec.run_id for spec in specs}) == 24

    def test_best_of_n_specs_reuse_sparse_winner_decode(self) -> None:
        specs = build_best_of_n_specs(
            date_stamp="2026-04-21",
            strategy="few-shot-cot",
            temperature=0.6,
            top_p=0.95,
        )
        assert len(specs) == 6
        assert {spec.best_of_n for spec in specs} == {8, 32}

    def test_run_id_is_stable(self) -> None:
        run_id = build_run_id(
            date_stamp="2026-04-21",
            strategy="zero-shot-cot",
            temperature=1.0,
            top_p=0.95,
            seed=11,
            best_of_n=32,
        )
        assert run_id == "prompt-sweep-2026-04-21-zero-shot-cot-t1-p0p95-s11-bon32"


class TestDeterminismHelpers:
    def test_stable_int_seed_is_repeatable(self) -> None:
        left = stable_int_seed("run-1", "e-1", 0)
        right = stable_int_seed("run-1", "e-1", 0)
        other = stable_int_seed("run-1", "e-1", 1)
        assert left == right
        assert left != other

    def test_majority_vote_breaks_ties_lexicographically(self) -> None:
        assert majority_vote(["B", "A", "B", "A"]) == "A"


class TestAggregation:
    def test_aggregate_rows_compute_mean_std_and_delta(self) -> None:
        rows = [
            {
                "run_id": "run-a",
                "strategy": "zero-shot-cot",
                "temperature": 0.6,
                "top_p": 0.9,
                "best_of_n": 1,
                "seed": 11,
                "accuracy": 0.60,
                "elapsed_seconds": 10.0,
            },
            {
                "run_id": "run-b",
                "strategy": "zero-shot-cot",
                "temperature": 0.6,
                "top_p": 0.9,
                "best_of_n": 1,
                "seed": 23,
                "accuracy": 0.70,
                "elapsed_seconds": 12.0,
            },
            {
                "run_id": "run-c",
                "strategy": "few-shot-cot",
                "temperature": 1.0,
                "top_p": 0.95,
                "best_of_n": 8,
                "seed": 11,
                "accuracy": 0.80,
                "elapsed_seconds": 25.0,
            },
            {
                "run_id": "run-d",
                "strategy": "few-shot-cot",
                "temperature": 1.0,
                "top_p": 0.95,
                "best_of_n": 8,
                "seed": 23,
                "accuracy": 0.82,
                "elapsed_seconds": 27.0,
            },
        ]

        aggregates = aggregate_sweep_results(rows, baseline_accuracy=0.50)

        assert len(aggregates) == 2
        best = aggregates[0]
        assert best.strategy == "few-shot-cot"
        assert best.best_of_n == 8
        assert best.accuracy_mean == pytest.approx(0.81)
        assert best.delta_vs_baseline == pytest.approx(0.31)
        assert best.significant is True

    def test_write_aggregate_csv_round_trips_headers(self, tmp_path: Path) -> None:
        rows = aggregate_sweep_results(
            [
                {
                    "run_id": "run-a",
                    "strategy": "zero-shot-cot",
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "best_of_n": 1,
                    "seed": 11,
                    "accuracy": 0.61,
                    "elapsed_seconds": 10.0,
                },
                {
                    "run_id": "run-b",
                    "strategy": "zero-shot-cot",
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "best_of_n": 1,
                    "seed": 23,
                    "accuracy": 0.63,
                    "elapsed_seconds": 11.0,
                },
            ],
            baseline_accuracy=0.5,
        )
        out_path = write_aggregate_csv(rows, tmp_path / "prompting_sweep_2026-04-21.csv")
        content = out_path.read_text(encoding="utf-8")
        assert "delta_vs_baseline" in content
        assert "zero-shot-cot" in content


class TestBaselineReference:
    def test_load_and_validate_baseline_reference(self, tmp_path: Path) -> None:
        repo_root = _baseline_repo(tmp_path)
        baseline = load_baseline_reference(repo_root, baseline_run_id="baseline-v1")
        val_rows = [_row("e-1"), _row("e-2")]
        validate_baseline_compatibility(
            baseline=baseline,
            val_rows=val_rows,
            expected_model_id="stub/model",
            expected_normalizer_id="strip_v1",
        )
        assert baseline.summary["accuracy"] == 0.5

    def test_validate_baseline_compatibility_rejects_dataset_mismatch(self, tmp_path: Path) -> None:
        repo_root = _baseline_repo(tmp_path)
        baseline = load_baseline_reference(repo_root, baseline_run_id="baseline-v1")
        bad_rows = [
            SplitArtifactRow(
                example_id="e-1",
                category="unit_conversion",
                prompt="prompt-e-1",
                gold="42",
                source="test",
                split="val",
                dataset_version="other-dataset",
                selection_seed=42,
                selection_rule="stratified-by-category",
                selection_reason="val:stratified-by-category; seed=42",
            )
        ]
        with pytest.raises(ValueError, match="dataset_version"):
            validate_baseline_compatibility(baseline=baseline, val_rows=bad_rows)

    def test_validate_baseline_compatibility_rejects_model_mismatch(self, tmp_path: Path) -> None:
        repo_root = _baseline_repo(tmp_path)
        baseline = load_baseline_reference(repo_root, baseline_run_id="baseline-v1")
        val_rows = [_row("e-1"), _row("e-2")]
        with pytest.raises(ValueError, match="model_id"):
            validate_baseline_compatibility(
                baseline=baseline,
                val_rows=val_rows,
                expected_model_id="other/model",
            )

    def test_validate_baseline_compatibility_rejects_example_id_drift(self, tmp_path: Path) -> None:
        repo_root = _baseline_repo(tmp_path)
        baseline = load_baseline_reference(repo_root, baseline_run_id="baseline-v1")
        drifted_rows = [_row("e-2"), _row("e-1")]
        with pytest.raises(ValueError, match="example_id order"):
            validate_baseline_compatibility(
                baseline=baseline,
                val_rows=drifted_rows,
            )


class TestFindingsRendering:
    def test_render_findings_markdown_includes_required_sections(self, tmp_path: Path) -> None:
        repo_root = _baseline_repo(tmp_path)
        baseline = load_baseline_reference(repo_root, baseline_run_id="baseline-v1")
        aggregate_rows = aggregate_sweep_results(
            [
                {
                    "run_id": "run-a",
                    "strategy": "few-shot-cot",
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "best_of_n": 1,
                    "seed": 11,
                    "accuracy": 0.81,
                    "elapsed_seconds": 20.0,
                },
                {
                    "run_id": "run-b",
                    "strategy": "few-shot-cot",
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "best_of_n": 1,
                    "seed": 23,
                    "accuracy": 0.79,
                    "elapsed_seconds": 22.0,
                },
            ],
            baseline_accuracy=0.5,
        )
        content = render_findings_markdown(
            date_stamp="2026-04-21",
            baseline=baseline,
            aggregate_rows=aggregate_rows,
            csv_path="experiments/prompting_sweep_2026-04-21.csv",
            golden_gate_passed=True,
            golden_summary="Golden gate: PASS (passed=20/20)",
            promoted_run_id="prompt-sweep-2026-04-21-few-shot-cot-t0p6-p0p95-s11",
            notes=["GPU run on RTX 3080."],
        )
        out_path = write_findings_markdown(content, tmp_path / "prompting_findings.md")
        text = out_path.read_text(encoding="utf-8")
        assert "# Prompting Findings" in text
        assert "Golden Regression Check" in text
        assert "delta vs baseline".lower() in text.lower()
        assert "GPU run on RTX 3080." in text
