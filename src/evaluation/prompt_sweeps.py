"""Prompt/decode sweep helpers for notebook 05.

This module keeps the notebook orchestration thin and pushes the
reproducibility-sensitive pieces into testable Python helpers:

- strict split-artifact loading (no synthetic fallback)
- deterministic run-id construction
- sparse-grid and Best-of-N spec generation
- majority-vote reduction
- aggregate result summarization / CSV writing
- findings markdown rendering

Real model loading and token generation stay in the notebook because
that runtime depends on GPU availability and local environment state.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.contracts import EvalRecord
from src.evaluation.config import EvalRunConfig, read_run_config
from src.evaluation.reporting import read_eval_records_jsonl, read_summary_json
from src.evaluation.splits import SplitArtifactRow, read_split_jsonl

VAL_SPLIT_FILENAME = "validation_200.jsonl"
GOLDEN_SPLIT_FILENAME = "golden_20.jsonl"

DEFAULT_STRATEGIES: tuple[str, ...] = (
    "zero-shot-cot",
    "few-shot-cot",
)
DEFAULT_DECODE_GRID: tuple[tuple[float, float], ...] = (
    (0.6, 0.9),
    (0.6, 0.95),
    (1.0, 0.9),
    (1.0, 0.95),
)
DEFAULT_SEEDS: tuple[int, ...] = (11, 23, 37)
DEFAULT_BEST_OF_N: tuple[int, ...] = (8, 32)


@dataclass(slots=True, frozen=True)
class BaselineReference:
    """Frozen handle to the baseline run used for delta computation."""

    run_dir: Path
    config: EvalRunConfig
    summary: dict[str, Any]
    records: tuple[EvalRecord, ...]


@dataclass(slots=True, frozen=True)
class SweepRunSpec:
    """One deterministic run specification for the prompt sweep."""

    run_id: str
    strategy: str
    temperature: float
    top_p: float
    seed: int
    best_of_n: int = 1

    @property
    def config_key(self) -> tuple[str, float, float, int]:
        return (
            self.strategy,
            self.temperature,
            self.top_p,
            self.best_of_n,
        )


@dataclass(slots=True, frozen=True)
class AggregateSweepRow:
    """Aggregate metrics for one unique sweep configuration."""

    strategy: str
    temperature: float
    top_p: float
    best_of_n: int
    run_count: int
    accuracy_mean: float
    accuracy_std: float
    latency_seconds_mean: float
    latency_seconds_std: float
    delta_vs_baseline: float
    significant: bool
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...]

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        return {
            "strategy": self.strategy,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "best_of_n": self.best_of_n,
            "run_count": self.run_count,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "latency_seconds_mean": self.latency_seconds_mean,
            "latency_seconds_std": self.latency_seconds_std,
            "delta_vs_baseline": self.delta_vs_baseline,
            "significant": self.significant,
            "run_ids": "|".join(self.run_ids),
            "seeds": "|".join(str(seed) for seed in self.seeds),
        }


def require_split_artifacts(repo_root: str | Path) -> tuple[Path, Path]:
    """Return the required val/golden split paths or fail clearly.

    Notebook 05 is intentionally strict: it must never fabricate or
    silently derive eval inputs once prompt sweeps begin.
    """
    root = Path(repo_root)
    val_path = root / "data" / "eval" / VAL_SPLIT_FILENAME
    golden_path = root / "data" / "eval" / GOLDEN_SPLIT_FILENAME
    missing = [path for path in (val_path, golden_path) if not path.exists()]
    if missing:
        rel = ", ".join(str(path.relative_to(root)) for path in missing)
        raise FileNotFoundError(
            "Prompt/decode sweeps require frozen split artifacts from issue #18. "
            f"Missing: {rel}. Create data/eval/validation_200.jsonl and "
            "data/eval/golden_20.jsonl before running notebook 05. "
            "Synthetic fallback is intentionally disabled for this phase."
        )
    return val_path, golden_path


def load_required_splits(
    repo_root: str | Path,
) -> tuple[list[SplitArtifactRow], list[SplitArtifactRow]]:
    """Load and validate the required val/golden split artifacts."""
    val_path, golden_path = require_split_artifacts(repo_root)
    val_rows = read_split_jsonl(val_path)
    golden_rows = read_split_jsonl(golden_path)
    if not val_rows:
        raise ValueError(f"Validation split is empty: {val_path}")
    if not golden_rows:
        raise ValueError(f"Golden split is empty: {golden_path}")
    return val_rows, golden_rows


def load_baseline_reference(
    repo_root: str | Path,
    *,
    baseline_run_id: str,
) -> BaselineReference:
    """Load the baseline run config + summary used for delta columns."""
    if not isinstance(baseline_run_id, str) or baseline_run_id == "":
        raise ValueError("baseline_run_id must be a non-empty str")
    root = Path(repo_root)
    run_dir = root / "data" / "eval" / "runs" / baseline_run_id
    config_path = run_dir / "run_config.json"
    records_path = run_dir / "eval_records.jsonl"
    summary_path = run_dir / "summary.json"
    missing = [
        path.name
        for path in (config_path, records_path, summary_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Baseline reference is incomplete for run_id "
            f"{baseline_run_id!r}: missing {', '.join(missing)} under {run_dir}"
        )
    config = read_run_config(config_path)
    records = tuple(read_eval_records_jsonl(records_path))
    summary = read_summary_json(summary_path)
    return BaselineReference(
        run_dir=run_dir,
        config=config,
        summary=summary,
        records=records,
    )


def validate_baseline_compatibility(
    *,
    baseline: BaselineReference,
    val_rows: Sequence[SplitArtifactRow],
    expected_model_id: str | None = None,
    expected_normalizer_id: str | None = None,
) -> None:
    """Fail closed if the selected baseline does not match the val split."""
    if baseline.config.split != "val":
        raise ValueError(
            "Baseline reference must target the val split, got "
            f"{baseline.config.split!r}"
        )
    if expected_model_id is not None and baseline.config.model_id != expected_model_id:
        raise ValueError(
            "Baseline model_id does not match the requested sweep model: "
            f"baseline={baseline.config.model_id!r}, "
            f"expected={expected_model_id!r}"
        )
    if (
        expected_normalizer_id is not None
        and baseline.config.normalizer_id != expected_normalizer_id
    ):
        raise ValueError(
            "Baseline normalizer_id does not match the sweep contract: "
            f"baseline={baseline.config.normalizer_id!r}, "
            f"expected={expected_normalizer_id!r}"
        )
    dataset_version = val_rows[0].dataset_version
    if baseline.config.dataset_version != dataset_version:
        raise ValueError(
            "Baseline dataset_version does not match current val split: "
            f"baseline={baseline.config.dataset_version!r}, "
            f"val={dataset_version!r}. Point BASELINE_RUN_ID to a run "
            "built on the same split artifact before computing deltas."
        )
    total = baseline.summary.get("total")
    if total != len(val_rows):
        raise ValueError(
            "Baseline total does not match current val split size: "
            f"baseline={total!r}, val_rows={len(val_rows)}. Use a baseline "
            "run produced from the same validation_200.jsonl artifact."
        )
    baseline_ids = tuple(record.example_id for record in baseline.records)
    val_ids = tuple(row.example_id for row in val_rows)
    if baseline_ids != val_ids:
        raise ValueError(
            "Baseline eval_records example_id order does not match the current "
            "val split artifact. Point BASELINE_RUN_ID to a run produced from "
            "this exact validation_200.jsonl selection."
        )


def build_run_id(
    *,
    date_stamp: str,
    strategy: str,
    temperature: float,
    top_p: float,
    seed: int,
    best_of_n: int = 1,
) -> str:
    """Construct a deterministic run id from the config tuple."""
    if best_of_n < 1:
        raise ValueError(f"best_of_n must be >= 1, got {best_of_n}")
    parts = [
        "prompt-sweep",
        date_stamp,
        _slug(strategy),
        f"t{_float_token(temperature)}",
        f"p{_float_token(top_p)}",
        f"s{seed}",
    ]
    if best_of_n > 1:
        parts.append(f"bon{best_of_n}")
    return "-".join(parts)


def build_sparse_sweep_specs(
    *,
    date_stamp: str,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    decode_grid: Sequence[tuple[float, float]] = DEFAULT_DECODE_GRID,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[SweepRunSpec]:
    """Expand the sparse sweep matrix into deterministic run specs."""
    specs: list[SweepRunSpec] = []
    for strategy in strategies:
        for temperature, top_p in decode_grid:
            for seed in seeds:
                specs.append(
                    SweepRunSpec(
                        run_id=build_run_id(
                            date_stamp=date_stamp,
                            strategy=strategy,
                            temperature=temperature,
                            top_p=top_p,
                            seed=seed,
                        ),
                        strategy=strategy,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        best_of_n=1,
                    )
                )
    return specs


def build_best_of_n_specs(
    *,
    date_stamp: str,
    strategy: str,
    temperature: float,
    top_p: float,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    best_of_n_values: Sequence[int] = DEFAULT_BEST_OF_N,
) -> list[SweepRunSpec]:
    """Expand Best-of-N follow-up runs for the chosen sparse winner."""
    specs: list[SweepRunSpec] = []
    for best_of_n in best_of_n_values:
        for seed in seeds:
            specs.append(
                SweepRunSpec(
                    run_id=build_run_id(
                        date_stamp=date_stamp,
                        strategy=strategy,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        best_of_n=best_of_n,
                    ),
                    strategy=strategy,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    best_of_n=best_of_n,
                )
            )
    return specs


def stable_int_seed(*parts: object) -> int:
    """Hash a tuple of values into a deterministic 31-bit seed."""
    digest = hashlib.sha256(
        "::".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def majority_vote(predictions: Sequence[str]) -> str:
    """Return the most common prediction, breaking ties lexicographically."""
    if not predictions:
        raise ValueError("majority_vote requires at least one prediction")
    counts = Counter(predictions)
    best_count = max(counts.values())
    winners = sorted(pred for pred, count in counts.items() if count == best_count)
    return winners[0]


def aggregate_sweep_results(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_accuracy: float,
) -> list[AggregateSweepRow]:
    """Aggregate seed-level sweep rows into one row per unique config."""
    buckets: dict[tuple[str, float, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["strategy"]),
            float(row["temperature"]),
            float(row["top_p"]),
            int(row.get("best_of_n", 1)),
        )
        buckets.setdefault(key, []).append(row)

    aggregates: list[AggregateSweepRow] = []
    for (strategy, temperature, top_p, best_of_n), bucket in buckets.items():
        accuracies = [float(item["accuracy"]) for item in bucket]
        latencies = [float(item["elapsed_seconds"]) for item in bucket]
        accuracy_mean = mean(accuracies)
        accuracy_std = stdev(accuracies) if len(accuracies) > 1 else 0.0
        latency_mean = mean(latencies)
        latency_std = stdev(latencies) if len(latencies) > 1 else 0.0
        delta = accuracy_mean - baseline_accuracy
        # The baseline currently carries only a point estimate, so use a
        # simple "winner is larger than two within-config std-devs" rule.
        significant = delta > (2.0 * accuracy_std)
        run_ids = tuple(sorted(str(item["run_id"]) for item in bucket))
        seeds = tuple(sorted(int(item["seed"]) for item in bucket))
        aggregates.append(
            AggregateSweepRow(
                strategy=strategy,
                temperature=temperature,
                top_p=top_p,
                best_of_n=best_of_n,
                run_count=len(bucket),
                accuracy_mean=accuracy_mean,
                accuracy_std=accuracy_std,
                latency_seconds_mean=latency_mean,
                latency_seconds_std=latency_std,
                delta_vs_baseline=delta,
                significant=significant,
                run_ids=run_ids,
                seeds=seeds,
            )
        )

    return sorted(
        aggregates,
        key=lambda row: (
            -row.accuracy_mean,
            row.accuracy_std,
            row.latency_seconds_mean,
            row.strategy,
            row.best_of_n,
        ),
    )


def write_aggregate_csv(
    rows: Sequence[AggregateSweepRow],
    path: str | Path,
) -> Path:
    """Write aggregate sweep rows to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "temperature",
        "top_p",
        "best_of_n",
        "run_count",
        "accuracy_mean",
        "accuracy_std",
        "latency_seconds_mean",
        "latency_seconds_std",
        "delta_vs_baseline",
        "significant",
        "run_ids",
        "seeds",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
    return out_path


def render_findings_markdown(
    *,
    date_stamp: str,
    baseline: BaselineReference,
    aggregate_rows: Sequence[AggregateSweepRow],
    csv_path: str | Path,
    golden_gate_passed: bool,
    golden_summary: str,
    promoted_run_id: str,
    notes: Iterable[str] = (),
) -> str:
    """Render the markdown summary written by notebook 05."""
    if not aggregate_rows:
        raise ValueError("aggregate_rows must not be empty")
    best = aggregate_rows[0]
    lines = [
        "# Prompting Findings",
        "",
        f"Date: `{date_stamp}`",
        "",
        "## Baseline",
        "",
        f"- Baseline run id: `{baseline.config.run_id}`",
        f"- Baseline model: `{baseline.config.model_id}`",
        f"- Baseline accuracy: `{baseline.summary['accuracy']:.4f}`",
        f"- Baseline dataset version: `{baseline.config.dataset_version}`",
        "",
        "## Ranked Configurations",
        "",
        "| Rank | Strategy | Temp | Top-p | Best-of-N | Mean Acc | Std | Delta vs Baseline | Significant | Run IDs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(aggregate_rows, start=1):
        lines.append(
            "| "
            f"{rank} | `{row.strategy}` | {row.temperature:.2f} | {row.top_p:.2f} | "
            f"{row.best_of_n} | {row.accuracy_mean:.4f} | {row.accuracy_std:.4f} | "
            f"{row.delta_vs_baseline:+.4f} | {row.significant} | "
            f"`{'<br>'.join(row.run_ids)}` |"
        )
    lines.extend(
        [
            "",
            "## Promotion Decision",
            "",
            f"- Promoted run id: `{promoted_run_id}`",
            f"- Best configuration: `{best.strategy}` at temp=`{best.temperature:.2f}`, top_p=`{best.top_p:.2f}`, best_of_n=`{best.best_of_n}`",
            f"- Aggregate accuracy: `{best.accuracy_mean:.4f} +/- {best.accuracy_std:.4f}`",
            f"- Golden gate passed: `{golden_gate_passed}`",
            "",
            "## Golden Regression Check",
            "",
            "```text",
            golden_summary.strip(),
            "```",
            "",
            "## Artifacts",
            "",
            f"- Aggregate CSV: `{Path(csv_path)}`",
            f"- Sweep run configs: `data/eval/runs/<run_id>/run_config.json`",
            f"- Sweep eval records: `data/eval/runs/<run_id>/eval_records.jsonl`",
        ]
    )
    note_lines = list(notes)
    if note_lines:
        lines.extend(["", "## Notes", ""])
        for note in note_lines:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_findings_markdown(content: str, path: str | Path) -> Path:
    """Write the findings markdown to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _float_token(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


__all__ = [
    "AggregateSweepRow",
    "BaselineReference",
    "DEFAULT_BEST_OF_N",
    "DEFAULT_DECODE_GRID",
    "DEFAULT_SEEDS",
    "DEFAULT_STRATEGIES",
    "GOLDEN_SPLIT_FILENAME",
    "SweepRunSpec",
    "VAL_SPLIT_FILENAME",
    "aggregate_sweep_results",
    "build_best_of_n_specs",
    "build_run_id",
    "build_sparse_sweep_specs",
    "load_baseline_reference",
    "load_required_splits",
    "majority_vote",
    "render_findings_markdown",
    "require_split_artifacts",
    "stable_int_seed",
    "validate_baseline_compatibility",
    "write_aggregate_csv",
    "write_findings_markdown",
]
