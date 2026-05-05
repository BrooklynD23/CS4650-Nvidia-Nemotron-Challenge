"""Synthetic data generation pipeline for SFT training (#10 / #24).

Generates :class:`~src.contracts.SFTExample` rows from retry-candidate
trajectories using a solver-first / LLM-fallback teacher policy. Includes
quality filters, cost caps, smoke-run mode, and dataset fingerprinting.

Running as a module::

    python -m src.data.synthetic --smoke --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.contracts import SFTExample
from src.evaluation.trajectory import TrajectoryRow

_BOXED_MARKER = r"\boxed{"
_MAX_TOKENS = 8192
_SMOKE_LIMIT = 50
_COST_CAP_USD = 20.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Runtime configuration for a generation run."""

    output_dir: Path
    categories: list[str]
    cost_cap_usd: float = _COST_CAP_USD
    max_tokens: int = _MAX_TOKENS
    verifier: Any = None  # Verifier Protocol instance (optional)
    llm_teacher_fn: Callable[[str], str] | None = None  # mock-able LLM call


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

class QualityFilter:
    """Stateful filter that tracks seen IDs for deduplication."""

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        self._seen: set[str] = set()

    def _fingerprint(self, ex: SFTExample) -> str:
        raw = ex.example_id + ex.category
        return hashlib.sha256(raw.encode()).hexdigest()

    def _has_boxed(self, ex: SFTExample) -> bool:
        return _BOXED_MARKER in ex.completion

    def _tokens_ok(self, ex: SFTExample) -> bool:
        total = sum(len(m.get("content", "")) for m in ex.messages)
        total += len(ex.completion)
        # Rough token estimate: chars / 4
        return (total // 4) <= self._config.max_tokens

    def _provenance_ok(self, ex: SFTExample) -> bool:
        required = {"teacher", "generated_at", "source_run_id"}
        return required.issubset(ex.provenance.keys())

    def _category_ok(self, ex: SFTExample) -> bool:
        return ex.category in self._config.categories

    def accept(self, ex: SFTExample) -> bool:
        """Return True if the example passes all quality gates."""
        fp = self._fingerprint(ex)
        if fp in self._seen:
            return False
        if not self._has_boxed(ex):
            return False
        if not self._tokens_ok(ex):
            return False
        if not self._provenance_ok(ex):
            return False
        if not self._category_ok(ex):
            return False
        self._seen.add(fp)
        return True

    def apply(self, examples: list[SFTExample]) -> list[SFTExample]:
        return [ex for ex in examples if self.accept(ex)]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _make_provenance(
    teacher: str,
    source_run_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov: dict[str, Any] = {
        "teacher": teacher,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run_id": source_run_id,
    }
    if extra:
        prov.update(extra)
    return prov


def _row_to_sft_example(
    row: TrajectoryRow,
    completion: str,
    teacher: str,
    source_run_id: str,
) -> SFTExample:
    return SFTExample(
        example_id=row.record.example_id,
        category=row.record.category,
        messages=[
            {"role": "user", "content": row.record.gold},
        ],
        completion=completion,
        source="synthetic",
        split="train",
        provenance=_make_provenance(teacher, source_run_id),
    )


def generate_from_retry_candidates(
    candidates: list[TrajectoryRow],
    config: SyntheticConfig,
    *,
    smoke: bool = False,
    dry_run: bool = False,
) -> list[SFTExample]:
    """Generate SFTExamples from retry-candidate trajectory rows.

    Teacher policy (in priority order):
    1. Verifier (if configured and answer is available from solver path)
    2. LLM teacher function (mock-able; in production: DeepSeek-R1 → Claude → GPT-4)

    Args:
        candidates: Recoverable-failure trajectory rows from Branch 1.
        config: Generation configuration including cost cap and output dir.
        smoke: If True, cap output at SMOKE_LIMIT (50) examples.
        dry_run: If True, print token/cost estimate and return empty list.

    Returns:
        Filtered SFTExample list; writes JSONL + SHA-256 fingerprint to config.output_dir.
    """
    target = candidates[:_SMOKE_LIMIT] if smoke else candidates

    if dry_run:
        est_tokens = sum(
            len(row.record.gold) // 4 + 200 for row in target
        )
        est_cost = est_tokens * 0.000002  # rough estimate
        print(f"[dry-run] examples to generate: {len(target)}")
        print(f"[dry-run] estimated tokens: {est_tokens}")
        print(f"[dry-run] estimated cost: ${est_cost:.4f}")
        return []

    source_run_id = candidates[0].record.run_id if candidates else "synthetic-run"
    quality_filter = QualityFilter(config)
    results: list[SFTExample] = []
    accumulated_cost = 0.0

    for row in target:
        if accumulated_cost >= config.cost_cap_usd:
            print(f"[synthetic] cost cap ${config.cost_cap_usd} reached; stopping at {len(results)} examples")
            break

        completion: str | None = None
        teacher_used = "none"

        # Teacher policy: verifier-first
        if config.verifier is not None:
            # If the record already has a recoverable answer, verify it
            pred = row.record.prediction
            gold = row.record.gold
            try:
                if config.verifier.verify(pred, gold):
                    completion = pred
                    teacher_used = "verifier"
            except Exception:
                pass

        # LLM teacher fallback
        if completion is None and config.llm_teacher_fn is not None:
            try:
                completion = config.llm_teacher_fn(row.record.gold)
                teacher_used = "llm_teacher"
                accumulated_cost += 0.01  # placeholder cost per call
            except Exception:
                pass

        if completion is None:
            continue

        example = _row_to_sft_example(row, completion, teacher_used, source_run_id)
        results.append(example)

    filtered = quality_filter.apply(results)
    return filtered


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_sft_examples_jsonl(examples: list[SFTExample], path: Path) -> None:
    """Write SFTExamples to a JSONL file and a sibling SHA-256 fingerprint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(dataclasses.asdict(ex), ensure_ascii=False))
            fh.write("\n")

    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = path.with_suffix(".sha256")
    sha_path.write_text(fingerprint + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline")
    parser.add_argument("--smoke", action="store_true", help=f"Cap at {_SMOKE_LIMIT} examples")
    parser.add_argument("--dry-run", action="store_true", help="Print estimates, no API calls")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    args = parser.parse_args(argv)

    config = SyntheticConfig(
        output_dir=Path(args.output_dir),
        categories=["bit_manipulation", "math", "code", "science"],
    )

    print(f"[synthetic] smoke={args.smoke}, dry_run={args.dry_run}")
    print(f"[synthetic] output_dir={config.output_dir}")
    print("[synthetic] No retry_candidates.jsonl provided — pass candidates programmatically.")


if __name__ == "__main__":
    _main()


__all__ = [
    "SyntheticConfig",
    "QualityFilter",
    "generate_from_retry_candidates",
    "write_sft_examples_jsonl",
]
