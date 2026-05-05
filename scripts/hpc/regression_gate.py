#!/usr/bin/env python3
"""Strict golden regression gate for checkpoint promotion.

Loads an EvalRecord JSONL and a golden SplitArtifactRow JSONL, reconstructs
the dataclass objects, calls evaluate_golden_gate() from
src.evaluation.golden_gate, prints a human-readable summary, and exits 0 on
pass / 1 on fail.

Usage:
    python scripts/hpc/regression_gate.py \\
        --eval-records /run/nemotron/eval/eval_records.jsonl \\
        --golden       /data/golden/golden_split.jsonl

Exit codes:
    0 — gate passed (all golden examples correct)
    1 — gate failed (one or more misses) or load/parse error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strict golden regression gate. "
            "Exits 0 if all golden examples are correct, 1 otherwise."
        ),
    )
    parser.add_argument(
        "--eval-records",
        type=Path,
        required=True,
        help="JSONL file of EvalRecord rows from the run under evaluation.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        required=True,
        help="JSONL file of frozen SplitArtifactRow golden-set rows.",
    )
    return parser.parse_args(argv)


def _load_eval_records(path: Path) -> list:
    from src.contracts import EvalRecord

    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in eval-records on line {lineno}: {exc}"
                ) from exc
            try:
                records.append(EvalRecord(**data))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot construct EvalRecord from line {lineno}: {exc}"
                ) from exc
    return records


def _load_golden_rows(path: Path) -> list:
    from src.evaluation.splits import SplitArtifactRow

    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in golden on line {lineno}: {exc}"
                ) from exc
            try:
                rows.append(SplitArtifactRow(**data))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot construct SplitArtifactRow from line {lineno}: {exc}"
                ) from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.eval_records.is_file():
        log.error("eval-records file not found: %s", args.eval_records)
        return 1
    if not args.golden.is_file():
        log.error("golden file not found: %s", args.golden)
        return 1

    try:
        from src.evaluation.golden_gate import evaluate_golden_gate, summarize_gate

        log.info("Loading eval records: %s", args.eval_records)
        eval_records = _load_eval_records(args.eval_records)
        log.info("Loaded %d eval record(s)", len(eval_records))

        log.info("Loading golden rows: %s", args.golden)
        golden_rows = _load_golden_rows(args.golden)
        log.info("Loaded %d golden row(s)", len(golden_rows))

        result = evaluate_golden_gate(eval_records, golden_rows)
        print(summarize_gate(result))

    except Exception as exc:
        log.error("Regression gate error: %s", exc)
        return 1

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
