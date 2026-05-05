#!/usr/bin/env python3
"""Scan a checkpoint directory and print the latest step path to stdout.

The printed path is suitable for passing directly to a HuggingFace Trainer's
``--resume_from_checkpoint`` argument.

Usage:
    python scripts/hpc/resume_from_latest.py \\
        --checkpoint-dir /run/nemotron/checkpoints

    # Or resolve via RUN_TAG + env vars:
    python scripts/hpc/resume_from_latest.py \\
        --run-tag my-run-001

    # Capture in shell:
    RESUME=$(python scripts/hpc/resume_from_latest.py --run-tag ${RUN_TAG})
    python train.py --resume_from_checkpoint "${RESUME}"

Exit codes:
    0 — a checkpoint was found; absolute path printed to stdout
    1 — no checkpoint found or configuration error
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_STEP_RE = re.compile(r"^step-(\d+)$")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the absolute path of the latest step-XXXXX checkpoint to stdout. "
            "Intended for shell capture to set --resume_from_checkpoint."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing step-XXXXX/ checkpoint subdirs.",
    )
    group.add_argument(
        "--run-tag",
        type=str,
        help=(
            "RUN_TAG; resolves checkpoint dir as "
            "${CHECKPOINT_ROOT}/${RUN_TAG}/checkpoints "
            "(requires CHECKPOINT_ROOT or RUN_ROOT env var)."
        ),
    )
    return parser.parse_args(argv)


def _resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir

    checkpoint_root = os.environ.get("CHECKPOINT_ROOT") or os.environ.get("RUN_ROOT")
    if not checkpoint_root:
        raise EnvironmentError(
            "Neither CHECKPOINT_ROOT nor RUN_ROOT is set. "
            "Provide one of these env vars or use --checkpoint-dir."
        )
    return Path(checkpoint_root) / args.run_tag / "checkpoints"


def _find_latest(checkpoint_dir: Path) -> Path | None:
    """Return the step-XXXXX subdir with the highest step number, or None."""
    if not checkpoint_dir.is_dir():
        return None

    best_step = -1
    best_path: Path | None = None

    for child in checkpoint_dir.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_RE.match(child.name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = child

    return best_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        checkpoint_dir = _resolve_checkpoint_dir(args)
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    latest = _find_latest(checkpoint_dir)
    if latest is None:
        print(
            f"ERROR: No step-XXXXX checkpoints found in {checkpoint_dir}",
            file=sys.stderr,
        )
        return 1

    print(str(latest.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
