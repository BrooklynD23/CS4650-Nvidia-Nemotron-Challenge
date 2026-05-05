#!/usr/bin/env python3
"""Checkpoint rotation and sidecar-file policy for the Nemotron pipeline.

Enforces the keep-last-3-plus-best policy defined in the HPC runbook:
  - Scans ``--checkpoint-dir`` for step-XXXXX/ subdirectories.
  - Keeps the three most recent steps plus the ``best/`` dir.
  - Deletes older step dirs (dry-run by default; pass --execute to delete).
  - Writes required sidecar files next to any checkpoint that lacks them.

Sidecar files written per checkpoint:
  trainer_state.json      — step number, directory name, created_at (ISO-8601 UTC)
  run_config.json         — copied from --run-config if provided and missing
  metrics.jsonl           — touched/left as-is (training loop appends here)
  git_sha.txt             — current HEAD SHA
  dataset_fingerprint.txt — copied from parent checkpoint dir if present

Usage:
    python scripts/hpc/checkpoint_policy.py \\
        --checkpoint-dir /run/nemotron/checkpoints \\
        --run-config /run/nemotron/checkpoints/run_config.json \\
        [--execute]

Exit 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^step-(\d+)$")
_KEEP_LAST_N = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply keep-last-3-plus-best checkpoint rotation and write sidecar files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing step-XXXXX/ checkpoint subdirs.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Path to run_config.json to copy into checkpoints that lack it.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete old checkpoints. Default is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicit dry-run flag (takes precedence over --execute).",
    )
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_sidecars(
    step_dir: Path,
    step: int,
    run_config_source: Path | None,
    parent_dir: Path,
) -> None:
    """Write required sidecar files into *step_dir* if they are absent."""
    state_path = step_dir / "trainer_state.json"
    if not state_path.exists():
        state = {
            "checkpoint_dir": step_dir.name,
            "step": step,
            "created_at": _utcnow(),
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        log.info("  wrote %s", state_path.name)

    rc_path = step_dir / "run_config.json"
    if not rc_path.exists() and run_config_source and run_config_source.is_file():
        shutil.copy2(run_config_source, rc_path)
        log.info("  copied run_config.json from %s", run_config_source)

    metrics_path = step_dir / "metrics.jsonl"
    if not metrics_path.exists():
        metrics_path.touch()
        log.info("  touched %s", metrics_path.name)

    sha_path = step_dir / "git_sha.txt"
    if not sha_path.exists():
        sha_path.write_text(_git_sha() + "\n")
        log.info("  wrote %s", sha_path.name)

    fp_src = parent_dir / "dataset_fingerprint.txt"
    fp_dst = step_dir / "dataset_fingerprint.txt"
    if not fp_dst.exists() and fp_src.is_file():
        shutil.copy2(fp_src, fp_dst)
        log.info("  copied dataset_fingerprint.txt")


def _apply_policy(
    checkpoint_dir: Path,
    run_config_source: Path | None,
    execute: bool,
) -> None:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    step_dirs: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_RE.match(child.name)
        if m:
            step_dirs.append((int(m.group(1)), child))

    step_dirs.sort(key=lambda t: t[0])
    log.info(
        "Found %d step checkpoint(s) in %s", len(step_dirs), checkpoint_dir
    )

    for step, sdir in step_dirs:
        log.info("Processing %s (step=%d)", sdir.name, step)
        _write_sidecars(sdir, step, run_config_source, checkpoint_dir)

    best_dir = checkpoint_dir / "best"
    if best_dir.is_dir() and not best_dir.is_symlink():
        log.info("Processing best/ checkpoint")
        _write_sidecars(best_dir, -1, run_config_source, checkpoint_dir)

    keep: set[Path] = {sdir for _, sdir in step_dirs[-_KEEP_LAST_N:]}
    to_delete = [(s, d) for s, d in step_dirs if d not in keep]

    if not to_delete:
        log.info(
            "Keep-last-%d policy: nothing to delete (%d step(s) total).",
            _KEEP_LAST_N,
            len(step_dirs),
        )
        return

    keeping_names = [d.name for _, d in step_dirs[-_KEEP_LAST_N:]]
    log.info(
        "Keep-last-%d policy: keeping %s; %d older step(s) eligible for deletion.",
        _KEEP_LAST_N,
        keeping_names,
        len(to_delete),
    )
    for step, sdir in to_delete:
        if execute:
            log.info("  DELETING %s", sdir)
            shutil.rmtree(sdir)
        else:
            log.info("  DRY-RUN: would delete %s", sdir)

    if not execute:
        log.info("Dry-run complete. Pass --execute to apply deletions.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    execute = args.execute and not args.dry_run

    log.info(
        "Mode: %s", "EXECUTE — deletions active" if execute else "DRY-RUN — no deletions"
    )

    try:
        _apply_policy(args.checkpoint_dir, args.run_config, execute)
    except Exception as exc:
        log.error("Checkpoint policy failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
