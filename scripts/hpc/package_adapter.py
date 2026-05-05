#!/usr/bin/env python3
"""Package a LoRA adapter into a Kaggle-ready submission bundle.

Thin wrapper around src.inference.submission.build_submission. The produced
``submission.zip`` contains EXACTLY ``adapter_config.json`` +
``adapter_model.safetensors`` at the zip root; provenance lives in
``submission.manifest.json`` beside the zip.

Usage:
    python scripts/hpc/package_adapter.py \\
        --adapter-dir   /run/nemotron/checkpoints/best \\
        --output-dir    /run/nemotron/package \\
        --base-model-id metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \\
        --adapter-rank  16

Optional:
    --dataset-version  v0         (default: "v0")
    --eval-sha         <sha>      (default: auto-detected HEAD)
    --git-sha          <sha>      (default: auto-detected HEAD)

Exits non-zero on any failure and prints an error to stderr.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference.submission import build_submission  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a LoRA adapter directory into a Kaggle-ready "
            "submission.zip with a sibling submission.manifest.json."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory containing adapter_config.json + adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for submission.zip and submission.manifest.json.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        required=True,
        help=(
            "HF model id of the base model "
            "(frozen: metric/nemotron-3-nano-30b-a3b-bf16/transformers/default)."
        ),
    )
    parser.add_argument(
        "--adapter-rank",
        type=int,
        required=True,
        help="LoRA rank used during fine-tuning (must be <= 32).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v0",
        help="Version tag for the training dataset (default: v0).",
    )
    parser.add_argument(
        "--eval-sha",
        type=str,
        default=None,
        help="SHA identifying the eval record; auto-detected via git if omitted.",
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Override for source git commit SHA; auto-detected if omitted.",
    )
    return parser.parse_args(argv)


def _detect_git_sha() -> str:
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    eval_sha = args.eval_sha or _detect_git_sha()

    try:
        bundle = build_submission(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            base_model_id=args.base_model_id,
            adapter_rank=args.adapter_rank,
            dataset_version=args.dataset_version,
            eval_sha=eval_sha,
            git_sha=args.git_sha,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    zip_bytes = bundle.submission_zip.stat().st_size
    print(
        f"OK: zip={bundle.submission_zip} "
        f"manifest={bundle.manifest_path} "
        f"zip_bytes={zip_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
