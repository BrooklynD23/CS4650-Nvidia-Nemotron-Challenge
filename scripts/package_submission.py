#!/usr/bin/env python3
"""Command-line entrypoint for packaging a Kaggle submission bundle.

Thin wrapper over :func:`src.inference.submission.build_submission`. The
produced ``submission.zip`` contains EXACTLY
``adapter_config.json`` + ``adapter_model.safetensors`` at the zip root;
provenance lives in ``submission.manifest.json`` beside the zip.

Usage:
    python scripts/package_submission.py \\
        --adapter-dir /path/to/adapter \\
        --output-dir experiments/submissions/<run_id> \\
        --base-model-id nvidia/Nemotron-Nano-9B-v2 \\
        --adapter-rank 32 \\
        --dataset-version v0 \\
        --eval-sha 0000000

Exits non-zero on any failure and prints an error message to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/package_submission.py`` to work without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference.submission import build_submission  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a LoRA adapter directory into a Kaggle-ready "
            "submission.zip with a sibling submission.manifest.json."
        )
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
        help="HF model id (or local path) of the base model, e.g. nvidia/Nemotron-Nano-9B-v2.",
    )
    parser.add_argument(
        "--adapter-rank",
        type=int,
        required=True,
        help="LoRA rank used during fine-tuning (int).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        required=True,
        help="Version tag of the training dataset (e.g. v0, 2026-04-20).",
    )
    parser.add_argument(
        "--eval-sha",
        type=str,
        required=True,
        help="SHA (git-style) identifying the eval record this adapter was validated against.",
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Override for source git commit SHA; auto-detected via `git rev-parse HEAD` if omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        bundle = build_submission(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            base_model_id=args.base_model_id,
            adapter_rank=args.adapter_rank,
            dataset_version=args.dataset_version,
            eval_sha=args.eval_sha,
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
