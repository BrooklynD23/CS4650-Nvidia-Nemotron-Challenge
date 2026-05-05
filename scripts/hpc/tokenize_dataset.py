#!/usr/bin/env python3
"""Tokenize a JSONL of SFTExample rows into .pt shards for SFT training.

Reads a JSONL file where each line is a JSON-serialised SFTExample, applies
the tokenizer's chat template, calls apply_loss_mask() to mask non-assistant
tokens, then writes fixed-size .pt shard files.

Usage:
    python scripts/hpc/tokenize_dataset.py \\
        --input  data/processed/train.jsonl \\
        --output data/processed/shards/ \\
        --config configs/lora_sft.yaml

Each output shard is a list of dicts saved via torch.save:
    [{"input_ids": [int, ...], "labels": [int, ...]}, ...]

A dataset_fingerprint.json manifest is written alongside the shards.

Exit 0 on success, 1 on any error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        description="Tokenize SFTExample JSONL into .pt shards for SFT training.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file (one SFTExample per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for .pt shard files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML training config (reads model_name_or_path, max_seq_len, shard_size).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Override: number of examples per shard (default: from config or 1000).",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        log.warning("PyYAML not available; using empty config defaults.")
        return {}
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


def _load_raw_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
    return rows


def _apply_chat_template_and_tokenize(
    messages: list[dict[str, str]],
    completion: str,
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with the full chat text tokenized."""
    all_messages = list(messages)
    if not all_messages or all_messages[-1].get("role") != "assistant":
        all_messages.append({"role": "assistant", "content": completion})

    text: str = tokenizer.apply_chat_template(
        all_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids: list[int] = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )
    return token_ids, list(token_ids)


def _run(
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    shard_size_override: int | None,
) -> None:
    import torch  # type: ignore[import]
    from transformers import AutoTokenizer  # type: ignore[import]

    from src.training.sft_trainer import apply_loss_mask

    model_name_or_path: str = config.get(
        "model_name_or_path",
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    )
    max_seq_len: int = int(config.get("max_seq_len", 8192))
    shard_size: int = shard_size_override or int(config.get("shard_size", 1000))

    log.info("Loading tokenizer from: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    log.info("Reading examples from: %s", input_path)
    raw_rows = _load_raw_rows(input_path)
    log.info("Loaded %d rows", len(raw_rows))

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer: list[dict[str, list[int]]] = []
    skipped = 0

    def _flush(buf: list[dict[str, list[int]]], idx: int) -> None:
        shard_path = output_dir / f"shard_{idx:05d}.pt"
        torch.save(buf, shard_path)
        log.info("Saved shard %05d (%d examples) -> %s", idx, len(buf), shard_path)

    for row_num, raw in enumerate(raw_rows):
        messages: list[dict[str, str]] = raw.get("messages", [])
        completion: str = raw.get("completion", "")
        if not messages or not completion:
            log.warning("Row %d missing messages or completion; skipping.", row_num)
            skipped += 1
            continue

        try:
            input_ids, labels = _apply_chat_template_and_tokenize(
                messages, completion, tokenizer, max_seq_len
            )
            masked_labels = apply_loss_mask(input_ids, labels, tokenizer)
        except Exception as exc:
            log.warning("Row %d tokenization/masking failed (%s); skipping.", row_num, exc)
            skipped += 1
            continue

        buffer.append({"input_ids": input_ids, "labels": masked_labels})
        if len(buffer) >= shard_size:
            _flush(buffer, shard_idx)
            shard_idx += 1
            buffer = []

    if buffer:
        _flush(buffer, shard_idx)
        shard_idx += 1

    kept = len(raw_rows) - skipped
    log.info(
        "Done: %d kept / %d skipped -> %d shards", kept, skipped, shard_idx
    )

    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha,
        "total_examples": len(raw_rows),
        "kept_examples": kept,
        "skipped_examples": skipped,
        "shard_count": shard_idx,
        "shard_size": shard_size,
        "max_seq_len": max_seq_len,
        "model_name_or_path": model_name_or_path,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = output_dir / "dataset_fingerprint.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info("Fingerprint manifest written: %s", manifest_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_file():
        log.error("Input file not found: %s", args.input)
        return 1
    if not args.config.is_file():
        log.error("Config file not found: %s", args.config)
        return 1

    try:
        config = _load_config(args.config)
        _run(args.input, args.output, config, args.shard_size)
    except Exception as exc:
        log.error("Tokenization failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
