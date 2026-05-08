#!/usr/bin/env python3
"""SFT training launcher for the Nemotron Challenge.

CLI entry point that:
  - Resolves extends: config chains (deep-merge child on top of parent)
  - Loads base model + optional 4-bit quantisation (QLoRA)
  - Wraps the model with a LoRA adapter via peft
  - Wraps pre-tokenized .pt shards in a Dataset (labels already masked)
  - Runs trl.SFTTrainer with a passthrough DataCollator (NO re-masking)
  - Applies checkpoint rotation policy post-training

The shards produced by tokenize_dataset.py already have apply_loss_mask
applied, so this script must NOT call apply_loss_mask again.

Exit 0 on success, 1 on any unhandled exception.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
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

IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT training launcher for Nemotron pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML training config (supports extends: chain resolution).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing pre-tokenized .pt shard files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path to resume from, or 'true' to let the Trainer "
            "auto-detect the latest checkpoint in --output-dir."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading with extends: chain resolution
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with *override* merged on top of *base* (child wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config, resolving any extends: chain recursively."""
    raw = _load_yaml(config_path)
    parent_path_str: str | None = raw.pop("extends", None)
    if parent_path_str is None:
        return raw

    parent_path = Path(parent_path_str)
    if not parent_path.is_absolute():
        parent_path = _REPO_ROOT / parent_path

    log.info("Resolving extends: %s -> %s", config_path.name, parent_path)
    parent_cfg = load_config(parent_path)
    return _deep_merge(parent_cfg, raw)


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]

    model_id: str = cfg.get("base_model") or cfg.get("model_name_or_path", "")
    if not model_id:
        raise ValueError("Config must specify 'base_model' or 'model_name_or_path'.")

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if cfg.get("load_in_4bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs.pop("torch_dtype", None)
        log.info("4-bit quantisation enabled (QLoRA mode).")

    log.info("Loading model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    log.info("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA / PEFT wrapping
# ---------------------------------------------------------------------------


def _apply_lora(model: Any, cfg: dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]

    if cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
        log.info("Model prepared for k-bit training.")

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules") or None,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset wrapping pre-tokenized .pt shards
# ---------------------------------------------------------------------------


class ShardDataset:
    """In-memory dataset over pre-tokenized .pt shard files.

    Each shard is a list of dicts: {"input_ids": [...], "labels": [...]}.
    Labels have already been masked by apply_loss_mask during tokenization.
    """

    def __init__(self, data_dir: Path) -> None:
        import torch  # type: ignore[import]

        shard_files = sorted(data_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No .pt shards found in {data_dir}")
        log.info("Loading %d shard file(s) from %s", len(shard_files), data_dir)

        self._examples: list[dict[str, list[int]]] = []
        for shard_path in shard_files:
            shard: list[dict[str, list[int]]] = torch.load(
                shard_path, weights_only=False
            )
            self._examples.extend(shard)

        log.info("Total examples: %d", len(self._examples))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._examples[idx]


# ---------------------------------------------------------------------------
# Passthrough DataCollator — NO re-masking
# ---------------------------------------------------------------------------


class PassthroughCollator:
    """Pad and stack pre-masked tensors without modifying labels.

    Padding positions use IGNORE_INDEX (-100) in labels so they are
    excluded from the loss — consistent with the pre-masking applied
    by tokenize_dataset.py.

    Asserts that the collated batch contains at least one -100 label,
    confirming that apply_loss_mask was already applied during tokenization.
    """

    def __call__(
        self, features: list[dict[str, list[int]]]
    ) -> dict[str, Any]:
        import torch  # type: ignore[import]

        max_len = max(len(f["input_ids"]) for f in features)

        padded_input_ids: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            padded_input_ids.append(f["input_ids"] + [0] * pad_len)
            padded_labels.append(f["labels"] + [IGNORE_INDEX] * pad_len)
            attention_masks.append([1] * seq_len + [0] * pad_len)

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

        assert (batch["labels"] == IGNORE_INDEX).any(), (
            "Expected pre-masked labels (== -100) in batch. "
            "tokenize_dataset.py should have applied apply_loss_mask — "
            "do NOT call apply_loss_mask again in run_sft.py."
        )
        return batch


# ---------------------------------------------------------------------------
# TrainingArguments from config
# ---------------------------------------------------------------------------


def _build_training_args(cfg: dict[str, Any], output_dir: Path) -> Any:
    from transformers import TrainingArguments  # type: ignore[import]

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        log.info("WANDB_API_KEY detected — reporting to wandb.")

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "save_steps": int(cfg.get("save_steps", 100)),
        "bf16": True,
        "report_to": report_to,
        # Required so our custom collator keys survive Trainer's column filter.
        "remove_unused_columns": False,
        "dataloader_drop_last": False,
    }

    for int_key in ("max_steps", "warmup_steps", "logging_steps", "eval_steps"):
        if int_key in cfg:
            kwargs[int_key] = int(cfg[int_key])

    for float_key in ("learning_rate", "weight_decay", "max_grad_norm"):
        if float_key in cfg:
            kwargs[float_key] = float(cfg[float_key])

    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint policy
# ---------------------------------------------------------------------------


def _run_checkpoint_policy(output_dir: Path) -> None:
    from scripts.hpc.checkpoint_policy import main as policy_main  # type: ignore[import]

    log.info("Running checkpoint rotation policy on: %s", output_dir)
    rc = policy_main(["--checkpoint-dir", str(output_dir), "--execute"])
    if rc != 0:
        log.warning("Checkpoint policy exited with non-zero code: %d", rc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        log.info("Resolving config: %s", args.config)
        cfg = load_config(args.config)
        log.info("Effective config: %s", cfg)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        model, _tokenizer = _load_model_and_tokenizer(cfg)
        model = _apply_lora(model, cfg)

        dataset = ShardDataset(args.data_dir)
        collator = PassthroughCollator()
        training_args = _build_training_args(cfg, args.output_dir)

        from trl import SFTTrainer  # type: ignore[import]

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        # Normalise --resume-from-checkpoint: the string "true" activates
        # Trainer's auto-detect behaviour (scans output_dir for checkpoints).
        resume: str | bool | None = args.resume_from_checkpoint
        if isinstance(resume, str) and resume.lower() == "true":
            resume = True

        log.info("Starting training (resume_from_checkpoint=%s).", resume)
        trainer.train(resume_from_checkpoint=resume or None)
        log.info("Training complete.")

        _run_checkpoint_policy(args.output_dir)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
