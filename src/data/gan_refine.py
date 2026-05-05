"""GAN-style refinement loop for synthetic data augmentation (#34 / Wave E).

Implements an iterative generate→evaluate→reject→negative-construction loop
using a fine-tuned Nemotron checkpoint as the generator and the Verifier
Protocol from src/inference/solver.py as the evaluator.

This module is CONDITIONAL: only activate after SFT produces a checkpoint
AND time/compute remain before the competition deadline.

Running as a module::

    python -m src.data.gan_refine --max-iterations 1 --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.contracts import SFTExample
from src.data.synthetic import write_sft_examples_jsonl


_DEFAULT_MAX_ITERATIONS = 3
_DEFAULT_CONFIDENCE_THRESHOLD = 0.7
_DEFAULT_COST_CAP_USD = 20.0


@dataclass
class GANConfig:
    """Runtime configuration for one GAN refinement run."""

    output_base_dir: Path
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD
    cost_cap_per_iteration_usd: float = _DEFAULT_COST_CAP_USD
    verifier: Any = None  # Verifier Protocol instance (required for real runs)
    generator_fn: Callable[[str, str], str] | None = None  # (prompt, category) -> completion


class GANLoop:
    """Generate → evaluate → reject → construct negatives → append to batch."""

    def __init__(self, config: GANConfig) -> None:
        self._config = config

    def _generate(self, prompt: str, category: str) -> str | None:
        if self._config.generator_fn is None:
            return None
        try:
            return self._config.generator_fn(prompt, category)
        except Exception:
            return None

    def _verify(self, pred: str, gold: str) -> bool:
        if self._config.verifier is None:
            return False
        try:
            return bool(self._config.verifier.verify(pred, gold))
        except Exception:
            return False

    def _make_negative(
        self,
        base_example: SFTExample,
        rejected_completion: str,
        iteration: int,
    ) -> SFTExample:
        """Construct a per-step negative from a rejected completion.

        The rejected completion is appended to `provenance` so downstream
        SFT masking can use it for contrastive training, while the `completion`
        field holds the gold answer (if known) or is left as-is.
        """
        prov = dict(base_example.provenance)
        prov["gan_iteration"] = iteration
        prov["rejected_completion"] = rejected_completion
        prov["teacher"] = f"gan_round_{iteration}"
        prov["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return SFTExample(
            example_id=base_example.example_id,
            category=base_example.category,
            messages=base_example.messages,
            completion=base_example.completion,
            source=base_example.source,
            split=base_example.split,
            provenance=prov,
        )

    def run(
        self,
        seed_examples: list[SFTExample],
        *,
        dry_run: bool = False,
    ) -> list[list[SFTExample]]:
        """Execute the GAN loop for up to max_iterations rounds.

        Args:
            seed_examples: Initial SFTExample batch (from Branch 3 or direct).
            dry_run: If True, print plan and return empty lists.

        Returns:
            One list of SFTExample per iteration (including negatives).
            Length ≤ max_iterations.
        """
        cfg = self._config

        if dry_run:
            print(f"[gan-refine] dry-run: max_iterations={cfg.max_iterations}")
            print(f"[gan-refine] cost_cap_per_iteration=${cfg.cost_cap_per_iteration_usd}")
            print(f"[gan-refine] seed_examples={len(seed_examples)}")
            print(f"[gan-refine] verifier={'set' if cfg.verifier else 'not set'}")
            print(f"[gan-refine] generator_fn={'set' if cfg.generator_fn else 'not set'}")
            return []

        all_rounds: list[list[SFTExample]] = []

        for iteration in range(1, cfg.max_iterations + 1):
            round_cost = 0.0
            round_batch: list[SFTExample] = []

            for example in seed_examples:
                if round_cost >= cfg.cost_cap_per_iteration_usd:
                    print(
                        f"[gan-refine] iteration {iteration}: cost cap "
                        f"${cfg.cost_cap_per_iteration_usd} reached; "
                        f"stopping at {len(round_batch)} examples"
                    )
                    break

                # Get prompt from messages (last user message)
                prompt = ""
                for msg in reversed(example.messages):
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break

                completion = self._generate(prompt, example.category)
                round_cost += 0.01  # placeholder cost per generator call

                if completion is None:
                    continue

                # Evaluate: if the generated completion passes verification, skip
                # (already correct); if it fails, construct a negative
                if self._verify(completion, example.completion):
                    # Generator produces correct answer → add as positive
                    round_batch.append(example)
                else:
                    # Rejected: construct per-step negative
                    negative = self._make_negative(example, completion, iteration)
                    round_batch.append(negative)

            if not round_batch:
                print(f"[gan-refine] iteration {iteration}: no examples produced; stopping early")
                break

            # Write round artifacts
            out_dir = cfg.output_base_dir / f"gan_round_{iteration}"
            out_path = out_dir / "batch.jsonl"
            write_sft_examples_jsonl(round_batch, out_path)
            print(f"[gan-refine] iteration {iteration}: {len(round_batch)} examples → {out_path}")

            all_rounds.append(round_batch)
            seed_examples = round_batch  # next iteration seeds from current round

        return all_rounds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GAN-style refinement loop")
    parser.add_argument(
        "--max-iterations", type=int, default=_DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations (default {_DEFAULT_MAX_ITERATIONS}; hard cap)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no generation")
    parser.add_argument("--output-dir", default="data/synthetic", help="Base output directory")
    args = parser.parse_args(argv)

    cfg = GANConfig(
        output_base_dir=Path(args.output_dir),
        max_iterations=min(args.max_iterations, _DEFAULT_MAX_ITERATIONS),
    )

    loop = GANLoop(cfg)
    print("[gan-refine] No seed examples provided — pass via API.")
    loop.run([], dry_run=args.dry_run or True)


if __name__ == "__main__":
    _main()


__all__ = [
    "GANConfig",
    "GANLoop",
]
