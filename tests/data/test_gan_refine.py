"""Tests for src/data/gan_refine.py (#34 / Wave E)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.contracts import SFTExample
from src.data.gan_refine import GANConfig, GANLoop


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_example(example_id: str = "ex1", category: str = "math") -> SFTExample:
    return SFTExample(
        example_id=example_id,
        category=category,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        completion=r"\boxed{4}",
        source="synthetic",
        split="train",
        provenance={
            "teacher": "test",
            "generated_at": "2026-05-04T00:00:00Z",
            "source_run_id": "run-test",
        },
    )


def _make_config(tmp_path: Path, **kwargs) -> GANConfig:
    return GANConfig(
        output_base_dir=tmp_path / "synthetic",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# GANLoop.run tests
# ---------------------------------------------------------------------------

def test_dry_run_returns_empty(tmp_path, capsys):
    examples = [_make_example(f"ex{i}") for i in range(5)]
    config = _make_config(tmp_path)
    loop = GANLoop(config)
    result = loop.run(examples, dry_run=True)
    assert result == []
    captured = capsys.readouterr()
    assert "dry-run" in captured.out


def test_loop_terminates_at_max_iterations(tmp_path):
    examples = [_make_example(f"ex{i}") for i in range(3)]
    config = _make_config(
        tmp_path,
        max_iterations=2,
        generator_fn=lambda prompt, cat: r"\boxed{99}",
    )
    # Verifier always returns False → all examples become negatives
    verifier = MagicMock()
    verifier.verify.return_value = False
    config.verifier = verifier

    loop = GANLoop(config)
    rounds = loop.run(examples)
    assert len(rounds) <= 2


def test_cost_cap_aborts_iteration(tmp_path, capsys):
    examples = [_make_example(f"ex{i}") for i in range(200)]
    config = _make_config(
        tmp_path,
        max_iterations=1,
        cost_cap_per_iteration_usd=0.005,  # stops after first generator call
        generator_fn=lambda prompt, cat: r"\boxed{99}",
    )
    verifier = MagicMock()
    verifier.verify.return_value = False
    config.verifier = verifier

    loop = GANLoop(config)
    rounds = loop.run(examples)
    captured = capsys.readouterr()
    assert "cost cap" in captured.out
    # Only 1 round started; may be empty or short
    assert len(rounds) <= 1


def test_rejection_collection_builds_negatives(tmp_path):
    examples = [_make_example(f"ex{i}") for i in range(3)]
    config = _make_config(
        tmp_path,
        max_iterations=1,
        generator_fn=lambda prompt, cat: r"\boxed{wrong}",
    )
    verifier = MagicMock()
    verifier.verify.return_value = False  # always rejected
    config.verifier = verifier

    loop = GANLoop(config)
    rounds = loop.run(examples)
    assert len(rounds) == 1
    for ex in rounds[0]:
        # Rejected completions become negatives with rejected_completion in provenance
        assert "rejected_completion" in ex.provenance
        assert ex.provenance["rejected_completion"] == r"\boxed{wrong}"


def test_per_step_negative_provenance_fields(tmp_path):
    example = _make_example()
    config = _make_config(
        tmp_path,
        max_iterations=1,
        generator_fn=lambda prompt, cat: r"\boxed{bad}",
    )
    verifier = MagicMock()
    verifier.verify.return_value = False
    config.verifier = verifier

    loop = GANLoop(config)
    rounds = loop.run([example])
    assert len(rounds) == 1
    neg = rounds[0][0]
    assert "gan_iteration" in neg.provenance
    assert neg.provenance["gan_iteration"] == 1
    assert neg.provenance["teacher"] == "gan_round_1"
    assert "generated_at" in neg.provenance


def test_no_generator_produces_empty_round(tmp_path, capsys):
    examples = [_make_example(f"ex{i}") for i in range(3)]
    config = _make_config(tmp_path, max_iterations=2, generator_fn=None)
    loop = GANLoop(config)
    rounds = loop.run(examples)
    # No generator → no completions → loop stops early
    assert rounds == []
    captured = capsys.readouterr()
    assert "stopping early" in captured.out


def test_output_files_written(tmp_path):
    examples = [_make_example(f"ex{i}") for i in range(2)]
    config = _make_config(
        tmp_path,
        max_iterations=1,
        generator_fn=lambda prompt, cat: r"\boxed{correct}",
    )
    verifier = MagicMock()
    verifier.verify.return_value = False
    config.verifier = verifier

    loop = GANLoop(config)
    loop.run(examples)

    out_path = tmp_path / "synthetic" / "gan_round_1" / "batch.jsonl"
    assert out_path.exists()
    sha_path = out_path.with_suffix(".sha256")
    assert sha_path.exists()
