"""Regression tests for HPC checkpoint and SFT launcher wiring."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from scripts.hpc import checkpoint_policy, resume_from_latest, run_sft


def test_resume_from_latest_finds_huggingface_checkpoint_dirs(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-00010").mkdir()
    (tmp_path / "checkpoint-00025").mkdir()
    (tmp_path / "step-99999").mkdir()

    latest = resume_from_latest._find_latest(tmp_path)

    assert latest == tmp_path / "checkpoint-00025"


def test_checkpoint_policy_rotates_huggingface_checkpoint_dirs(tmp_path: Path) -> None:
    for step in (10, 20, 30, 40):
        (tmp_path / f"checkpoint-{step:05d}").mkdir()
    (tmp_path / "step-99999").mkdir()

    checkpoint_policy._apply_policy(tmp_path, run_config_source=None, execute=True)

    assert not (tmp_path / "checkpoint-00010").exists()
    assert (tmp_path / "checkpoint-00020").exists()
    assert (tmp_path / "checkpoint-00030").exists()
    assert (tmp_path / "checkpoint-00040").exists()
    assert (tmp_path / "step-99999").exists()


def test_shard_dataset_honors_dataset_max_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_path = tmp_path / "shard_00000.pt"
    shard_path.touch()

    fake_torch = types.SimpleNamespace(
        load=lambda path, weights_only=False: [
            {"input_ids": [idx], "labels": [-100]} for idx in range(5)
        ]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    dataset = run_sft.ShardDataset(tmp_path, max_rows=2)

    assert len(dataset) == 2
    assert dataset[0]["input_ids"] == [0]
    assert dataset[1]["input_ids"] == [1]


def test_passthrough_collator_uses_tokenizer_pad_token_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor(list):
        def __eq__(self, other: object) -> "FakeTensor":
            return FakeTensor(
                [
                    value == other
                    for row in self
                    for value in (row if isinstance(row, list) else [row])
                ]
            )

        def any(self) -> bool:
            return any(bool(value) for value in self)

    fake_torch = types.SimpleNamespace(
        long="long",
        tensor=lambda values, dtype=None: FakeTensor(values),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    collator = run_sft.PassthroughCollator(pad_token_id=123)
    batch = collator(
        [
            {"input_ids": [9, 8, 7], "labels": [-100, 8, 7]},
            {"input_ids": [6], "labels": [-100]},
        ]
    )

    assert batch["input_ids"][1] == [6, 123, 123]
    assert batch["labels"][1] == [-100, -100, -100]


def test_validate_config_rejects_bad_config_before_model_load() -> None:
    with pytest.raises(ValueError, match="dataset_max_rows"):
        run_sft.validate_config({"base_model": "model", "dataset_max_rows": 0})


def test_submit_sft_uses_checkpoint_dir_for_training_and_resume() -> None:
    script = Path("scripts/hpc/submit_sft.sbatch").read_text()

    assert 'export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in script
    assert 'resume_from_latest.py --checkpoint-dir "$CHECKPOINT_DIR"' in script
    assert '--output-dir "$CHECKPOINT_DIR"' in script
    assert '"${RESUME_ARG[@]}"' in script
    assert "checkpoint_policy.py" not in script.split("=== Step 4:", 1)[-1]
