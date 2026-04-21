"""Regression checks for notebook 05 prompt/decode sweep contract."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "05_prompting_and_decode_sweeps.ipynb"
)


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_requires_real_split_artifacts() -> None:
    source = _notebook_source()
    assert "load_required_splits(REPO_ROOT)" in source
    assert "_SYNTHETIC_VAL" not in source
    assert "_SYNTHETIC_GOLDEN" not in source


def test_notebook_reuses_eval_runner_and_golden_gate() -> None:
    source = _notebook_source()
    assert "run_baseline_eval(" in source
    assert "evaluate_golden_gate(" in source
    assert "expected_model_id=MODEL_ID" in source
    assert "expected_normalizer_id=NORMALIZER_ID" in source


def test_notebook_writes_csv_and_findings_outputs() -> None:
    source = _notebook_source()
    assert "prompting_sweep_" in source
    assert "prompting_findings.md" in source
    assert "delta_vs_baseline" in source


def test_notebook_uses_explicit_input_device_for_generation() -> None:
    source = _notebook_source()
    assert "INPUT_DEVICE = model.get_input_embeddings().weight.device" in source
    assert "value.to(INPUT_DEVICE)" in source
    assert 'torch.Generator(device=str(INPUT_DEVICE))' in source
