"""Regression checks for notebook 04 evaluation contract.

These tests are intentionally lightweight and text-based so they can run in
CI without executing the full notebook kernel.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "04_baseline_eval_and_normalization.ipynb"
)


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_does_not_special_case_golden_prefix() -> None:
    source = _notebook_source()
    assert 'startswith("gold-")' not in source


def test_notebook_roundtrip_checks_all_records() -> None:
    source = _notebook_source()
    assert "for rec in reloaded_records[:3]:" not in source
    assert "for rec in reloaded_records:" in source
