"""Regression tests for the git-hook wrapper script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_commit_guard_wrapper_imports_repo_module() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/commit_guard.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined
    assert "usage:" in combined
