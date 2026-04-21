"""Pytest config: ensure the repo root is on sys.path so ``import src``
works whether pytest is invoked from the repo root or a subdirectory.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
