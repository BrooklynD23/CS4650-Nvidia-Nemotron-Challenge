"""Artifact contract helpers for reserved splits.

The split artifacts under ``data/eval/`` must be *versioned* so the
golden set stays immutable after first approval (Issue #18 policy).
This module owns two concerns:

1. A path check so reviewers and CI can catch an unversioned
   ``golden.jsonl`` before it is committed.
2. A tiny selection-manifest format that lives beside each split and
   records the provenance needed to reproduce it on a clean machine.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


# Accepted forms of the version suffix on the golden artifact's stem.
# ``golden_v1.jsonl`` and ``golden_20.jsonl`` both pass; ``golden.jsonl``
# fails (no suffix at all).
GOLDEN_ARTIFACT_VERSION_PATTERN: str = r"^v\d+$"
"""Regex for a canonical ``v<N>`` version suffix on the stem.

Callers MAY also use a bare numeric suffix (``golden_20``) to indicate
a row-count-based pin; see :func:`is_immutable_golden_path`.
"""

_GOLDEN_STEM_PATTERN: re.Pattern[str] = re.compile(
    r"^golden_(?:v\d+|\d+)$"
)

_MANIFEST_FIELDS: tuple[str, ...] = (
    "dataset_version",
    "selection_seed",
    "selection_rule",
    "row_count",
    "created_at",
)


def is_immutable_golden_path(path: Path) -> bool:
    """Return True if ``path`` is a properly versioned golden artifact.

    A file is considered immutable-safe when the stem matches
    ``golden_v<N>`` or ``golden_<N>`` (N an integer). A bare
    ``golden.jsonl`` fails because it invites in-place edits.

    The caller is responsible for ensuring the file lives under the
    expected ``data/eval/`` root; this function only checks the name
    pattern so tests can exercise it with tmp paths.
    """
    if not isinstance(path, Path):
        raise TypeError(
            "is_immutable_golden_path: expected Path, got "
            f"{type(path).__name__}"
        )
    if path.suffix != ".jsonl":
        return False
    return bool(_GOLDEN_STEM_PATTERN.match(path.stem))


def write_selection_manifest(
    *,
    path: Path,
    dataset_version: str,
    selection_seed: int,
    selection_rule: str,
    row_count: int,
) -> None:
    """Write the sidecar JSON manifest for a split artifact.

    ``created_at`` is stamped at write time in ISO-8601 UTC. The
    manifest sits beside the JSONL split file so an eval run can
    validate provenance without loading the split body.
    """
    if not isinstance(path, Path):
        raise TypeError(
            "write_selection_manifest: expected Path, got "
            f"{type(path).__name__}"
        )
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("write_selection_manifest: dataset_version required")
    if not isinstance(selection_rule, str) or not selection_rule:
        raise ValueError("write_selection_manifest: selection_rule required")
    if isinstance(selection_seed, bool) or not isinstance(
        selection_seed, int
    ):
        raise TypeError(
            "write_selection_manifest: selection_seed must be int, got "
            f"{type(selection_seed).__name__}"
        )
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError(
            "write_selection_manifest: row_count must be int, got "
            f"{type(row_count).__name__}"
        )
    if row_count < 0:
        raise ValueError(
            f"write_selection_manifest: row_count must be >= 0, got "
            f"{row_count}"
        )

    payload: dict[str, object] = {
        "dataset_version": dataset_version,
        "selection_seed": selection_seed,
        "selection_rule": selection_rule,
        "row_count": row_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_selection_manifest(path: Path) -> dict:
    """Load and validate the sidecar JSON manifest at ``path``.

    Raises:
        ValueError: if any required field is missing or has the wrong
            type.
    """
    if not isinstance(path, Path):
        raise TypeError(
            "load_selection_manifest: expected Path, got "
            f"{type(path).__name__}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"load_selection_manifest: {path} did not contain a JSON "
            f"object (got {type(data).__name__})"
        )
    missing = [f for f in _MANIFEST_FIELDS if f not in data]
    if missing:
        raise ValueError(
            f"load_selection_manifest: {path} missing fields {missing}"
        )
    # Type sanity: catch corruption early.
    if not isinstance(data["dataset_version"], str):
        raise ValueError(
            "load_selection_manifest: dataset_version must be str"
        )
    if isinstance(data["selection_seed"], bool) or not isinstance(
        data["selection_seed"], int
    ):
        raise ValueError("load_selection_manifest: selection_seed must be int")
    if not isinstance(data["selection_rule"], str):
        raise ValueError("load_selection_manifest: selection_rule must be str")
    if isinstance(data["row_count"], bool) or not isinstance(
        data["row_count"], int
    ):
        raise ValueError("load_selection_manifest: row_count must be int")
    if not isinstance(data["created_at"], str):
        raise ValueError("load_selection_manifest: created_at must be str")
    return {k: data[k] for k in _MANIFEST_FIELDS}


__all__ = [
    "GOLDEN_ARTIFACT_VERSION_PATTERN",
    "is_immutable_golden_path",
    "write_selection_manifest",
    "load_selection_manifest",
]
