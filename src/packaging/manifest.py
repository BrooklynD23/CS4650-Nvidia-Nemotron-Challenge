"""Construct and (de)serialize :class:`PackageManifest` provenance cards."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import MANIFEST_VERSION, PackageManifest


def make_package_manifest(
    *,
    base_model_id: str,
    adapter_rank: int,
    dataset_version: str,
    eval_sha: str,
    artifact_paths: dict[str, str],
    manifest_version: str = MANIFEST_VERSION,
    created_at: str | None = None,
) -> PackageManifest:
    """Build a :class:`PackageManifest` with sensible defaults.

    ``created_at`` defaults to the current UTC time in ISO-8601 format
    when not explicitly provided.
    """
    effective_created_at = (
        created_at
        if created_at is not None
        else datetime.now(timezone.utc).isoformat()
    )
    return PackageManifest(
        manifest_version=manifest_version,
        base_model_id=base_model_id,
        adapter_rank=adapter_rank,
        dataset_version=dataset_version,
        eval_sha=eval_sha,
        artifact_paths=dict(artifact_paths),
        created_at=effective_created_at,
    )


def write_package_manifest(manifest: PackageManifest, path: str | Path) -> None:
    """Write ``manifest`` to ``path`` as a pretty-printed JSON object."""
    if not isinstance(manifest, PackageManifest):
        raise TypeError(
            "write_package_manifest: expected PackageManifest, got "
            f"{type(manifest).__name__}"
        )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_package_manifest(path: str | Path) -> PackageManifest:
    """Read a JSON file and return a validated :class:`PackageManifest`."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"read_package_manifest: {in_path} did not contain a JSON "
            f"object (got {type(data).__name__})"
        )
    return PackageManifest(**data)


__all__ = [
    "make_package_manifest",
    "write_package_manifest",
    "read_package_manifest",
]
