"""Tests for :mod:`src.packaging.manifest`."""

from __future__ import annotations

from pathlib import Path

from src.contracts import MANIFEST_VERSION, PackageManifest
from src.packaging.manifest import (
    make_package_manifest,
    read_package_manifest,
    write_package_manifest,
)


def _kwargs() -> dict:
    return dict(
        base_model_id="nemotron-mini",
        adapter_rank=16,
        dataset_version="v0.1",
        eval_sha="abc123",
        artifact_paths={"adapter": "adapters/v1/"},
    )


def test_make_package_manifest_defaults_manifest_version() -> None:
    manifest = make_package_manifest(**_kwargs())
    assert manifest.manifest_version == MANIFEST_VERSION


def test_make_package_manifest_auto_populates_created_at() -> None:
    manifest = make_package_manifest(**_kwargs())
    # ISO-8601 datetimes contain a 'T' separator.
    assert "T" in manifest.created_at
    assert manifest.created_at != ""


def test_make_package_manifest_accepts_explicit_created_at() -> None:
    manifest = make_package_manifest(
        **_kwargs(), created_at="2026-04-20T00:00:00+00:00"
    )
    assert manifest.created_at == "2026-04-20T00:00:00+00:00"


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_package_manifest(
        **_kwargs(), created_at="2026-04-20T00:00:00+00:00"
    )
    write_package_manifest(manifest, path)

    loaded = read_package_manifest(path)
    assert isinstance(loaded, PackageManifest)
    assert loaded == manifest


def test_write_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "manifest.json"
    manifest = make_package_manifest(**_kwargs())
    write_package_manifest(manifest, path)
    assert path.exists()
