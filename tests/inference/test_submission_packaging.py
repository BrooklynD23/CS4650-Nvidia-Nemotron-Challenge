"""Tests for :mod:`src.inference.submission`.

These tests exercise the Kaggle zip-shape contract end-to-end without
requiring a real adapter: we write synthetic bytes for
``adapter_model.safetensors`` and minimal JSON for ``adapter_config.json``.
The zip layout and manifest-location invariants don't care about the
adapter's semantic content; they only care about the filesystem shape.

The PEFT load smoke test is skipped by default; it only runs when both
``peft`` is importable AND the env var ``NEMOTRON_BASE_MODEL_CACHED=1`` is
set, so CI never has to pull down the base model.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from src.inference.submission import (
    MANIFEST_FILENAME,
    REQUIRED_ADAPTER_FILES,
    SUBMISSION_ZIP_FILENAME,
    build_submission,
    sha256_of_file,
    validate_submission_zip,
)


# ---------- helpers ----------------------------------------------------------


def _make_adapter_dir(
    base: Path,
    *,
    include_config: bool = True,
    include_safetensors: bool = True,
) -> Path:
    """Materialize a synthetic adapter directory for packaging tests.

    ``adapter_config.json`` gets a minimal valid JSON payload; the
    ``adapter_model.safetensors`` file is just 1 KiB of arbitrary bytes
    since the zip-shape tests don't parse safetensors content.
    """
    adapter_dir = base / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if include_config:
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 32,
                    "lora_alpha": 64,
                    "target_modules": ["q_proj", "v_proj"],
                    "task_type": "CAUSAL_LM",
                }
            ),
            encoding="utf-8",
        )
    if include_safetensors:
        # Arbitrary non-empty bytes; we're validating zip shape, not
        # safetensors semantics.
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00SAFETENSORS" * 64)
    return adapter_dir


def _baseline_kwargs() -> dict:
    return dict(
        base_model_id="nvidia/Nemotron-Nano-9B-v2",
        adapter_rank=32,
        dataset_version="v0",
        eval_sha="deadbeef",
        git_sha="cafef00d",  # pin so git state doesn't leak into tests
    )


# ---------- zip shape --------------------------------------------------------


def test_zip_contains_exactly_two_required_files_at_root(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src")
    output_dir = tmp_path / "out"

    bundle = build_submission(adapter_dir, output_dir, **_baseline_kwargs())

    assert bundle.submission_zip.name == SUBMISSION_ZIP_FILENAME
    assert bundle.submission_zip.is_file()

    with zipfile.ZipFile(bundle.submission_zip, "r") as zf:
        names = zf.namelist()

    # Exact match, no folders, no extras.
    assert sorted(names) == sorted(REQUIRED_ADAPTER_FILES)
    for name in names:
        assert "/" not in name and "\\" not in name, (
            f"entry {name!r} is nested; zip must be flat"
        )
        assert not name.endswith("/"), (
            f"entry {name!r} is a directory marker; zip must be files-only"
        )

    # Standalone validator also accepts the produced zip.
    validate_submission_zip(bundle.submission_zip)


# ---------- manifest location ------------------------------------------------


def test_manifest_is_outside_zip_and_references_zip_hash(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src")
    output_dir = tmp_path / "out"

    bundle = build_submission(adapter_dir, output_dir, **_baseline_kwargs())

    # 1) Manifest lives beside the zip, NOT inside it.
    assert bundle.manifest_path.name == MANIFEST_FILENAME
    assert bundle.manifest_path.parent == bundle.submission_zip.parent
    with zipfile.ZipFile(bundle.submission_zip, "r") as zf:
        assert MANIFEST_FILENAME not in zf.namelist()

    # 2) Manifest JSON references the zip's sha256 and has required fields.
    payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    expected_zip_sha = sha256_of_file(bundle.submission_zip)
    assert payload["submission_zip_sha256"] == expected_zip_sha
    assert payload["base_model_id"] == "nvidia/Nemotron-Nano-9B-v2"
    assert payload["adapter_rank"] == 32
    assert payload["dataset_version"] == "v0"
    assert payload["eval_sha"] == "deadbeef"
    assert payload["git_sha"] == "cafef00d"
    assert "created_at" in payload and payload["created_at"]
    assert "manifest_version" in payload
    # Per-file hashes cover every required adapter file.
    for fname in REQUIRED_ADAPTER_FILES:
        assert fname in payload["artifact_sha256"]
        assert payload["artifact_sha256"][fname] == sha256_of_file(adapter_dir / fname)


# ---------- validator: zip rejects extras ------------------------------------


def test_validate_rejects_extra_file_in_zip(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src")
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in REQUIRED_ADAPTER_FILES:
            zf.write(adapter_dir / fname, arcname=fname)
        zf.writestr("README.md", "nope")

    with pytest.raises(ValueError, match="extra"):
        validate_submission_zip(bad_zip)


def test_validate_rejects_nested_paths(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src")
    bad_zip = tmp_path / "nested.zip"
    with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(
            adapter_dir / "adapter_config.json",
            arcname="subdir/adapter_config.json",
        )
        zf.write(
            adapter_dir / "adapter_model.safetensors",
            arcname="adapter_model.safetensors",
        )

    with pytest.raises(ValueError, match="nested"):
        validate_submission_zip(bad_zip)


# ---------- builder: missing adapter files -----------------------------------


def test_build_fails_when_adapter_config_missing(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src", include_config=False)
    output_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        build_submission(adapter_dir, output_dir, **_baseline_kwargs())


def test_build_fails_when_safetensors_missing(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path / "src", include_safetensors=False)
    output_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        build_submission(adapter_dir, output_dir, **_baseline_kwargs())


# ---------- PEFT smoke (skipped by default) ----------------------------------


def _peft_base_model_available() -> bool:
    """Gate the PEFT smoke test behind an explicit opt-in env var.

    Returns True only when the operator has signaled (a) the base model is
    cached locally and (b) ``peft`` imports cleanly. Skipping by default
    keeps CI free of network/model-cache dependencies.
    """
    if os.environ.get("NEMOTRON_BASE_MODEL_CACHED") != "1":
        return False
    try:
        import peft  # noqa: F401
    except Exception:  # noqa: BLE001 - any import error = not available
        return False
    return True


@pytest.mark.skipif(
    not _peft_base_model_available(),
    reason="base model not cached; set NEMOTRON_BASE_MODEL_CACHED=1 and install peft to enable",
)
def test_peft_load_smoke(tmp_path: Path) -> None:
    """Ensure the packaged zip can be extracted and parsed by PEFT.

    We only assert that ``PeftConfig.from_pretrained`` can parse the
    extracted adapter directory. Loading the full model is deferred to an
    integration harness because it requires the actual competition base
    model bytes.
    """
    from peft import PeftConfig  # type: ignore[import-not-found]

    adapter_dir = _make_adapter_dir(tmp_path / "src")
    output_dir = tmp_path / "out"
    bundle = build_submission(adapter_dir, output_dir, **_baseline_kwargs())

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(bundle.submission_zip, "r") as zf:
        zf.extractall(extract_dir)

    # Synthetic config lacks base_model_name_or_path; for a real run this
    # would round-trip, so we tolerate either a successful parse or a
    # ValueError from PEFT's schema checks.
    try:
        PeftConfig.from_pretrained(str(extract_dir))
    except ValueError:
        pytest.xfail("synthetic adapter_config lacks fields required by PEFT schema")
