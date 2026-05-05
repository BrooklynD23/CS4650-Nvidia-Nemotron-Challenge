"""Submission packaging for the Kaggle Nemotron Challenge.

The Kaggle ``submission.zip`` archive MUST contain EXACTLY two files at
its root, matching the konbu17 reference layout
(``data/external/konbu17/cells/cell04.py`` / ``cell15.py`` / ``cell17.py``):

* ``adapter_config.json``
* ``adapter_model.safetensors``

All provenance metadata (git SHA, base model id, adapter rank, dataset
version, eval SHA, artifact hashes, artifact paths, created_at) is written
to ``submission.manifest.json`` BESIDE the zip (never inside it). This keeps
the Kaggle archive byte-identical to the reference shape while remaining
fully auditable out-of-band.

Failure modes (all fail-closed):
    * Adapter directory missing either required file --> ``FileNotFoundError``
    * Zip contains extra entries, nested paths, or folder entries --> ``ValueError``
    * Manifest missing required fields (enforced by :class:`PackageManifest`)

Related:
    * ``docs/execution/plans/issue-20-submission-packaging-and-provenance.md``
    * ``docs/architecture/ARCHITECTURE.md`` (PackageManifest contract)
    * ``src/packaging/manifest.py`` (canonical manifest factory, owned by #17)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the canonical manifest contract landed by issue #17.
# If that module is ever removed, packaging should fail loudly here (the
# import error surfaces the broken dependency immediately) rather than
# silently fall back to a divergent local schema.
from src.contracts import PackageManifest
from src.packaging.manifest import make_package_manifest


REQUIRED_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
"""Exact file names that MUST live at the zip root."""

MANIFEST_FILENAME: str = "submission.manifest.json"
"""On-disk name of the provenance manifest (written beside the zip)."""

SUBMISSION_ZIP_FILENAME: str = "submission.zip"
"""Canonical name for the Kaggle-facing zip archive."""


@dataclass(frozen=True)
class SubmissionBundle:
    """Filesystem handles for a packaged submission."""

    adapter_dir: Path
    submission_zip: Path
    manifest_path: Path


def validate_adapter_dir(adapter_dir: Path) -> None:
    """Raise ``FileNotFoundError`` if any required adapter file is missing.

    The error message names the specific missing file so the caller (CLI
    or notebook) can surface it without extra plumbing.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found or not a directory: {adapter_dir}"
        )
    resolved_adapter_dir = adapter_dir.resolve()
    for fname in REQUIRED_ADAPTER_FILES:
        fpath = adapter_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Missing required adapter file: {fpath} "
                f"(expected one of {list(REQUIRED_ADAPTER_FILES)} in {adapter_dir})"
            )
        if fpath.is_symlink():
            raise ValueError(
                f"Required adapter file must not be a symlink: {fpath}"
            )
        resolved_file = fpath.resolve()
        try:
            resolved_file.relative_to(resolved_adapter_dir)
        except ValueError as exc:
            raise ValueError(
                "Required adapter file resolves outside adapter directory: "
                f"{fpath} -> {resolved_file}"
            ) from exc


def validate_submission_zip(zip_path: Path) -> None:
    """Assert the zip contains EXACTLY the two required files at root.

    Rejects:
        * extra entries
        * missing entries
        * directory entries (names ending with ``/``)
        * nested paths (anything containing ``/`` or ``\\``)

    Raises:
        FileNotFoundError: zip does not exist.
        ValueError: zip is malformed or does not match the required shape.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"submission zip not found: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"submission zip is not a valid zip archive: {zip_path}"
        ) from exc

    for name in names:
        if name.endswith("/"):
            raise ValueError(
                f"submission zip contains a directory entry {name!r}; "
                "only flat files at the archive root are allowed"
            )
        if "/" in name or "\\" in name:
            raise ValueError(
                f"submission zip contains nested path {name!r}; "
                "all entries must live at the archive root"
            )

    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "submission zip contains duplicate entries: "
            f"{duplicates}"
        )

    required = set(REQUIRED_ADAPTER_FILES)
    actual = set(names)
    if len(names) != len(REQUIRED_ADAPTER_FILES) or actual != required:
        missing = required - actual
        extra = actual - required
        raise ValueError(
            "submission zip does not match required layout: "
            f"expected {sorted(required)}, got {names} "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA256 digest of ``path``.

    Reads the file in 1 MiB chunks so large safetensors files don't load
    into memory all at once.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_git_sha(repo_dir: Path | None = None) -> str:
    """Best-effort ``git rev-parse HEAD`` with a safe fallback.

    Returns ``"unknown"`` if git is not available, the cwd is not a
    repository, or the command fails for any reason.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_submission_zip(adapter_dir: Path, zip_path: Path) -> None:
    """Write the Kaggle-shaped zip with flat root entries only."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in REQUIRED_ADAPTER_FILES:
            src = adapter_dir / fname
            zf.write(src, arcname=fname)


def _manifest_payload(
    manifest: PackageManifest,
    *,
    artifact_sha256: dict[str, str],
    submission_zip_sha256: str,
    submission_zip_bytes: int,
    git_sha: str,
) -> dict[str, Any]:
    """Serialize a :class:`PackageManifest` plus submission-only fields.

    The canonical :class:`PackageManifest` contract (owned by #17) is
    deliberately narrow; the packaging layer augments it with hashes and
    git provenance as out-of-band fields so reviewers can verify the exact
    bytes shipped to Kaggle. These extra fields live under explicit
    top-level keys rather than mutating the frozen dataclass.
    """
    payload: dict[str, Any] = asdict(manifest)
    payload["artifact_sha256"] = dict(artifact_sha256)
    payload["submission_zip_sha256"] = submission_zip_sha256
    payload["submission_zip_bytes"] = submission_zip_bytes
    payload["git_sha"] = git_sha
    return payload


def build_submission(
    adapter_dir: Path,
    output_dir: Path,
    *,
    base_model_id: str,
    adapter_rank: int,
    dataset_version: str,
    eval_sha: str,
    extra_artifact_paths: dict[str, str] | None = None,
    git_sha: str | None = None,
) -> SubmissionBundle:
    """Package an adapter directory into a Kaggle-ready submission bundle.

    Steps (all fail-closed):

        1. :func:`validate_adapter_dir` confirms both required files exist.
        2. ``output_dir`` is created if needed.
        3. ``submission.zip`` is written with ONLY the two adapter files at
           the zip root (no folders, no manifest, no README).
        4. SHA256 digests are computed for each adapter file and for the
           zip itself.
        5. A :class:`PackageManifest` is serialized (plus hash + git SHA
           extensions) to ``submission.manifest.json`` BESIDE the zip.
        6. :func:`validate_submission_zip` re-checks the zip shape before
           returning.

    Args:
        adapter_dir: directory containing the trained adapter.
        output_dir: destination directory; the zip and manifest are written here.
        base_model_id: HF model id or local path of the base model.
        adapter_rank: LoRA rank used during fine-tuning.
        dataset_version: version tag for the training dataset.
        eval_sha: git/commit-style SHA identifying the eval record.
        extra_artifact_paths: optional mapping of logical name to filesystem
            path for auxiliary artifacts (logs, eval reports); recorded in
            the manifest only, never embedded in the zip.
        git_sha: override for the commit SHA; auto-detected from the repo
            when ``None``.

    Returns:
        :class:`SubmissionBundle` with concrete paths for the zip and manifest.
    """
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)

    validate_adapter_dir(adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / SUBMISSION_ZIP_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    _write_submission_zip(adapter_dir, zip_path)

    artifact_sha256 = {
        fname: sha256_of_file(adapter_dir / fname)
        for fname in REQUIRED_ADAPTER_FILES
    }
    zip_sha = sha256_of_file(zip_path)
    zip_bytes = zip_path.stat().st_size

    artifact_paths: dict[str, str] = {
        "adapter_dir": str(adapter_dir.resolve()),
        "submission_zip": str(zip_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    for fname in REQUIRED_ADAPTER_FILES:
        artifact_paths[fname] = str((adapter_dir / fname).resolve())
    if extra_artifact_paths:
        for key, value in extra_artifact_paths.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "extra_artifact_paths must be dict[str, str]; "
                    f"offending entry: {key!r}: {value!r}"
                )
            artifact_paths[key] = value

    resolved_git_sha = git_sha if git_sha is not None else _detect_git_sha()

    manifest = make_package_manifest(
        base_model_id=base_model_id,
        adapter_rank=adapter_rank,
        dataset_version=dataset_version,
        eval_sha=eval_sha,
        artifact_paths=artifact_paths,
        created_at=_utcnow_iso(),
    )

    payload = _manifest_payload(
        manifest,
        artifact_sha256=artifact_sha256,
        submission_zip_sha256=zip_sha,
        submission_zip_bytes=zip_bytes,
        git_sha=resolved_git_sha,
    )

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # Final gate: re-validate the zip we just produced so packaging fails
    # closed if ZIP_DEFLATED or the caller's filesystem injected anything
    # unexpected (e.g., `__MACOSX/`, hidden dirs).
    validate_submission_zip(zip_path)

    return SubmissionBundle(
        adapter_dir=adapter_dir,
        submission_zip=zip_path,
        manifest_path=manifest_path,
    )


__all__ = [
    "REQUIRED_ADAPTER_FILES",
    "MANIFEST_FILENAME",
    "SUBMISSION_ZIP_FILENAME",
    "PackageManifest",
    "SubmissionBundle",
    "validate_adapter_dir",
    "validate_submission_zip",
    "sha256_of_file",
    "build_submission",
]
