"""Validation for the beginner-facing ``docs/learn`` content pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REQUIRED_METADATA: tuple[str, ...] = (
    "title",
    "audience",
    "page_type",
    "status",
    "last_reviewed",
    "repo_sources",
    "external_sources",
)

VALID_PAGE_TYPES: frozenset[str] = frozenset(
    {
        "index",
        "concept",
        "project-status",
        "roadmap",
        "glossary",
        "source-policy",
        "source-ledger",
        "template",
    }
)
VALID_STATUSES: frozenset[str] = frozenset(
    {"implemented", "in_progress", "planned", "conceptual"}
)
REQUIRED_HEADINGS: tuple[str, ...] = ("## Why This Page Exists", "## Sources")
STATUS_HEADINGS: dict[str, str] = {
    "implemented": "## What Exists Today",
    "in_progress": "## What Is In Progress",
    "planned": "## What Is Planned Next",
}


def validate_learn_docs_tree(root: Path) -> list[str]:
    """Validate the entire ``docs/learn`` tree."""
    if not root.exists():
        return [f"missing docs tree: {root}"]
    if not root.is_dir():
        return [f"docs tree is not a directory: {root}"]

    errors: list[str] = []
    markdown_files = sorted(root.rglob("*.md"))
    if not markdown_files:
        return [f"no markdown files found under {root}"]

    ledger_urls = _ledger_urls(root / "sources" / "citation-ledger.md")
    for path in markdown_files:
        errors.extend(validate_learn_doc(path, ledger_urls=ledger_urls))
    return errors


def validate_learn_doc(
    path: Path,
    *,
    ledger_urls: set[str] | None = None,
) -> list[str]:
    """Validate one ``docs/learn`` page."""
    errors: list[str] = []
    metadata, body = _load_frontmatter(path, errors)
    if metadata is None:
        return errors

    errors.extend(_validate_metadata(path, metadata))
    errors.extend(_validate_body(path, metadata, body))
    errors.extend(_validate_ledger_coverage(path, metadata, ledger_urls))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="docs/learn",
        help="Path to the docs/learn root.",
    )
    args = parser.parse_args(argv)

    errors = validate_learn_docs_tree(Path(args.root))
    if errors:
        for error in errors:
            print(f"learn-docs-guard: {error}", file=sys.stderr)
        return 1
    return 0


def _load_frontmatter(
    path: Path,
    errors: list[str],
) -> tuple[dict[str, Any] | None, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        errors.append(f"{path}: missing YAML frontmatter")
        return None, text

    try:
        _, raw_frontmatter, body = text.split("---\n", 2)
    except ValueError:
        errors.append(f"{path}: malformed YAML frontmatter")
        return None, text

    try:
        loaded = _parse_frontmatter_mapping(raw_frontmatter)
    except ValueError as exc:
        errors.append(f"{path}: invalid YAML frontmatter: {exc}")
        return None, body

    if not isinstance(loaded, dict):
        errors.append(f"{path}: frontmatter must be a mapping")
        return None, body
    return loaded, body


def _parse_frontmatter_mapping(raw_frontmatter: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_list_key: str | None = None

    for line_number, raw_line in enumerate(raw_frontmatter.splitlines(), start=1):
        if not raw_line.strip():
            continue

        if raw_line.startswith("  - "):
            if current_list_key is None:
                raise ValueError(
                    f"line {line_number}: list item without a preceding key"
                )
            metadata[current_list_key].append(_parse_scalar(raw_line[4:].strip()))
            continue

        current_list_key = None
        if ":" not in raw_line:
            raise ValueError(f"line {line_number}: expected 'key: value'")

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError(f"line {line_number}: empty key")

        if value == "":
            metadata[key] = []
            current_list_key = key
            continue

        if value == "[]":
            metadata[key] = []
            continue

        metadata[key] = _parse_scalar(value)

    return metadata


def _parse_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _validate_metadata(path: Path, metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_METADATA:
        if field not in metadata:
            errors.append(f"{path}: missing required frontmatter field {field!r}")

    audience = metadata.get("audience")
    if audience != "beginner":
        errors.append(f"{path}: audience must be 'beginner'")

    page_type = metadata.get("page_type")
    if page_type not in VALID_PAGE_TYPES:
        errors.append(
            f"{path}: page_type must be one of {sorted(VALID_PAGE_TYPES)!r}"
        )

    status = metadata.get("status")
    if status not in VALID_STATUSES:
        errors.append(
            f"{path}: status must be one of {sorted(VALID_STATUSES)!r}"
        )

    for list_field in ("repo_sources", "external_sources"):
        value = metadata.get(list_field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            errors.append(f"{path}: {list_field} must be a list[str]")

    return errors


def _validate_body(path: Path, metadata: dict[str, Any], body: str) -> list[str]:
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in body:
            errors.append(f"{path}: missing required heading {heading!r}")

    page_type = metadata.get("page_type")
    if page_type in {"concept", "glossary"} and not metadata.get(
        "external_sources"
    ):
        errors.append(
            f"{path}: concept/glossary pages must declare external_sources"
        )

    if page_type in {"project-status", "roadmap"}:
        if "## Current Repo Evidence" not in body:
            errors.append(
                f"{path}: project/roadmap pages must include '## Current Repo Evidence'"
            )
        status = metadata.get("status")
        required_heading = STATUS_HEADINGS.get(status)
        if required_heading and required_heading not in body:
            errors.append(
                f"{path}: status {status!r} requires heading {required_heading!r}"
            )

    for source in metadata.get("repo_sources", []):
        if source not in body:
            errors.append(f"{path}: repo source {source!r} missing from body")
    for source in metadata.get("external_sources", []):
        if source not in body:
            errors.append(f"{path}: external source {source!r} missing from body")

    return errors


def _ledger_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return {
        token.strip("()<>`")
        for token in text.split()
        if token.startswith("http://") or token.startswith("https://")
    }


def _validate_ledger_coverage(
    path: Path,
    metadata: dict[str, Any],
    ledger_urls: set[str] | None,
) -> list[str]:
    if ledger_urls is None or path.name == "citation-ledger.md":
        return []
    errors: list[str] = []
    for source in metadata.get("external_sources", []):
        if source not in ledger_urls:
            errors.append(
                f"{path}: external source {source!r} is missing from citation-ledger.md"
            )
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
