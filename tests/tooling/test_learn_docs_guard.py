"""Tests for :mod:`src.tooling.learn_docs_guard`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tooling.learn_docs_guard import (
    validate_learn_doc,
    validate_learn_docs_tree,
)


ROOT = Path(__file__).resolve().parents[2]


def test_repo_learn_docs_tree_is_valid() -> None:
    errors = validate_learn_docs_tree(ROOT / "docs" / "learn")
    assert errors == []


def test_rejects_missing_required_frontmatter(tmp_path: Path) -> None:
    doc = tmp_path / "broken.md"
    doc.write_text("# Missing frontmatter\n", encoding="utf-8")

    errors = validate_learn_doc(doc)
    assert any("frontmatter" in error.lower() for error in errors)


def test_rejects_concept_page_without_external_sources(tmp_path: Path) -> None:
    doc = tmp_path / "concept.md"
    doc.write_text(
        """---
title: Test Concept
audience: beginner
page_type: concept
status: conceptual
last_reviewed: 2026-04-21
repo_sources:
  - docs/architecture/ARCHITECTURE.md
external_sources: []
---

## Why This Page Exists

Concept page.

## Sources

- Repo: `docs/architecture/ARCHITECTURE.md`
""",
        encoding="utf-8",
    )

    errors = validate_learn_doc(doc)
    assert any("external_sources" in error for error in errors)


def test_rejects_project_status_page_without_status_heading(tmp_path: Path) -> None:
    doc = tmp_path / "status.md"
    doc.write_text(
        """---
title: Status
audience: beginner
page_type: project-status
status: implemented
last_reviewed: 2026-04-21
repo_sources:
  - docs/execution/SPRINTS.md
external_sources:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
---

## Why This Page Exists

Status page.

## Sources

- Repo: `docs/execution/SPRINTS.md`
- External: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
""",
        encoding="utf-8",
    )

    errors = validate_learn_doc(doc)
    assert any("What Exists Today" in error for error in errors)

