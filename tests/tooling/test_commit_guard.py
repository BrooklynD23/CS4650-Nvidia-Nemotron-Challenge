"""Tests for :mod:`src.tooling.commit_guard`."""

from __future__ import annotations

from src.tooling.commit_guard import (
    extract_closing_issue_refs,
    extract_issue_refs,
    infer_touched_issues,
    validate_commit_message,
    validate_pre_commit,
)


def test_infer_touched_issues_from_issue_19_files() -> None:
    paths = [
        "src/evaluation/runner.py",
        "tests/evaluation/test_runner.py",
    ]
    assert infer_touched_issues(paths) == {19}


def test_infer_touched_issues_handles_shared_harness_files() -> None:
    paths = [
        ".github/ISSUE_TEMPLATE/agent-execution.md",
        "docs/execution/ISSUE_REVIEW_HARNESS.md",
    ]
    assert infer_touched_issues(paths) == {13, 15}


def test_pre_commit_requires_accountability_doc_for_issue_code() -> None:
    errors = validate_pre_commit(["src/evaluation/runner.py"])
    assert len(errors) == 1
    assert "documentation" in errors[0].lower()
    assert "#19" in errors[0]


def test_pre_commit_accepts_issue_code_with_docs_update() -> None:
    errors = validate_pre_commit(
        [
            "src/evaluation/runner.py",
            "docs/execution/NOTEBOOKS.md",
        ]
    )
    assert errors == []


def test_extract_issue_refs_finds_plain_and_closing_refs() -> None:
    message = "feat(#19): add eval runner\n\nRefs #19\nCloses #18"
    assert extract_issue_refs(message) == {18, 19}


def test_extract_closing_issue_refs_only_finds_closure_keywords() -> None:
    message = "feat(#19): add eval runner\n\nRefs #19\nResolves #18"
    assert extract_closing_issue_refs(message) == {18}


def test_commit_message_requires_issue_ref_for_touched_issue() -> None:
    errors = validate_commit_message(
        [
            "src/evaluation/runner.py",
            "docs/execution/NOTEBOOKS.md",
        ],
        "feat: add eval runner",
    )
    assert len(errors) == 1
    assert "missing issue references" in errors[0].lower()
    assert "#19" in errors[0]


def test_commit_message_accepts_matching_issue_ref() -> None:
    errors = validate_commit_message(
        [
            "src/evaluation/runner.py",
            "docs/execution/NOTEBOOKS.md",
        ],
        "feat(#19): add eval runner",
    )
    assert errors == []


def test_commit_message_rejects_closing_unrelated_issue() -> None:
    errors = validate_commit_message(
        [
            "src/evaluation/runner.py",
            "docs/execution/NOTEBOOKS.md",
        ],
        "feat(#19): add eval runner\n\nCloses #20",
    )
    assert len(errors) == 1
    assert "closing keywords" in errors[0].lower()
    assert "#20" in errors[0]
