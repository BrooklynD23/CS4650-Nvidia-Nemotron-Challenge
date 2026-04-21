"""Git commit guard for documentation and issue accountability.

This module powers lightweight local hooks:

- ``pre-commit`` blocks implementation changes that do not stage a
  corresponding accountability-document update.
- ``commit-msg`` blocks commits that touch mapped issue files without a
  matching issue reference in the message.

The checks are intentionally local and deterministic so commits do not
depend on live network/API access.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path


ISSUE_FILE_PATTERNS: dict[int, tuple[str, ...]] = {
    13: (
        "notebooks/README.md",
        "docs/execution/NOTEBOOKS.md",
        "docs/execution/ISSUE_REVIEW_HARNESS.md",
        "scripts/scaffold_notebooks.py",
    ),
    14: (
        "notebooks/00_competition_constraints_and_rules.ipynb",
        "docs/execution/plans/issue-14-constraints-freeze.md",
        "docs/architecture/COMPETITION.md",
    ),
    15: (
        ".github/ISSUE_TEMPLATE/*",
        "docs/execution/ISSUE_REVIEW_HARNESS.md",
        "docs/execution/plans/issue-15-review-harness.md",
    ),
    16: (
        "notebooks/01_external_baselines_and_design_deltas.ipynb",
        "docs/execution/plans/issue-16-external-baselines-delta.md",
        "data/external/konbu17/**",
    ),
    17: (
        "notebooks/02_dataset_schema_and_eda.ipynb",
        "docs/execution/plans/issue-17-schema-and-eda.md",
        "src/contracts.py",
        "src/data/schema_mapping.py",
        "src/data/schema_eda.py",
        "src/packaging/manifest.py",
        "tests/test_contracts.py",
        "tests/test_schema_mapping.py",
        "tests/test_schema_eda.py",
        "tests/test_manifest.py",
    ),
    18: (
        "notebooks/03_validation_and_golden_set.ipynb",
        "docs/execution/plans/issue-18-validation-and-golden-set.md",
        "src/evaluation/splits.py",
        "src/evaluation/golden_gate.py",
        "tests/evaluation/test_splits.py",
        "tests/evaluation/test_golden_gate.py",
    ),
    19: (
        "notebooks/04_baseline_eval_and_normalization.ipynb",
        "docs/execution/plans/issue-19-baseline-eval-and-normalization.md",
        "src/evaluation/config.py",
        "src/evaluation/normalization.py",
        "src/evaluation/reporting.py",
        "src/evaluation/runner.py",
        "src/evaluation/scoring.py",
        "tests/evaluation/test_normalization.py",
        "tests/evaluation/test_reporting.py",
        "tests/evaluation/test_runner.py",
        "tests/evaluation/test_scoring.py",
    ),
    20: (
        "notebooks/10_submission_packaging_and_provenance.ipynb",
        "docs/execution/plans/issue-20-submission-packaging-and-provenance.md",
        "src/evaluation/artifacts.py",
        "src/inference/submission.py",
        "tests/evaluation/test_artifact_manifest.py",
        "tests/inference/test_submission_packaging.py",
    ),
}

ACCOUNTABILITY_DOC_PATTERNS: tuple[str, ...] = (
    "notebooks/*.ipynb",
    "notebooks/README.md",
    "docs/execution/*.md",
    "docs/execution/plans/*.md",
    "docs/analysis/*.md",
    ".github/ISSUE_TEMPLATE/*",
)

DOC_ENFORCED_ISSUES: frozenset[int] = frozenset({13, 15, 17, 18, 19, 20})
IMPLEMENTATION_PREFIXES: tuple[str, ...] = ("src/", "tests/")

ISSUE_REF_RE = re.compile(r"#(\d+)\b")
CLOSING_REF_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b",
    re.IGNORECASE,
)


def infer_touched_issues(paths: list[str]) -> set[int]:
    """Infer issue ids touched by ``paths`` from the repo mapping."""
    touched: set[int] = set()
    normalized = _normalize_paths(paths)
    for path in normalized:
        for issue_id, patterns in ISSUE_FILE_PATTERNS.items():
            if _matches_any(path, patterns):
                touched.add(issue_id)
    return touched


def extract_issue_refs(message: str) -> set[int]:
    """Return every ``#<number>`` reference in ``message``."""
    return {int(match) for match in ISSUE_REF_RE.findall(message)}


def extract_closing_issue_refs(message: str) -> set[int]:
    """Return issue ids referenced by closing keywords."""
    return {int(match) for match in CLOSING_REF_RE.findall(message)}


def validate_pre_commit(paths: list[str]) -> list[str]:
    """Return blocking pre-commit errors for ``paths``."""
    normalized = _normalize_paths(paths)
    issues_needing_docs = _issues_requiring_docs(normalized)
    if not issues_needing_docs:
        return []
    if _has_accountability_doc_update(normalized):
        return []
    issue_list = _format_issue_list(issues_needing_docs)
    return [
        "Documentation/accountability drift: staged implementation changes "
        f"touch {issue_list} but no accountability doc update is staged. "
        "Stage a related notebook, plan doc, registry, review-harness, or "
        "analysis note update alongside the code."
    ]


def validate_commit_message(paths: list[str], message: str) -> list[str]:
    """Return blocking commit-message errors for ``paths`` and ``message``."""
    normalized = _normalize_paths(paths)
    touched = _issues_requiring_issue_refs(normalized)
    if not touched:
        return []

    referenced = extract_issue_refs(message)
    missing = sorted(touched - referenced)
    errors: list[str] = []
    if missing:
        errors.append(
            "Missing issue references: this commit touches "
            f"{_format_issue_list(missing)} but the commit message does not "
            "reference all touched issue ids."
        )

    closing = extract_closing_issue_refs(message)
    unrelated_closing = sorted(closing - touched)
    if unrelated_closing:
        errors.append(
            "Closing keywords drift: the commit message closes "
            f"{_format_issue_list(unrelated_closing)} but those issues are "
            "not mapped to the staged files."
        )
    return errors


def git_staged_paths() -> list[str]:
    """Return staged file paths from git."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("pre-commit", help="Run staged-file doc drift checks.")

    commit_msg_parser = subparsers.add_parser(
        "commit-msg",
        help="Run issue-reference checks against a commit message file.",
    )
    commit_msg_parser.add_argument("message_file", type=Path)

    args = parser.parse_args(argv)
    paths = git_staged_paths()

    if args.command == "pre-commit":
        errors = validate_pre_commit(paths)
    else:
        message = args.message_file.read_text(encoding="utf-8")
        errors = validate_commit_message(paths, message)

    if errors:
        for error in errors:
            print(f"commit-guard: {error}", file=sys.stderr)
        return 1
    return 0


def _normalize_paths(paths: list[str]) -> list[str]:
    return sorted({path.strip() for path in paths if path.strip()})


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _issues_requiring_docs(paths: list[str]) -> list[int]:
    issues: set[int] = set()
    for path in paths:
        if not path.startswith(IMPLEMENTATION_PREFIXES):
            continue
        for issue_id in DOC_ENFORCED_ISSUES:
            patterns = ISSUE_FILE_PATTERNS[issue_id]
            if _matches_any(path, patterns):
                issues.add(issue_id)
    return sorted(issues)


def _has_accountability_doc_update(paths: list[str]) -> bool:
    return any(_matches_any(path, ACCOUNTABILITY_DOC_PATTERNS) for path in paths)


def _issues_requiring_issue_refs(paths: list[str]) -> set[int]:
    primary_paths = [
        path
        for path in paths
        if not _matches_any(path, ACCOUNTABILITY_DOC_PATTERNS)
    ]
    primary_issues = infer_touched_issues(primary_paths)
    if primary_issues:
        return primary_issues
    return infer_touched_issues(paths)


def _format_issue_list(issue_ids: list[int] | set[int]) -> str:
    ordered = sorted(issue_ids)
    return ", ".join(f"#{issue_id}" for issue_id in ordered)


if __name__ == "__main__":
    raise SystemExit(main())
