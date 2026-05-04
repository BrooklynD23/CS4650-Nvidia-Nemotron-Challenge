---
name: doc-sync
---

Run the doc orchestrator over a commit range.

- `WORKTREE` pre-commit sync mode: inspect the current worktree diff and sync docs before commit
- Commit-range mode: `/doc-sync [range]` with optional `<range>`; default `HEAD~1..HEAD`
- Dispatch the same orchestrator logic used by the post-commit hook
