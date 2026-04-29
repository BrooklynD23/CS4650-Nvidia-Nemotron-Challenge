# data/eval — Eval Artifacts

## Canonical file names

| Artifact | Filename | Notebook that creates it |
|----------|----------|--------------------------|
| Validation split | `validation_200.jsonl` | Notebook 03 |
| Golden regression set | `golden_20.jsonl` | Notebook 03 |

## Tracking policy

These files are **not committed to git** (covered by the `data/**` gitignore
rule). They are reproducible from the canonical training rows via Notebook 03.
Store them locally or on shared HPC storage. Record the dataset version,
seed, and row count in the sidecar manifest produced by Notebook 03.

## Immutability rule

`golden_20.jsonl` must never be edited in place after first approval.
A new selection requires a new versioned filename (e.g., `golden_40.jsonl`).
See `src/evaluation/artifacts.py::is_immutable_golden_path` for enforcement.
