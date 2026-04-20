"""Generate the notebook-first foundation scaffolds for the Nemotron repo."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


COMMON_REPRO = """\
This notebook is scaffolded to execute cleanly before the implementation code exists.

- Python: 3.11+
- Run from repo root or open directly in Jupyter
- Expected companion issue and registry entry: `docs/execution/NOTEBOOKS.md`
"""


NOTEBOOKS = [
    {
        "filename": "00_competition_constraints_and_rules.ipynb",
        "title": "00. Competition Constraints and Rules",
        "issue": "#14",
        "summary": "Freeze the official Kaggle constraints before downstream implementation starts.",
        "audience": "Project leads, reviewers, and notebook authors who need one verified source of truth for deadlines, submission limits, base model assumptions, and scoring rules.",
        "decision": "Treat all current competition constraints as provisional until verified from Kaggle and official NVIDIA sources. Downstream notebooks must inherit those verified values instead of re-stating rumors.",
        "method": [
            "Record the verified competition dates, scoring rules, and submission artifact layout.",
            "Compare those facts against the assumptions in the planning and architecture docs.",
            "List required follow-up changes for any mismatches.",
        ],
        "outputs": [
            "Constraint table",
            "Mismatch log against current repo assumptions",
            "Go/no-go recommendation for downstream implementation",
        ],
        "risks": [
            "Public mirrors and community write-ups may be stale or incomplete.",
            "Rumored limits such as LoRA rank may differ from the official rules.",
        ],
        "sources": [
            ("Kaggle competition page", "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge"),
            ("Kaggle submission demo", "https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo"),
            ("Competition facts doc", "docs/architecture/COMPETITION.md"),
            ("Execution plan v0.2", "docs/planning/plan_v0.2.md"),
        ],
    },
    {
        "filename": "01_external_baselines_and_design_deltas.ipynb",
        "title": "01. External Baselines and Design Deltas",
        "issue": "#16",
        "summary": "Review external solutions and extract ideas worth carrying into this repo without importing unverified assumptions.",
        "audience": "Technical reviewers and non-technical stakeholders who need to understand why the team is not simply copying a public notebook.",
        "decision": "Use public pipelines as design references, not as ground truth. Every borrowed idea must be labeled as verified, inferred, or speculative.",
        "method": [
            "Summarize the Tong repository and the konbu17 notebook at a workflow level.",
            "Compare those workflows against the current architecture and adversarial review.",
            "Capture concrete deltas the team should adopt, defer, or reject.",
        ],
        "outputs": [
            "Comparison table",
            "Decision log for adopted and rejected ideas",
            "Follow-up issue candidates for architecture-impact changes",
        ],
        "risks": [
            "Community notebooks may depend on hidden data or private implementation details.",
            "A strong leaderboard score does not automatically imply a portable design.",
        ],
        "sources": [
            ("Tong Hui Kang repo", "https://github.com/tonghuikang/nemotron"),
            ("konbu17 Kaggle notebook", "https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2"),
            ("Adversarial review", "docs/analysis/ADVERSARIAL_REVIEW.md"),
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
        ],
    },
    {
        "filename": "02_dataset_schema_and_eda.ipynb",
        "title": "02. Dataset Schema and EDA",
        "issue": "#17",
        "summary": "Explain the dataset shape, category mix, and normalization contract before code paths hard-code assumptions.",
        "audience": "Data contributors, reviewers, and non-technical readers who need to see what the benchmark actually looks like.",
        "decision": "Normalize all dataset sources into the shared `ReasoningExample` contract and emphasize category-specific task shape over generic math-only framing.",
        "method": [
            "Inspect the official dataset or mirror columns and sample prompts.",
            "Document category distribution and the input/output format.",
            "Show how raw records map into the canonical schema.",
        ],
        "outputs": [
            "Schema mapping examples",
            "Category inventory",
            "Normalization checklist",
        ],
        "risks": [
            "Mirror datasets may not preserve all metadata from Kaggle.",
            "Category names and answer formatting may evolve as the official data is refreshed.",
        ],
        "sources": [
            ("Competition facts doc", "docs/architecture/COMPETITION.md"),
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
            ("HF mirror cited in repo docs", "https://huggingface.co/datasets/Taurine511/nvidia-nemotron-model-reasoning-challenge"),
        ],
    },
    {
        "filename": "03_validation_and_golden_set.ipynb",
        "title": "03. Validation and Golden Set Design",
        "issue": "#18",
        "summary": "Define how the team will hold out validation data and protect against regressions.",
        "audience": "Anyone evaluating results or reviewing training claims.",
        "decision": "Reserve a stratified validation split and a smaller golden regression set before any serious training work starts.",
        "method": [
            "Document the validation split policy and category coverage expectations.",
            "Describe the golden-set role in catching regressions that aggregate metrics can hide.",
            "Tie these choices back to the plan review findings.",
        ],
        "outputs": [
            "Validation split policy",
            "Golden-set checklist",
            "Measurement caveats for small category slices",
        ],
        "risks": [
            "If the hold-out split is too small, improvements may be noise.",
            "If the golden set is poorly chosen, it may fail to catch real regressions.",
        ],
        "sources": [
            ("Execution plan v0.2", "docs/planning/plan_v0.2.md"),
            ("Plan review", "docs/planning/plan_review.md"),
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
        ],
    },
    {
        "filename": "04_baseline_eval_and_normalization.ipynb",
        "title": "04. Baseline Evaluation and Normalization",
        "issue": "#19",
        "summary": "Define the baseline evaluation contract so later improvements can be measured without self-deception.",
        "audience": "Evaluation owners, reviewers, and stakeholders comparing model changes.",
        "decision": "Use deterministic evaluation and store every prediction as an `EvalRecord`, with normalization rules that reflect the verified competition metric.",
        "method": [
            "Document normalization rules for answers and category-specific edge cases.",
            "Define the eval artifact shape and minimum metrics to report.",
            "Explain how regression gates and failure slices are produced.",
        ],
        "outputs": [
            "Normalization policy",
            "EvalRecord contract examples",
            "Baseline report template",
        ],
        "risks": [
            "If normalization diverges from Kaggle scoring, local gains will mislead the team.",
            "Category-specific parsing shortcuts may hide genuine failures.",
        ],
        "sources": [
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
            ("Adversarial review", "docs/analysis/ADVERSARIAL_REVIEW.md"),
            ("Competition notebook after verification", "notebooks/00_competition_constraints_and_rules.ipynb"),
        ],
    },
    {
        "filename": "05_prompting_and_decode_sweeps.ipynb",
        "title": "05. Prompting and Decode Sweeps",
        "issue": "#21",
        "summary": "Measure low-cost gains from prompt templates and decoding parameters before training.",
        "audience": "Experiment owners choosing whether to spend compute on training or inference-time improvements.",
        "decision": "Keep prompting experiments grounded in the baseline eval harness and compare results by category, not just overall accuracy.",
        "method": [
            "Enumerate prompt template and decode variants to test.",
            "Define the evaluation protocol for mean/std reporting.",
            "Document criteria for promoting a prompt configuration into the baseline.",
        ],
        "outputs": [
            "Sweep matrix",
            "Decision threshold for adoption",
            "Risk notes for thinking-budget and latency trade-offs",
        ],
        "risks": [
            "Prompting gains may overfit the held-out split.",
            "Latency and token usage can erase the practical value of small accuracy gains.",
        ],
        "sources": [
            ("Execution plan v0.2", "docs/planning/plan_v0.2.md"),
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
            ("Baseline evaluation notebook", "notebooks/04_baseline_eval_and_normalization.ipynb"),
        ],
    },
    {
        "filename": "06_trajectory_collection_and_error_slices.ipynb",
        "title": "06. Trajectory Collection and Error Slices",
        "issue": "#22",
        "summary": "Capture failure slices that guide targeted solver, data, and training work.",
        "audience": "Model developers and reviewers deciding where the next engineering dollar should go.",
        "decision": "Log raw completions, extracted answers, and correctness labels per category so the team can target the most recoverable failures first.",
        "method": [
            "Describe which trajectory fields must be stored and why.",
            "Define useful failure slices by category, prompt pattern, and formatting errors.",
            "Link those slices to follow-up solver and synthetic-data work.",
        ],
        "outputs": [
            "Trajectory field inventory",
            "Failure-slice taxonomy",
            "Downstream work recommendations",
        ],
        "risks": [
            "Collecting too little metadata makes failures hard to reproduce.",
            "Collecting too much metadata can slow iteration if the format is not standardized.",
        ],
        "sources": [
            ("Adversarial review", "docs/analysis/ADVERSARIAL_REVIEW.md"),
            ("Trajectory dataset reference", "https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj"),
            ("Prompting notebook", "notebooks/05_prompting_and_decode_sweeps.ipynb"),
        ],
    },
    {
        "filename": "07_solver_framework_design.ipynb",
        "title": "07. Solver Framework Design",
        "issue": "#23",
        "summary": "Specify the plugin interface for category-aware solvers and their verification hooks.",
        "audience": "Agents building solvers, reviewers checking extensibility, and stakeholders evaluating the project’s differentiation strategy.",
        "decision": "Use a category-aware `solve` and `verify` contract with explicit confidence and metadata so solvers and LLM teachers can coexist.",
        "method": [
            "Explain why one monolithic reasoning prompt is insufficient for this benchmark.",
            "Define the shared solver interface and failure metadata.",
            "Describe how solver outputs feed synthetic data and training.",
        ],
        "outputs": [
            "Plugin interface",
            "Fallback-to-LLM policy",
            "Open design questions for category-specific implementations",
        ],
        "risks": [
            "Over-engineering the interface too early may slow the first solver implementation.",
            "A weak verification contract will let incorrect teacher labels into the dataset.",
        ],
        "sources": [
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
            ("Adversarial review", "docs/analysis/ADVERSARIAL_REVIEW.md"),
            ("Tong Hui Kang repo", "https://github.com/tonghuikang/nemotron"),
        ],
    },
    {
        "filename": "08_synthetic_data_recipe.ipynb",
        "title": "08. Synthetic Data Recipe",
        "issue": "#24",
        "summary": "Define how synthetic data should be generated, filtered, and documented before large runs start.",
        "audience": "Data generation owners, cost reviewers, and non-technical stakeholders evaluating risk and provenance.",
        "decision": "Synthetic data should only be promoted when answers are verifiable, costs are bounded, and provenance is explicit.",
        "method": [
            "Define teacher options and selection criteria.",
            "Document the filtering stages and provenance fields.",
            "Set cost and quality gates before generation runs begin.",
        ],
        "outputs": [
            "Teacher selection policy",
            "Quality filter checklist",
            "Provenance requirements for generated records",
        ],
        "risks": [
            "Incorrect teacher outputs can poison training data.",
            "API-based generation can become costly if runs are not bounded in advance.",
        ],
        "sources": [
            ("Execution plan v0.2", "docs/planning/plan_v0.2.md"),
            ("NVIDIA reasoning training blog", "https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/"),
            ("SYNTHETIC-1", "https://www.primeintellect.ai/blog/synthetic-1"),
        ],
    },
    {
        "filename": "09_sft_runbook_and_masking.ipynb",
        "title": "09. SFT Runbook and Masking",
        "issue": "#25",
        "summary": "Define the adapter training runbook, masking policy, and checkpoint rules before the first training execution.",
        "audience": "Training owners, reviewers, and stakeholders who need a traceable explanation of what the adapter is learning and why.",
        "decision": "Start with a conservative LoRA/QLoRA plan, keep target modules configurable, and make loss masking and checkpointing explicit artifacts rather than tacit trainer settings.",
        "method": [
            "Document the planned training stack and hardware modes.",
            "Describe the masking policy and why it matters.",
            "Define checkpoint, regression, and provenance requirements.",
        ],
        "outputs": [
            "Runbook outline",
            "Masking decision log",
            "Training risk register",
        ],
        "risks": [
            "Competition constraints may invalidate aggressive LoRA settings.",
            "Poor masking decisions can waste gradient budget or overfit formatting.",
        ],
        "sources": [
            ("Execution plan v0.2", "docs/planning/plan_v0.2.md"),
            ("PEFT LoRA docs", "https://huggingface.co/docs/peft/en/package_reference/lora"),
            ("TRL docs", "https://huggingface.co/docs/trl/main/en/index"),
        ],
    },
    {
        "filename": "10_submission_packaging_and_provenance.ipynb",
        "title": "10. Submission Packaging and Provenance",
        "issue": "#20",
        "summary": "Define the dry-run packaging flow so future submissions are valid, reproducible, and reviewable.",
        "audience": "Submission owners, reviewers, and project leads managing leaderboard attempts.",
        "decision": "Every submission artifact must carry a `PackageManifest` and fail fast if the repo cannot prove which model, data, and eval run produced it.",
        "method": [
            "Describe the submission folder layout and required metadata.",
            "Define packaging checks and rollback expectations.",
            "Tie provenance requirements back to issue and notebook artifacts.",
        ],
        "outputs": [
            "Packaging checklist",
            "Manifest field examples",
            "Rollback and review expectations",
        ],
        "risks": [
            "An adapter can appear valid locally but still fail Kaggle if layout assumptions drift.",
            "Missing provenance makes leaderboard results hard to trust or reproduce.",
        ],
        "sources": [
            ("Architecture doc", "docs/architecture/ARCHITECTURE.md"),
            ("Kaggle submission demo", "https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo"),
            ("Competition notebook", "notebooks/00_competition_constraints_and_rules.ipynb"),
        ],
    },
]


def markdown_cell(text: str, cell_id: str) -> dict:
    normalized = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in normalized.splitlines()],
    }


def code_cell(text: str, cell_id: str) -> dict:
    normalized = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in normalized.splitlines()],
    }


def build_notebook(spec: dict) -> dict:
    sources = "\n".join(f"- [{label}]({url})" for label, url in spec["sources"])
    method_items = "\n".join(f"- {item}" for item in spec["method"])
    output_items = "\n".join(f"- {item}" for item in spec["outputs"])
    risk_items = "\n".join(f"- {item}" for item in spec["risks"])

    cells = [
        markdown_cell(
            f"""
            # {spec['title']}

            - Parent issue: `{spec['issue']}`
            - Status: `scaffolded`
            - Summary: {spec['summary']}
            """,
            "overview",
        ),
        markdown_cell(
            f"""
            ## Audience and Why It Matters

            {spec['audience']}
            """,
            "audience",
        ),
        markdown_cell(
            f"""
            ## Decision / Hypothesis

            {spec['decision']}
            """,
            "decision",
        ),
        markdown_cell(
            f"""
            ## Environment and Reproduction

            {COMMON_REPRO}
            """,
            "environment",
        ),
        code_cell(
            """
            from pathlib import Path
            import platform

            REPO_ROOT = Path.cwd()

            print(f"Repo root: {REPO_ROOT}")
            print(f"Python platform: {platform.platform()}")
            """,
            "environment-check",
        ),
        markdown_cell(
            f"""
            ## Method and Outputs

            ### Planned method
            {method_items}

            ### Expected outputs
            {output_items}
            """,
            "method",
        ),
        code_cell(
            """
            planned_tasks = [
                "replace scaffold text with verified findings",
                "link concrete artifacts produced during execution",
                "update docs/execution/NOTEBOOKS.md when status changes",
            ]

            for item in planned_tasks:
                print(f"- {item}")
            """,
            "task-list",
        ),
        markdown_cell(
            f"""
            ## Results / Open Risks

            This scaffold intentionally records the current open questions before implementation code exists.

            {risk_items}
            """,
            "risks",
        ),
        markdown_cell(
            f"""
            ## Sources

            {sources}
            """,
            "sources",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in NOTEBOOKS:
        notebook_path = NOTEBOOKS_DIR / spec["filename"]
        notebook = build_notebook(spec)
        notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {notebook_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
