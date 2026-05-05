# Project TODO Board

This file is the PM control board for the CS4650 NVIDIA Nemotron Challenge repo.
It connects the live GitHub issue set to concrete remaining work, dependencies,
handoff warnings, and verification commands.

Live GitHub inventory checked during planning:

- Open epics: `#1` through `#12`
- Closed execution issues: `#13` through `#20`
- Open execution issues: `#21` through `#25`
- No assignees or milestones were set in the live issue list at the time of
  inspection.

This TODO does not replace GitHub issues. Use it to coordinate the next work,
then keep issue comments and repo docs updated as artifacts become real.

## Operating Rules

- If a task has `Blocked by`, the owner must check the general chat before
  starting.
- If a task changes a shared contract, it needs human review and, when noted,
  architecture review.
- Do not run real SFT/QLoRA training unless the run conforms to the verified
  `#14` base model, scoring contract, LoRA constraints, and submission rules.
- Do not promote prompt-sweep results until the output-parser contract is
  resolved and comparisons use compatible scoring.
- Do not promote or submit adapters unless the golden gate from `#18` and the
  packaging checks from `#20` pass.
- Closed execution issues may still have TODOs here when their code exists but
  real artifacts, status docs, or downstream evidence are missing.
- Keep large artifacts out of git. Before producing small reproducibility
  artifacts under `data/eval/`, decide whether they are committed, copied to a
  tracked docs artifact, or stored locally with a manifest.

## General Chat Dependency Check

Use this message before starting any task with dependencies:

```text
Dependency check: I am about to start <task>. This depends on <issue/person/artifact>. Is it ready?
Please confirm: file path, commit/branch, artifact version, and known caveats.
```

Do not begin the dependent task until the upstream owner confirms:

- artifact path
- current branch or commit
- dataset/config/run version
- known blockers
- whether the artifact is final, provisional, or still changing

## Immediate P0 Fixes

- [x] Fix the current failing test:
  `tests/tooling/test_learn_docs_guard.py::test_rejects_concept_page_without_external_sources`.
  - Issue link: process/docs guard, related to `#12`, `#13`, `#15`
  - Fix shipped in commit `d3267c9`; test confirmed passing.
- [x] Run the full test suite after the guard fix.
  - Command:
    `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider`
  - Result: `189 passed, 1 skipped, 0 failed`
- [x] Resolve root status drift.
  - Issue links: `#12`, `#13`, `#15`
  - Fix: `AGENTS.md` "Repo Reality Check" section updated to describe Wave A/B
    foundation (`src/`, `tests/`, `notebooks/`); item 9 added to canonical docs
    list pointing to `docs/learn/project/implemented-today.md`.
- [x] Resolve split artifact naming drift before generating real eval files.
  - Issue links: `#18`, `#19`, `#21`
  - Decision: canonical names are `validation_200.jsonl` (validation split) and
    `golden_20.jsonl` (golden set). `VAL_SPLIT_FILENAME` and
    `GOLDEN_SPLIT_FILENAME` in `src/evaluation/prompt_sweeps.py` updated; all
    doc references in `docs/analysis/`, `docs/learn/foundations/`, and
    `docs/learn/project/` updated to match.
- [x] Decide eval artifact tracking policy.
  - Issue links: `#3`, `#4`, `#18`, `#19`
  - Decision: artifacts are gitignored (not committed); reproducible via
    Notebook 03. Policy documented in `data/eval/README.md`; `.gitignore`
    updated to allow that README to be committed.

## Issue Map

| Issue | Live status | Role | Canonical child work |
|---|---:|---|---|
| `#1` | Open | Epic | `#14` Kaggle constraints |
| `#2` | Open | Epic | `#16` external baseline review |
| `#3` | Open | Epic | `#17` dataset/schema |
| `#4` | Open | Epic | `#18`, `#19` validation + eval |
| `#5` | Open | Epic | `#20` packaging |
| `#6` | Open | Epic | `#21` prompt/decode sweeps |
| `#7` | Open | Epic | `#22` trajectories/failure slices |
| `#8` | Open | Epic | `#23` solver framework |
| `#9` | Open | Epic | No execution child yet; needs `#26+` |
| `#10` | Open | Epic | `#24` synthetic data |
| `#11` | Open | Epic | `#25` SFT runbook/training |
| `#12` | Open | Epic | `#13`, `#15` runbook/review harness |
| `#13` | Closed | Execution | Notebook template + citation rubric |
| `#14` | Closed | Execution | Verified constraints contract |
| `#15` | Closed | Execution | Agent/human review harness |
| `#16` | Closed | Execution | External baseline deltas; must conform to `#14` |
| `#17` | Closed | Execution | Schema/code foundation |
| `#18` | Closed | Execution | Split/golden code, real artifacts missing |
| `#19` | Closed | Execution | Eval code, real baseline missing |
| `#20` | Closed | Execution | Packaging code, real adapter package missing |
| `#21` | Open | Execution | Prompt sweeps, execution blocked |
| `#22` | Open | Execution | Trajectory/failure slices, scaffold only |
| `#23` | Open | Execution | Solver framework, scaffold only |
| `#24` | Open | Execution | Synthetic data recipe, scaffold only |
| `#25` | Open | Execution | SFT runbook/training, plan only |

## Issue TODOs

### `#1` / `#14` - Kaggle Constraints Freeze

Owner lane: PM / constraints

Verified 2026-04-29 from competition overview + submission demo notebook:

- [x] Confirm the evaluator base model ID and revision.
  - KaggleHub path: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
  - HF slug still unconfirmed; use KaggleHub path in Kaggle environments.
- [x] Confirm the base-model load recipe (core confirmed from demo):
  `trust_remote_code=True`, `dtype=torch.bfloat16`, `device_map="auto"`.
  - Tokenizer special tokens, chat template, attention backend, and
    thinking-control flags are **not confirmed** (evaluator-internal).
- [x] Confirm scoring and answer-normalization rules.
  - Answer in `\boxed{}`. Metric extracts boxed content, falls back to
    heuristics. Correct = exact string match OR within relative tolerance 1e-3.
- [x] Confirm whether output should be answer-only, boxed, or allowed to include
  reasoning.
  - Reasoning is allowed; only the extracted final answer is graded.
- [~] Confirm LoRA-only submission rule, rank cap, target-module restrictions,
  dtype limits, and file-size limits.
  - LoRA-only: confirmed. Rank cap `r ≤ 32`: confirmed (evaluator enforces).
  - Demo target modules: `in_proj|out_proj|up_proj|down_proj` (4 modules).
  - Dtype bf16: confirmed from demo. Competition-enforced dtype limits: unconfirmed.
  - File-size limits: **unconfirmed**.
- [x] Confirm exact `submission.zip` layout and accepted root filenames.
  - Root must contain `adapter_config.json` + `adapter_model.safetensors`.
- [x] Confirm final deadline, entry deadline, daily submission limits, and final
  submission selection behavior.
  - Final deadline: 2026-06-15 23:59 (treat as UTC).
  - Entry deadline: 2026-05-10 (confirmed by team).
  - Daily limit: 5 submissions per day (confirmed by team).
  - Final-submission selection rule: out of scope — assume last submission counts.
- [x] Update `docs/architecture/COMPETITION.md` with a clearly labeled
  `Verified` section. — Done 2026-04-29.
- [x] Propagate frozen facts to `docs/architecture/COMPETITION.md` and this
  TODO. — Done 2026-04-29.
  - `docs/architecture/ARCHITECTURE.md` and `docs/planning/plan_v0.2.md`
    propagation deferred; no blocking dependency for Wave B start.

Out of scope / unable to confirm (noted in COMPETITION.md, not blocking):

- HF slug for base model — use KaggleHub path.
- Evaluator tokenizer/chat-template/thinking flags — design as configurable.
- Competition-enforced target-module restrictions — treat demo 4-module set as reference.
- Dtype enforcement — treat bf16 as required.
- File-size limit — monitor; flag if adapter > ~1 GB.
- Final-submission selection rule — assume last submission counts.
- Final deadline timezone — treat as UTC.

General chat warning:

- Before any training, scoring promotion, or packaging promotion, confirm with
  the PM that the verified `#14` contract above is still the working contract.
- The 4-module target set from the demo is the reference; do not expand to
  konbu17's 9-module set without PM sign-off.

### `#2` / `#16` - External Baseline Review

Owner lane: research / baseline comparison

- [x] Create `docs/analysis/EXTERNAL_BASELINE_REVIEW.md` for the bounded
  Tong (`tonghuikang`) + konbu17 review.
- [x] Capture konbu17 notebook facts from existing repo-local evidence:
  base model, load recipe, LoRA config, target modules, masking behavior,
  dataset, eval protocol, packaging.
- [x] Review Tong (`tonghuikang`) public pipeline for masking, augmentation,
  weighting, solver/teacher patterns, and packaging ideas.
- [x] Produce an Adopt / Reject / Gate matrix.
- [x] Link each Adopt/Gate decision to downstream issues:
  `#19`, `#20`, `#21`, `#23`, `#24`, `#25`.
- [x] Identify any baseline assumptions that conflict with exact-match scoring
  or rank `<=32`.

Blocked by:

- None for the bounded Tong + konbu17 review. Future reproduction work remains
  blocked by source availability, license/provenance capture, and PM signoff for
  any baseline choice that deviates from the verified `#14` contract.

General chat warning:

- Before using an external baseline as implementation guidance, ask PM whether
  the baseline's model/output/rank assumptions match the frozen Kaggle rules.

### `#3` / `#17` - Dataset Ingestion + Canonical Schema

Owner lane: data

- [x] Download official Kaggle data after rules acceptance, or select an
  approved mirror for provisional local development.
- [x] Normalize rows into `ReasoningExample`:
  `id`, `category`, `prompt`, `answer`, `source`, `split`, `metadata`.
- [x] Preserve extra columns under `metadata`.
- [x] Produce dataset version string and source provenance.
- [x] Run EDA for row counts, category distribution, prompt length, answer
  shape, and missing-value anomalies.
- [x] Make `src/contracts.py` the single contract source unless the team
  chooses a separate schema module.
- [x] Update status docs so notebook `02` status matches actual execution and
  artifact state.

Blocked by:

- Kaggle data/rules access if the official dataset cannot be downloaded.

General chat warning:

- Before `#18`, `#19`, `#22`, `#24`, or `#25` consumes data, confirm the dataset
  version and canonical schema artifact path in general chat.

### `#4` / `#18` - Validation Split + Golden Set

Owner lane: evaluation

- [x] Choose canonical eval artifact names and update all loaders/docs.
- [x] Generate the real validation split from canonical training rows.
- [x] Generate the real golden regression set.
- [x] Record seed, selection rule, dataset version, category distribution, and
  source split per row.
- [x] Ensure the golden set is immutable; replacement must create a new version.
- [x] Add a manifest or index if artifacts are not committed.
- [ ] Resolve the documented `#18` / `#19` dependency cycle.

Blocked by:

- `#17` canonical dataset rows.
- Verified `#14` scoring contract must be applied if selection depends on
  answer normalization.

General chat warning:

- Before `#19` baseline or `#21` sweeps start, confirm exact split paths and
  whether artifacts are final or provisional.

### `#4` / `#19` - Baseline Eval + Normalization

Owner lane: evaluation

- [ ] Replace synthetic/fallback-only notebook execution with real split loading
  once `#18` artifacts exist.
- [ ] Run a real baseline eval under `data/eval/runs/<run_id>/`.
- [ ] Ensure `EvalRecord` includes:
  `run_id`, `example_id`, `model_id`, `prompt_template_id`, `normalizer_id`,
  `category`, `split`, `gold`, `prediction`, `normalized_prediction`,
  `correct`, `latency_ms`, `tokens_in`, `tokens_out`, `seed`, `decode_config`.
- [ ] Produce overall accuracy, per-category accuracy, failure examples, latency,
  token usage, and normalizer ID.
- [ ] Centralize answer normalization so later issues import one implementation.
- [ ] Add category-specific parsers only behind explicit normalizer IDs.

Blocked by:

- `#17` canonical rows.
- `#18` real eval artifacts.

General chat warning:

- Before `#21`, `#22`, `#25`, or `#20` compares against baseline, confirm
  baseline run ID, normalizer ID, output parser, and dataset version.

### `#5` / `#20` - Adapter Packaging + Provenance

Owner lane: packaging / release

- [ ] Keep `submission.zip` minimal unless Kaggle rules say otherwise.
- [ ] Use root-only archive entries:
  `adapter_config.json`, `adapter_model.safetensors`.
- [ ] Keep provenance manifest outside the zip.
- [ ] Resolve notebook 10 conflict that mentions manifest/README inside the zip.
- [ ] Validate zip contents, duplicate entries, symlinks, nested paths, and
  adapter config parseability.
- [ ] Define where real submission attempts live, for example
  `experiments/submissions/<run_id>/`.
- [ ] Run packaging tests after any packaging change.

Blocked by:

- `#19` eval provenance inputs.
- `#25` real adapter output for final package.

General chat warning:

- Before packaging a real adapter, ask training owner for adapter path,
  base-model ID, LoRA rank, dataset version, eval summary, and promotion status.

### `#6` / `#21` - Prompt Template + Decode Sweeps

Owner lane: prompting

- [ ] Resolve the pending architecture review for `final-answer-line-v1`.
- [ ] Decide whether to accept, reject, or defer the prompt-sweep output-parser
  contract.
- [ ] Run notebook `05` only after real `#18` splits and a compatible `#19`
  baseline exist.
- [ ] Store sweep run records with prompt template, decode config, model ID,
  normalizer ID, output parser, seed, and run ID.
- [ ] Produce ranked findings and update `docs/analysis/prompting_findings.md`.
- [ ] Do not promote a sweep config if it only wins because it uses a different
  parser than the baseline.

Blocked by:

- `#18` validation/golden files.
- `#19` baseline run.
- Pending parser-contract review.

General chat warning:

- Before running sweeps, ask evaluation owner to confirm split paths, baseline
  run ID, normalizer ID, and output-parser compatibility.

### `#7` / `#22` - Trajectory Collection + Error Slices

Owner lane: failure analysis

- [ ] Implement notebook `06` beyond scaffold.
- [ ] Define a trajectory/trace artifact contract if `EvalRecord` is not enough.
- [ ] Generate failure slices by category, prompt pattern, answer type, and
  recoverability.
- [ ] Produce retry candidates for synthetic-data targeting.
- [ ] Align output paths with `#24`; current docs disagree between
  `data/analysis/*` and `data/errors/*`.
- [ ] Document hardware/runtime plan for trajectory collection.

Blocked by:

- `#19` real baseline artifacts.
- `#21` best prompt/decode config.

General chat warning:

- Before starting trajectory runs, ask prompting owner for the promoted prompt
  config and evaluation owner for baseline artifact paths.

### `#8` / `#23` - Teacher/Solver Framework

Owner lane: solver design

- [ ] Implement notebook `07` beyond scaffold.
- [ ] Define solver interface:
  `solve(prompt) -> {answer, confidence, metadata}`.
- [ ] Define verifier interface:
  `verify(pred, gold) -> bool`.
- [ ] Define fallback policy for solver failure and LLM teacher usage.
- [ ] Define logging and failure-mode taxonomy.
- [x] Decide whether `#15` is a real dependency for this issue or just process
  boilerplate. Resolved: no — `#15` is the review harness only; `src/inference/solver.py` is not blocked by it.
- [ ] Link downstream implementation issues for individual category solvers.

Blocked by:

- `#22` error taxonomy and failure slices.
- `#17` category taxonomy.

General chat warning:

- Before locking solver interfaces, ask failure-analysis owner which categories
  and error modes are highest ROI.

### `#9` - First Solver Implementation

Owner lane: solver implementation

- [ ] Create a child execution issue, likely `#26`, for bit-manipulation solver
  implementation.
- [ ] Define input fixtures from real or mirrored prompts.
- [ ] Implement search over plausible operations:
  shifts, rotates, xor, and, or, not, masks, base conversion, simple numeric
  transforms.
- [ ] Verify candidate rule against all in-prompt examples before predicting.
- [ ] Return answer only when confidence threshold is met.
- [ ] Add unit tests for successful inference, ambiguity, no-solution, and
  formatting.
- [ ] Wire solver into the framework from `#23`.

Blocked by:

- `#23` solver interface.
- `#17` category schema.
- `#22` failure evidence if prioritization changes.

General chat warning:

- Before implementing a category solver, ask solver-framework owner whether the
  interface is frozen and ask data owner for canonical prompt fixtures.

### `#10` / `#24` - Synthetic Data Generator + Quality Filters

Owner lane: synthetic data

- [ ] Implement notebook `08` beyond scaffold.
- [ ] Define teacher sources and solver-first / LLM-fallback policy.
- [ ] Define generation prompts per category.
- [ ] Define quality filters:
  dedupe, answer validity, category validity, length limits, solver confidence,
  provenance completeness.
- [x] Define output schema as `SFTExample` or a documented derivative.
  Resolved: `SFTExample` is defined and frozen in `src/contracts.py:97`; use as-is.
- [ ] Produce small smoke synthetic set before large generation.
- [ ] Add cost caps for paid APIs and dry-run token/cost estimates.
- [ ] Document artifact paths and dataset fingerprint.

Blocked by:

- `#22` failure slices.
- `#23` solver/teacher framework.
- Verified `#14` scoring/output format must be applied if completions include
  reasoning or boxed answers.

General chat warning:

- Before generating synthetic data, ask failure-analysis owner for target
  categories and solver owner for verifier availability.

### `#11` / `#25` - SFT Training Pipeline + Masking

Owner lane: training

- [x] Remove or revise stale `r=64` plan/config assumptions in coordination
  docs; default to rank `<=32` per the verified `#14` contract.
- [ ] Implement expected HPC scripts from the runbook:
  - `scripts/hpc/preflight.sh`
  - `scripts/hpc/tokenize_dataset.py`
  - `scripts/hpc/submit_sft.sbatch`
  - `scripts/hpc/submit_rl.sbatch`
  - `scripts/hpc/checkpoint_policy.py`
  - `scripts/hpc/regression_gate.py`
  - `scripts/hpc/package_adapter.py`
  - `scripts/hpc/resume_from_latest.py`
- [ ] Add training configs only when they use the verified `#14` base model and
  adapter constraints.
- [ ] Implement masking tests so prompt/user tokens are not trained unless
  explicitly intended.
- [ ] Run a tiny SFT smoke job before any production run.
- [ ] Save checkpoints with sidecar metadata:
  trainer state, run config, metrics, git SHA, dataset fingerprint.
- [ ] Promote only checkpoints that pass golden and validation gates.

Blocked by:

- `#19` eval/golden gate.
- `#23` solver framework if training data uses solver outputs.
- `#24` synthetic data recipe for generated data.

General chat warning:

- Before submitting any GPU job, ask PM whether the run uses the current `#14`
  contract and ask evaluation owner whether golden/validation gates are ready.

### `#12` / `#13` / `#15` - Runbook, Templates, Review Harness

Owner lane: PM / process

- [ ] Verify GitHub labels and issue templates still render correctly.
- [ ] Update issue bodies if dependency corrections are accepted.
- [ ] Keep `docs/execution/SPRINTS.md`,
  `docs/execution/ISSUE_REVIEW_HARNESS.md`, and
  `docs/execution/NOTEBOOKS.md` synchronized.
- [x] Treat `docs/execution/NOTEBOOKS.md` as the canonical status source.
- [ ] Update stale notebook plan statuses or explicitly mark them as historical
  plans.
- [ ] Document reproducible submission flow after first valid adapter.

Blocked by:

- None for process cleanup, but artifact status updates depend on each owner.

General chat warning:

- Before closing any architecture-impact child issue, ask for human and
  architecture review explicitly.

## Cross-Issue Drift TODOs

- [ ] Reconcile issue parent/dependency metadata against
  `docs/execution/SPRINTS.md`.
- [ ] Break the `#18` / `#19` dependency cycle.
- [x] Resolve `#16` / `#14` dependency: the bounded baseline review is not
  blocked by `#14`, but every recommendation must conform to the verified `#14`
  contract.
- [ ] Decide whether `#17` formally depends on `#15`.
- [ ] Decide whether `#20` should list `#25` as a real dependency for final
  packaging, while keeping dummy packaging independent.
- [ ] Align `#20` zip contract across all docs.
- [ ] Align golden threshold policy: strict `20/20` vs any tolerated miss.
- [x] Propagate the verified `#14` eval-format policy: boxed extraction with
  exact match or `1e-3` numeric tolerance.
- [ ] Add a producer issue for curated training data if `#22`/`#24`/`#25`
  require `data/processed/training_curated.jsonl`.
- [ ] Align `#22` output paths with `#24` input paths.
- [ ] Add hardware gates for heavy runs:
  trajectory collection, DeepSeek-style generation, QLoRA, SFT, optional RL.
- [ ] Add runtime dependency matrix:
  WSL2, Colab, Kaggle, HPC.
- [ ] Decide optional dependency strategy for Unsloth, vLLM, NeMo/Curator, and
  CUDA-specific PyTorch wheels.
- [ ] Make `scripts/kaggle.sh` less dependent on repo-local `.venv` and `.env`
  if it must run in hosted environments.

## Suggested New Child Issues

Use new child issues instead of expanding epics directly.

- [ ] `#26`: Bit-manipulation solver implementation and tests.
  - Parent: `#9`
  - Blocked by: `#23`
- [ ] `#27`: Runtime/dependency reproducibility matrix and lock strategy.
  - Parent: `#12` or new process child under `#11`
  - Blocked by: PM decision on supported environments.
- [ ] `#28`: Prompt-sweep output parser contract resolution.
  - Parent: `#6`
  - Blocked by: `#19`
- [ ] `#29`: Real eval artifact generation and storage policy.
  - Parent: `#4`
  - Blocked by: `#17`, artifact tracking decision.
- [ ] `#30`: Synthetic teacher/augmentation pipeline implementation.
  - Parent: `#10`
  - Blocked by: `#23`, `#24`
- [ ] `#31`: SFT masking smoke-run implementation.
  - Parent: `#11`
  - Blocked by: `#19`; must conform to the verified `#14` contract.
- [ ] `#32`: Accepted dummy or first-adapter Kaggle submission evidence.
  - Parent: `#5`, `#12`
  - Blocked by: `#20`; must conform to the verified `#14` contract.
- [ ] `#33`: External baseline reproduction pass for konbu17 and Tong
  (`tonghuikang`) ideas.
  - Parent: `#2`
  - Blocked by: baseline source availability, license/provenance capture, and
    PM signoff for deviations from the verified `#14` contract.
- [ ] `#34`: Optional RL/GRPO planning.
  - Parent: future training/research epic
  - Blocked by: strong SFT baseline and stable reward/eval contract.

## Four-Person Assignment Guide

### Member 1 - PM / Constraints / Process

Primary issues:

- `#1`, `#12`, `#14`, `#15`, `#27`, `#28`, `#32`

Immediate tasks:

- [x] Freeze Kaggle constraints in `docs/architecture/COMPETITION.md`.
- [ ] Fix or assign process/test failure.
- [ ] Resolve status/dependency drift.
- [ ] Own general-chat dependency checks and review gates.

Must check with:

- Member 2 before changing eval artifact names.
- Member 4 before changing packaging/training constraints.

### Member 2 - Data + Evaluation

Primary issues:

- `#3`, `#4`, `#17`, `#18`, `#19`, `#29`

Immediate tasks:

- [ ] Produce canonical dataset rows.
- [ ] Produce validation/golden artifacts.
- [ ] Produce real baseline eval run.
- [ ] Own normalizer and eval record compatibility.

Must check with:

- Member 1 before finalizing scoring/normalization.
- Member 3 before declaring sweep inputs ready.
- Member 4 before evaluating candidate adapters.

### Member 3 - Prompting + Failure Analysis + Solver Design

Primary issues:

- `#6`, `#7`, `#8`, `#9`, `#21`, `#22`, `#23`, `#26`, `#28`

Immediate tasks:

- [ ] Resolve prompt output-parser contract.
- [ ] Run prompt sweeps after eval inputs exist.
- [ ] Produce failure slices.
- [ ] Define solver framework and first solver child issue.

Must check with:

- Member 2 for baseline run and split artifacts.
- Member 1 for output-format constraints.
- Member 4 before recommending SFT data strategy.

### Member 4 - Synthetic Data + Training + Packaging

Primary issues:

- `#5`, `#10`, `#11`, `#20`, `#24`, `#25`, `#30`, `#31`, `#32`

Immediate tasks:

- [ ] Keep packaging tests green.
- [ ] Implement training preflight against the verified `#14` constraints.
- [ ] Prepare synthetic-data recipe after solver/failure slices exist.
- [ ] Run tiny SFT smoke job before production training.

Must check with:

- Member 1 before GPU jobs or packaging changes.
- Member 2 before promoting checkpoints.
- Member 3 before using synthetic/failure-slice data.

## Validation Commands

Run these after changes relevant to each area:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/tooling/test_learn_docs_guard.py -q -p no:cacheprovider
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/inference/test_submission_packaging.py -q -p no:cacheprovider
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_prompt_sweeps.py tests/evaluation/test_notebook_05_contract.py -q -p no:cacheprovider
```

```bash
git diff --check
```

Expected current risk:

- `git diff --check` may report many existing whitespace/line-ending issues in
  the dirty working tree. Do not mix broad whitespace cleanup with feature work
  unless the PM explicitly scopes it.

## Definition of Project MVP

The project has an MVP only when all are true:

- [ ] Kaggle constraints are frozen or explicitly accepted as provisional with
  PM signoff.
- [ ] Official or approved mirror dataset is normalized to canonical schema.
- [ ] Real validation and golden artifacts exist.
- [ ] Real baseline eval artifacts exist.
- [ ] Prompt sweeps have real results or are explicitly deferred.
- [ ] At least one adapter package can be produced and validated.
- [ ] Golden gate blocks bad checkpoints.
- [ ] Submission runbook can reproduce the artifact from source inputs.

Until then, the repo should be described as a tested foundation, not a completed
competition MVP.
