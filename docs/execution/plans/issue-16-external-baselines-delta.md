# Issue 16 — External Baselines Delta (Planning Only)

**Parent issue:** `#16`  
**Deliverable path:** `docs/execution/plans/issue-16-external-baselines-delta.md`  
**Dependencies:** `#13` (review workflow), `#14` (constraints freeze gating)  
**Agent owner:** Orchestrator (this session)  
**Human reviewer:** Repo owner / architecture reviewer  
**Architecture reviewer:** Required (touches shared assumptions; may trigger contract edits)  
**Status:** `decision-complete (with explicit gates)`

## 1) Goal

Produce a decision-complete plan to review external baselines (starting with `konbu17` and Tong’s public pipeline), extract the transferable design ideas, and record *explicit deltas* vs our repo’s current architecture + plan assumptions **without implementing notebooks yet**.

Primary output is a delta matrix plus gated recommendations, so Waves B–D do not build on the wrong base model, output format, or masking assumptions.

## 2) Non-goals

- Do not implement `notebooks/01_external_baselines_and_design_deltas.ipynb` yet
- Do not run long training or sweeps
- Do not “pick a winner” baseline without passing the `#14` freeze gates (base model + scoring + adapter constraints)

## 3) Decisions required (and gates)

| Decision | Status | Why it matters | Upstream gate |
|---|---|---|---|
| Which external baselines are in-scope | frozen | Keeps scope bounded | This plan |
| Baseline extraction checklist (what to capture) | frozen | Prevents missing the high-leverage deltas | This plan |
| Whether to adopt `\\boxed{}` or exact-match output style | gated | Must match Kaggle scoring | `#14` scoring gate |
| Base model ID for reproduction | gated | Drives tokenizer/template/module names | `#14` base model gate |
| LoRA defaults (rank/targets) adopted as provisional | frozen provisional | Prevents illegal rank | `#14` adapter constraints gate |

## 4) Files to create / modify

### Planning artifacts (now)

- `docs/execution/plans/issue-16-external-baselines-delta.md` (this file)

### Analysis / coordination artifacts (later, after execution)

- Update: `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` (append a delta summary under Agent D or a new “Agent — External Baselines Delta” section)
- Optional update: `docs/architecture/ARCHITECTURE.md` if the baseline review forces contract changes (must be architecture-reviewed)

### Notebook artifact (later; not in this planning phase)

- Create (later): `notebooks/01_external_baselines_and_design_deltas.ipynb`
- Update (later): `docs/execution/NOTEBOOKS.md` status and pointers

## 5) Interfaces / contracts referenced

- Constraint freeze variables from `docs/execution/plans/issue-14-constraints-freeze.md`
- Raw/eval/package schema contracts referenced in `docs/architecture/ARCHITECTURE.md`
- Minimal submission zip layout from `docs/execution/plans/issue-20-submission-packaging-and-provenance.md`

## 6) Step-by-step tasks

### Task 1 — Freeze “baseline review checklist” (what to extract)

For each baseline (konbu17, Tong):

- [ ] Base model ID + load recipe (`trust_remote_code`, dtype, chat template, thinking flag)
- [ ] Output contract (does it require `\\boxed{}`? any stop tokens? any `</think>` boundary?)
- [ ] Dataset inputs (official Kaggle vs mirror vs synthetic), plus split policy
- [ ] LoRA config (rank, alpha, dropout, bias, task type, target modules)
- [ ] Training loop (TRL/transformers version assumptions, masking policy, max length)
- [ ] Inference/eval method (local metric, normalization rules, failure slices)
- [ ] Packaging/submission artifact layout
- [ ] Environment/deps hacks (Kaggle runtime quirks, wheels, Triton/ptxas, etc.)

### Task 2 — Extract “konbu17” baseline facts from repo snapshot

Repo-local evidence already exists under `data/external/konbu17/`.

- [ ] Create a “konbu17 baseline facts” section with file references:
  - base model pull + load args (`cells/cell10.py`)
  - LoRA config (`cells/cell11.py`)
  - training recipe + formatting (`cells/cell13.py`)
  - packaging (`cells/cell04.py`, `cells/cell17.py`)
- [ ] Record any implicit assumptions (e.g., TRL default masking behavior) as risks.

### Task 3 — Extract Tong pipeline facts (gated by external availability)

If Tong’s repo is not vendored locally, capture facts manually:

- [ ] Identify Tong’s key contracts: masking, prompt templates, data augmentation, eval protocol.
- [ ] Record a “manual evidence” checklist so a human can paste/quote the relevant sections into repo docs if needed.

### Task 4 — Build a delta matrix vs our repo contracts

- [ ] Compare baselines against:
  - `docs/planning/plan_v0.2.md` assumptions (notably 4B vs 30B; r=64 vs r<=32; boxed formatting)
  - `docs/architecture/ARCHITECTURE.md` contracts (exact match, masking, eval record shape, packaging manifest)
  - `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` drift matrix and gates
- [ ] Produce a delta table with “Adopt / Reject / Gate” per item.
- [ ] For each “Adopt”, name the exact downstream issue(s) that should consume the change (e.g., `#19` normalization, `#20` packaging).

### Task 5 — Emit explicit recommendations with stop conditions

- [ ] Recommendations must be either:
  - “safe provisional default” (repo-evidence backed), or
  - “BLOCKED until `#14` is resolved”
- [ ] Do not recommend final training hyperparameters until base model + scoring + adapter constraints are frozen.
- [ ] Do not recommend `\\boxed{}` or `<think>` defaults until scoring is verified.

## 7) Verification commands (and expected outputs)

| Command | Expected output |
|---|---|
| `ls data/external/konbu17/cells | head` | Shows `cell10.py`, `cell11.py`, `cell13.py`, etc. |
| `rg -n \"metric/nemotron-3-nano|apply_chat_template|enable_thinking\" data/external/konbu17` | Points to baseline base model + thinking usage |
| `rg -n \"lora_\" data/external/konbu17/cells/cell11.py` | Shows LoRA rank/alpha/targets |
| `rg -n \"submission\\.zip|adapter_model\\.safetensors|adapter_config\\.json\" data/external/konbu17` | Shows packaging logic and required files |
| `git diff --check` | No whitespace errors |

## 8) Acceptance checklist (aligned to `docs/execution/ISSUE_REVIEW_HARNESS.md`)

- [ ] Baseline review checklist is explicit and complete.
- [ ] konbu17 extraction uses repo-local evidence with file references.
- [ ] Tong baseline extraction is either evidence-backed or explicitly BLOCKED with manual capture steps.
- [ ] Delta matrix clearly compares baseline vs repo contracts and marks each item as Adopt/Reject/Gate.
- [ ] Recommendations include explicit stop-conditions tied to `#14` freeze gates.
- [ ] No notebooks were created or modified.

## 9) Risks / open questions

- Tong baseline may require web access or manual copying if not vendored.
- Konbu17 uses boxed formatting and “thinking” template features; if Kaggle scoring is strict exact-match, adopting this could be harmful.
- TRL default masking behavior may differ by version; if masking is critical, this must be explicitly set in our future training scripts.

## 10) Sources

Repo-local:
- `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
- `docs/planning/plan_v0.2.md`
- `docs/architecture/ARCHITECTURE.md`
- `docs/architecture/COMPETITION.md`
- `data/external/konbu17/cells/cell10.py`
- `data/external/konbu17/cells/cell11.py`
- `data/external/konbu17/cells/cell13.py`
- `data/external/konbu17/cells/cell04.py`
- `data/external/konbu17/cells/cell17.py`

External sources to verify (manual capture may be required):
- Tong Hui Kang public repo: `https://github.com/tonghuikang/nemotron`
- Kaggle notebook: `https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2`

