# Issue 14 — Constraints Freeze

**Parent Issue**: `#14`  
**Deliverable path**: `docs/execution/plans/issue-14-constraints-freeze.md`  
**Dependencies**: `docs/architecture/COMPETITION.md`, `docs/planning/plan_v0.2.md`, `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`, `data/external/konbu17/**`  
**Agent owner**: Agent 1  
**Human reviewer**: Project owner / Kaggle account holder  
**Architecture reviewer**: Required  
**Status**: `decision-complete (gated by Kaggle rules capture)`

## 1. Goal

Freeze the competition-facing contract set that downstream notebooks must obey:

- base model identity and load recipe
- scoring / normalization behavior
- LoRA adapter constraints
- required `submission.zip` layout

This issue does **not** implement training, evaluation, or packaging code. It only locks the decisions and names the blockers that prevent a fully authoritative freeze from repo evidence alone.

## 2. Non-Goals

- No notebook code
- No model training or inference implementation
- No Kaggle API automation
- No changes to other plan docs
- No changes to other agents’ report sections

## 3. Decisions Required

| Decision | Status | Why it matters | Upstream gate |
|---|---|---|---|
| Base model ID | `BLOCKED` | Wrong model ID means wrong tokenizer, module names, and adapter target set | Kaggle rules / submission demo notebook |
| Base-model load recipe | `BLOCKED` | `trust_remote_code`, `enable_thinking`, dtype, and attention backend affect reproducibility | Kaggle rules / submission demo notebook |
| Scoring / normalization contract | `BLOCKED` | Local eval must match Kaggle scoring, or every downstream metric is misleading | Kaggle rules text |
| LoRA adapter cap | `frozen provisional: rank <= 32` | A higher rank may violate submission rules | Kaggle rules text; confirmed by repo evidence only as a working cap |
| Allowed adapter targets | `frozen provisional: konbu17 9-module set` | Module choice affects training cost and submission validity | Kaggle rules text |
| Required submission layout | `frozen provisional: root-only adapter files` | Extra files can break submission packaging | Kaggle rules text; konbu17 baseline |

## 4. Constraint Table

| Variable | Frozen value (or BLOCKED) | Evidence | Confidence | Manual human action |
|---|---|---|---|---|
| Base model ID | `BLOCKED` | `docs/architecture/COMPETITION.md` marks it unresolved; `docs/planning/plan_v0.2.md` hard-claims `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`; `data/external/konbu17/cells/cell10.py` loads `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` | low | Open the Kaggle rules page or submission demo notebook and record the exact HF slug + revision/sha |
| Base-model load recipe | `BLOCKED` | `plan_v0.2.md` uses `trust_remote_code=True`, BF16, and `enable_thinking=True`; `cell10.py` uses BF16, `trust_remote_code=True`, `attn_implementation="eager"`, and `pad_token=eos_token` fallback | low | Capture the exact Kaggle demo load cell and preserve it as the canonical recipe |
| Scoring / normalization | `BLOCKED` | `docs/architecture/ARCHITECTURE.md` centers exact-match normalization; `docs/planning/plan_v0.2.md` hardcodes `\\boxed{}` extraction and rewards; `docs/analysis/ADVERSARIAL_REVIEW.md` says the task looks like rule inference with plain-string answers | low | Copy the Kaggle scoring / answer-format text into the repo and resolve whether reasoning text is ignored, normalized, or rejected |
| LoRA-only constraint | `LoRA adapter only` | `docs/architecture/COMPETITION.md` says LoRA adapter only; `docs/planning/plan_v0.2.md` also frames the submission as a LoRA adapter | medium | Verify the rules page for any exception or extra artifact requirement |
| LoRA rank cap | `rank <= 32` (provisional freeze) | `docs/architecture/COMPETITION.md` flags `LoRA rank <= 32`; `data/external/konbu17/cells/cell11.py` uses `r=32`; `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` treats this as the safe cap | medium | Confirm the cap on Kaggle rules before any larger sweep |
| LoRA target modules | `["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj","lm_head"]` (provisional baseline) | `data/external/konbu17/cells/cell11.py` uses this exact 9-module set | medium | Check whether Kaggle restricts target modules or whether a narrower subset is preferred |
| Adapter dtype / precision | `bf16` (provisional baseline) | `data/external/konbu17/cells/cell10.py` loads BF16; `docs/planning/plan_v0.2.md` and `COMPETITION.md` both describe BF16-era model handling | medium | Verify whether Kaggle evaluation requires BF16, FP16, or accepts either |
| Required `submission.zip` layout | Root contains only `adapter_config.json` and `adapter_model.safetensors` | `data/external/konbu17/cells/cell17.py` zips exactly those two files at the root | high | Confirm whether Kaggle wants any extra metadata file or a different archive root |
| Provenance placement | Out-of-band only | `docs/architecture/ARCHITECTURE.md` defines `PackageManifest`; `cell17.py` keeps the submission zip minimal | high | Keep manifests in docs / experiments, not inside the submission artifact, unless rules say otherwise |

## 5. Explicit Blockers

These constraints remain blocked until a human captures the Kaggle rules or demo notebook text in the repo:

1. Exact base model ID and revision.
2. Exact scoring / normalization contract.
3. Any submission size or filename limits not visible in repo evidence.
4. Any target-module restrictions beyond the provisional konbu17 baseline.

If Kaggle rules or the submission demo are inaccessible from this environment, the manual path is:

1. Open the competition page in a browser.
2. Accept the rules.
3. Open the submission demo notebook.
4. Copy the relevant rule text into the repo issue or docs.
5. Reconcile the copied rules against the provisional freezes above.

## 6. Files To Create or Modify Later

These files are downstream consumers of the frozen contract. They are not changed by this issue, but they should be updated after the gates clear.

- `src/evaluation/normalize.py`
- `src/evaluation/harness.py`
- `src/inference/submission.py`
- `configs/lora_baseline.yaml`
- `configs/lora_qlora.yaml`
- `tests/test_normalize.py`
- `tests/test_submission_zip.py`
- `docs/architecture/ARCHITECTURE.md`
- `docs/execution/NOTEBOOKS.md`
- `docs/execution/SPRINTS.md`

## 7. Interfaces / Contracts Referenced

- `ReasoningExample` canonical raw row contract
- `EvalRecord` evaluation artifact contract
- `PackageManifest` packaging contract
- `normalize_answer()` / exact-match evaluation policy
- `LoRAConfig` adapter contract
- `submission.zip` root artifact contract

## 8. Step-by-Step Tasks

1. Compare the plan, architecture notes, and konbu17 baseline to isolate all competition-facing assumptions.
2. Record each assumption as either frozen or blocked in the constraint table.
3. Mark any repo-visible provisional values as provisional, not authoritative.
4. Write explicit human actions for all blocked decisions.
5. Enumerate the downstream files that will consume these frozen values.
6. Leave the report in `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` under `<!-- AGENT_1_REPORT -->`.

## 9. Verification Commands

These are the minimum checks to run once the human has supplied the missing Kaggle facts.

| Command | Expected output |
|---|---|
| `rg -n "BLOCKED|rank <= 32|adapter_config.json|adapter_model.safetensors" docs/execution/plans/issue-14-constraints-freeze.md docs/analysis/PLAN_V0_2_REVIEW_PLAN.md` | The frozen constraints and blockers are visible in both docs |
| `git diff --check` | No whitespace or patch-format errors |
| `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw` | No `403 Forbidden`; competition files download successfully |
| `kaggle competitions files -c nvidia-nemotron-model-reasoning-challenge` | `train.csv` and `test.csv` are listed |
| `python -m pytest tests/test_normalize.py` | Normalization tests pass after the contract is implemented |
| `python -m pytest tests/test_submission_zip.py` | Submission layout tests pass after packaging is implemented |

## 10. Acceptance Checklist

Aligned to `docs/execution/ISSUE_REVIEW_HARNESS.md`.

- [ ] Parent issue is declared.
- [ ] Deliverable path is explicit.
- [ ] Dependencies are explicit.
- [ ] Agent owner is explicit.
- [ ] Human reviewer is explicit.
- [ ] Architecture reviewer is explicit because the issue changes shared contracts.
- [ ] Acceptance checklist is present.
- [ ] Sources to verify are listed.
- [ ] Risks / open questions are listed or clearly embedded in blockers.
- [ ] Each frozen constraint has evidence, confidence, and a manual human action.
- [ ] Explicit blockers are separated from provisional freezes.
- [ ] No notebook code was added.

## 11. Risks / Open Questions

- Exact base model slug may still differ from the konbu17 baseline.
- Kaggle scoring may normalize answers differently from the current exact-match assumption.
- Kaggle may impose extra adapter or packaging limits that are not visible in repo evidence.
- `LoRA rank <= 32` is the safest repo-backed working cap, but it still needs human confirmation.

## 12. Sources Used

- `docs/architecture/COMPETITION.md`
- `docs/planning/plan_v0.2.md`
- `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
- `docs/architecture/ARCHITECTURE.md`
- `data/external/konbu17/cells/cell10.py`
- `data/external/konbu17/cells/cell11.py`
- `data/external/konbu17/cells/cell13.py`
- `data/external/konbu17/cells/cell17.py`
