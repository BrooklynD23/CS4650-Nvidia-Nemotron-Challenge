# Issue 14 — Constraints Freeze

**Parent Issue**: `#14`  
**Deliverable path**: `docs/execution/plans/issue-14-constraints-freeze.md`  
**Dependencies**: `docs/architecture/COMPETITION.md`, `docs/planning/plan_v0.2.md`, `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`, `data/external/konbu17/**`  
**Agent owner**: Agent 1  
**Human reviewer**: Project owner / Kaggle account holder  
**Architecture reviewer**: Required  
**Status**: `verified-frozen (Kaggle rules captured 2026-04-29; see docs/architecture/COMPETITION.md "Verified" section)`

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
| Base model ID | `VERIFIED: KaggleHub metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` | Wrong model ID means wrong tokenizer, module names, and adapter target set | Resolved by the 2026-04-29 freeze in `docs/architecture/COMPETITION.md` |
| Base-model load recipe | `VERIFIED: trust_remote_code=True, torch.bfloat16, device_map="auto"` | `trust_remote_code`, dtype, and attention backend affect reproducibility | Resolved by the 2026-04-29 freeze in `docs/architecture/COMPETITION.md` |
| Scoring / normalization contract | `VERIFIED: \boxed{} extraction, exact match or 1e-3 numeric tolerance, reasoning allowed` | Local eval must match Kaggle scoring, or every downstream metric is misleading | Resolved by the 2026-04-29 freeze in `docs/architecture/COMPETITION.md` |
| LoRA adapter cap | `VERIFIED: rank <= 32 (evaluator enforces max_lora_rank=32)` | A higher rank fails the evaluator’s LoRA cap | Resolved by the 2026-04-29 freeze |
| Allowed adapter targets | `VERIFIED demo target modules: in_proj|out_proj|up_proj|down_proj (4 modules); broader sets allowed if r <= 32 holds` | Module choice affects training cost and submission validity | Resolved by the 2026-04-29 freeze |
| Required submission layout | `VERIFIED: adapter_config.json + adapter_model.safetensors at zip root` | Extra files break submission packaging | Resolved by the 2026-04-29 freeze |
| Evaluator decode params | `VERIFIED: max_tokens=7680, temperature=0.0, top_p=1.0, max_model_len=8192` | Local eval must mirror evaluator decoding to be predictive | Resolved by the 2026-04-29 freeze |
| Deadlines & submission cap | `VERIFIED: entry 2026-05-10, daily limit 5/day, final 2026-06-15 23:59 UTC` | Anchors the execution timeline | Resolved by the 2026-04-29 freeze |

## 4. Constraint Table

| Variable | Frozen value | Evidence | Confidence | Manual human action |
|---|---|---|---|---|
| Base model ID | `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (KaggleHub) | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29); matches `data/external/konbu17/cells/cell10.py` | high | None — frozen |
| Base-model load recipe | `trust_remote_code=True`, `torch.bfloat16`, `device_map="auto"` | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| Scoring / normalization | `\boxed{}` extraction; exact match or `1e-3` numeric tolerance; reasoning text allowed but ignored outside the box | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| LoRA-only constraint | `LoRA adapter only` | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| LoRA rank cap | `r <= 32` (evaluator enforces `max_lora_rank=32`) | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| LoRA target modules | Demo set: `in_proj|out_proj|up_proj|down_proj` (4 modules) | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29). Broader sets are allowed only if `r <= 32` is preserved. | high | None — frozen for the demo target set |
| Adapter dtype / precision | `bf16` | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| Required `submission.zip` layout | Root contains only `adapter_config.json` and `adapter_model.safetensors` | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29); konbu17 baseline matches | high | None — frozen |
| Evaluator decode params | `max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_model_len=8192` | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| Entry deadline / submission cap / final deadline | Entry: 2026-05-10; daily limit: 5 submissions/day; final: 2026-06-15 23:59 UTC | Verified via `docs/architecture/COMPETITION.md` "Verified" section (snapshot 2026-04-29) | high | None — frozen |
| Provenance placement | Out-of-band only | `docs/architecture/ARCHITECTURE.md` defines `PackageManifest`; `cell17.py` keeps the submission zip minimal; reaffirmed by the verified zip layout | high | Keep manifests in docs / experiments, not inside the submission artifact |

## 5. Resolved Blockers (historical)

All blockers from the original 2026-04-20 draft are resolved by the 2026-04-29 freeze recorded in `docs/architecture/COMPETITION.md` "Verified" section:

1. Base model ID and load recipe — resolved (KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`, BF16, `trust_remote_code=True`, `device_map="auto"`).
2. Scoring / normalization contract — resolved (`\boxed{}` extraction, exact match or `1e-3` numeric tolerance, reasoning allowed).
3. Submission size and filename layout — resolved (root-only `adapter_config.json` + `adapter_model.safetensors`).
4. Target-module restrictions — resolved (demo set `in_proj|out_proj|up_proj|down_proj`; broader sets allowed under `r <= 32`).

The historical manual-capture path is preserved here only for traceability; downstream work should consume the verified contract directly from `docs/architecture/COMPETITION.md`.

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

These are the minimum checks to run against the verified `#14` freeze (snapshot 2026-04-29).

| Command | Expected output |
|---|---|
| `rg -n "VERIFIED|rank <= 32|adapter_config.json|adapter_model.safetensors|metric/nemotron-3-nano-30b-a3b-bf16" docs/architecture/COMPETITION.md docs/execution/plans/issue-14-constraints-freeze.md` | The verified constraints (base model, rank cap, zip layout) are visible in both docs |
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

The previously-listed risks are resolved by the 2026-04-29 freeze:

- Base model slug confirmed (KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`, matches konbu17 baseline).
- Scoring contract confirmed (`\boxed{}` extraction, exact match or `1e-3` numeric tolerance, reasoning allowed).
- Adapter cap confirmed (`r <= 32`, evaluator enforces `max_lora_rank=32`); packaging limited to `adapter_config.json` + `adapter_model.safetensors` at zip root.

Residual operational risks (out of scope for this issue, tracked elsewhere):

- Final-deadline timezone is documented as UTC; human owner should re-confirm before the 2026-06-15 cutoff.
- The 5/day submission cap means iteration speed is the constraint, not legality of any single attempt.

## 12. Sources Used

- `docs/architecture/COMPETITION.md`
- `docs/planning/plan_v0.2.md`
- `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
- `docs/architecture/ARCHITECTURE.md`
- `data/external/konbu17/cells/cell10.py`
- `data/external/konbu17/cells/cell11.py`
- `data/external/konbu17/cells/cell13.py`
- `data/external/konbu17/cells/cell17.py`
