# Plan v0.2 Review Plan (Constraints + MVP Alignment)

**Date:** 2026-04-20  
**Inputs:** `docs/planning/plan_v0.2.md`, `docs/planning/plan_review.md`, `docs/architecture/{COMPETITION,ARCHITECTURE}.md`, `docs/analysis/ADVERSARIAL_REVIEW.md`, `docs/execution/{SPRINTS,NOTEBOOKS,ISSUE_REVIEW_HARNESS}.md`, external baseline under `data/external/konbu17/`  
**Output:** This document is the single shared place to log mismatches and the resolution checklist.

## Verified Kaggle Facts (Authoritative)

These facts were verified via Kaggle CLI in this repo on **2026-04-20**.

### Competition deadline

```text
ref                                                                            deadline             category       reward  teamCount  userHasEntered
-----------------------------------------------------------------------------  -------------------  --------  -----------  ---------  --------------
https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge  2026-06-15 23:59:00  Featured  106,388 Usd       2284           False
```

### Competition data files exposed via the API

```text
name             size  creationDate
---------  ----------  --------------------------
test.csv         1461  2026-03-14 17:56:54.418000
train.csv     3069304  2026-03-14 17:56:54.414000
```

### Known verification gaps (still blocking)

- Kaggle CLI in this environment does **not** support `kaggle competitions join`; attempts to download data returned `403 Forbidden`. This implies we likely need manual acceptance of rules on Kaggle web UI before the API allows downloads.
- The Kaggle CLI does not expose rules details (rank cap, output parsing rules, base model ID) via `kaggle competitions ...` subcommands available here.

## Drift Matrix (Plan vs Architecture vs External Baselines vs Kaggle)

| Area | What docs/baselines currently imply | Risk | What we need to freeze |
|---|---|---|---|
| Base model ID | Plan hard-claims `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`; architecture/competition docs mark it unresolved; konbu17 uses KaggleHub `metric/nemotron-3-nano-30b-a3b-bf16/...` | Training/eval on the wrong model/tokenizer/chat template; wrong LoRA target modules; wasted compute | **Authoritative evaluator base model ID + revision** (and the exact load recipe used in submission demo) |
| Output/scoring contract | Plan optimizes `\\boxed{}` and `<think>`; architecture/analysis expects exact-match string answers for rule induction categories | “Improvement” locally but zero leaderboard gain if formatting mismatches or scoring is strict exact match | **Exact normalization rules** (whitespace/case rules, category-specific parsers, whether reasoning text is ignored or penalized) |
| LoRA constraints | Plan defaults to `r=64`; competition notes “rank<=32” (unverified); konbu17 uses `r=32` | Hard invalid submission if a cap exists | **Rank cap and any adapter restrictions** (allowed target modules, dtype/quantization rules, file size limits) |
| Packaging artifact | Plan is vague; architecture defines `PackageManifest`; konbu17 packages `submission.zip` with only `adapter_config.json` + `adapter_model.safetensors` at zip root | Submission rejection if zip layout differs; provenance may leak into submission if mixed in | **Kaggle-required zip layout**; keep provenance manifests **out-of-band** unless explicitly allowed |
| Dataset schema | Architecture defines `ReasoningExample` but plan examples use `question/expected_answer` and synthetic uses `input/output` | Glue code drift; silent eval bugs; hard to slice failures | **Canonical raw example schema** + explicit mappings to eval and SFT formats |
| Eval artifacts | Architecture defines record-level `EvalRecord`; plan mostly talks aggregate mean/std | No failure slicing; irreproducible comparisons across prompt/decode runs | **Record-level eval output** (JSONL/Parquet) with `example_id` + run config attribution |
| Dependencies/repro | Plan lists `unsloth`, `vllm`, NeMo Curator; `requirements.txt` omits them and is loosely pinned | “It worked on my machine” failures across WSL2/Colab/HPC/Kaggle | **Supported runtime matrix + pinned lock strategy** (torch + CUDA build matters) |

## Decisions To Freeze (Go/No-Go Gates)

These are the decisions that block “real” training runs.

1. **Base model gate (BLOCKER):** confirm the evaluator base model ID (30B-A3B vs 4B) and load recipe (tokenizer/chat template, `trust_remote_code`, thinking enablement).
2. **Scoring gate (BLOCKER):** confirm answer parsing rules and whether output must be plain answer vs can include reasoning.
3. **Adapter constraints gate (BLOCKER):** confirm LoRA rank cap and any packaging limits.
4. **Submission artifact gate (BLOCKER):** confirm required zip file layout (use `konbu17` packaging as a compatibility baseline unless Kaggle rules contradict it).

## Required Contract Changes (Schemas / Interfaces)

These changes are required to stop schema drift across notebooks and eventual `src/` code.

1. **Keep `ReasoningExample` as the canonical raw row contract** (model-agnostic): `id`, `category`, `prompt`, `answer`, `source`, `split`, `metadata`.
2. **Define a first-class SFT contract** (suggested name: `SFTExample`) derived from `ReasoningExample`. Fields: `example_id`, `category`, `messages`, `completion`, `source`, `split`, `provenance`. Rationale: plan’s synthetic `input/output` format should become this, instead of inventing a third schema.
3. **Make evaluation artifacts record-level** and link back to the raw example:
   Add `example_id` (or rename `id` consistently) to `EvalRecord`. Capture decode/thinking settings either directly in each `EvalRecord` or via an `EvalRunConfig` referenced by `run_id`.
4. **Treat normalization as versioned and swappable**:
   Default should match Kaggle scoring (likely exact match). Keep `\\boxed{}` extraction only as a category-specific normalizer, not the global default.
5. **Separate Kaggle submission zip from provenance**:
   Submission zip must contain only minimal required files. Provenance (`PackageManifest` + eval summary + config snapshots) should be stored alongside, but not inside the Kaggle zip unless explicitly allowed.

## Dependency + Runtime Risk Audit

Immediate risks to address before implementation hardens:

- `requirements.txt` omits `unsloth` and `vllm` referenced in `plan_v0.2`; NeMo/Curator paths are not dependency-backed yet.
- Version pinning is incomplete (`torch==2.4.0` pinned but CUDA build is environment-dependent; most other libs are `>=` or unpinned).
- Notebook scaffolds assume Python 3.11; there is no repo-level Python pin/lock.
- `scripts/kaggle.sh` assumes a repo-local `.venv` and `.env` sourcing, which will not hold on Kaggle/Colab/HPC by default.
- `.gitignore` currently ignores `data/**`, which would also ignore small but reproducibility-critical eval artifacts if stored under `data/eval/` as described in the plan.

## Resolution Checklist (Action Items + Owners)

### P0 (Blockers)

- [ ] **Freeze base model ID + load recipe** (Owner: Issue `#14` notebook).
Evidence sources: Kaggle submission demo notebook; Kaggle rules tab (manual if needed); external baseline `data/external/konbu17/cells/cell10.py`.
- [ ] **Freeze scoring/normalization contract** (Owner: Issue `#14` + `#19` notebook). Define `normalize_answer()` rules for local eval to match Kaggle.
- [ ] **Freeze LoRA constraints** (Owner: Issue `#14`). Assume `r<=32` until proven otherwise; plan must not hardcode `r=64`.
- [ ] **Freeze submission zip layout** (Owner: Issue `#20`). Start from konbu17: zip root contains `adapter_config.json` + `adapter_model.safetensors`.

### P1 (Schema/contract alignment)

- [ ] **Document explicit schema mappings** between plan examples and `ReasoningExample`/`EvalRecord` in `ARCHITECTURE.md` or a dedicated `docs/architecture/contracts.md`.
- [ ] **Add `example_id` + run config attribution** to the evaluation artifact contract so error slicing is possible.
- [ ] **Define `SFTExample`** (messages+completion) so synthetic/teacher data isn’t a third ad-hoc schema.

### P2 (Dependency/repro hardening)

- [ ] **Define a supported runtime matrix** (Local WSL2, Colab, HPC, Kaggle) and a pin/lock strategy.
- [ ] **Decide whether Unsloth/vLLM/NeMo are required** (if yes, add explicit dependency sets and test them).
- [ ] **Fix reproducibility artifact storage**: golden set/validation sets should be trackable (not lost under ignored directories), while raw datasets remain ignored.

## Agent Reports (Append-Only)

### Agent A — Competition vs Plan

## Agent A — Competition vs Plan

- **Findings:** `docs/architecture/COMPETITION.md` treats the **base model ID as unresolved** (likely “Nemotron-3 Nano baseline” could mean 30B-A3B), while `docs/planning/plan_v0.2.md` states the base model is **definitively** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and builds assumptions on top of it (hybrid module names, `enable_thinking`, `<think>` handling). `COMPETITION.md` also flags a likely **LoRA rank cap <= 32**, but `plan_v0.2.md` hard-codes **`r=64`** (repo structure comment + Phase 4.1 “NVIDIA Recommended” config). `plan_v0.2.md` enforces **math-centric output formatting** (`\\boxed{}` smoke test, `extract_boxed_answer`, “Structured output forcing `\\boxed{}`”, RL “format_reward” for `<think>` + `\\boxed{}`), while `COMPETITION.md` + `docs/analysis/ADVERSARIAL_REVIEW.md` indicate the task is **I/O rule inference** with **plain-string answers** across categories like `bit_manipulation`, `text_cipher`, etc (likely strict exact match). `plan_v0.2.md`’s primary eval framing leans on **MATH500/AIME25/GPQA** and boxed extraction, whereas competition scoring is (per `COMPETITION.md`) **accuracy on the hidden benchmark** (details like macro vs micro, normalization unknown). `plan_v0.2.md` leaves the **deadline as an open question**, while `COMPETITION.md` provides working dates (**Entry: 2026-06-08; Final: 2026-06-15**, unverified).
- **Impact:** If the Kaggle evaluator uses a different base checkpoint than the plan, you risk training against the wrong tokenizer/chat template/module names and producing an adapter that either won’t load or won’t generalize to the eval environment. If rank is capped at 32, any `r=64` work is invalid (wasted training time, potential submission rejection). If Kaggle scoring is strict exact-match on short answers, enforcing `\\boxed{}` / encouraging verbose `<think>` traces can drive near-zero accuracy even when the model “knows” the answer. If local eval is benchmark-math-centric, you can get misleading “improvements” that don’t move leaderboard accuracy. Unanchored deadlines increase the chance you don’t reach a working submission before 2026-06-08/2026-06-15.
- **Recommendations:** Treat the Kaggle base model as a **hard external dependency**: verify the exact HF repo ID (and revision if specified) via the Kaggle submission demo notebook/rules, and make `base_model_id` a single config knob everywhere (don’t bake in `<think>` token IDs or module names until confirmed). Enforce **competition-safe LoRA configs** now: set max rank to **32** across the plan/configs and do a sweep like `r in {8,16,32}` (tune `lora_alpha` accordingly) rather than assuming “r=64 sweet spot.” Replace `\\boxed{}`-centric plumbing with **competition-aligned answer handling**: generate “answer-only” outputs by default, implement a `normalize_answer()` that matches Kaggle’s exact rules (and only the rules Kaggle uses), and remove format rewards that don’t map to leaderboard scoring. Make the competition metric the primary objective: build local eval on the **competition-style dataset** (official or mirror) with **overall + per-category exact-match accuracy**, and treat MATH500/AIME/GPQA as optional diagnostics only. Anchor execution to dates: assume **2026-06-08 entry / 2026-06-15 final** until verified, and prioritize (1) a dummy submission, (2) a valid `r<=32` adapter submission, (3) tight iteration loops over anything long-horizon.
- **Open Questions:** What exact **base model ID** does Kaggle evaluation load (4B vs 30B-A3B, exact HF slug, revision/sha, `trust_remote_code` expectations)? Are the adapter rules truly **LoRA-only** and is **rank<=32** confirmed (and are there limits on target modules, dtype/quantization, file naming/structure)? What is the **output parsing/scoring contract** (answer-only vs allowing reasoning, exact match vs normalized, whitespace/case rules, category-specific canonicalization)? What is the definitive **metric definition** (micro overall accuracy vs macro by category, any partial credit)? What are the exact **deadlines with timezones** and submission limits on Kaggle (per-day caps, final submission selection rules)?

### Agent B — Architecture Contracts

## Agent B — Architecture Contracts

- Findings: `ARCHITECTURE.md` defines an exact-match-first normalization posture, but `plan_v0.2.md` hardcodes `\\boxed{}` as the expected format and verification mechanism (smoke test + regex extractor + RL reward). This conflicts with the notebook scaffolds’ “exact-match over math formatting” stance.
- Findings: `ReasoningExample` contract (`prompt`/`answer`/`id`/`source`/`split`/`metadata`) is not matched by plan examples, which use `problem["question"]` + `problem["expected_answer"]` in evaluation and a separate synthetic schema with `input`/`output`/`reasoning`. There’s no defined mapping layer between these shapes.
- Findings: `EvalRecord` exists in architecture, but the plan’s evaluation protocol produces only aggregate stats (mean/std) and uses fields (`question`, `expected_answer`) that don’t align with `EvalRecord` (`gold`, `normalized_prediction`, `prompt_template_id`, etc.). The on-disk eval artifact format/location is not defined.
- Findings: `EvalRecord` lacks an explicit linkage key back to `ReasoningExample` (e.g., `example_id`) and lacks a place to capture decode/thinking settings; this is a gap for `#22` failure slicing and for the plan’s decode/prompt sweeps, where you need per-example drilldowns and config attribution.
- Findings: `PackageManifest` is defined in architecture, but `plan_v0.2.md` never references generating a manifest (or an adapter config card), even though the sprint gate says the “package manifest” must be stable in Wave B and the packaging notebook explicitly targets provenance metadata.
- Findings: Architecture references teacher outputs (`teacher_answer`, optional `teacher_reasoning`) but provides no concrete contract for “teacher trace” / “SFT training sample”; meanwhile the plan’s synthetic format is chat-template-centric (`input`/`output`) and will likely diverge from `ReasoningExample` unless formalized.

- Impact: Different parts of the pipeline will serialize “the same thing” with different keys (`prompt` vs `question`, `answer` vs `expected_answer`, `prompt`/`answer` vs `input`/`output`), increasing glue code, silent bugs, and irreproducible evaluations.
- Impact: If Kaggle evaluation is exact-match (as ARCHITECTURE/NOTEBOOKS/SPRINTS are prepared for), the plan’s boxed-answer normalization can mis-score and push the project to optimize the wrong target.
- Impact: Without record-level `EvalRecord` artifacts linked to example IDs + run config, failure slicing (`#22`) and regression analysis will be brittle, and “what changed?” questions will be hard to answer.
- Impact: Packaging/provenance is at risk of becoming ad-hoc (weights-only), which undermines the sprint gate (“manifest stable”) and slows review/debug of submissions.

- Recommendations: Pick one canonical “raw example” schema (keep `ReasoningExample`), and update plan-facing examples to either use it directly or explicitly document a mapping (`question`→`prompt`, `expected_answer`→`answer/gold`, add `id/source/split/metadata`).
- Recommendations: Split contracts into (1) `ReasoningExample` (raw, model-agnostic) and (2) a chat/SFT contract (e.g., `SFTExample` with `messages` + `completion`), both carrying `example_id`, `category`, and provenance fields; treat the plan’s synthetic JSON as the starting point for `SFTExample`.
- Recommendations: Extend `EvalRecord` (or add an `EvalRunConfig` artifact referenced by `run_id`) to capture: `example_id`, `split`, `normalizer_id`, `enable_thinking`, `temperature/top_p/max_new_tokens/seed`, and `prompt_template_id` provenance, so prompt/decode sweeps are attributable.
- Recommendations: Make `PackageManifest` concrete in the plan: require it in Phase 8 and in notebook `#20`, define required file paths (weights, adapter config, eval summary, contracts version), and clarify what `eval_sha` hashes (eval code commit vs eval records content hash).
- Recommendations: Align normalization guidance: treat `\\boxed{}` extraction as a category-specific normalizer behind a switch, not the default global contract, until `#14` freezes the true scoring format.

- Open Questions: What is the competition’s authoritative answer-matching rule (exact string match, whitespace/case normalization, category-specific parsing, boxed math, etc.) that `#14` is supposed to freeze?
- Open Questions: Should `ReasoningExample.prompt` be the raw “question” text, or the fully rendered chat prompt (post `apply_chat_template`)? If raw, where is the rendered prompt stored and versioned (`prompt_template_id`)?
- Open Questions: What exactly should `eval_sha` represent in `PackageManifest`: git commit of eval code, hash of the eval dataset, hash of the produced `EvalRecord` file(s), or a composite?
- Open Questions: Do we need a first-class “Trajectory/Trace” artifact contract (for `#22` and synthetic teacher traces) distinct from `EvalRecord`, to store `<think>` text, intermediate steps, and verifier metadata?

### Agent C — Dependencies & Repro

## Agent C — Dependencies & Repro

- Findings: `requirements.txt` matches the plan’s HF/TRL core but **omits `unsloth` and `vllm`** explicitly listed in `docs/planning/plan_v0.2.md` Phase 0.2; the plan’s **NeMo Curator/NeMo** path (e.g., Phase 3.2 / Phase 4.3) has **no corresponding dependencies** (e.g., `nemo-toolkit` / curator tooling) and `configs/` is currently a placeholder; only **`torch==2.4.0` is pinned** while most of the stack is `>=` or unpinned (plus torch’s **CUDA wheel selection is implicit**, so “same version” is not actually reproducible across machines); notebook scaffolds hard-assume **Python 3.11** (`scripts/scaffold_notebooks.py` sets notebook metadata to 3.11) but there’s no repo-level Python pin/lock; `scripts/kaggle.sh` hardcodes **`.venv/bin/kaggle`** and sources a repo-local **`.env`**, which is a local-dev assumption that won’t hold on Kaggle/Colab/HPC by default; `.gitignore` ignores **`data/**`**, which would also ignore the plan’s `data/eval/validation_200.jsonl` and `data/eval/golden_20.jsonl` if you follow the plan literally.
- Impact: Plan phases relying on Unsloth/vLLM/NeMo are **not runnable from a clean checkout** without manual installs and compatibility debugging; “same repo” can still produce **CPU-only torch or differing CUDA builds** across WSL2/Colab/HPC, creating irreproducible OOM/perf/behavior changes; hosted runtimes (Kaggle/Colab) will likely break on helper tooling (`scripts/kaggle.sh`) unless they mirror the local `.venv` layout; reproducibility artifacts the plan calls for (frozen configs + tracked eval splits + provenance) are easy to lose because key small datasets/configs aren’t currently check-in friendly.
- Recommendations: Define a tested **environment matrix** (Local-WSL2, Colab, HPC, Kaggle-notebook) with a known-good Python+CUDA+torch combo per target; adopt a lock strategy (fully pin `transformers/datasets/accelerate/peft/trl/bitsandbytes/wandb/kaggle` to a tested set and generate a lockfile with hashes) and treat torch as **“version + CUDA build”** rather than just `2.4.0`; split heavy/fragile stacks into optional extras or separate requirement sets (`unsloth`, `vllm`, NeMo/Curator) and explicitly gate plan phases on them; make helper scripts environment-agnostic (prefer `kaggle` from PATH or `python -m kaggle`, avoid requiring a repo-local `.env` in hosted runs); allow tracking small reproducibility-critical artifacts (golden/validation sets + minimal config YAMLs) somewhere not excluded by `.gitignore`, even if large raw datasets remain ignored.
- Open Questions: Which environment is the “source of truth” for compatibility (HPC vs Colab vs local WSL2)? Will you actually run the **NeMo Curator/NeMo** path (if yes, which packages/versions), or treat it as reference-only? Is **Unsloth** required for Phase 4, and does it support the pinned `torch==2.4.0` for this model? Will you rely on **vLLM** for hybrid Mamba-2 inference or standardize on HF `generate()`? What Python version must be supported end-to-end (3.10 vs 3.11) given Kaggle/Colab/HPC constraints? Does your intended `kaggle` version/auth flow actually support `KAGGLE_API_TOKEN`, or should you standardize on `KAGGLE_USERNAME/KAGGLE_KEY` + `kaggle.json`?

### Agent D — External Baselines (konbu17)

## Agent D — External Baselines (konbu17)

**Findings**
- Base model is pulled via KaggleHub as `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` and loaded with `torch_dtype=torch.bfloat16`, `trust_remote_code=True`, `attn_implementation="eager"`; tokenizer `pad_token` is set to `eos_token` if missing. (`data/external/konbu17/cells/cell10.py`)
- LoRA config is explicitly `r=32`, `lora_alpha=32`, `lora_dropout=0.0`, `bias="none"`, `task_type=CAUSAL_LM`, targeting 9 modules: `["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj","lm_head"]`. (`data/external/konbu17/cells/cell11.py`)
- Training recipe (Kaggle mode) uses TRL `SFTTrainer` with `num_train_epochs=1`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=64` (effective batch 64), `learning_rate=2e-4` (comment notes `2.4e-4`), `lr_scheduler_type="linear"`, `warmup_ratio=0.0`, `weight_decay=0.0`, `bf16=True`, `max_length=4096` (comment: “should be 8192 but OOM”), `gradient_checkpointing=True` with `use_reentrant=False`, and effectively no grad clipping (`max_grad_norm=1e9`). (`data/external/konbu17/cells/cell13.py`)
- Prompt/formatting is “Tong-style CoT”: user prompt appends `Please put your final answer inside \\boxed{}`; assistant text is forced to include a `</think>` boundary and a final `\\boxed{answer}`; formatting uses `tokenizer.apply_chat_template(..., enable_thinking=True)`. (`data/external/konbu17/cells/cell13.py`)
- Masking strategy is not explicitly configured (no custom data collator / completion-only masking shown); behavior depends on TRL defaults for the installed version. (`data/external/konbu17/cells/cell13.py`, `cell08.py`)
- Data strategy includes hard-coded per-category downsampling ratios and an optional “priority upweighting” that duplicates “hard” examples based on a precomputed priority list; there’s also commented code for generating that list via vLLM `prompt_logprobs` min-logprob mining (`CHUNK_SIZE=500`, “TP=4”). (`data/external/konbu17/cells/cell13.py`, `cell18.py`, `cell02.py`)
- Submission packaging creates `/kaggle/working/submission.zip` containing exactly `adapter_config.json` and `adapter_model.safetensors` at the zip root; it can package either freshly trained `/kaggle/working/sft_adapter` or a pretrained adapter dataset (default path set to `/kaggle/input/datasets/konbu17/exp026-s012-lora`). (`data/external/konbu17/cells/cell04.py`, `cell15.py`, `cell17.py`)
- Kaggle/Blackwell environment hacks exist for training mode: installs Triton from a wheel under `/kaggle/input`, sets `TRITON_PTXAS_PATH` to `ptxas-blackwell`, and installs `mamba_ssm` / `causal_conv1d` from wheels. (`data/external/konbu17/cell07.py`, `cells/cell06.py`, `cells/cell07.py`, `cells/cell09.py`)

**Impact**
- `plan_v0.2.md` states the base is **Nemotron-3-Nano-4B**, but this baseline is built around **Nemotron-3-Nano-30B-A3B-BF16** (and `COMPETITION.md` already flags base-model uncertainty); if Kaggle eval truly uses 30B, our plan’s “4B-first” assumptions will mis-spec module names, memory budgets, and training feasibility.
- `plan_v0.2.md` recommends `r=64` / `alpha=128`, which likely violates `COMPETITION.md`’s “LoRA rank <= 32” constraint; konbu17’s `r=32` aligns with the suspected cap.
- `ARCHITECTURE.md` says “avoid `\\boxed{}`” and “always mask prompt/user tokens”; konbu17 explicitly trains a `\\boxed{}` contract and does not show prompt masking configuration, so our docs and this baseline disagree on two high-leverage training/eval contracts.
- Our docs are currently non-specific on Kaggle packaging; konbu17 provides a concrete, minimal `submission.zip` shape that we should mirror for compatibility (and keep extra metadata manifests out-of-band if Kaggle is strict).

**Recommendations**
- Update our “source of truth” assumptions to treat `metric/nemotron-3-nano-30b-a3b-bf16` (or its HF equivalent) as the *most likely* Kaggle base model until proven otherwise, and make `plan_v0.2.md` stop hard-claiming “4B” in the intro.
- Align Phase 4 LoRA defaults to competition-likely constraints: start with `r=32`; consider adopting the 9-module target list (including `lm_head`) as a baseline “known-good” fallback while still implementing programmatic module discovery as `ARCHITECTURE.md` intends.
- Decide the output contract explicitly: either commit to exact-match (no `\\boxed{}`) per `ARCHITECTURE.md`, or adopt a `\\boxed{}`-style training target; don’t leave plan/architecture in conflict. If exact-match is required, treat konbu17’s `\\boxed{}` suffix as a cautionary anti-pattern to avoid.
- Mirror konbu17’s submission artifact exactly (`adapter_config.json` + `adapter_model.safetensors` in `submission.zip` root) and keep any manifest/logs outside the submitted zip.
- Consider adopting the “priority upweighting” idea: compute hardness via min completion logprob on training data and duplicate/weight those examples; it’s a concrete, low-engineering approach to curriculum that fits our planned ablations.

**Open Questions**
- Which base model does Kaggle evaluation actually merge against (30B-A3B vs 4B), and what is the exact canonical model ID on Kaggle/HF?
- Does Kaggle scoring expect an exact raw string answer (likely) or tolerate/ignore formatting like `\\boxed{}`? (This determines whether konbu17’s output contract is helpful or harmful.)
- What TRL version is assumed in the baseline, and does its `SFTTrainer` default to full-sequence loss vs completion-only loss for chat templates (i.e., are prompt/user tokens contributing to loss here)?
- Are LoRA targets like `lm_head` allowed/beneficial under competition size/perf constraints, or should we restrict to attention/MLP/Mamba projections only?
- Are `attn_implementation="eager"` and the Triton/ptxas-blackwell setup necessary in the Kaggle runtime for inference as well, or only for training kernels?

## Sprint Plan (Implementation-Planning Only, Pre-Notebook)

**Scope:** produce decision-complete implementation plans + freeze/gate constraints **before** any notebook implementation.

### Timeline (2026-04-20 → 2026-05-08)

| Date | Gate | Exit criteria |
|---|---|---|
| 2026-04-20 | Kickoff | Subagents dispatched; plan doc skeleton paths reserved; GitHub issues synced/created |
| 2026-04-22 | Gate A0 — Kaggle access | Either (a) rules accepted + API downloads work, or (b) explicit “BLOCKED” checklist written with manual steps |
| 2026-04-24 | Gate A — Wave A plans ready | Issues `#14-#16` planning complete; constraints freeze plan includes stop-conditions; external baseline delta plan names exact files + acceptance tests |
| 2026-04-30 | Gate B — Wave B plans ready | Issues `#17-#20` plans are decision-complete and reference frozen/gated contracts |
| 2026-05-02 | Gate HPC — queue runbook drafted | SLURM runbook plan exists with placeholders for base-model/rank constraints if still blocked |
| 2026-05-08 | PM cutoff | Constraints frozen **or** explicitly blocked; Wave A/B plan docs linked from this doc + GitHub; MVP execution path planned end-to-end (ingest → eval → package → dummy adapter submission) |

### Wave A/B plan doc index (created 2026-04-20)

| Issue | Wave | Plan doc | Status |
|---|---|---|---|
| `#14` | A | `docs/execution/plans/issue-14-constraints-freeze.md` | ready (gated by Kaggle rules capture) |
| `#15` | A | `docs/execution/plans/issue-15-review-harness.md` | ready |
| `#16` | A | `docs/execution/plans/issue-16-external-baselines-delta.md` | ready (gated by `#14`) |
| `#17` | B | `docs/execution/plans/issue-17-schema-and-eda.md` | ready |
| `#18` | B | `docs/execution/plans/issue-18-validation-and-golden-set.md` | ready |
| `#19` | B | `docs/execution/plans/issue-19-baseline-eval-and-normalization.md` | ready |
| `#20` | B | `docs/execution/plans/issue-20-submission-packaging-and-provenance.md` | ready |

### HPC plan doc index

| Issue | Plan doc | Status |
|---|---|---|
| `#25` | `docs/execution/plans/issue-25-hpc-queue-runbook.md` | ready (hyperparams gated by `#14`) |

### MVP execution path (planned end-to-end; pre-notebook)

This is the minimal runnable path we will implement *after* the constraints gate clears:

1. **Ingest:** `#17` produces canonical `ReasoningExample` rows from `train.csv` (or a mirror), plus a dataset version string.
2. **Reserve eval sets:** `#18` produces `data/eval/validation_*.jsonl` and immutable `data/eval/golden_*.jsonl`.
3. **Baseline eval:** `#19` produces record-level `EvalRecord` artifacts + per-category metrics with a versioned normalizer.
4. **Dummy / first adapter:** `#25` defines a minimal SFT smoke job that emits a PEFT adapter directory (checkpoint → adapter export).
5. **Package:** `#20` turns the adapter directory into a Kaggle-safe `submission.zip` + out-of-band `submission.manifest.json`.

Stop condition: do not promote or submit any adapter unless `#18` golden gate passes and `#20` zip validation passes.

### Manual human actions (expected blockers)

If Kaggle API downloads are `403 Forbidden`:

1. Open the competition page in a browser and accept the rules (Kaggle UI).
2. Confirm “Joined” status on the competition page.
3. Re-run `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw`.
4. Copy/paste (or screenshot) the **Rules** tab sections that specify:
   - base model / allowed models
   - adapter constraints (LoRA rank cap, file limits)
   - answer parsing / scoring rules
   - submission file requirements (zip layout)

### Subagent dispatch (2026-04-20)

Each subagent must:
1) append their report into this doc under their section below, and  
2) create/update the plan doc(s) they own under `docs/execution/plans/`.

#### Agent 1 — Constraints Freeze

Owned plan doc: `docs/execution/plans/issue-14-constraints-freeze.md`

<!-- AGENT_1_REPORT -->
## Agent 1 — Constraints Freeze

- **Findings:** The repo does **not** contain an authoritative Kaggle rules snapshot, so the exact base model ID and scoring contract remain blocked. `docs/architecture/COMPETITION.md` explicitly marks the base model unresolved, `docs/planning/plan_v0.2.md` hard-claims `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, and `data/external/konbu17/cells/cell10.py` loads `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`; that conflict is why the base-model row is BLOCKED. On the other hand, the repo does provide a strong provisional adapter signal: `docs/architecture/COMPETITION.md` says LoRA-only and `rank <= 32`, and `data/external/konbu17/cells/cell11.py` uses `r=32`, `lora_alpha=32`, `bias="none"`, and the 9-module target set. Packaging is also concrete in the baseline: `data/external/konbu17/cells/cell17.py` writes `submission.zip` with only `adapter_config.json` and `adapter_model.safetensors` at the root. For scoring, the repo is still split between `\\boxed{}`-centric plan text and exact-match-first architecture/review notes, so the normalization contract remains BLOCKED pending Kaggle rules text.
- **Impact:** Downstream notebooks must not hard-code the 4B claim from `plan_v0.2.md` or the boxed-answer workflow as if it were authoritative. The safest frozen defaults for implementation are: keep the base model and normalization contract behind a gate, cap LoRA at `r<=32`, use the konbu17 9-module target list as the provisional baseline, and mirror the minimal `submission.zip` layout. Anything more specific than that requires a human to capture the Kaggle rules or submission demo text.
- **Report:** `docs/execution/plans/issue-14-constraints-freeze.md` now records a constraint table with frozen vs BLOCKED rows, explicit blockers, downstream file paths, interfaces/contracts referenced, verification commands, and an issue-review-harness-aligned acceptance checklist. The doc deliberately avoids notebook code and limits itself to decision freezing. The remaining manual action is to copy the Kaggle rules / demo notebook text into the repo and resolve the currently BLOCKED rows.
- **Open Questions:** What is the exact Kaggle base model slug and revision? Does Kaggle scoring ignore reasoning text, strip formatting, or require plain exact-match answers? Are the LoRA target modules or adapter dtype further restricted beyond the provisional `r<=32` cap? Is the two-file `submission.zip` layout the final requirement, or do rules require an extra manifest or metadata file?

#### Agent 2 — Schema + Contracts

Owned plan doc: `docs/execution/plans/issue-17-schema-and-eda.md`

<!-- AGENT_2_REPORT -->
## Agent 2 — Schema + Contracts

- **Findings:** `ReasoningExample` is already the canonical raw contract in `ARCHITECTURE.md`, but the plan still leaks alternate names (`question`/`expected_answer`) and the synthetic path uses `input`/`output`/`reasoning`; `EvalRecord` exists, but it is not yet example-linked or config-attributed; `PackageManifest` exists, but the plan does not yet treat it as a first-class provenance artifact; and `SFTExample` is still missing as a named, persistent contract.
- **Decisions:** Keep `ReasoningExample` raw and model-agnostic with `id`, `category`, `prompt`, `answer`, `source`, `split`, and `metadata`; define `SFTExample` as the only training contract with `example_id`, `category`, `messages`, `completion`, `source`, `split`, and `provenance`; extend `EvalRecord` with `example_id`, `split`, `normalizer_id`, and `decode_config` so every result is traceable; and version `PackageManifest` with `manifest_version` so exports cannot drift silently.
- **Mapping rules:** Raw CSV rows map directly into `ReasoningExample` (`id`/`prompt`/`answer`) while extra columns stay in `metadata`; `question`/`expected_answer` are aliases that must collapse to `prompt`/`answer` at ingest time; `ReasoningExample` then renders into `SFTExample.messages` plus `completion`, with answer-only completions as the default and reasoning traces allowed only when a teacher trace is explicitly present; eval inputs come from `ReasoningExample.prompt` plus a versioned prompt template, with gold answers copied into `EvalRecord.gold` and predictions normalized through a named normalizer before scoring.
- **Minimal contract edits:** Add the `example_id` link and config attribution to `EvalRecord`; add a `manifest_version` and explicit `artifact_paths` to `PackageManifest`; and make the prompt-template boundary explicit so rendered chat prompts never replace the raw `ReasoningExample.prompt`. The plan doc should also list the future schema files (`src/contracts.py`, `src/data/schema_mapping.py`, `src/data/schema_eda.py`, `src/evaluation/records.py`, `src/packaging/manifest.py`) so implementation stays focused.
- **Open questions:** The only remaining ambiguity is how narrow the final normalizer has to be once the competition scoring rules are frozen. If Kaggle introduces category-specific parsing, that should stay behind `normalizer_id` rather than changing the core schema again.

#### Agent 3 — Eval Harness Plan

Owned plan docs:
- `docs/execution/plans/issue-18-validation-and-golden-set.md`
- `docs/execution/plans/issue-19-baseline-eval-and-normalization.md`

<!-- AGENT_3_REPORT -->
Completed the eval-harness planning pass for issues `#18` and `#19`. The two plans now close the main drift gaps called out in the review matrix: `#18` freezes the validation/golden split contract, defines immutable golden regression behavior, and requires versioned provenance on the reserved eval rows; `#19` defines the full ingest → predict → normalize → score → report pipeline, with record-level artifacts, run-config attribution, and versioned normalization/parsing.

Key choices recorded:
- Validation and golden artifacts are row-level JSONL with `example_id`, `category`, `split`, `dataset_version`, and selection provenance.
- Golden regression is strict: any single golden miss blocks promotion, regardless of aggregate validation gains.
- Baseline eval artifacts must capture `run_id`, `model_id`, `prompt_template_id`, `normalizer_id`, `seed`, `latency_ms`, `tokens_in`, and `tokens_out`.
- Normalization is versioned and swappable so scoring drift becomes an explicit contract change instead of a hidden code change.

The plans also include the required harness metadata fields, issue-specific acceptance criteria, and test cases that would catch normalization/parsing drift before it contaminates leaderboard comparisons. Sources used were limited to the repo docs named in the task, especially `docs/architecture/ARCHITECTURE.md`, `docs/architecture/COMPETITION.md`, `docs/execution/ISSUE_REVIEW_HARNESS.md`, `docs/planning/plan_v0.2.md`, and the existing review-plan matrix.

#### Agent 4 — Submission Packaging Plan

Owned plan doc: `docs/execution/plans/issue-20-submission-packaging-and-provenance.md`

<!-- AGENT_4_REPORT -->
- The submission archive is now decision-complete: `submission.zip` must contain only `adapter_config.json` and `adapter_model.safetensors` at the zip root, matching the konbu17 packaging logic in `data/external/konbu17/cells/cell04.py` and the required-file checks in `cell15.py`.
- Provenance stays out-of-band in `submission.manifest.json` beside the zip, not inside it; the manifest records `base_model_id`, `adapter_rank`, `dataset_version`, `eval_sha`, `artifact_paths`, `created_at`, git SHA, and SHA256 hashes for the adapter files and zip.
- Acceptance tests are frozen as: exact zip-root validation, archive-integrity validation, and adapter reload smoke tests from extracted contents with PEFT; the load test may skip only when the competition base model is not cached locally.
- I saved the implementation plan at `docs/execution/plans/issue-20-submission-packaging-and-provenance.md` and kept it aligned with the architecture contract while resolving the packaging-vs-provenance split.

#### Agent 5 — HPC Queue/Runbook Plan

Owned plan doc (new): `docs/execution/plans/issue-25-hpc-queue-runbook.md`

<!-- AGENT_5_REPORT -->
## Agent 5 — HPC Queue / Runbook Plan

- **Findings:** The repo already treats HPC as the long-run training target for Phase 4 (SFT) and Phase 6 (GRPO), but the operational path is still implicit: `plan_v0.2.md` only gives high-level guidance on checkpoint cadence, early stopping, and where to save outputs. The new runbook plan should make SLURM the explicit orchestration layer and keep prep/tokenize/train/eval/package as separate job types so each stage can be retried independently.
- **Findings:** The runbook must stay behind the freeze gates from `#14` and `ARCHITECTURE.md`: do not finalize any SFT or RL hyperparameters until the base model ID/revision and LoRA constraints are frozen. That means `r`, `alpha`, target modules, max sequence length, and decode settings stay provisional until the contract is locked.
- **Findings:** Artifact handling needs to be first-class, not ad hoc. `ARCHITECTURE.md` already defines `EvalRecord` and `PackageManifest`, so the HPC flow should store checkpoints, eval outputs, and provenance outside git, with a clear promotion path from checkpoint root → best adapter → Kaggle-safe package. The submitted zip should stay minimal, while manifests and logs live alongside it out of band.
- **Findings:** The queue plan should include a strict resume story: keep last 3 checkpoints plus best, write atomic checkpoint directories, and only resume from checkpoints that already passed the most recent golden/validation gate. Early stopping should fire on validation stagnation, golden regressions, NaNs, repeated OOM, or broken tokenizer/model reload.
- **Findings:** Optional RL belongs behind the SFT gate, not beside it. The plan should treat RL as a second queue path that reuses the same artifact conventions but cannot start until the SFT baseline is frozen and the RL reward / eval contract is stable.

### Job types

| Job type | Purpose | Resource range | Notes |
|---|---|---|---|
| `prep` | Validate env vars, paths, git SHA, and frozen contracts | 1 CPU node, 2-4 cores, 8-16 GB RAM | fast fail only |
| `tokenize` | Build cached tokenized shards and dataset fingerprints | 1 CPU node, 4-16 cores, 16-64 GB RAM | one-time per dataset version |
| `train` (SFT) | Run supervised fine-tuning with checkpoints and eval hooks | 1 GPU node, 1-4 GPUs, 64-128 GB RAM | primary HPC workload |
| `train` (RL, optional) | Run RL only after SFT passes all gates | 1-4 GPU nodes, 64-256 GB RAM | skip unless SFT is green |
| `eval` | Run golden/validation regression checks on candidates | 1 GPU node or cached CPU eval, 16-64 GB RAM | gate promotion |
| `package` | Produce Kaggle-safe zip plus manifest/checksums | 1 CPU node, 2-4 cores, 4-8 GB RAM | final handoff only |

### Resource estimates

| Stage | Range | Assumptions |
|---|---:|---|
| `prep` | 5-15 min | config validation only |
| `tokenize` | 15-90 min | cached tokenizer, CPU parallelism |
| SFT smoke | 0.5-2 GPU-hours | small subset, 100-500 steps |
| SFT production | 8-48 GPU-hours | `r<=32`, checkpointing on, validation every few hundred steps |
| Optional RL | 24-96 GPU-hours | smaller sweep count, stable reward contract |
| `eval` | 30 min-4 h | golden set + validation slice |
| `package` | 5-20 min | copy/hash/zip/manifest only |

### Exact scripts / commands to create later

- `scripts/hpc/preflight.sh`
- `scripts/hpc/tokenize_dataset.py`
- `scripts/hpc/submit_sft.sbatch`
- `scripts/hpc/submit_rl.sbatch`
- `scripts/hpc/checkpoint_policy.py`
- `scripts/hpc/regression_gate.py`
- `scripts/hpc/package_adapter.py`
- `scripts/hpc/resume_from_latest.py`

Planned invocation shapes:

```bash
sbatch --job-name=nemotron-sft \
  --export=ALL,RUN_TAG=${RUN_TAG},BASE_MODEL_ID=${BASE_MODEL_ID},LORA_RANK=${LORA_RANK} \
  scripts/hpc/submit_sft.sbatch
```

```bash
python scripts/hpc/regression_gate.py \
  --checkpoint "${CHECKPOINT_ROOT}/${RUN_TAG}/best" \
  --golden "${GOLDEN_SET_PATH}" \
  --validation "${VALIDATION_SET_PATH}" \
  --normalizer "${NORMALIZER_ID}"
```

### Required env vars + storage paths

| Category | Vars / paths |
|---|---|
| Auth | `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| SLURM | `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`, `SLURM_TIME_LIMIT`, `SLURM_MEM_GB`, `SLURM_GPUS`, `SLURM_CPUS_PER_TASK` |
| Frozen contract | `BASE_MODEL_ID`, `BASE_MODEL_REVISION`, `TOKENIZER_ID`, `ENABLE_THINKING`, `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`, `MAX_SEQ_LEN`, `MAX_NEW_TOKENS`, `SEED` |
| Roots | `DATA_ROOT`, `RUN_ROOT`, `CHECKPOINT_ROOT`, `ARTIFACT_ROOT`, `LOG_ROOT`, `HF_HOME`, `TRANSFORMERS_CACHE`, `WANDB_DIR`, `TMPDIR` |
| Eval | `NORMALIZER_ID`, `GOLDEN_SET_PATH`, `VALIDATION_SET_PATH`, `RUN_TAG` |

Storage convention: raw data stays under `${DATA_ROOT}/raw/`, tokenized caches under `${DATA_ROOT}/processed/`, checkpoints under `${RUN_ROOT}/${RUN_TAG}/checkpoints/`, eval artifacts under `${RUN_ROOT}/${RUN_TAG}/eval/`, and the final adapter mirror under `adapters/issue-25/${RUN_TAG}/`.

### Explicit gates

1. **Freeze gate:** no final hyperparameter selection until base model and LoRA constraints are frozen.
2. **Smoke gate:** tiny SFT run must load, checkpoint, and reload cleanly.
3. **Golden gate:** checkpoint promotion stops if the golden set regresses.
4. **Validation gate:** keep only checkpoints within the frozen tolerance band.
5. **Packaging gate:** final zip must match the minimal Kaggle layout and keep provenance out of the bundle.

### Report summary

This plan turns HPC into a repeatable queue workflow instead of a loose notebook dependency chain. It gives SFT the first-class path, keeps RL optional and gated, and defines where checkpoints, eval outputs, and submission artifacts live so the project can resume cleanly after interruptions without losing provenance.
