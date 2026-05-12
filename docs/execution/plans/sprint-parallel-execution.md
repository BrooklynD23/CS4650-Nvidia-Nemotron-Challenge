# Sprint: Parallel Execution — Implementation Plan

**Branch:** `sprint/parallel-execution` (off `main`)
**Deadline:** 2026-06-15 (Kaggle final submission)
**Created:** 2026-05-07
**Status of repo at plan time:** 248 tests passing, all infrastructure built, **zero real training artifacts**.

Touches issues: #25 (HPC training), #21 (prompt sweeps), #18 (splits), #20 (packaging), #24 (synthetic data smoke).
Coordinates with: in-flight teammate PR for #19 (baseline eval).

---

## Context

The codebase now has the full SFT launcher path implemented end-to-end
(`submit_sft.sbatch` → `run_sft.py` → checkpoint rotation). The remaining work
in this sprint is validation and docs cleanup, not replacing a stub.

This sprint runs two tracks in parallel using the available compute pool (RTX 3080 + HPC + Kaggle + Colab Pro) so that one human can drive both training pipeline validation (Track 1) and data/eval/notebook completion (Track 2) without serializing on shared GPU access. A teammate is independently driving #19 (baseline eval) on a separate PR; we run our own #21 (prompt sweeps) so we can compare and keep the best config.

Outcome at end of sprint: a real `submission.zip` containing a real LoRA adapter trained on real data, a #21 sweep findings doc, and notebooks 02/03/05/09 in a `validated` state.

---

## Pre-Work (do before any track starts)

1. **Capture design choices from stale orchestrator prompts, then discard the line-ending drift.**
   The 6 modified files under `.claude/orchestrator/workers/feature-*.prompt` are pure CRLF↔LF noise on top of prompts describing already-merged Wave D work (synthetic data + HPC scripts, commits `94db71f` and `801e309`). Spawn a Haiku 4.5 subagent to read each prompt's "## Your Task" block and write a single ~120-line retrospective at `docs/learn/project/wave-d-design-choices.md` capturing: lora_r=32 rationale, target_modules choice, `apply_loss_mask` API contract, keep-last-3-plus-best policy, QualityFilter solver-confidence gate, smoke=50/cost-cap=$20 limits. Then `git checkout .claude/orchestrator/workers/feature-*.prompt` to discard the line-ending diff.

2. **Create branch and push.**
   ```bash
   git checkout -b sprint/parallel-execution
   git push -u origin sprint/parallel-execution
   ```
   Coordinate with the in-flight #19 PR by rebasing onto `main` once it lands (no shared file conflicts expected — #19 touches `data/eval/runs/` and a notebook; this sprint touches `scripts/hpc/`, training configs, and other notebooks).

---

## Track 1 — Training Pipeline (Local RTX 3080 → HPC)

**Owner:** training. Sequential within track, blocks on T1.3 before HPC submission.

### T1.1 — Validate `scripts/hpc/run_sft.py` (implemented)

The training launcher now:
- Parses `--config <yaml>` (resolves `extends:` chain so `smoke_sft.yaml` inherits from `lora_baseline.yaml`).
- Loads base model + tokenizer per config (`trust_remote_code=True`, `dtype=bfloat16`, `device_map="auto"`); honors `load_in_4bit` for the QLoRA path.
- Wraps with `peft.LoraConfig` using `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules` from config.
- Loads pre-tokenized `.pt` shards from `--data-dir` (output of `scripts/hpc/tokenize_dataset.py`). **These shards already have labels masked** by `apply_loss_mask` during tokenization, so `run_sft.py` does NOT re-mask. The collator now pads from `tokenizer.pad_token_id`.
- Instantiates `trl.SFTTrainer` with `TrainingArguments` derived from config (`max_steps`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `save_steps`, `output_dir=CHECKPOINT_DIR`, `bf16=True`, `report_to="none"` unless `WANDB_API_KEY` is set).
- Reads `--resume-from-checkpoint <path>` from CLI and forwards to `trainer.train(resume_from_checkpoint=…)`.
- Validates config before model load so bad config fails fast before any weight loading starts.
- Applies checkpoint rotation once at the end of training; the SLURM wrapper does not call the policy a second time.
- Exit 0 on success, 1 on any unhandled exception (with traceback to stderr).

**Reuses (do not duplicate):**
- `src/training/sft_trainer.py::apply_loss_mask` — already correct; called only inside `tokenize_dataset.py`.
- `scripts/hpc/checkpoint_policy.py` — already writes `trainer_state.json`, `run_config.json`, `metrics.jsonl`, `git_sha.txt`, `dataset_fingerprint.txt` and is invoked once by `run_sft.py`.
- `src/contracts.py::SFTExample` — input row schema, frozen.

### T1.2 — Wire `submit_sft.sbatch` to call `run_sft.py`

Edit `scripts/hpc/submit_sft.sbatch` lines 122-162. Replace the entire `python - <<'PYEOF' ... PYEOF` heredoc block with:

```bash
RESUME_ARGS=()
if LATEST=$(python scripts/hpc/resume_from_latest.py --checkpoint-dir "$CHECKPOINT_DIR" 2>/dev/null); then
    [ -n "$LATEST" ] && RESUME_ARGS=(--resume-from-checkpoint "$LATEST")
fi

python scripts/hpc/run_sft.py \
    --config "$CONFIG_PATH" \
    --data-dir "$TOKENIZED_DIR" \
    --output-dir "$CHECKPOINT_DIR" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$RUN_DIR/train.log"
```

Keep the surrounding env-var resolution, `PYTHONPATH` export, log header, and
exit-code handling. `run_sft.py` now validates config before model load and
applies checkpoint rotation once at the end of training, so the wrapper does
not need a second checkpoint-policy call. Update the script's top comment block
to remove "stub" language.

### T1.3 — Local smoke validation on RTX 3080 with a stand-in model

Goal: prove the pipeline plumbing (tokenize → train → checkpoint → package) end-to-end before burning HPC queue time. The frozen 30B base model will not fit on a 3080; smoke uses a tiny stand-in.

**New file: `configs/smoke_sft_local.yaml`** — extends `configs/smoke_sft.yaml` but overrides:
- `base_model: Qwen/Qwen2.5-0.5B-Instruct` (~1 GB in bf16, fits comfortably).
- `lora_r: 8`, `lora_alpha: 16`.
- `target_modules: [q_proj, k_proj, v_proj, o_proj]` (Qwen-compatible).
- `dataset_max_rows: 100`, `max_steps: 20`, `save_steps: 10`.
- Header comment: "LOCAL SMOKE ONLY — NOT FOR SUBMISSION. Validates pipeline; uses non-frozen base model."

`run_sft.py` validates config before model load, so smoke failures at launch
are config issues rather than weight-loading surprises. The collator now pads
from `tokenizer.pad_token_id`; the smoke tokenizer must define a pad token.

**Edit `scripts/hpc/preflight.sh`** — add a `--local` flag that skips the frozen `BASE_MODEL_ID` equality check and the model-weights-path check (still runs CUDA, disk space, package importability, dataset SHA256). Default behavior unchanged for HPC use. The local path also supplies defaults for `LORA_RANK`, `LORA_TARGET_MODULES`, and `NORMALIZER_ID`.

**Smoke run sequence (verifies everything except real-model VRAM):**
```bash
bash scripts/hpc/preflight.sh --local
python scripts/hpc/tokenize_dataset.py --input data/processed/synthetic_train.jsonl --output /tmp/smoke_tok --config configs/smoke_sft_local.yaml
python scripts/hpc/run_sft.py --config configs/smoke_sft_local.yaml --data-dir /tmp/smoke_tok --output-dir /tmp/smoke_run/checkpoints
python scripts/hpc/package_adapter.py --adapter-dir /tmp/smoke_run/checkpoints/best --output-dir /tmp/smoke_pkg --base-model-id Qwen/Qwen2.5-0.5B-Instruct --adapter-rank 8
```

**Hard gate:** T1.4 cannot start until T1.3 passes the verification checklist below.

### T1.4 — HPC full SFT run

```bash
# On cluster:
git clone <repo> && cd <repo> && git checkout sprint/parallel-execution
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -e .
bash scripts/hpc/preflight.sh                        # no --local; full frozen-vars check
sbatch scripts/hpc/submit_sft.sbatch configs/lora_baseline.yaml
# Monitor: squeue -u $USER ; tail -f $RUN_ROOT/$RUN_TAG/train.log
```

After job completes:
```bash
python scripts/hpc/regression_gate.py --eval-records <baseline_eval.jsonl> --golden data/golden_20.jsonl
python scripts/hpc/package_adapter.py --adapter-dir $RUN_DIR/checkpoints/best --output-dir submissions/$RUN_TAG --base-model-id metric/nemotron-3-nano-30b-a3b-bf16/transformers/default --adapter-rank 32
```
`run_sft.py` already applied checkpoint rotation during the job, so there is no
second policy call here.

### T1.5 — Submit

Verify `submission.zip` (see verification section), then submit via `scripts/kaggle.sh` or the Kaggle web UI. Daily limit is 5; final selection rule is "last submission counts" — plan accordingly.

---

## Track 2 — Data + Eval + Notebooks (Kaggle + Colab)

**Owner:** data/eval. Largely parallel to Track 1; T2.2 blocks on T2.1.

### T2.1 — Generate validation_200 + golden_20 splits (Kaggle)

Pre-req: Kaggle dataset access (rules accepted).

Execute `notebooks/03_validation_and_golden_set.ipynb` end-to-end (it consumes the normalized output of `notebooks/02_dataset_schema_and_eda.ipynb`, which is also fully implemented). Produces:
- `data/validation_200.jsonl` (200 rows, 33-34 per category, stratified)
- `data/golden_20.jsonl` (20 rows, 3-4 per category, immutable)
- `data/eval/splits_manifest.json` (provenance: seed, dataset version, source split per row)
- SHA-256 sidecars for each JSONL

Update `docs/execution/NOTEBOOKS.md`: bump notebook 02 and 03 status from `scaffolded` → `validated` (resolves status drift; both notebooks have been fully implemented for some time).

### T2.2 — #21 prompt sweeps (Kaggle GPU or Colab Pro)

Notebook `notebooks/05_prompting_and_decode_sweeps.ipynb` is fully implemented (~900 lines; sparse 2×4×3=24 grid + Best-of-N follow-up at N∈{8,32}). Its `load_required_splits()` requires **both** `validation_200.jsonl` and `golden_20.jsonl` — so this is hard-blocked on T2.1 (the "start on golden_20 alone" idea from the brainstorm doesn't work without a code patch, which is not worth it since both splits are produced together).

Run on Kaggle (T4 free tier should fit Nemotron-3-Nano in 4-bit) or Colab Pro if Kaggle GPU times out. Outputs:
- `experiments/prompting_sweep_<date>.csv`
- `docs/analysis/prompting_findings.md` (ranked top-K configs)
- `data/eval/runs/<run_id>/` artifacts

When teammate's #19+#21 PR lands, compare both findings docs side-by-side; keep the winning config (whoever's). Document the comparison briefly at the bottom of `prompting_findings.md`.

### T2.3 — Notebook 07 (solver framework) — DEFER unless PM confirms

`notebooks/07_solver_framework.ipynb` is currently a documented scaffold (~300 lines) marked `active` in `NOTEBOOKS.md`. The actual framework code lives in `src/inference/solver.py` and is fully tested. The brainstorm asked us to "fill" it, but the existing intent (per `NOTEBOOKS.md`) appears to be that the notebook is design documentation, not an execution artifact. **Action: skip in this sprint unless PM explicitly says otherwise.** If extending later: add cells exercising `CategoryRouter` on `validation_200` examples and report verifier coverage per category.

### T2.4 — Notebook 09 (SFT runbook) — fill it

`notebooks/09_sft_runbook_and_masking.ipynb` is genuinely a minimal stub (~200 lines, no executable training content). Add cells covering:
- Masking walkthrough: load a real `SFTExample`, show tokenizer output, call `apply_loss_mask`, render which token positions are kept vs masked.
- Pipeline diagram (markdown): `preflight → tokenize_dataset → run_sft → checkpoint_policy → regression_gate → package_adapter`.
- Reference invocation cells (do NOT execute training in the notebook; just show the CLI commands for `run_sft.py` with `configs/lora_baseline.yaml` and the sbatch submission).
- Resume semantics: how `resume_from_latest.py --checkpoint-dir ...` feeds `--resume-from-checkpoint`.
- Promotion criteria: golden gate must pass before `package_adapter.py` is run.

Update `NOTEBOOKS.md`: notebook 09 status `scaffolded` → `active`.

### T2.5 — Synthetic data generation (Colab/Kaggle)

`notebooks/08_synthetic_data_recipe.ipynb` is executable in smoke mode. CLI:
```bash
python -m src.data.synthetic --smoke --dry-run         # cost estimate only
python -m src.data.synthetic --smoke                   # 50-row real generation
```
Produces `data/processed/synthetic_train.jsonl` + `.sha256`. Real (non-smoke) generation requires `data/analysis/retry_candidates.jsonl` from #22 — that input does not yet exist; if it's still missing when this step runs, skip non-smoke generation and use the smoke output as a sanity dataset for the local pipeline smoke (T1.3).

---

## Files to Modify / Create

| Path | Action | Track |
|---|---|---|
| `scripts/hpc/run_sft.py` | **NEW** (~200 LOC) | 1 |
| `scripts/hpc/submit_sft.sbatch` | **EDIT** lines 122-162 | 1 |
| `scripts/hpc/preflight.sh` | **EDIT** add `--local` flag | 1 |
| `configs/smoke_sft_local.yaml` | **NEW** (small-model smoke) | 1 |
| `notebooks/09_sft_runbook_and_masking.ipynb` | **EXTEND** beyond stub | 2 |
| `data/validation_200.jsonl` (+sidecar) | **GENERATE** via nb 03 | 2 |
| `data/golden_20.jsonl` (+sidecar) | **GENERATE** via nb 03 | 2 |
| `data/eval/splits_manifest.json` | **GENERATE** via nb 03 | 2 |
| `experiments/prompting_sweep_<date>.csv` | **GENERATE** via nb 05 | 2 |
| `docs/analysis/prompting_findings.md` | **GENERATE** via nb 05 | 2 |
| `docs/execution/NOTEBOOKS.md` | **EDIT** status: 02/03→validated, 09→active | 2 |
| `docs/learn/project/wave-d-design-choices.md` | **NEW** (Haiku-summarized retrospective) | pre-work |
| `data/processed/synthetic_train.jsonl` (+sidecar) | **GENERATE** (smoke OK) | 2 |
| `submissions/<run_id>/submission.zip` | **GENERATE** on HPC | 1 |

## Files explicitly NOT to modify

These are correct as-is; new code reuses them:
- `src/training/sft_trainer.py` (`apply_loss_mask` is library-only and correct).
- `scripts/hpc/tokenize_dataset.py` (already invokes `apply_loss_mask` and writes shards + fingerprint).
- `scripts/hpc/checkpoint_policy.py` (sidecar writer + rotation).
- `scripts/hpc/regression_gate.py` (delegates to `src.evaluation.golden_gate.evaluate_golden_gate`).
- `scripts/hpc/package_adapter.py` (wraps `src.inference.submission`).
- `scripts/hpc/resume_from_latest.py` (already returns highest-numbered checkpoint-XXXXX path).
- `src/evaluation/prompt_sweeps.py`, `src/evaluation/golden_gate.py`, `src/inference/submission.py`, `src/contracts.py`.

---

## Verification

**Track 1 local smoke (RTX 3080):**
- `bash scripts/hpc/preflight.sh --local` exits 0.
- `python scripts/hpc/tokenize_dataset.py …` produces ≥1 `.pt` shard + `dataset_fingerprint.json`.
- `python scripts/hpc/run_sft.py …` stdout shows monotonically (or near-monotonically) decreasing loss across the 20 steps.
- `ls /tmp/smoke_run/checkpoints/` lists at least one `checkpoint-*` directory plus `best/`.
- Spot-check: load a checkpoint's `labels` from a tokenized shard and assert `(labels == -100).sum() > 0` for non-assistant positions.
- `python scripts/hpc/package_adapter.py …` produces a zip with exactly `adapter_config.json` + `adapter_model.safetensors` at root.
- Full test suite still green: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` → 248 passed.

**Track 1 HPC full run:**
- `bash scripts/hpc/preflight.sh` (no `--local`) exits 0.
- `squeue` shows job in `R` then disappears with no `FAIL`/`TIMEOUT`/`OOM` in `train.log`.
- `python scripts/hpc/regression_gate.py …` exits 0 against `data/golden_20.jsonl`.
- `unzip -l submissions/$RUN_TAG/submission.zip` shows exactly two entries, no nested paths, no symlinks.
- `wc -c < submissions/$RUN_TAG/submission.zip` < 100 MB.

**Track 2 data:**
- `wc -l data/validation_200.jsonl` = 200; `wc -l data/golden_20.jsonl` = 20.
- `sha256sum -c data/validation_200.jsonl.sha256` and `… golden_20.jsonl.sha256` both pass.
- `data/eval/splits_manifest.json` has `dataset_version`, `seed`, per-row source-split annotation.

**Track 2 #21:**
- `experiments/prompting_sweep_<date>.csv` exists with ≥24 sparse rows + Best-of-N rows.
- `docs/analysis/prompting_findings.md` has a ranked top-K table and a "Promoted config" line.

**Track 2 notebooks:**
- `jupyter nbconvert --to notebook --execute notebooks/02_dataset_schema_and_eda.ipynb --inplace --ExecutePreprocessor.timeout=600` exits 0.
- Same for notebook 03 and (post-fill) notebook 09.

---

## Sequencing & Dependencies

```
Pre-work (1h, branch + retrospective)
   ├─ Track 1 ─────────────────────────────────────────────────
   │   T1.1 run_sft.py (4h)
   │     └─ T1.2 sbatch wire (30m)
   │           └─ T1.3 local smoke RTX 3080 (3h)  ← HARD GATE
   │                 └─ T1.4 HPC submit + queue (variable, hours-days)
   │                       └─ T1.5 Kaggle submit (1h)
   │
   └─ Track 2 ─────────────────────────────────────────────────
       T2.1 splits via nb 03 on Kaggle (1h)
         ├─ T2.2 #21 sweep (4h compute on Kaggle/Colab)
         └─ (T1.4 also consumes golden_20 for regression_gate)
       T2.4 notebook 09 fill (2h, depends on T1.1/T1.2 for accurate CLI refs)
       T2.5 synthetic smoke (1h, mostly independent)
       T2.3 notebook 07 — DEFER unless PM confirms
```

Critical path: Pre-work → T1.1 → T1.2 → T1.3 → T1.4 (queue) → T1.5.

---

## Out of Scope

- **RL training (`submit_rl.sbatch`)** — stretch only if Track 1 finishes with cluster time to spare. Will require its own plan (reward function, eval contract, ALLOW_RL=1 gate is already present in the script).
- **Notebook 07 extension** — gated on PM decision; current state is intentional per `NOTEBOOKS.md`.
- **Specific HPC cluster name, weight pre-staging, queue partition** — runtime config; preflight handles validation.
- **Patching #21 sweep code to allow golden-only input** — bypassed by running T2.1 (which produces both splits in one notebook execution).
- **Coordinating teammate's #19 PR merge timing** — rebase as needed when their PR lands.
- **Closing #19 ourselves** — owned by teammate.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| RTX 3080 smoke uses a different model than HPC, so a 30B-only bug (e.g., MoE-specific config) won't be caught locally. | Smoke is for *plumbing* not *model behavior*; HPC `preflight.sh` checks the frozen model load path before sbatch even queues training. |
| Tokenized shards already have masked labels, but TRL `SFTTrainer` may try to re-mask via its own `formatting_func`. | Use a passthrough `DataCollator` that returns the pre-tokenized batch directly; do NOT pass `formatting_func` or `dataset_text_field` to `SFTTrainer`. Add a one-line assert in `run_sft.py` that any sampled batch has `(labels == -100).any()`. |
| Kaggle GPU may not have enough VRAM for Nemotron-3-Nano in 4-bit during sweeps. | Notebook 05 already calls `require_cuda()`; if Kaggle T4 OOMs, switch to Colab Pro A100/L4 — the sweep code is environment-agnostic. |
| HPC queue time exceeds remaining sprint window. | Submit early; smoke validation on RTX 3080 means the HPC job is unlikely to fail for code reasons; preflight catches env issues before sbatch. |
| `data/analysis/retry_candidates.jsonl` (input to real synthetic generation) doesn't exist (it's a #22 output that hasn't been built). | Use empty-list fallback baked into notebook 08 + the smoke output; full synthetic data generation is not on the critical path for first submission. |
