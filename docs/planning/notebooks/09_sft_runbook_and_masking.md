# Notebook 09: SFT LoRA Runbook and Loss Masking

**Parent Issue**: `#25`  
**Plan Phase**: Phase 4 (Supervised Fine-Tuning with LoRA)  
**Scaffold**: `notebooks/09_sft_runbook_and_masking.ipynb`  
**Status**: `planned`  
**Dependencies (upstream)**: `#14` (constraints), `#15` (review harness), `#19` (eval), `#23` (solver), `#24` (synthetic recipe)  
**Consumers (downstream)**: `#20` (submission packaging)

---

## 1. Objective

Codify the LoRA/QLoRA training runbook before executing long runs on the HPC cluster: default to the verified `#14` base model and `r <= 32`, start from the official 4-module demo target set (`in_proj`, `out_proj`, `up_proj`, `down_proj`), gate any broader konbu17-style target set behind PM signoff, implement explicit completion-only masking so prompt/user tokens are ignored by loss, define checkpoint and early-stopping policies, and produce configuration files plus a minimal smoke-test training run that validates the entire pipeline on a small curated subset.

## 2. Why It Matters

- **Competition**: Masking decisions determine whether gradients are wasted on formatting tokens or focused on reasoning; wrong masking can regress accuracy or waste compute.
- **Capstone learning**: Understanding LoRA target module selection (hybrid Mamba-2 + Transformer architecture) and loss masking is critical for fine-tuning success.
- **Upstream blocking**: Notebook #23 (solver) and #24 (synthetic recipe) produce training data that will be consumed here; #19 (eval) provides the golden_20 regression set.
- **Downstream blocker**: #20 (submission packaging) cannot start until we have validated adapters and a runbook for reproducibility.

## 3. Strategy — How We Aim To Accomplish It

1. **Confirm LoRA target modules by introspection**: Load the verified `#14` base model, dump `model.named_modules()`, start from the official demo set (`in_proj`, `out_proj`, `up_proj`, `down_proj`), and document any proposed expansion separately for PM signoff.
2. **Verify tokenizer chat template and reasoning tokens**: Confirm `<think>` = token 12, `</think>` = token 13; validate that `apply_chat_template(..., enable_thinking=True)` produces expected token sequences.
3. **Implement loss masking**: Write masking utility to set prompt token labels to -100 (ignored by loss), keep reasoning/answer tokens; unit test with synthetic input showing correct -100 mask alignment.
4. **Codify LoRA configs**: Write `configs/lora_baseline.yaml` with `r=32` or lower, bf16, explicit target modules, dropout/bias/task type, and completion-only masking; write `configs/lora_qlora.yaml` with 4-bit quantization for Colab/local smoke tests.
5. **Define checkpoint/early-stopping policy**: Log checkpoint intervals (every 500 steps), eval interval (every 100 steps), early-stopping patience (3 evals without improvement), validation on golden_20.
6. **Smoke-run sign-off**: Execute 100-step training on 1k curated subset using Unsloth on RTX 3080 with QLoRA, log loss curve to WandB, verify golden_20 pass rate ≥ 100%, save adapter as safetensors and load it back.

## 4. MVP (Minimum Viable Notebook)

**Inputs**: Verified `#14` base model, 1k curated samples from #24, golden_20 validation set, requirements.txt pinning torch/transformers/peft/unsloth/trl.

**Cells**:
- Cell 1: Environment check (GPU, torch, transformers versions)
- Cell 2: Load base model, dump module names, confirm target modules
- Cell 3: Load tokenizer, verify chat template and reasoning tokens
- Cell 4: Implement masking utility and unit test
- Cell 5: Load configs from YAML, instantiate LoRA + trainer
- Cell 6: Load 1k curated dataset, apply masking, create DataLoader
- Cell 7: Run 100-step smoke training with Unsloth, log to WandB
- Cell 8: Evaluate adapter on golden_20, verify 100% pass rate (20/20)
- Cell 9: Save adapter as safetensors, load back, verify loading succeeds
- Cell 10: Summary: log final loss, checkpoint count, runtime; write execution log

**Outputs**:
- `configs/lora_baseline.yaml` (full LoRA for HPC)
- `configs/lora_qlora.yaml` (QLoRA 4-bit for Colab/local)
- `src/training/sft_trainer.py` (wrapper + masking logic)
- `tests/test_masking.py` (unit tests for loss masking)
- `docs/execution/SFT_RUNBOOK.md` (step-by-step runbook with hardware routing)
- `adapters/smoke_<date>/adapter_model.safetensors` (smoke run adapter weights)
- WandB run with tag `smoke_run_validation`

**Verification**: Loss decreases over 100 steps; golden_20 pass rate = 20/20 (100%); adapter loads successfully with base model; masking unit test confirms all prompt tokens have label=-100.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: RTX 3080 (10GB), Unsloth installed, 1k curated samples, base model cached
- **Action**: Load model → confirm target modules → verify tokenizer → test masking → run 100-step smoke training → evaluate golden_20 → save/load adapter
- **Expected**: Final loss ~0.5-1.0 (regression-free vs baseline); golden_20 pass rate = 20/20; adapter loads and runs inference without error; masking unit test logs all prompt tokens as -100

### 5.2 Alternative / Fallback

If Unsloth does not support the hybrid Mamba-2 + Transformer architecture on installed version:
- **Setup**: Same hardware, but fall back to HF PEFT + TRL SFTTrainer
- **Action**: Use PEFT `LoraConfig` + TRL `SFTTrainer` (smaller batch size, longer wall-clock time)
- **Expected**: Same golden_20 pass rate; loss curve comparable; document version gap in `docs/execution/SFT_RUNBOOK.md`

### 5.3 Regression Guardrails

- **Golden set freeze**: golden_20 pass rate must never drop below 20/20 (100%)
- **Loss curve validation**: Final loss in smoke run must be ≤ 2.0; if loss increases over first 20 steps, stop and debug
- **Adapter loading**: After saving as safetensors, re-load and verify forward pass succeeds (no tensor shape mismatches)

## 6. Success Criteria (Done When)

- [ ] `configs/lora_baseline.yaml` written with `r<=32`, verified target module names, and PM signoff for any expansion beyond the official 4-module demo set
- [ ] `configs/lora_qlora.yaml` written with 4-bit quantization params
- [ ] `src/training/sft_trainer.py` contains masking utility; all prompt tokens set to label=-100
- [ ] `tests/test_masking.py` passes with 100% prompt coverage in unit tests
- [ ] 100-step smoke run completes on RTX 3080 with final loss ≤ 2.0
- [ ] Golden_20 evaluation shows 100% pass rate (20/20) on smoke adapter
- [ ] `docs/execution/SFT_RUNBOOK.md` documents launch command, hardware routing (HPC vs Colab), OOM ladder, and checkpoint policy
- [ ] Smoke adapter saved as safetensors and reloaded successfully
- [ ] WandB run tagged `smoke_run_validation` with loss curve plot
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`

## 7. Risks & Open Questions

| Risk | Mitigation |
|------|-----------|
| **Mamba-2 module names differ from docs** | Dump `model.named_modules()` early; update YAML if names don't match q_proj, v_proj, x_proj, in_proj, out_proj |
| **Masking miscount due to special tokens** | Unit test with synthetic input; verify -100 labels align with prompt boundaries in chat template |
| **OOM on RTX 3080 even with QLoRA** | Implement OOM ladder: start batch=8, then 4, then 2, then inference-only fallback |
| **vLLM unsupported for Mamba-2 inference** | Fall back to HF `model.generate()` for evaluation; document in runbook |
| **Unsloth version incompatible with Nemotron** | Use HF PEFT + TRL as fallback; test early in smoke run |

| Open Question | Who Answers |
|---------------|-------------|
| What is the exact max_tokens for training batches on HPC? | Notebook author (depends on cluster config) |
| Should we use cosine or constant LR schedule? | Check plan_v0.2 Phase 4; use cosine with 5% warmup as specified |
| What early-stopping patience is optimal? | Default to 3 evals without improvement; tune if smoke run shows instability |

## 8. Artifacts & Handoff

**Produces**:
- `configs/lora_baseline.yaml` — Full LoRA (`r<=32`) for HPC
- `configs/lora_qlora.yaml` — QLoRA 4-bit for Colab/RTX 3080
- `src/training/sft_trainer.py` — SFT trainer wrapper + masking logic
- `tests/test_masking.py` — Unit tests for loss masking (prompt -100 alignment)
- `docs/execution/SFT_RUNBOOK.md` — Runbook: launch commands, hardware routing, OOM ladder
- `adapters/smoke_<date>/adapter_model.safetensors` — Smoke run adapter checkpoint
- WandB run `smoke_run_validation` — Loss curve, golden_20 eval results

**Consumed by**: Notebook #20 (submission packaging); HPC training job submission; any future fine-tuning iterations.

**External references**: [PEFT LoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora), [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer), [Unsloth](https://github.com/unslothai/unsloth), plan_v0.2 Phase 4.

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Module introspection + masking impl | 2 | RTX 3080 |
| Config writing + smoke run | 1.5 | RTX 3080 |
| Golden_20 eval + test coverage | 1 | RTX 3080 |
| Runbook + artifact cleanup | 0.5 | Local CPU |
| **MVP Total** | **5** | **RTX 3080** |
| Alternative (Unsloth → PEFT fallback) | +1.5 | RTX 3080 |
| Full polish (OOM ladder, early-stopping tuning) | +1 | RTX 3080 |
