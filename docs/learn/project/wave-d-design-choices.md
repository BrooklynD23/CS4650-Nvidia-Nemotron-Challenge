---
title: Wave D Design Choices Retrospective
audience: beginner
page_type: concept
status: implemented
last_reviewed: 2026-05-08
repo_sources:
  - docs/execution/plans/issue-25-hpc-queue-runbook.md
  - docs/execution/plans/sprint-parallel-execution.md
  - docs/learn/project/implemented-today.md
  - scripts/hpc/checkpoint_policy.py
  - scripts/hpc/resume_from_latest.py
  - scripts/hpc/run_sft.py
  - scripts/hpc/submit_sft.sbatch
  - src/training/sft_trainer.py
external_sources:
  - https://huggingface.co/papers/2106.09685
  - https://huggingface.co/papers/2312.00752
---

# Wave D — Design Choices Retrospective

## Why This Page Exists

Consolidated rationale for key implementation decisions made during Wave D
(SFT/RL pipeline scaffolding). Intended as a permanent reference for future
sprints; forward-looking decisions belong in `docs/planning/`.

---

## 1. LoRA rank: `lora_r = 32`

**Config reference:** `configs/lora_baseline.yaml`

```yaml
lora_r: 32
lora_alpha: 64   # 2x lora_r — standard 2x scaling rule
lora_dropout: 0.05
```

`lora_r = 32` balances expressiveness against parameter budget for the
Nemotron-3-Nano-30B-A3B model:

- **Rank 16** (common default) leaves too little headroom for math/code tasks
  that require fine-grained reasoning-path shifts.
- **Rank 64+** doubles VRAM footprint for the adapter matrices without
  consistent quality gains at this scale.
- `lora_alpha = 64` (= 2 x lora_r) applies the standard 2x scaling convention
  so the effective learning rate for the adapter does not need retuning when
  rank changes later.
- `lora_dropout = 0.05` is a light regulariser to avoid over-fitting on the
  synthetic data distribution.

---

## 2. `target_modules` choice

**Config reference:** `configs/lora_baseline.yaml`

```yaml
target_modules:
  - in_proj
  - out_proj
  - up_proj
  - down_proj
```

The Nemotron-3-Nano model uses a Mamba2 / MLP hybrid architecture. The four
projection names map to:

| Module | Layer type | Rationale |
|--------|-----------|-----------|
| `in_proj` | SSM input projection | Controls what information enters the state-space branch |
| `out_proj` | SSM output projection | Maps hidden state back to residual stream |
| `up_proj` | MLP gate / expansion | Primary pathway for factual and reasoning updates |
| `down_proj` | MLP contraction | Bottleneck where new knowledge is written back |

Attention-style QKV projections are absent because Mamba2 blocks do not use
explicit attention; `in_proj`/`out_proj` target the analogous
information-routing pathway. `trust_remote_code: true` is required for the
custom Mamba2 kernel.

---

## 3. `apply_loss_mask` API contract

**Source:** `src/training/sft_trainer.py`

### Signature

```python
def apply_loss_mask(
    input_ids: list[int],
    labels: list[int],
    tokenizer: object,
    mask_role: str = "user",
) -> list[int]:
```

### Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `input_ids` | `list[int]` | Full tokenised conversation (all turns concatenated) |
| `labels` | `list[int]` | Positionally aligned label IDs, typically a copy of `input_ids` |
| `tokenizer` | object | Must expose `role_start_ids: dict[str, list[int]]` mapping role name to the token sequence that opens that role's block |
| `mask_role` | `str` | Kept for API compatibility; all non-`"assistant"` roles are masked regardless of this value |

### Output

A **new** `list[int]` of the same length as `labels` where every token
position belonging to a non-assistant turn is replaced with `IGNORE_INDEX = -100`.

### What gets masked

- Scans `input_ids` for each role-boundary marker in `tokenizer.role_start_ids`.
- Boundaries are sorted by position; each span runs from one boundary to the
  next (or to end-of-sequence for the last span).
- Any span whose role is not `"assistant"` is masked (label set to `-100`).
- `IGNORE_INDEX = -100` is the PyTorch cross-entropy sentinel — those
  positions are excluded from the loss so the model does not learn to predict
  prompt tokens.

### Error condition

Raises `ValueError` if no assistant turn is found, preventing silent training
on a fully-masked sequence where the model would receive no gradient signal.

### Immutability guarantee

Always returns a **new list**; the input `labels` argument is never mutated.

---

## 4. Keep-last-3-plus-best checkpoint policy

**Source:** `scripts/hpc/checkpoint_policy.py`

```python
_KEEP_LAST_N = 3
```

### Policy rules

1. **Scan** `--checkpoint-dir` for subdirectories matching `checkpoint-\d+`.
2. **Sort** by checkpoint number ascending.
3. **Keep** the three most recent `checkpoint-XXXXX/` directories.
4. **Keep** the `best/` directory unconditionally (best validation-metric
   checkpoint regardless of recency).
5. **Delete** all older `checkpoint-XXXXX/` directories (`--execute` required;
   default is dry-run to prevent accidents).

### Sidecar files written per checkpoint

Each surviving checkpoint receives the following if absent:

| File | Contents |
|------|----------|
| `trainer_state.json` | Step number, directory name, `created_at` ISO-8601 UTC |
| `run_config.json` | Copied from `--run-config` source |
| `metrics.jsonl` | Touched / left as-is (training loop appends) |
| `git_sha.txt` | `git rev-parse HEAD` at policy-run time |
| `dataset_fingerprint.txt` | Copied from parent checkpoint dir if present |

### Rationale

- 3 recent checkpoints covers the typical debugging window: one checkpoint back diagnoses
  divergence, two gives a trend.
- `best/` is always kept because it is the model shipped downstream; losing
  it is unrecoverable without rerunning training.
- Dry-run default prevents accidental deletion in development.
- The current wrapper writes checkpoints under `CHECKPOINT_DIR` and applies
  this rotation once after training, so the SLURM layer does not duplicate the
  policy call.

## Sources

- Repo: [docs/execution/plans/issue-25-hpc-queue-runbook.md](../../execution/plans/issue-25-hpc-queue-runbook.md)
- Repo: [docs/execution/plans/sprint-parallel-execution.md](../../execution/plans/sprint-parallel-execution.md)
- Repo: [docs/learn/project/implemented-today.md](implemented-today.md)
- Repo: [scripts/hpc/checkpoint_policy.py](../../../scripts/hpc/checkpoint_policy.py)
- Repo: [scripts/hpc/resume_from_latest.py](../../../scripts/hpc/resume_from_latest.py)
- Repo: [scripts/hpc/run_sft.py](../../../scripts/hpc/run_sft.py)
- Repo: [scripts/hpc/submit_sft.sbatch](../../../scripts/hpc/submit_sft.sbatch)
- Repo: [src/training/sft_trainer.py](../../../src/training/sft_trainer.py)
- External: https://huggingface.co/papers/2106.09685
- External: https://huggingface.co/papers/2312.00752

---

## 5. QualityFilter solver-confidence gate

**Source:** `src/data/synthetic.py`

### Configuration

```python
@dataclass
class SyntheticConfig:
    solver_confidence_threshold: float = 0.0  # 0.0 = disabled
```

### How it is applied

Inside `QualityFilter.accept()`:

```python
if self._config.solver_confidence_threshold > 0:
    conf = ex.provenance.get("solver_confidence", 0.0)
    if conf < self._config.solver_confidence_threshold:
        return False
```

- The gate is **opt-in**: `0.0` (default) disables it so no examples are
  filtered on this criterion during development.
- When enabled, reads `ex.provenance["solver_confidence"]` — a float written
  by the solver teacher path when it produces a verified answer.
- Examples where `solver_confidence` is absent default to `0.0`, failing any
  non-zero threshold (safe-by-default: uncertain provenance is rejected).

### Full gate set in `QualityFilter.accept()`

| Gate | Criterion |
|------|-----------|
| Deduplication | SHA-256 of `example_id + category`; rejects duplicates |
| `\boxed{}` presence | Completion must contain the LaTeX boxed-answer marker |
| Token budget | Estimated token count <= `max_tokens` (chars / 4 heuristic) |
| Provenance completeness | `provenance` must contain `teacher`, `generated_at`, `source_run_id` |
| Category allow-list | `category` must be in `SyntheticConfig.categories` |
| Solver confidence | `provenance["solver_confidence"]` >= threshold (when threshold > 0) |

---

## 6. Smoke run and cost-cap limits

**Source:** `src/data/synthetic.py`

```python
_SMOKE_LIMIT = 50
_COST_CAP_USD = 20.0
```

### Smoke mode (`--smoke` flag)

```python
target = candidates[:_SMOKE_LIMIT] if smoke else candidates
```

- Slices the candidate list to the first **50 rows** before generation begins.
- Validates the full pipeline (tokenisation, teacher call, quality filter,
  JSONL write, fingerprint) without full API cost.
- 50 rows was chosen as enough to catch schema/format bugs while keeping a
  typical smoke run under 2 minutes.

### Cost cap (`cost_cap_usd`)

```python
if accumulated_cost >= config.cost_cap_usd:
    print(f"[synthetic] cost cap ${config.cost_cap_usd} reached; stopping …")
    break
```

- Hard-stops generation when accumulated LLM-teacher spend reaches **$20**.
- Resets per `generate_from_retry_candidates` call (per pipeline run, not per
  process lifetime).
- The `$0.01` per-call placeholder is a stub; production code must inject
  actual token-count-based pricing.
- Combined with `--dry-run` (prints token and cost estimates, no API calls),
  these two mechanisms prevent runaway spend during iteration.

---

*Generated 2026-05-08. Update when thresholds or architecture change.*
