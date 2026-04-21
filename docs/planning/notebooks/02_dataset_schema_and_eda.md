# Notebook 02: Dataset Schema and EDA

**Parent Issue**: `#17`
**Plan Phase**: Phase 1.3 (Dataset Exploration) and Phase 3 prep (Data Curation)
**Scaffold**: `notebooks/02_dataset_schema_and_eda.ipynb`
**Status**: `scaffolded`
**Dependencies (upstream)**: `#14` (constraints frozen)
**Consumers (downstream)**: `#18` (validation/golden set), `#19` (eval normalization), `#22`, `#24`, `#25`

---

## 1. Objective

Define a canonical data schema across competition-provided test data and public training datasets (Llama-Nemotron Post-Training 32M+, Puzzle-KD v2 851K). Produce field inventory (types, null rates, token-length distributions, category enums), quantify reasoning-on/off and answer-format mixes, and draft a normalization spec (`src/data/schema.py`) that downstream notebooks import.

## 2. Why It Matters

- **Leaderboard**: Correct normalization of field formats (`\boxed{}`, reasoning tokens) directly impacts exact-match accuracy in evaluation.
- **Capstone learning**: Forces explicit schema thinking before data pipelines hard-code assumptions.
- **Handoff**: Notebooks #18, #19, #22, #24, #25 all depend on unified category labels and field names; schema drift causes silent failures.

## 3. Strategy — How We Aim To Accomplish It

1. Load 1k-row sample from competition test set (via Kaggle or HF mirror) and describe dtypes, null rates, categories.
2. Load `nvidia/Llama-Nemotron-Post-Training-Dataset` (streaming mode if >130GB) and 10k-row slice; extract field names, note nested structures, infer category taxonomy.
3. Load `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` full subset and describe structure (verify 851K count, 95/5 split, reasoning=off assumption).
4. Cluster samples by category; inspect answer-format mix (`\boxed{}` prevalence, free-form, multiple-choice, reasoning-on/off token ratios).
5. Compute token-length histogram for all sources (key metric: 95th percentile ≤ 8192, plan's `max_token_count`).
6. Draft normalization rules: strip whitespace, unify category labels, enforce token limits, validate UTF-8.
7. Produce `data/eval/schema.json` (canonical field spec) and `src/data/schema.py` stub listing field names, enums, types.

## 4. MVP (Minimum Viable Notebook)

- **Inputs**: HF dataset mirrors or Kaggle CLI downloads; `transformers.AutoTokenizer` for Nemotron.
- **Cells**:
  1. Environment setup, imports, seed.
  2. Load and describe competition test set (1k rows).
  3. Load and describe Llama-Nemotron subset (10k rows, streaming).
  4. Load and describe Puzzle-KD v2 (full 851K).
  5. Category inventory and answer-format frequency table.
  6. Token-length histogram (all sources, per-category breakdown).
  7. Normalization rules pseudocode + test cases.
  8. Write `data/eval/schema.json` and `src/data/schema.py` stub.
- **Outputs**:
  - `data/eval/schema.json` — canonical field dict (names, types, enums, constraints).
  - `experiments/eda_<date>.md` — summary stats table (category count, avg/95th tokens, reasoning % by source).
  - `experiments/figures/token_length_histogram.png` — overlay histogram (all sources, log scale).
  - `src/data/schema.py` — Python dataclass or TypedDict listing ReasoningExample contract.
- **Verification**: Printed schema table ≥ 5 sources × 6 fields (category, answer_format, reasoning_on, token_count, language, null_rate); no assertion, human review.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: HF datasets library installed, `HUGGINGFACE_TOKEN` set or public mirror accessible, 50GB free disk for 10k-row download.
- **Action**: Run all 8 cells end-to-end with seed=42.
- **Expected**: 
  - Schema table printed with category enum ≥ 8 values (math, code, science, logic, chat, safety, etc.).
  - Token-length 95th percentile ≤ 8192 for each source (warn if ≥ 7500).
  - `data/eval/schema.json` contains well-formed JSON with 6+ fields per category.
  - Reasoning-on % reported (plan expects 75% for curated training set; baseline data may vary).

### 5.2 Alternative / Fallback

- **Setup**: HF `datasets.load_dataset(..., streaming=True)` if local storage < 20GB. No Kaggle access (use public mirror).
- **Action**: Load Llama-Nemotron in streaming mode, iterate 10k samples sequentially, accumulate stats in rolling counter (mean, m2 for variance, histogram buckets).
- **Expected**: Same schema JSON output as primary; `experiments/eda_<date>.md` notes "streaming mode (no full download)"; histogram smoothed but bin counts match ±5%.

### 5.3 Regression Guardrails

- **Re-running with seed=42 produces byte-identical stats**: category counts, reasoning % per source match to ±0 (discrete counts) or ≤ 0.1% (continuous metrics).
- **Golden-set contract**: If `data/eval/golden_20.jsonl` exists from #18, schema must accommodate all golden-set fields without adaptation.

## 6. Success Criteria (Done When)

- [ ] 1k-row sample from competition test set loaded and described (all fields with non-null count).
- [ ] Llama-Nemotron 10k sample loaded and category taxonomy extracted (enum ≥ 5 distinct categories).
- [ ] Puzzle-KD v2 confirmed 851K samples, reasoning-off assumption validated (≤1% samples with non-empty `<think>` tokens).
- [ ] Token-length histogram computed for all sources; 95th percentile ≤ 8192 confirmed or risk flagged.
- [ ] Answer-format mix quantified: % `\boxed{}`, % free-form, % multiple-choice per category.
- [ ] `data/eval/schema.json` written with canonical field spec; validates against 3 randomly sampled records from each source.
- [ ] `src/data/schema.py` stub created with ReasoningExample dataclass (fields: id, category, prompt, answer, reasoning_on, source, language, tokens_used).
- [ ] `experiments/eda_<date>.md` summary committed with reproducibility note.
- [ ] Artifact(s) linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk**: Undisclosed private-test categories in competition data (category enum mismatch). | **Mitigation**: #14 freeze issues describe competition constraints; if new categories appear in #18, revise schema.json with additive-only rule.
- **Risk**: Non-UTF-8 strings in Llama-Nemotron (mojibake, emoji, invalid unicode). | **Mitigation**: Detect in EDA; add encoding exception handler in normalization spec; report count to WandB.
- **Risk**: 130GB Llama-Nemotron too large for WSL2. | **Mitigation**: Use streaming mode (Alternative 5.2); store schema.json on HPC cluster or external drive; prioritize Puzzle-KD v2 for Phase 3 curation.
- **Open question**: Does "reasoning_on" in Puzzle-KD v2 label match token-counting check (presence of `<think>` tags)? | **Who answers**: Notebook execution; add manual spot-check of 10 samples.

## 8. Artifacts & Handoff

- **Produces**:
  - `data/eval/schema.json` — canonical schema dict (category enum, field types, constraints).
  - `experiments/eda_<date>.md` — summary stats table (source, total rows, category distribution, mean/95th token length, reasoning %, answer-format mix).
  - `experiments/figures/token_length_histogram.png` — overlay histogram (all sources, bins=[256, 512, 1024, 2048, 4096, 8192, 16384, ∞]).
  - `src/data/schema.py` — Python module exporting ReasoningExample dataclass and normalization function stub.
- **Consumed by**: 
  - `#18` (validation/golden set) — imports schema for field validation.
  - `#19` (eval normalization) — uses category enum and token-length policy.
  - `#22`, `#24`, `#25` — reference schema.py for data loading.
- **External references cited**: 
  - [Llama-Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
  - [Puzzle-KD v2](https://huggingface.co/datasets/nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2)
  - [NVIDIA blog: Train a Reasoning Capable LLM](https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/)
  - [Kaggle competition submission demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| MVP (cells 1–8) | 4–5 | Colab Pro or RTX 3080 + 50GB disk |
| Alternative path (streaming) | 6–8 | Colab CPU or local CPU + rolling stats |
| Full polish (figures, markdown export) | 2–3 | Local |
| **Total** | **7–11** | **Colab Pro recommended** |
