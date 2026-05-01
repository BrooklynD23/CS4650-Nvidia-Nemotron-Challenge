# Notebook 01: External Baselines and Design Deltas

**Parent Issue**: `#16`
**Plan Phase**: Phase 1 (baseline) + Phase 4 (SFT)
**Scaffold**: `notebooks/01_external_baselines_and_design_deltas.ipynb`
**Status**: `validated`
**Dependencies (upstream)**: `#13` (template); `#14` is the verified official contract source, not a blocker
**Consumers (downstream)**: `#19` (baseline eval), `#23` (solver), `#25` (SFT runbook)

---

## 1. Objective

Compare the bounded external reference set for `#16`: Tong (`tonghuikang/nemotron`) and konbu17. Kishan and aitherium remain optional future references. The review extracts transferable ideas for masking, augmentation, solver/teacher patterns, packaging, and SFT defaults, then records adoption decisions in `docs/analysis/EXTERNAL_BASELINE_REVIEW.md`.

## 2. Why It Matters

- **Leaderboard impact**: Public reference techniques may transfer to private test set; incorrectly copying them risks overfitting, rule drift, or license/provenance friction.
- **Capstone learning**: Comparing external code teaches pattern recognition in real LoRA pipelines versus theory.
- **Downstream coupling**: Notebooks 04, 07, 09 depend on this review to lock masking strategy and eval rules before training loops begin.

## 3. Strategy — How We Aim To Accomplish It

1. **Review Tong (`tonghuikang/nemotron`)** from GitHub and extract base model, masking, dataset/corpus, evaluation, packaging, augmentation, weighting, and solver/teacher patterns.
2. **Review konbu17** from the public Kaggle notebook URL and existing repo-local notebook evidence for base model, load recipe, LoRA config, target modules, masking behavior, dataset, eval protocol, and packaging.
3. **Anchor decisions to `#14`** so the Kaggle/NVIDIA official contract is the default when external baselines differ.
4. **Record Adopt / Reject / Gate decisions** with downstream issue mapping in `docs/analysis/EXTERNAL_BASELINE_REVIEW.md`.

## 4. MVP (Minimum Viable Notebook)

- **Inputs**: Tong GitHub repo, konbu17 Kaggle notebook URL, existing repo-local konbu17 evidence, and `docs/architecture/COMPETITION.md`.
- **Cells**:
  1. Environment check and repo clone/API fetch
  2. Extract config/training script signatures from each source
  3. Build a markdown decision matrix
  4. Write Adopt / Reject / Gate decisions with downstream issue mapping
  5. Commit `docs/analysis/EXTERNAL_BASELINE_REVIEW.md`
- **Outputs**: `docs/analysis/EXTERNAL_BASELINE_REVIEW.md`.
- **Verification**: Every decision names a downstream issue and does not override the verified `#14` contract.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Tong repo cloneable from GitHub; konbu17 notebook publicly readable or covered by repo-local evidence.
- **Action**: Cells fetch repo metadata, extract masking/eval approach from each source's training script or markdown README.
- **Expected**: Delta matrix covers Tong and konbu17, including source identity, masking strategy, eval/metric notes, packaging, and adoption decision.

### 5.2 Alternative / Fallback

If a source becomes private or unextractable:
- **Setup**: Archive links or PDF snapshots available from prior Kaggle snapshot API call.
- **Action**: Cells extract decision from description-level metadata (public notebook metadata API, blog post summary, arXiv abstract) instead of source code inspection.
- **Expected**: Row marked "evidence incomplete"; masking/approach columns filled from abstract or notebook title/description only; decision logic explicitly flags as "inference from public claim only."

### 5.3 Regression Guardrails

- **Golden-set integrity**: No "adopt" decision silently changes plan_v0.2 baseline eval rules (exact-match normalization, prompt template); changes require a new GitHub issue with `[BREAKING]` label.
- **Dataset leakage check**: No row references private or contest-internal datasets as adoption source; Kishan public dataset is OK; any internal Kaggle-only data marked as "risk: leakage."
- **Citation freshness**: Every URL in the matrix must be dated or linked to a specific commit SHA / Kaggle kernel version timestamp; dead links logged with mitigation step.

## 6. Success Criteria (Done When)

- [x] Tong and konbu17 have entries in the review artifact.
- [x] The review distinguishes public reference baselines from official Kaggle/NVIDIA sources.
- [x] At least one Adopt, Reject, and Gate decision is recorded with downstream issue mapping.
- [x] `docs/analysis/EXTERNAL_BASELINE_REVIEW.md` is committed as the review artifact.
- [x] No external baseline overrides the verified `#14` contract.
- [x] Artifact is linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk: Leaderboard overfitting** | **Mitigation**: Flag any technique that directly copies public reference behavior as "high specificity risk"; default to gate unless it is backed by local validation.
- **Risk: License incompatibility** | **Mitigation**: Audit Tong repo LICENSE file; if GPL or incompatible with repo license, mark adoption as "forbidden" and document in decision log.
- **Risk: Kaggle notebook extraction restricted** | **Mitigation**: Use repo-local evidence only when it already captures exact notebook facts; otherwise mark the source evidence incomplete.
- **Open question: Do Tong's augmentation, weighting, and solver-first teacher patterns generalize to private test set?** | **Who answers**: Phase 4 checkpoint after source/commit/license capture and local validation.

## 8. Artifacts & Handoff

- **Produces**:
  - `docs/analysis/EXTERNAL_BASELINE_REVIEW.md` — main review and decision table
- **Consumed by**:
  - `#19` (04_baseline_eval): Confirms eval metric alignment from Tong/konbu17
  - `#23` (07_solver): Tong solver/teacher patterns
  - `#24` (08_synthetic_data): Tong augmentation patterns
  - `#25` (09_sft_runbook): masking strategy, LoRA rank, target-module gates
- **External references cited**:
  - https://github.com/tonghuikang/nemotron (commit SHA captured in review artifact)
  - https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2 (kernel version / last edit date)

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| Repo clones + API fetches + skim | 2 | CPU (no GPU) |
| Extract configs & build matrix | 1.5 | CPU |
| Score & rank rows | 1 | CPU |
| Write decision log + CSV export | 1 | CPU |
| **MVP total** | **5.5** | **CPU** |
| Fallback (metadata-only delta) | +1 | CPU |
| Full polish (cross-check refs, archive links) | +1.5 | CPU |
