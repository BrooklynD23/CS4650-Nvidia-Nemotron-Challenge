# Notebook 01: External Baselines and Design Deltas

**Parent Issue**: `#16`
**Plan Phase**: Phase 1 (baseline) + Phase 4 (SFT)
**Scaffold**: `notebooks/01_external_baselines_and_design_deltas.ipynb`
**Status**: `scaffolded`
**Dependencies (upstream)**: `#13` (template)
**Consumers (downstream)**: `#19` (baseline eval), `#23` (solver), `#25` (SFT runbook)

---

## 1. Objective

Compare four external reference implementations (Tong, konbu17, Kishan, aitherium) against plan_v0.2 assumptions to extract transferable ideas for masking, augmentation, trajectory capture, and distillation. Produce a delta matrix documenting {approach, dataset, masking, eval, outcome} for each source with adoption decisions and implementation risk/cost scores. Output a prioritized decision log as `experiments/external_review_matrix.csv`.

## 2. Why It Matters

- **Leaderboard impact**: Public winners' techniques (ConstitutionalAI masking, synthetic trajectory augmentation) may transfer to private test set; incorrectly copying them risks overfitting or license friction.
- **Capstone learning**: Comparing external code teaches pattern recognition in real LoRA pipelines versus theory.
- **Downstream coupling**: Notebooks 04, 07, 09 depend on this review to lock masking strategy and eval rules before training loops begin.

## 3. Strategy — How We Aim To Accomplish It

1. **Clone or skim each repository** (Tong via GitHub, konbu17 + Kishan via Kaggle snapshot API, aitherium via blog summary) and extract {base model, dataset version, masking method, eval metric, final score} for each.
2. **Tabulate in a delta matrix** (4 rows × 6 columns: source, approach, dataset, masking, eval, outcome).
3. **Map each finding to plan_v0.2 phase** (Phase 1: baseline prompt & eval; Phase 4: LoRA + SFT masking rules).
4. **Score relevance (1–5), implementation risk (1–5), re-implementation cost in hours (1–40)** for each cell; rank by (relevance − 0.5 × risk) / cost.
5. **Record adoption decisions** with ≤2-sentence justification and upstream citation; mark "adopt," "adapt," or "reject" with downstream notebook mapping.

## 4. MVP (Minimum Viable Notebook)

- **Inputs**: GitHub/Kaggle snapshots of Tong, konbu17 repos; blog link for aitherium; public Kishan dataset metadata.
- **Cells**:
  1. Environment check and repo clone/API fetch
  2. Extract config/training script signatures from each source
  3. Build delta matrix (markdown table, then CSV export)
  4. Score and rank rows by adoption priority
  5. Write decision log with citations
  6. Commit `experiments/external_review_matrix.csv` and matrix markdown cell output
- **Outputs**: `experiments/external_review_matrix.csv`; inline markdown table; decision log as notebook cell text.
- **Verification**: Every row has dated URL + plan-phase mapping; every "adopt" row lists downstream notebook ID.

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Tong repo cloneable from GitHub; konbu17 notebook publicly readable; aitherium blog accessible; Kishan dataset visible on Kaggle.
- **Action**: Cells fetch repo metadata, extract masking/eval approach from each source's training script or markdown README.
- **Expected**: Delta matrix with all 4 rows populated, each containing {source name, masking strategy, eval metric, outcome score, adoption decision}; no empty cells in "approach" or "masking" columns.

### 5.2 Alternative / Fallback

If a source code becomes private (konbu17 Kaggle notebook deleted, Kishan dataset restricted):
- **Setup**: Archive links or PDF snapshots available from prior Kaggle snapshot API call.
- **Action**: Cells extract decision from description-level metadata (public notebook metadata API, blog post summary, arXiv abstract) instead of source code inspection.
- **Expected**: Row marked "evidence incomplete"; masking/approach columns filled from abstract or notebook title/description only; decision logic explicitly flags as "inference from public claim only."

### 5.3 Regression Guardrails

- **Golden-set integrity**: No "adopt" decision silently changes plan_v0.2 baseline eval rules (exact-match normalization, prompt template); changes require a new GitHub issue with `[BREAKING]` label.
- **Dataset leakage check**: No row references private or contest-internal datasets as adoption source; Kishan public dataset is OK; any internal Kaggle-only data marked as "risk: leakage."
- **Citation freshness**: Every URL in the matrix must be dated or linked to a specific commit SHA / Kaggle kernel version timestamp; dead links logged with mitigation step.

## 6. Success Criteria (Done When)

- [ ] All 4 external sources (Tong, konbu17, Kishan, aitherium) have row entries in the delta matrix.
- [ ] Every row has a dated URL citation, plan-phase mapping, and adoption decision with ≤2-sentence rationale.
- [ ] At least one "adopt" decision with downstream notebook ID; at least one "reject" decision with risk justification.
- [ ] `experiments/external_review_matrix.csv` committed with columns: `source | approach | dataset | masking_strategy | eval_metric | outcome | adoption_decision | downstream_notebook | risk_score | cost_hours | decision_rationale | citation_url | citation_date`.
- [ ] No breaking changes to plan_v0.2 baseline assumptions without a corresponding GitHub issue.
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk: Leaderboard overfitting** | **Mitigation**: Flag any technique that directly copies public winner masking as "high specificity risk"; default to reject unless re-implementation cost ≤ 4 hours and relevance ≥ 4/5.
- **Risk: License incompatibility** | **Mitigation**: Audit Tong repo LICENSE file; if GPL or incompatible with repo license, mark adoption as "forbidden" and document in decision log.
- **Risk: Kaggle dataset restricted** | **Mitigation**: Check Kishan dataset access level before execution; if access denied, switch to fallback (metadata-only delta, mark "evidence incomplete").
- **Open question: Does Tong's ConstitutionalAI masking generalize to private test set?** | **Who answers**: Phase 4 checkpoint; compare Phase 1 baseline vs. Phase 4 SFT with adopted masking on golden set.

## 8. Artifacts & Handoff

- **Produces**:
  - `experiments/external_review_matrix.csv` — main delta table
  - Inline markdown table in notebook cell (rendered for quick review)
  - Decision log summary (notebook markdown cell)
- **Consumed by**:
  - `#19` (04_baseline_eval): Confirms eval metric alignment from Tong/konbu17
  - `#23` (07_solver): Trajectory augmentation patterns from Kishan dataset analysis
  - `#25` (09_sft_runbook): Masking strategy and distillation safeguards from aitherium review
- **External references cited**:
  - https://github.com/tonghuikang/nemotron (commit SHA or release tag at review date)
  - https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2 (kernel version / last edit date)
  - https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj (dataset version / last update date)
  - https://aitherium.com/blog/nemotron-reasoning-challenge-mirothinker-distillation/ (blog publish date)

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
