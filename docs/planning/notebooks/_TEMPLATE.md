# Notebook NN: <Title>

**Parent Issue**: `#XX`
**Plan Phase**: <phase name from docs/planning/plan_v0.2.md>
**Scaffold**: `notebooks/NN_<slug>.ipynb`
**Status**: `planned` | `scaffolded` | `active` | `validated` | `superseded`
**Dependencies (upstream)**: `NN`, `NN`
**Consumers (downstream)**: `NN`, `NN`

---

## 1. Objective

One paragraph. What decision, artifact, or learning does this notebook exist to produce? Written so a reviewer who has never opened the notebook can state the goal in one sentence after reading it.

## 2. Why It Matters

Two-to-four bullets. Tie the objective back to (a) competition leaderboard, (b) the capstone learning goal, (c) an upstream/downstream notebook that needs this artifact.

## 3. Strategy — How We Aim To Accomplish It

Ordered steps, 4-8 bullets. Keep each step executable in a single cell or short cell block. Name the concrete tool (e.g., `transformers.AutoModelForCausalLM`, `nemo_curator`, `trl.GRPOTrainer`, a specific dataset id) wherever possible.

## 4. MVP (Minimum Viable Notebook)

The smallest end-to-end version that counts as "done" for a first committable pass. Scope it so one focused session (3-6 hours) finishes it.

- **Inputs**: datasets, configs, prior artifacts
- **Cells**: minimal cell list (numbered)
- **Outputs**: files written under `data/`, `configs/`, `experiments/`, or WandB runs
- **Verification**: one concrete assertion or printed metric that proves success

## 5. Test Cases

Document the test matrix the notebook must pass. Each case states Setup -> Action -> Expected.

### 5.1 Primary (MVP path)

- **Setup**: <preconditions, env, data files>
- **Action**: <what the cells do>
- **Expected**: <assertion / printed value>

### 5.2 Alternative / Fallback

What we run if the primary path is blocked (data unavailable, GPU OOM, vendor API down, numerical mismatch). Same Setup/Action/Expected format.

### 5.3 Regression Guardrails

What must not break when this notebook is re-run or edited. Typically the golden-set or a frozen-schema check.

## 6. Success Criteria (Done When)

- [ ] ...
- [ ] ...
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`
- [ ] WandB run (if applicable) tagged and linked

## 7. Risks & Open Questions

- **Risk**: ... | **Mitigation**: ...
- **Open question**: ... | **Who answers**: ...

## 8. Artifacts & Handoff

- **Produces**:
  - `data/...`
  - `configs/...`
  - `experiments/...`
- **Consumed by**: notebook(s) NN, NN
- **External references cited**: <links>

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| MVP | | |
| Alternative path | | |
| Full polish | | |
