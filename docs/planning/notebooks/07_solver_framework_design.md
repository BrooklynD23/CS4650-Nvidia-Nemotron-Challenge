# Notebook 07: Solver Framework Design (Category-Aware, with Fallback)

**Parent Issue**: `#23`
**Plan Phase**: Phase 4 (SFT targeting) informs; Phase 8 (submission composition) consumes
**Scaffold**: `notebooks/07_solver_framework_design.ipynb`
**Status**: `planned`
**Dependencies (upstream)**: `#15` (review harness), `#22` (trajectories/error slices)
**Consumers (downstream)**: `#24` (synthetic data), `#25` (SFT runbook), `#20` (submission)

---

## 1. Objective

Specify a category-aware solver interface with a fallback policy for low-confidence cases. The competition benchmark is treated as category-specific rule induction, so the solver contract is: given (question, category), return (answer, trace, confidence). Each category gets a dedicated adapter configuration and decode settings; a fallback policy (retry with alternative prompt → increase N → accept last) handles edge cases. Produce `src/inference/solver.py` (typed stubs), `configs/solver_routing.yaml`, and `docs/architecture/SOLVER_DESIGN.md`.

## 2. Why It Matters

1. **Competition leaderboard**: A robust solver framework ensures consistent answer extraction and prevents format collapse under RL or synthetic data misalignment.
2. **Capstone learning**: This notebook demonstrates how to structure a reasoning system with explicit category awareness—key for domain-specific fine-tuning evaluation.
3. **Upstream dependencies**: Consumes error slice taxonomy from `#22` and validation harness from `#15` to define routing rules and test fallback paths.
4. **Downstream dependencies**: `#25` (SFT runbook) uses the solver interface to score synthetic data; `#20` (submission) uses the fallback policy to guarantee valid answers at submission time.

## 3. Strategy — How We Aim To Accomplish It

1. **Enumerate categories** from `#17` (EDA) and `#22` (trajectory slices); confirm category count and schema.
2. **Define the `Solver` protocol** as a Python `typing.Protocol` with methods `solve(question: str, category: str) -> SolverOutput` and configuration keys for per-category decode settings.
3. **Design routing logic**: category → `(model_id, decode_config, post_process_fn)` tuple; validate routes cover all categories or have a default fallback.
4. **Pick a confidence signal**: use `prob_of_boxed_answer` (presence of valid `\boxed{}` format) + `agreement_across_best_of_N` (how many of top-5 samples agree on final answer).
5. **Spec the fallback ladder**: (1) retry with alternative system prompt, (2) increase N and re-sample, (3) accept last valid answer or raise.
6. **Create stubs and config**: write type-hinted `solver.py` that type-checks with `mypy --strict`; generate YAML routing config from category list.
7. **Document rationale** in `SOLVER_DESIGN.md` with decision dates, category justifications, and regression risk mitigations.

## 4. MVP (Minimum Viable Notebook)

**Inputs**:
- Category list from `#22` (or fallback: assume ["math", "code", "science"] if #22 incomplete)
- Golden set from `#18` for validation
- Base model `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (already loaded from Phase 1)

**Cells** (7–9 cells):
1. Load category taxonomy; confirm count and names.
2. Define `SolverOutput` TypedDict and `Solver` Protocol with docstrings.
3. Implement `CategoryRouter` class that maps category → decode config (temperature, top_p, max_tokens).
4. Implement confidence scorer with `prob_of_boxed` and `agreement_score` functions.
5. Implement fallback policy as a state machine: retry_count, fallback_stage, action.
6. Create dummy solver (round-trip test): instantiate router, score 2–3 dummy problems per category.
7. Export solver.py stub, routes.yaml, and decision document.

**Outputs** (files written):
- `src/inference/solver.py` (stubs + Protocol + Router class)
- `configs/solver_routing.yaml` (category → decode config mapping)
- `docs/architecture/SOLVER_DESIGN.md` (decision write-up)

**Verification**: Print that solver stub type-checks (`mypy --strict src/inference/solver.py` must pass) and that YAML parses without error; golden_20 invocations through the solver should return identical boxed answers as baseline direct model calls (regression check).

## 5. Test Cases

### 5.1 Primary (MVP path)

**Setup**:
- Category list available (from notebook `#17` or hardcoded fallback: ["math", "code", "science"]).
- Base model loaded from Phase 1 (cached locally or on Colab).
- Golden set loaded: ≥ 10 problems across ≥ 2 categories.

**Action**:
- Instantiate `CategoryRouter` with default config.
- For each category in the list, generate answer to a golden problem via the solver.
- Call `confidence_score(answer, category)` on the output.
- Invoke fallback policy with confidence < 0.3 (simulate low-confidence case).

**Expected**:
- `solver.py` passes `mypy --strict` with no errors.
- `solver_routing.yaml` parses as valid YAML with ≥ 3 category entries.
- For each category, solver returns `SolverOutput` with `answer`, `confidence`, `trace`, and `metadata` fields.
- Golden problems return identical boxed answers on re-run (deterministic with seed=42).

### 5.2 Alternative / Fallback

**Setup**:
- If category count from `#22` is ≤ 2 (unlikely but possible).

**Action**:
- Simplify router to a single-path solver with only decode-config variation (no per-category adapter specialization).
- Document the simplification in `SOLVER_DESIGN.md` with rationale: "Category count too low to warrant per-adapter routing; decode settings provide sufficient variation."

**Expected**:
- Solver still type-checks and produces valid output.
- Single route in `solver_routing.yaml` labeled `"default"` with fallback config.

### 5.3 Regression Guardrails

**Setup**:
- Golden set loaded; baseline inference results cached from Phase 1.

**Action**:
- Run solver on golden_20 with deterministic generation (temperature=0.0, seed=42).
- Extract boxed answers and compare to Phase 1 baseline.

**Expected**:
- 100% match rate: all golden_20 boxed answers from the solver match the baseline (solver introduces zero regression).
- No format errors in any output (all answers wrapped in `\boxed{...}`).

## 6. Success Criteria (Done When)

- [ ] Category taxonomy enumerated and confirmed in notebook; category count logged.
- [ ] `src/inference/solver.py` created with `Solver` Protocol, `SolverOutput` TypedDict, `CategoryRouter` class, and confidence scoring functions (all with docstrings and type hints).
- [ ] `src/inference/solver.py` passes `mypy --strict` with zero errors.
- [ ] `configs/solver_routing.yaml` created with entries for all categories (or explicit `"default"` if count ≤ 2).
- [ ] Fallback policy documented as a decision tree (retry → N-increase → accept) with concrete code stubs.
- [ ] `docs/architecture/SOLVER_DESIGN.md` written with: category justifications, routing rules, confidence estimator design, fallback policy, and risk mitigations.
- [ ] Golden set regression check passes: 100% match on boxed answers (solver adds no regression).
- [ ] Two dummy categories successfully round-trip through solver interface (instantiate, score, verify output shape).
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`.

## 7. Risks & Open Questions

- **Risk**: Over-engineering routing when a single adapter may work. | **Mitigation**: Start with default fallback; if category count is ≤ 2, collapse to single-path solver and document in rationale section.
- **Risk**: Category taxonomy mismatch between training data (#22 error slices) and test set. | **Mitigation**: Verify category list against both #17 (EDA) and #22 (slices); if mismatch, document as "Open" in decision write-up and revisit in Phase 4 (SFT).
- **Risk**: Confidence estimator miscalibration (false positives or high rejection rate). | **Mitigation**: Use two complementary signals (`prob_of_boxed` + `agreement`); log empirical confidence distribution on golden_20 and validate fallback is only triggered for genuinely low-confidence cases.
- **Open question**: How many categories does #22 discover? | **Who answers**: Issue #22 author; clarify in opening cell of notebook 07.
- **Open question**: Should per-category adapters specialize, or should decode settings be the only axis of variation? | **Who answers**: Phase 4 (SFT) results; if SFT shows no per-category loss divergence, collapse to single adapter with category-aware decode config only.

## 8. Artifacts & Handoff

**Produces**:
- `src/inference/solver.py` — Protocol definition, routing logic, confidence scoring, fallback policy
- `configs/solver_routing.yaml` — Category → decode config mapping (temperature, top_p, max_tokens, max_new_tokens)
- `docs/architecture/SOLVER_DESIGN.md` — Design rationale, decision dates, risk mitigations, and open questions

**Consumed by**:
- Notebook `#24` (synthetic data): uses `Solver` interface to verify synthetic examples are valid before including in training
- Notebook `#25` (SFT runbook): uses category-aware routing to apply per-category loss masks and adapter config
- Notebook `#20` (submission): uses fallback policy to ensure final submission never has unanswerable questions

**External references cited**:
- `#14` (competition rules): verify answer format contract (`\boxed{}`)
- `#17` (EDA): category schema and distribution
- `#22` (error slices): category-specific failure modes and taxonomy refinement
- `#18` (golden set): regression guardrails
- `docs/architecture/ARCHITECTURE.md`: "Future solver contract" section

## 9. Estimated Effort

| Step | Hours | Hardware |
|---|---|---|
| MVP (enumerate categories, write Protocol stubs, create dummy routes) | 3 | Laptop + cached model |
| Confidence scoring design + fallback policy codification | 2 | Laptop |
| Regression testing on golden_20 | 1 | Colab (if local runs slow) |
| Decision write-up (`SOLVER_DESIGN.md`) | 1.5 | Laptop |
| **Total** | **7.5** | Minimal GPU |
