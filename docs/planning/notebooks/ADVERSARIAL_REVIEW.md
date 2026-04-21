# Adversarial Review — Notebook Plans 00-10
Date: 2026-04-20
Reviewer: Codex (adversarial)
Scope: docs/planning/notebooks/*.md

---

## 1. Verdict Summary

**Ship-with-fixes — but with several blockers that make the current set un-executable without prior
agreement on contested facts.**

The planner produced eleven structurally consistent plans that follow the template faithfully and
demonstrate genuine awareness of upstream/downstream coupling. That is not nothing. However,
structural consistency is not the same as executability. Four compounding problems threaten the
whole stack: (1) the eval contract — what "correct" means — is deliberately deferred to notebook 00,
but notebook 00 has no mechanism to actually determine it, only to record whatever Kaggle's public
page says; if that page is silent, the contract stays open and every downstream test case is built
on a void; (2) the golden-set construction in plan 03 runs base-model inference five times, which
means it will take 2–4 GPU-hours on an RTX 3080 and may silently fail the entropy requirements
if the base model is already near-perfect on easy problems; (3) plan 06 references
`data/processed/training_curated.jsonl` — a file that is produced by a curation pipeline
described nowhere in these plans, not in plan 02 (EDA only), not in plan 03, not in plan 08
(synthetic); the notebook that would produce it (a data-curation notebook) is simply missing from
the 11-plan set, and plan 06 cannot run without it; (4) the provenance chain from plan 08 (teacher
provenance fields) through plan 09 (SFT data loader) through plan 10 (submission manifest) is
described in words but the field names are never agreed on across plans, so integration will
require manual negotiation that will break any automation.

Plans 00, 02, 04, 06, and 09 have the most structural problems. Plans 03, 05, and 10 are in
reasonable shape and could ship with minor fixes. Plans 01, 07, and 08 are in the middle tier:
conceptually sound, but contain execution blockers that require targeted repair.

---

## 2. Blockers (must-fix before any plan executes)

**B1 — Plan 06 references a file that no plan produces.**
`06_trajectory_collection_and_error_slices.md` §4 Inputs lists
`data/processed/training_curated.jsonl` as a required input. This file is produced by a
data-curation pipeline. Plan 02 does EDA only. Plan 03 reserves eval sets. Plan 08 generates
synthetic data. None of these plans write a curated training file. The curation pipeline from
plan_v0.2.md Phase 3.2–3.3 (NeMo Curator, deduplication, quality filtering) has no corresponding
notebook plan in this 11-plan set. Before plan 06 can execute, either (a) a notebook 00b / 02b
for data curation must be added, or (b) plan 06 must be updated to accept a raw dataset slice
instead of the curated file, with explicit instructions for producing it.

**B2 — The eval contract is deferred to plan 00, which has no mechanism to resolve it from first
principles.**
Plans 04, 07, 09, and 10 all state they implement "boxed extraction via the plan_v0.2 regex."
Plan 04 §5.2 Alternative says "if #14 reveals strict exact-match without boxed extraction, add
`strict_em` branch." Plan 04 §7 Open Questions says "does competition use strict exact-match or
boxed extraction? Who answers: #14." But plan 00 §3 Strategy says only "fetch Kaggle competition
page" and "scan forum posts." Kaggle's competition pages routinely leave scoring details
ambiguous. If the page is silent on boxed vs. plain text, plan 00 will record confidence="Low"
and proceed. Every downstream plan will then inherit an unresolved branching condition and will
have to maintain both code paths forever. The planner must decide: either commit to boxed
extraction as the default now (because the submission demo notebook uses it, which is actually
the authoritative signal), or write an explicit decision rule into plan 00 that resolves the
branch using the demo notebook rather than waiting for the Kaggle rules page.

**B3 — Plan 03 does not specify a candidate pool for golden-set selection, making the 20-problem
guarantee unreachable.**
Plan 03 §4 MVP Cell 2 says "stratified sample 200 problems from eligible held-out source data."
It does not name the source. Plan 03 §4 Inputs says "eligible source data (e.g., Puzzle-KD
Dataset v2, competition benchmark problems, publicly sourced math/code problems — NOT
training-set samples)." The word "eligible" is doing enormous work here. If the only available
sources are the Puzzle-KD dataset (851K, reasoning=off) and public competition problems, and if
those problems must be disjoint from training data, the student needs to have already decided on
their training set before they can define eligibility. But plan 03 runs in Wave B, before any
training-data decisions are locked. Without a named, concrete source with a known schema, plan 03
Cell 2 cannot be automated. This blocks everything downstream from plan 03 (plans 04–10).

**B4 — Plan 09 runs a 100-step smoke train on an RTX 3080 (10 GB) with QLoRA, but provides no
estimate of whether this is feasible within the VRAM budget.**
Plan 09 §5.1 Primary says "Setup: RTX 3080 (10GB), Unsloth installed, 1k curated samples, base
model cached." The base model is 4B parameters in BF16 = ~8 GB just for weights. With QLoRA 4-bit
(bitsandbytes), the quantized model is ~2 GB, but optimizer states, activations, and gradient
accumulation during a 100-step training loop push peak VRAM to 9–12 GB in practice (based on
plan_v0.2 §4.4 which estimates "batch_size=16-32, gradient_accumulation=8" for RTX 3080 QLoRA).
Plan 09 gives no specific batch size for the smoke run, no max_seq_length cap, and no assertion
about whether the smoke run will fit. If it OOMs mid-run, the "100-step smoke-run sign-off"
produces no adapter and blocks plan 10. The plan must include explicit VRAM budget arithmetic or
reference the OOM ladder before the smoke run cell.

---

## 3. Major Issues (should-fix before the notebook is opened)

**M1 — Plan 02 §4 Verification is not mechanically checkable.**
"Printed schema table ≥ 5 sources × 6 fields ... no assertion, human review." The README
authoring conventions (which this planner wrote) state "Every Success Criteria item must be
mechanically verifiable." Plan 02 explicitly carves out a human-review escape. This propagates:
if the schema is wrong and only caught by a human reviewer who skims it, plans 04, 06, 08, and 09
will all inherit silent field-name mismatches. The fix is one assert statement:
`assert set(schema.keys()) >= REQUIRED_FIELD_SET`.

**M2 — Plan 04 effort estimate is 6.5–7.5 hours, but the template says MVP = "3–6 hours".**
The template (§4 MVP) specifies "one focused session (3–6 hours)." Plan 04 §9 estimates 6.5–7.5
hours MVP. This overflows by 25%. The primary path includes implementing normalize.py,
harness.py, unit tests, baseline on golden_20, and 3-seed eval on validation_200 (200 × 3 = 600
inferences). 600 inferences of a 4B model at ~2 tokens/second on RTX 3080 is ~50 minutes of
generation, not accounting for prompt encoding. The effort estimate is plausible but plan 04 needs
to either (a) scope the MVP down to normalization + smoke test only, deferring full 3-seed eval
to a follow-on cell, or (b) acknowledge the template violation and justify it.

**M3 — Plan 05 §5.1 Primary Expected states "Exactly one row beats baseline with delta > 2σ."**
This is a false precondition masquerading as an expected result. The test case asserts a scientific
hypothesis ("at least one strategy will be better") as a boolean pass/fail gate. If every strategy
is statistically equal to or worse than baseline — a plausible outcome for a model at 95.4%
MATH500 near ceiling — the test "fails" even though nothing is wrong with the notebook execution.
A test case should assert mechanical properties of the output (correct schema, completed inference,
values in range), not assert experimental outcomes. The expected line should read: "Table ≥8 rows,
each with accuracy_mean ∈ [0,1], accuracy_std ≤ 0.1, delta_vs_baseline filled; statistical
significance flag populated."

**M4 — Plan 06 hard-codes expected accuracy numbers in the test case.**
Plan 06 §5.1 Primary Expected contains: "Printed summary: 'Correct: 752/1200 (62.7%) | Incorrect:
448/1200 (37.3%)'" and "Top-3 error types: e.g., format_miss (156), arithmetic_slip (142)." These
are not expected values — they are the planner's guesses about what the base model will do.
Including them as expected values means the test case will "fail" unless the model happens to
achieve exactly 62.7%. The expected section must be replaced with mechanically verifiable
assertions only: "JSONL has 1,200 rows," "every row has required keys," "at least 3 named error
types with ≥5 examples each." The example distribution must be labeled "illustrative only."

**M5 — Plan 07 references upstream `#15` (review harness) but plan 15 is not in the 11-plan set.**
Plan 07 §1 header: "Dependencies (upstream): `#15` (review harness), `#22` (trajectories)."
Issue #15 is defined in SPRINTS.md as "Agent and human review harness (`.github/ISSUE_TEMPLATE/*`,
execution doc)." It is not a notebook plan and does not appear in this folder. However, plan 07
§4 Inputs says "base model already loaded from Phase 1" and makes no use of any review harness
artifact. The `#15` dependency appears to be a copy-paste from SPRINTS.md that is not actually
used. If #15 produces nothing that plan 07 consumes, remove it. If it does (e.g., a golden_20
hash manifest), name what is consumed explicitly.

**M6 — Plan 08 §4 MVP says the pilot generates 500 samples from DeepSeek-R1 "via HPC or Colab."
DeepSeek-R1 (671B MoE) cannot run on Colab Pro A100 (40 GB).**
The full DeepSeek-R1 model requires ~350 GB+ of VRAM. The plan_v0.2 §5.1 explicitly names
"DeepSeek-R1 / R1-0528" under "HF (free, self-hosted)" and lists it as requiring only "GPU time."
This is technically true for HPC with multi-GPU, but plan 08 §9 Effort lists "HPC (optional) or
Colab" for the 3-hour pilot run. Colab Pro A100 is 40 GB — far insufficient. The plan must
either (a) use a distilled DeepSeek-R1 variant (e.g., DeepSeek-R1-Distill-Qwen-32B, which fits in
~64 GB and could run on multi-GPU HPC), or (b) default to the SYNTHETIC-1 fallback and note that
self-hosting the full R1 requires HPC multi-GPU with explicit setup instructions, not "optional."

**M7 — Plan 09 §5.1 Primary Expected "final loss ~0.5–1.0 (regression-free vs baseline)" is not a
regression check.**
A smoke run of 100 steps on 1k samples is nowhere near convergence. Final loss after 100 steps
depends entirely on initial loss (which varies by model and data distribution). Stating "~0.5–1.0"
as an expected value implies either (a) the planner has run this before and knows the range, or
(b) the planner invented a range. Neither constitutes a regression check. The mechanically
verifiable criterion is: "loss at step 100 < loss at step 1" (descending), which is what
plan_v0.2 §4 "Done When" item 1 actually states ("final loss < initial loss by >50%"). Replace
the absolute range with a relative check.

**M8 — Plan 10 §5.2 Alternative drops the golden_20 regression threshold from 20/20 to 19/20
("allow 1 question drop from stochasticity") with no justification.**
Plan 03 §5.3 Regression says "golden_20 must be 20/20; any drop is regression and blocks
submission." Plan 04 §5.1 says "golden_20 accuracy = 100% (or investigation required)." Plan 09
§5.3 says "never drop below 20/20." Then plan 10 §5.2 quietly lowers the bar to 19/20. There is
no cited reason for the relaxation. "Stochasticity" is not an acceptable reason for a regression
guardrail relaxation — stochasticity is controlled by setting temperature=0. Either plan 10 should
hold the 20/20 standard, or it needs to document a formal policy change and retroactively update
plans 03, 04, and 09.

**M9 — Plan 01 success criteria require "at least one adopt decision" as a checkpoint gate, but
this is a tautology that can always be satisfied trivially.**
Plan 01 §6 Success Criteria: "At least one 'adopt' decision with downstream notebook ID; at least
one 'reject' decision with risk justification." Any analysis of 4 sources where the analyst
decides "I'll adopt anything" trivially satisfies this. The gate should be: "At least one 'adopt'
decision where the downstream notebook ID is confirmed by that notebook's plan maintainer as
accepting the change" — or, more practically, the gate should require specific adoption targets
(e.g., masking strategy confirmed, eval metric confirmed) rather than existence of any adoption.

---

## 4. Minor Nits

**Schema / naming inconsistencies:**
- Plan 02 §6 produces `src/data/schema.py` with `ReasoningExample` dataclass. Plan 09 §4 MVP
  inputs says "1k curated samples from #24." Plan 08 §3 says output is in "canonical schema with
  provenance fields." None of these three plans import `ReasoningExample` from `src/data/schema`.
  The planner says schema.py is the canonical contract but never enforces it downstream.
- Plan 06 §8 produces `data/analysis/trajectories_<date>.jsonl` but §4 defines the output schema
  as `{input, raw_output, extracted_answer, correct, error_type, token_length}`. Plan 07 §4
  Inputs does not list this file (it lists golden_20 from #18 but not the trajectories). Plan 08
  §3 Strategy Step 2 references `data/errors/category_slices.jsonl` — a different path than plan
  06's stated output path. Path mismatch.

**Effort estimate inconsistencies:**
- Plan 02 total effort is 7–11 hours. Plan 08 total is 11 hours MVP. Both exceed the 3–6 hour
  template target. Plans 02 and 08 are the most GPU-intensive plans in Wave B and D respectively;
  the template 3–6 hour target was clearly written for CPU-only plans. Either the template or the
  plans need to acknowledge this distinction.

**Circular dependency in plan 08 provenance logic:**
- Plan 08 §3 Step 6 says "check for overlap with curated data from notebook 03." Notebook 03
  produces the golden set and validation set, not curated training data. The correct reference
  should be "notebook 02" (EDA/schema) or the missing curation notebook. This suggests plan 08
  was written with a mental model of a different notebook numbering than the final set.

**Plan 03 success criteria are pre-checked:**
- Plan 03 §6 shows all success criteria marked with `[x]` (checked). These are planning artifacts
  that have not been executed. Pre-checking boxes in a planning document undermines the gate
  function of the checklist entirely. The planner probably copy-pasted from a "what would done
  look like" draft and forgot to uncheck.

**Plan 04 has a duplicate "Artifact(s) committed" line in §6:**
- Plan 04 §6 Success Criteria lists both "Artifact(s) committed and linked in
  `docs/execution/NOTEBOOKS.md`" and "WandB run (if applicable) tagged and linked" as separate
  items — this matches the template's final two lines. But above them, it also includes "WandB run
  tagged `baseline` and linked in docs/execution/NOTEBOOKS.md" as an earlier item. The WandB
  criterion appears twice with slightly different wording.

**Plan 06 §4 Verification uses a checkmark emoji (`✓`):**
- This is cosmetically inconsistent with all other plans which use plain text. Minor, but signals
  the plan was written by a different sub-agent with different defaults.

**Plan 07 §2 lists "Upstream dependencies" as #15 and #22, but the dependency graph in README.md
shows 07 depends on #15 and #22, while plan 07 §1 header also lists #15 and #22 — consistent.
However, README.md dependency graph shows 07 also feeds 10 directly, but plan 10 §1 header lists
dependencies as only `#14` and `#19` — 07 is not listed. This means plan 10 implicitly uses
solver.py from plan 07 (for format checking) without declaring the dependency.**

---

## 5. Per-Plan Scrutiny

### 5.00 Competition Constraints and Rules

- **Lens 1 (Objective falsifiability):** WEAK — "Verify and freeze all official Kaggle competition
  constraints" is measurable only if every field has an authoritative source. But the plan's own
  §7 Risks table acknowledges "rumored LoRA rank cap unverified." If the rank cap stays at
  confidence="Low" (which is likely given the Kaggle rules page is sparse), the plan ends with an
  open constraint, not a frozen one. "Freeze" implies closure; the plan should be called "snapshot
  with confidence ratings" if that's what actually happens.

- **Lens 2 (Strategy concreteness):** WEAK — Strategy bullets 4 and 5 ("Review submission demo
  notebook" and "Scan recent Kaggle forum posts") name no concrete tool, API, or code path.
  Kaggle's forum API is unauthenticated JSON; at minimum the plan should show a `requests.get`
  target URL. Without this, "scan forum posts" is manual work that takes unbounded time.

- **Lens 3 (MVP executability):** PASS — 2 hours for fetch+verify+export on local+browser is
  realistic. The Kaggle dataset API is well-documented and the HF model card is JSON-queryable.

- **Lens 4 (Test-case teeth):** WEAK — §5.1 Expected: "Constraint table with all
  downstream-critical fields filled with non-null values and citations." This is a vibe, not an
  assertion. A mechanically checkable version would be:
  `assert set(constraints.keys()) == REQUIRED_FIELDS` where `REQUIRED_FIELDS` is defined in the
  notebook. The "Alternative" §5.2 fallback says "proceed with provisional values if snapshots are
  ≤1 week old" but does not assert a freshness check on the snapshot date.

- **Lens 5 (Alternative realism):** PASS — The fallback (cached COMPETITION.md) is specific and
  bounded. The staleness threshold (7 days) is concrete.

- **Lens 6 (Dependency honesty):** FAIL — The plan states it produces the eval contract that all
  downstream plans depend on, but it does not name the mechanism by which an ambiguous page
  outcome gets resolved. If Kaggle's page is silent on exact-match vs. boxed, the plan will mark
  the field as Open and move on. All 10 downstream plans will then inherit an unresolved contract.
  This is the core blocker B2.

- **Lens 7 (Artifact discipline):** PASS — Output paths (`data/eval/competition_constraints.json`,
  `.csv`, `.txt`) are stable and version-safe. No mutation concern.

- **Lens 8 (Risk honesty):** WEAK — The plan mentions "rumored LoRA rank cap" but does not
  mention the most concrete risk: the submission demo notebook itself may be the authoritative
  source for submission format (not the rules page), and if the demo notebook changes between
  plan writing and execution, the constraints are stale. The plan should fetch the demo notebook's
  last-modified timestamp.

- **Top unaddressed risk:** If the competition scoring uses a private test set that includes
  non-math categories (science, code, logic), and plan 00 only captures the public benchmark
  categories, then every category assumption downstream is wrong. Plan 00 should explicitly
  attempt to infer hidden test set composition from the public dataset and forum discussion.

- **Verdict: ship-with-fixes** (B2 blocker, Lens 6)

---

### 5.01 External Baselines and Design Deltas

- **Lens 1 (Objective falsifiability):** PASS — "Delta matrix documenting {approach, dataset,
  masking, eval, outcome} for each source with adoption decisions" is concrete. You can fail it
  if the matrix has empty cells.

- **Lens 2 (Strategy concreteness):** WEAK — "Clone or skim each repository (Tong via GitHub,
  konbu17 + Kishan via Kaggle snapshot API, aitherium via blog summary)" — the Kaggle "snapshot
  API" is not a real API name. Kaggle has a `kaggle kernels pull` command and a datasets API;
  the planner should name one.

- **Lens 3 (MVP executability):** WEAK — 5.5 hours for cloning 4 repos, extracting configs,
  building a matrix, scoring, and writing a decision log is aggressive but plausible on CPU.
  The real risk: aitherium blog may not expose a training script — only a narrative description.
  The plan acknowledges this in the fallback but does not address whether the fallback produces
  enough signal to make an informed adoption decision.

- **Lens 4 (Test-case teeth):** WEAK — §5.1 Expected: "no empty cells in 'approach' or 'masking'
  columns." This is the sharpest assertion in the test case and is genuinely checkable. However,
  there is no assertion about the content quality — a cell containing "unknown" or "N/A" satisfies
  the non-empty check. The regression guardrail in §5.3 ("no 'adopt' decision silently changes
  plan_v0.2 baseline eval rules without a [BREAKING] issue") is good but unenforceable
  mechanically; it requires human review.

- **Lens 5 (Alternative realism):** PASS — The metadata-only fallback is specific. "Evidence
  incomplete" label with justification is better than most plans' fallbacks.

- **Lens 6 (Dependency honesty):** PASS — The plan correctly identifies that this notebook has
  no data pipeline dependencies, only external URLs. The `#13` template dependency is trivial.

- **Lens 7 (Artifact discipline):** PASS — `experiments/external_review_matrix.csv` with 13
  named columns is specific. The column list is the clearest artifact spec in any of the 11 plans.

- **Lens 8 (Risk honesty):** WEAK — The plan lists license incompatibility and Kaggle access
  as risks but misses the largest risk: the external notebooks may have been tuned to a specific
  version of the competition data or an earlier evaluation protocol. Adopting their masking or
  eval approach may be retroactively invalid if the competition rules changed between when they
  published and now. The plan should cross-check each source's publication date against constraint
  snapshot date from plan 00.

- **Top unaddressed risk:** Tong's repo targets an earlier Nemotron architecture or a different
  tokenizer version. If the chat template changed between the repo's last commit and the current
  model card, the "adopted" masking rules are silently wrong. Plan 01 should assert: "each source
  references the same base model ID and tokenizer version as plan_v0.2; reject if mismatched."

- **Verdict: ship-with-fixes** (M9)

---

### 5.02 Dataset Schema and EDA

- **Lens 1 (Objective falsifiability):** PASS — "Define canonical data schema... produce field
  inventory (types, null rates, token-length distributions, category enums)" is concrete. You can
  fail it if schema.json has fewer than 6 fields per category.

- **Lens 2 (Strategy concreteness):** PASS — Strategy names `datasets.load_dataset(streaming=True)`,
  `transformers.AutoTokenizer`, specific dataset IDs, specific metric (95th percentile ≤ 8192).

- **Lens 3 (MVP executability):** WEAK — §4 Inputs: "50GB free disk for 10k-row download." A 10k
  row download from Llama-Nemotron Post-Training (130 GB total) via streaming does not require
  50 GB of disk — streaming mode buffers a few hundred MB at most. The 50 GB estimate suggests the
  planner is confused between downloading the full dataset and streaming a slice. This overstates
  the disk requirement by 200x and could cause a student to abort the notebook when they see the
  50 GB requirement and their disk shows 30 GB free, even though streaming mode would work fine.

- **Lens 4 (Test-case teeth):** FAIL — §4 Verification: "Printed schema table ≥ 5 sources × 6
  fields ... no assertion, human review." This is the explicit admission that the plan's primary
  verification step is not mechanically checkable. See M1.

- **Lens 5 (Alternative realism):** PASS — Streaming mode fallback is concrete; "histogram
  smoothed but bin counts match ±5%" is a real tolerance.

- **Lens 6 (Dependency honesty):** WEAK — Plan 02 depends on `#14` (constraints frozen), but
  the only constraint from plan 00 that plan 02 actually uses is the base model tokenizer for
  token length counting. This is listed in §8 Consumed-by as "uses category enum and token-length
  policy." The dependency is real but the plan doesn't clarify which constraint facts it needs
  (only: "base model tokenizer version" and "whether competition test set categories are known").

- **Lens 7 (Artifact discipline):** PASS — `data/eval/schema.json`, `experiments/eda_<date>.md`,
  `experiments/figures/token_length_histogram.png`, `src/data/schema.py` — all stable paths with
  date-stamping where appropriate.

- **Lens 8 (Risk honesty):** WEAK — The plan notes "undisclosed private-test categories" as a
  risk but does not address the symmetric risk: the Puzzle-KD dataset's "reasoning=off" assumption
  (plan §3 Step 3) is treated as verified by a ≤1% `<think>` token check. But "reasoning=off"
  in the dataset metadata may mean something different from "no reasoning trace" — it may mean
  the model was not prompted with reasoning mode, but could still have produced reasoning tokens
  spontaneously. The ≤1% threshold is a concrete check, but the plan should assert whether that
  1% is a documentation inconsistency or real data to route to reasoning=on training.

- **Top unaddressed risk:** `src/data/schema.py` exports a `ReasoningExample` dataclass with
  fields `[id, category, prompt, answer, reasoning_on, source, language, tokens_used]`. This
  schema does not include `error_type` (produced by plan 06), `teacher_id` or `verifier_pass`
  (produced by plan 08), or `training_config_hash` (produced by plan 10). As provenance fields
  accumulate, the `ReasoningExample` schema will either (a) need to be extended, breaking
  previously validated files, or (b) be replaced by a different schema in each notebook, causing
  fragmentation. The plan should define a versioning strategy for schema.py from the start.

- **Verdict: ship-with-fixes** (M1, Lens 4)

---

### 5.03 Validation Split and Golden Set

- **Lens 1 (Objective falsifiability):** PASS — "Commit JSONL files... with SHA-256 hashes; any
  adapter that breaks golden problems is rejected" is falsifiable and binary.

- **Lens 2 (Strategy concreteness):** WEAK — Step 1 says "stratified random sampling (seed=42)
  from eligible held-out source data." The word "eligible" is undefined. This is blocker B3.
  Without naming a concrete source with a schema that the planner has already verified in plan 02,
  "stratified sample" cannot be automated.

- **Lens 3 (MVP executability):** WEAK — 5 inference runs of a 4B model on "candidate problems"
  to select golden set. How many candidates? If there are 300 candidates and each run takes
  5 seconds on RTX 3080, that is 5 × 300 × 5 = 7,500 seconds = ~2.1 hours for the golden-set
  selection alone, not counting the 200-problem validation set construction. The 2–3 hour estimate
  for "MVP (stratified sample + golden-set infer)" is likely underestimated by 50%.

- **Lens 4 (Test-case teeth):** PASS — §5.1 Expected: "validation_200.jsonl contains exactly 200
  items with fields matching schema #17; Category distribution: math ≥50, code ≥40, science ≥40,
  logic ≥30; golden_20.jsonl contains exactly 20 items, all with correct answer on 5/5 inference
  runs; SHA-256 printed and recorded." These are all mechanically checkable. This is the best
  test-case section in the 11 plans.

- **Lens 5 (Alternative realism):** PASS — "downgrade to ≥80% correct (3 inference runs)" is
  specific and bounded. The EVAL_POLICY.md documentation requirement for the relaxation is a good
  audit trail.

- **Lens 6 (Dependency honesty):** WEAK — Plan 03 depends on plan 02's schema (#17) to validate
  JSONL fields. But plan 02 might not have been run yet if the student executes the plans in
  Wave B simultaneously. The plan should assert that schema.json exists and is loadable before
  building the validation set, not just list #17 as an upstream issue number.

- **Lens 7 (Artifact discipline):** PASS — SHA-256 hashes committed to manifest, hash-check
  guards in downstream notebooks, `scripts/build_eval_sets.py` for reproducibility. This is the
  best artifact discipline in the 11 plans.

- **Lens 8 (Risk honesty):** WEAK — The plan lists "source data overlaps with Kaggle test set"
  as a risk, with mitigation "use only public-domain source problems." But the competition's novel
  benchmark is unpublished — there is no way to verify disjointness. The mitigation is wishful.
  A more honest mitigation: "accept this risk; document the source provenance thoroughly so any
  overlap is detectable in post-competition analysis."

- **Top unaddressed risk:** If the base model already achieves >90% consistency on every easy
  problem in the candidate pool, the golden set will be trivially dominated by easy problems (the
  ones the model never misses). These easy problems have no discriminative power as a regression
  tripwire for a stronger adapter. The plan should enforce a difficulty floor: golden problems
  must come from at least 3 difficulty tiers to be useful regression probes.

- **Verdict: ship-with-fixes** (B3, Lens 2)

---

### 5.04 Baseline Eval and Normalization

- **Lens 1 (Objective falsifiability):** PASS — "Publish `src/evaluation/normalize.py`,
  `harness.py`, and an initial baseline score card with deterministic evaluation" — all
  mechanically verifiable artifacts.

- **Lens 2 (Strategy concreteness):** PASS — Strategy names specific regex
  (`r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'`), function signatures (`extract_boxed_answer`,
  `normalize`, `equal`), seeds {42, 43, 44}, and WandB config.

- **Lens 3 (MVP executability):** WEAK — 6.5–7.5 hours exceeds the 3–6 hour template target.
  See M2. The bottleneck is 200-problem × 3-seed = 600 inferences. On RTX 3080, this is
  realistically 1–2 hours of GPU time, which is fine, but the planner should say so explicitly
  rather than lumping it into a vague 3–4 hour estimate.

- **Lens 4 (Test-case teeth):** PASS — §5.1 Expected: "assert `extract_boxed_answer('...\\boxed{2}.')
  == '2'` → PASS; golden_20 accuracy = 100%; validation_200 mean±std logged to WandB with
  per-problem predictions in JSON." These are all mechanically verifiable. The regression guardrail
  (determinism: run twice, assert identical per-problem scores) is excellent.

- **Lens 5 (Alternative realism):** PASS — The `strict_em` branch (if competition uses plain
  exact-match) is concrete: "conditional `if competition_uses_plain_em: use_strict_em_path()`."
  Both paths are described as code, not words.

- **Lens 6 (Dependency honesty):** WEAK — Plan 04 depends on `golden_20.jsonl` from plan 03,
  but plan 03's golden-set construction depends on base-model inference (plan 04's primary
  artifact — the loaded model). If plan 03 encounters an issue and returns a golden set with
  relaxed confidence (3 runs, ≥80%), plan 04's requirement "golden_20 accuracy = 100%" may not
  hold even for the base model. Plan 04 should assert at load time: "if golden_set confidence
  is 'relaxed', log warning and use 95% pass threshold instead of 100%."

- **Lens 7 (Artifact discipline):** PASS — `experiments/baseline_<date>.json` with per-seed
  per-problem predictions is well-specified. WandB run tagged and linked.

- **Lens 8 (Risk honesty):** PASS — The plan lists: nested-brace regex edge cases, Mamba-2
  nondeterminism at temperature=0, competition eval contract ambiguity, and golden_20 data quality
  issues. This is the most complete risk section of all 11 plans.

- **Top unaddressed risk:** `temperature=0` on Mamba-2 may not be deterministic. The plan
  acknowledges "reported in some systems" but the mitigation is vague ("record any nondeterminism;
  if detected, use fixed random seed at generation kernel level if supported"). If temperature=0
  is nondeterministic on this architecture, the 3-seed eval design is invalid — all three seeds
  will produce different results even with the same seed input. The plan should include a
  2-run determinism check as the first eval step, before committing to the 3-seed protocol.

- **Verdict: ship-with-fixes** (M2, B2 dependency)

---

### 5.05 Prompting and Decode Sweeps

- **Lens 1 (Objective falsifiability):** PASS — "Comparison table of {strategy × decode config
  × accuracy mean±std × compute cost}" is falsifiable.

- **Lens 2 (Strategy concreteness):** PASS — Sparse grid (zero-shot CoT + few-shot CoT × {temp
  0.6, 1.0} × {top_p 0.9, 0.95} = 8 seeds, 3 runs each) is fully specified.

- **Lens 3 (MVP executability):** WEAK — §4 Verification: "Exactly one row beats baseline with
  delta > 2σ." See M3. This is a precondition disguised as a test assertion.

- **Lens 4 (Test-case teeth):** FAIL — §5.1 Expected "Exactly one row has delta > baseline + 2σ"
  is the M3 issue. This assertion will fail legitimately if the model is near ceiling (95.4%
  MATH500) and no strategy helps. See M3 for the fix. The rest of the expected section
  (schema, accuracy_std ≤ 0.05, no golden regression) is properly checkable.

- **Lens 5 (Alternative realism):** PASS — "Downscale to stratified random 100 subsample" if
  inference is too slow is concrete. "Mark results as 'noisy — to be re-run on Colab A100'" is
  an honest deferral, not a fake pass criterion.

- **Lens 6 (Dependency honesty):** PASS — Depends on plan 04 (eval harness, golden_20,
  validation_200). All three are explicitly named.

- **Lens 7 (Artifact discipline):** PASS — `experiments/prompting_sweep_<date>.csv` with 9
  named columns is well-specified.

- **Lens 8 (Risk honesty):** PASS — Overfitting to validation_200, inference latency ballooning,
  NVIDIA defaults being globally optimal — all three are real risks with concrete early-exit
  criteria.

- **Top unaddressed risk:** Few-shot CoT examples are injected into the context (plan §4 Cell 3:
  "3-5 solved examples"). If those examples are drawn from the training data that plan 09 will
  eventually train on, there is a subtle data-leakage risk: the few-shot examples effectively
  participate in the prompting experiment AND in training, inflating both prompting and SFT
  baselines. The plan should specify where the few-shot examples come from (public math problems
  not in the training set, or examples from validation_200 that are held out from both training
  and prompting sweep scoring).

- **Verdict: ship-with-fixes** (M3, Lens 4)

---

### 5.06 Trajectory Collection and Error Slices

- **Lens 1 (Objective falsifiability):** PASS — "Cluster failures into error types... export JSONL
  trajectory dataset and markdown error-slice taxonomy report" is falsifiable: you can fail it if
  fewer than 3 named error types have ≥5 examples.

- **Lens 2 (Strategy concreteness):** PASS — Error classification taxonomy (6 types), regex
  heuristics for format/truncation/empty reasoning, optional LLM-judge, per-category clustering.

- **Lens 3 (MVP executability):** FAIL — See B1. The primary input `data/processed/training_curated.jsonl`
  does not exist as a plan-produced artifact. No notebook in the 11-plan set produces it. This
  is not recoverable by adjusting the plan's scope — it requires adding a missing notebook or
  explicitly sourcing a raw dataset slice here.

- **Lens 4 (Test-case teeth):** FAIL — §5.1 Expected contains "Printed summary: 'Correct:
  752/1200 (62.7%) | Incorrect: 448/1200 (37.3%)'" — see M4. This is a fabricated expected value,
  not a mechanically verifiable assertion.

- **Lens 5 (Alternative realism):** PASS — Heuristic-only fallback if LLM-judge budget is
  exhausted is specific and realistic.

- **Lens 6 (Dependency honesty):** FAIL — The plan silently depends on a curation artifact that
  no upstream plan produces. This is blocker B1.

- **Lens 7 (Artifact discipline):** WEAK — Plan §8 Produces section lists
  `data/analysis/trajectories_2026-04-20.jsonl` (hardcoded date). Every other plan uses a
  `<date>` placeholder. Hardcoding today's date means re-runs produce a new file at a different
  path without updating the hardcoded reference. The filename should be
  `trajectories_<date>.jsonl`.

- **Lens 8 (Risk honesty):** PASS — OOM risk on RTX 3080 for 16k token samples, golden-set
  leakage risk, LLM-judge bias — all real and all listed.

- **Top unaddressed risk:** The heuristic error classification (regex for format, truncation
  detection) has no precision/recall estimate. If the heuristics are poorly calibrated, "format
  miss" errors are over- or under-counted, and the synthetic data recipe (plan 08) targets the
  wrong failure mode. The plan should include a 20-sample manual validation of the heuristic
  classifier before scaling to 1,200 samples.

- **Verdict: rewrite** (B1, Lens 3 and 6)

---

### 5.07 Solver Framework Design

- **Lens 1 (Objective falsifiability):** WEAK — "Specify a category-aware solver interface with
  fallback policy" — the deliverable is code stubs and a design doc, not a running model.
  The success criterion "solver.py passes mypy --strict" is falsifiable, but "confidence signal
  calibrated" is not.

- **Lens 2 (Strategy concreteness):** WEAK — Strategy Step 4 "pick a confidence signal: use
  `prob_of_boxed_answer` (presence of valid `\boxed{}` format) + `agreement_across_best_of_N`."
  These two signals are described as function names but not as formulas. `prob_of_boxed_answer`
  is a binary (present/absent), not a probability — naming it "prob" is misleading. The plan
  should define: `conf_format = int(bool(extract_boxed_answer(answer)))` and `conf_agreement
  = majority_vote_count / N`. Without formulas, the confidence scorer will be reimplemented
  differently in plans 08, 09, and 10, causing drift.

- **Lens 3 (MVP executability):** PASS — 3 hours for writing stubs + YAML routing + design doc
  on CPU is realistic. The golden_20 regression check (1 hour) is reasonable.

- **Lens 4 (Test-case teeth):** PASS — §5.1 Expected: "solver.py passes mypy --strict; YAML
  parses with ≥3 entries; SolverOutput returned with answer/confidence/trace/metadata; golden
  problems deterministic with seed=42." All mechanically checkable.

- **Lens 5 (Alternative realism):** PASS — "If category count ≤2, collapse to single-path solver"
  is specific and the expected output (single 'default' route) is concrete.

- **Lens 6 (Dependency honesty):** WEAK — Plan 07 header lists `#15` (review harness) as an
  upstream dependency. But the MVP Inputs do not use anything from #15 — they use golden_20
  (from #18), category list (from #22 or hardcoded fallback), and the base model. See M5.

- **Lens 7 (Artifact discipline):** PASS — Three specific output files, all with stable paths.
  SOLVER_DESIGN.md is documentation, not data, so no hash requirement is appropriate.

- **Lens 8 (Risk honesty):** PASS — Over-engineering routing, category taxonomy mismatch,
  confidence miscalibration — all legitimate risks with concrete mitigations.

- **Top unaddressed risk:** The solver framework is designed assuming the competition has clearly
  defined categories. But plan 00's §7 Open Questions includes "hidden test set composition" as
  an unresolved item. If the competition benchmark uses an unknown category distribution, the
  CategoryRouter will fall through to the default route for all questions, making the category
  routing logic dead code. The plan should include a stub for handling "unknown" category and
  a test that the solver degrades gracefully when category is None.

- **Verdict: ship-with-fixes** (M5, Lens 6)

---

### 5.08 Synthetic Data Recipe

- **Lens 1 (Objective falsifiability):** PASS — "500-sample pilot run with ≥90% verifier pass
  rate, ≤$20 cost, 5 provenance fields" — all falsifiable binary conditions.

- **Lens 2 (Strategy concreteness):** PASS — Strategy names DeepSeek-R1, SYNTHETIC-1, sympy for
  math verification, exact-match for logic, Nemotron chat template tokens.

- **Lens 3 (MVP executability):** FAIL — See M6. DeepSeek-R1 (671B) cannot run on Colab Pro
  A100 (40 GB). The effort table lists "HPC (optional) or Colab" for the 3-hour pilot run,
  which is wrong. The SYNTHETIC-1 fallback is actually the only realistic path for a student
  without multi-GPU HPC access. The plan inverts the priority: SYNTHETIC-1 should be the primary
  path, and self-hosted DeepSeek-R1 should be the "advanced" path with a clear VRAM/GPU
  requirement stated.

- **Lens 4 (Test-case teeth):** PASS — §4 Verification assertions are concrete:
  `assert len(samples) == 500`, `assert pass_rate >= 0.90`, `assert total_cost <= 20.0`,
  `assert all(s["teacher_id"] and s["verifier_pass"] is not None for s in samples)`. These are
  the cleanest assertions in the 11 plans.

- **Lens 5 (Alternative realism):** PASS — SYNTHETIC-1 subsample fallback is well-specified:
  "filter to problems matching #22 error slice topics; apply same verification pipeline; zero
  generation cost; output schema matches primary." The schema compatibility check is correctly
  required.

- **Lens 6 (Dependency honesty):** WEAK — §3 Strategy Step 6 says "check for overlap with
  curated data from notebook 03." Plan 03 produces golden_20 and validation_200 — curated
  training data. The correct reference is to the missing curation notebook (B1 again), or
  explicitly to `data/eval/validation_200.jsonl` and `data/eval/golden_20.jsonl`. See also the
  minor nit about the path mismatch (`data/errors/category_slices.jsonl` vs.
  `data/analysis/trajectories_<date>.jsonl`).

- **Lens 7 (Artifact discipline):** PASS — `data/synthetic/pilot_<YYYYMMDD>.jsonl` with date
  stamping, `configs/synthetic_recipe.yaml` checked-in, provenance fields defined.

- **Lens 8 (Risk honesty):** PASS — Cost blowout, teacher-student mismatch, license conflict,
  data leakage — all real risks with specific mitigations. The HPC feasibility question is
  correctly flagged as open.

- **Top unaddressed risk:** The `sympy` verification approach for math answers has known failure
  modes: sympy cannot evaluate all valid math expressions (e.g., non-symbolic matrix problems,
  combinatorics, some geometry), and may raise exceptions on valid but unusual syntax. A >10%
  sympy parse failure rate would silently depress the "pass rate" metric below 90% even if the
  answers are correct. The plan should specify a fallback verification path (string normalization
  → numeric comparison → skip) for sympy failures, and report sympy-failure rate separately from
  logical-incorrectness rate.

- **Verdict: ship-with-fixes** (M6, B1 indirect, Lens 3)

---

### 5.09 SFT LoRA Runbook and Masking

- **Lens 1 (Objective falsifiability):** PASS — "100-step smoke training run... validate pipeline,
  save adapter as safetensors and load it back" — all falsifiable.

- **Lens 2 (Strategy concreteness):** PASS — Target modules listed by exact name
  (q_proj, v_proj, x_proj, in_proj, out_proj), reasoning tokens (12, 13), configs by filename.

- **Lens 3 (MVP executability):** FAIL — See B4. The smoke run on RTX 3080 with QLoRA has no
  VRAM estimate. The base model at 4-bit is ~2 GB; with gradient checkpointing and a 1k-sample
  loader, peak VRAM is hard to predict without testing. The plan should include a cell that
  runs `torch.cuda.memory_reserved()` after model load and before training, and aborts with a
  clear error if VRAM > 8 GB (leaving headroom for OS + other processes).

- **Lens 4 (Test-case teeth):** WEAK — §5.1 Primary Expected: "Final loss ~0.5–1.0
  (regression-free vs baseline)." See M7. This absolute range is not a regression check. The
  mechanically correct version: "loss at step 100 < 0.95 × loss at step 1."

- **Lens 5 (Alternative realism):** PASS — "Fall back to HF PEFT + TRL SFTTrainer if Unsloth
  doesn't support Mamba-2" is specific and actionable.

- **Lens 6 (Dependency honesty):** WEAK — Plan 09 header lists `#24` (synthetic recipe) as an
  upstream dependency. But §4 Inputs says "1k curated samples from #24" — it uses synthetic
  data as training data for the smoke run. If plan 08 has not yet produced `pilot_<date>.jsonl`,
  the plan must fall back to raw curated data (from the missing curation notebook) or public data.
  The plan should name a concrete fallback for the smoke-run training data when plan 08 has not
  run.

- **Lens 7 (Artifact discipline):** PASS — `adapters/smoke_<date>/adapter_model.safetensors`,
  two YAML configs, `src/training/sft_trainer.py`, `tests/test_masking.py` — all specific paths.
  The smoke adapter is saved to a date-stamped directory, preventing overwrites.

- **Lens 8 (Risk honesty):** PASS — Mamba-2 module name mismatch, masking miscount, OOM, vLLM
  incompatibility, Unsloth version gap — all real and listed.

- **Top unaddressed risk:** The plan sets `<think>` = token 12 and `</think>` = token 13 as
  facts. This is from plan_v0.2 §1 Competition Summary table. But plan 00 has not confirmed these
  token IDs against the actual tokenizer — they are stated as facts in the competition summary
  without a citation to the model card. If these token IDs are wrong, loss masking will mask the
  wrong tokens and the smoke run will appear to work (loss decreases) while training on garbage.
  Plan 09 Cell 3 says "verify that apply_chat_template(..., enable_thinking=True) produces
  expected token sequences" — this is good — but the plan does not specify what "expected" means
  (i.e., what the expected token sequence is). Add a concrete assertion:
  `assert tokenizer.convert_ids_to_tokens([12, 13]) == ["<think>", "</think>"]`.

- **Verdict: ship-with-fixes** (B4, M7, Lens 3 and 4)

---

### 5.10 Submission Packaging and Provenance

- **Lens 1 (Objective falsifiability):** PASS — "Manifest JSON valid; golden_20 = 20/20;
  SHA256 re-hash matches manifest; git SHA from HEAD; zip unzips cleanly" — all falsifiable.

- **Lens 2 (Strategy concreteness):** PASS — Strategy names `git rev-parse HEAD`, `hashlib.sha256`,
  specific manifest fields (8 named), specific output directory structure.

- **Lens 3 (MVP executability):** PASS — 4 hours for cells 1–9 with a dummy adapter on local CPU
  is realistic. No GPU required for packaging.

- **Lens 4 (Test-case teeth):** PASS — §5.1 Expected: "all three dry-run checks pass; golden_20
  = 20/20; validation_50 mean accuracy within ±2σ of #19 baseline; zip contains
  adapter_model.safetensors, manifest.json, README.md." These are mechanically checkable.

- **Lens 5 (Alternative realism):** WEAK — The Alternative path (use real Phase 4/6 adapter)
  drops the golden_20 threshold to 19/20, contradicting plans 03, 04, and 09. See M8.

- **Lens 6 (Dependency honesty):** WEAK — Plan 10 header lists only `#14` and `#19` as upstream
  dependencies. But plan 10 packages an adapter that came from plan 09 (SFT), uses the solver
  from plan 07 for format checking, and validates against a schema defined in plan 02. None of
  plans 07, 09, or 02 are listed as dependencies. The dependency list is too narrow.

- **Lens 7 (Artifact discipline):** PASS — `docs/schemas/manifest_v1.0.json` with JSON Schema
  v7, SHA256 of adapter verified post-zip, git-cleanliness check before packaging. This is the
  best artifact discipline of all terminal notebooks.

- **Lens 8 (Risk honesty):** PASS — Competition format change, timezone drift, adapter size
  limit, SHA256 mismatch after zip — all real and addressed.

- **Top unaddressed risk:** The manifest includes `training_config_hash` but plan 09 does not
  define how this hash is computed (hash of which YAML file? which Python config dict? which
  step?). If the hash algorithm or input differs between plan 09 (which writes the adapter) and
  plan 10 (which hashes the config), the manifest will contain a hash that cannot be reproduced
  or verified. The two plans need to agree on: `training_config_hash = sha256(open(
  "configs/lora_baseline.yaml").read().encode()).hexdigest()` or equivalent.

- **Verdict: ship-with-fixes** (M8, Lens 5 and 6)

---

## 6. Cross-Plan Findings

### Schema agreement

Plans that share data schemas: 02 (defines), 03 (validates), 04 (validates at load), 06
(extends), 08 (adds provenance), 09 (consumes for training).

**Divergences found:**

- **`data/eval/schema.json` vs. `src/data/schema.py`**: Plan 02 produces both. The JSON
  schema is described as "canonical field dict (names, types, enums, constraints)"; the Python
  module `schema.py` exports `ReasoningExample` dataclass with fields `[id, category, prompt,
  answer, reasoning_on, source, language, tokens_used]`. Plan 04 says it validates
  `normalize.py` functions against "competition data format (once verified in #14)" — it does not
  import `ReasoningExample`. Plan 06 adds fields `[raw_output, extracted_answer, correct,
  error_type, token_length]` to each trajectory row without extending the dataclass. Plan 08
  adds `[teacher_id, teacher_version, prompt_hash, generation_date, verifier_pass]` without
  extending the dataclass. By plan 09, the "1k curated samples from #24" have a different schema
  than `ReasoningExample`, but `sft_trainer.py` presumably uses one schema to build the
  DataLoader. **No plan nominates who owns schema evolution.** This is fragmentation in slow
  motion.

- **Answer format**: Plan 04 uses `r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'` from plan_v0.2. Plan
  06 §4 Inference says "extract `<think>...</think>` and final answer" using `extract_boxed_answer`
  — implicitly the same regex. Plan 07 says "use `prob_of_boxed_answer`" as a confidence signal
  without specifying whether it uses the same regex. Plan 08 says "verify final `\boxed{...}`
  answer: math via sympy or numeric comparison" — this implies sympy takes the content of
  `\boxed{}` as input, not the raw output string. Plan 09 says "masking utility... keep
  reasoning/answer tokens" — it does not reference the boxed regex at all. Plan 10 says
  "verify `\boxed{2}` format." The normalization regex is defined in plan 04's `normalize.py`
  but plans 06, 07, and 10 each reference boxed-answer extraction without explicitly importing
  from `src/evaluation/normalize.py`. This means there are potentially 3–4 independent
  re-implementations of the same regex.

### Eval-contract agreement

Plans 04, 07, and 10 must agree on what "correct" means.

- **Plan 04**: Implements `extract_boxed_answer` + `normalize` + `equal`; states "if competition
  uses plain exact-match, add strict_em branch"; both paths configurable via flag from plan 00.
- **Plan 07**: Uses "presence of valid `\boxed{}` format" as confidence signal — assumes boxed
  extraction. Does not reference plan 04's normalize.py or plan 00's contract flag.
- **Plan 10**: Dry-run smoke test "verify `\boxed{2}` format" — assumes boxed. Does not
  reference the strict_em flag from plan 04.

If plan 00 determines the competition uses plain exact-match (e.g., answers are integers or
decimals without LaTeX), plan 04 flips the flag, but plans 07 and 10 do not read the flag.
The eval contract is not consistently propagated across the three plans. **Plan 04 is the
authoritative source; plans 07 and 10 must import from it.**

### Golden-set discipline

Plans 03 (creates), 04 (establishes baseline), 05 (regression check), 09 (regression gate),
10 (dry-run check).

- **Same file**: All plans reference `data/eval/golden_20.jsonl` — consistent path. Good.
- **Hash check**: Plan 03 §4 Cell 5 computes SHA-256. Plan 03 §3 Step 6 says "all eval
  notebooks (04, 05, 07, 09, 10) implement hash-check guards on load." Plan 07 does not mention
  a hash check. Plan 08 does not consume golden_20. Plan 10 mentions "golden_20 accuracy = 20/20"
  as a test but does not show a hash load assertion.
- **Mutation risk**: Plan 05 §4 Cells section describes running best prompting strategy on
  golden_20 for regression. Plan 06 §5.3 Regression says "verify golden_20 is NOT in trajectories."
  Neither plan 05 nor plan 06 writes to golden_20, so the no-write policy holds. Good.
- **Threshold inconsistency**: Plans 03, 04, 09 require 20/20. Plan 10 §5.2 Alternative drops
  to 19/20 without a policy change. This is M8.

### Provenance discipline

Plans 08 (synthetic data provenance), 09 (SFT adapter provenance), 10 (submission manifest).

- **Plan 08 provenance fields**: `teacher_id`, `teacher_version`, `prompt_hash`,
  `generation_date`, `verifier_pass` — defined in §6 Success Criteria.
- **Plan 09 provenance**: No provenance fields defined for the SFT adapter itself. The plan
  produces `adapters/smoke_<date>/adapter_model.safetensors` and a WandB run, but no structured
  provenance metadata file alongside the adapter. Plan 10 expects a `training_config_hash` in
  the manifest — but plan 09 does not write one.
- **Plan 10 manifest fields**: `git_sha`, `adapter_sha256`, `base_model`, `training_config_hash`,
  `eval_scores`, `decode_config`, `author`, `date` — defined in §4 Cell 2.
- **Composition gap**: Plan 10 cannot automatically derive `training_config_hash` from plan 09's
  adapter directory because plan 09 does not write a machine-readable provenance sidecar.
  The chain is broken. Plan 09 must write `adapters/<run>/provenance.json` with at minimum:
  `{training_config_hash, base_model_id, dataset_version, seed, wandb_run_url}`.

### Hardware routing

The RTX 3080 is 10 GB VRAM. The Colab Pro A100 is 40 GB. The HPC cluster is unspecified but
assumed 40–80 GB.

- **Plan 03**: 5 inference runs of 4B BF16 model per candidate. BF16 at 4B = ~8 GB weights +
  ~1–2 GB KV cache. This barely fits on RTX 3080 at batch_size=1. The plan says "Colab Pro or
  RTX 3080" without noting that RTX 3080 is at the edge. Risk: PASS (just barely).
- **Plan 04**: 600 inferences (200 × 3 seeds) on RTX 3080. Same memory profile as plan 03.
  Inference-only (no gradient storage), so 10 GB is tight but feasible with greedy decoding.
  Risk: PASS.
- **Plan 06**: 1,200 inferences on RTX 3080 with enable_thinking=True (may generate long traces,
  up to 16k tokens per plan §6). At 16k tokens output per sample, KV cache alone is ~6 GB.
  Combined with weights (8 GB), this will OOM on RTX 3080. The plan says "token_length ≤ 16k"
  as a success criterion but does not cap generation at 4k or 8k in the inference cell.
  **Hardware routing FAIL for plan 06 on RTX 3080.** Should require Colab Pro A100 or RTX 3060
  with shorter max_new_tokens.
- **Plan 08**: DeepSeek-R1 generation "HPC (optional) or Colab." FAIL — see M6. Colab cannot
  run the full DeepSeek-R1 model.
- **Plan 09**: RTX 3080 QLoRA smoke run. See B4. No VRAM assertion.
- **Plans 02, 05, 07, 10**: CPU-heavy or minimal GPU. Routing is appropriate.

### Scope-creep check

- **Plan 07 (Solver Framework Design)** is explicitly scoped to "stubs and config files" per its
  MVP — this is appropriate for the discovery phase, but it has grown to include a confidence
  scorer with two signals, a fallback state machine, a decision document, and a mypy-strict
  type check. The actual running solver is deferred to plan_v0.2 Phase 8 (submission). There is
  a risk the solver stubs become a mini-framework that plan 09 and 10 depend on in ways not
  planned for. The scope is at the edge but still defensible.
- **Plan 06 (Trajectory Collection)** doubles as a data quality validation notebook (it validates
  that curated training data is error-free). This is scope that belongs in the missing data
  curation notebook. Plan 06's scope should be "error analysis of base model outputs on eval sets
  only," not "validate training data quality."
- **Plan 10 (Submission)** implements a `dry_run_validation_slice()` that runs the model on 50
  validation problems. This is an eval step that belongs in plan 04 (eval harness). Plan 10's
  scope should be packaging and provenance only, not evaluation. Running inference in the packaging
  notebook creates a dependency on GPU access at submission time, which is an unnecessary
  operational risk.

---

## 7. Prioritized Fix List

The original planner must work through these in order. Items 1–4 are blockers (execution-halting).
Items 5–10 are major issues (quality-degrading). Items 11–15 are important but non-blocking.

1. **Add a data curation notebook plan** (address B1). The 11-plan set is missing the notebook
   that produces `data/processed/training_curated.jsonl`. This is plan_v0.2 Phase 3.2–3.3
   (NeMo Curator pipeline). Plan 06 cannot execute without this artifact. Add a notebook 02b or
   reassign the curation work to plan 02 with an explicit curation step.

2. **Resolve the eval contract in plan 00 using the submission demo notebook** (address B2).
   Do not wait for Kaggle's rules page to answer whether the scorer uses boxed extraction or plain
   exact-match. The authoritative source is the submission demo notebook from Ryan Holbrook. Plan
   00 §3 Strategy must add a step: "Inspect submission demo inference cell; extract answer parsing
   logic; treat it as ground truth for eval contract." Propagate the decision as a flag in
   `competition_constraints.json` and document the code path in plan 04.

3. **Name the source dataset for plan 03 golden-set construction** (address B3). Replace
   "eligible source data (e.g., ...)" with a specific dataset ID, subset name, and version.
   Given the current plan set, the only confirmed available source with schema alignment is
   Puzzle-KD Dataset v2. Name it explicitly: "stratified sample from
   `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` validation split (5% held-out,
   ~42K samples), seed=42."

4. **Add a VRAM budget cell to plan 09 before the smoke run** (address B4). The cell must:
   load QLoRA model, call `torch.cuda.memory_reserved()`, compare against threshold (8 GB for
   RTX 3080), and abort with a human-readable error if exceeded. Include a suggested
   max_seq_length and batch_size for the smoke run (e.g., max_seq_length=2048, batch_size=2,
   gradient_accumulation=32).

5. **Replace the fabricated expected accuracy in plan 06 §5.1 with mechanically verifiable
   assertions** (address M4). Delete "Correct: 752/1200 (62.7%)" and all absolute error-type
   counts from the Expected section. Replace with: JSONL row count matches 1,200; all rows have
   required keys; ≥3 named error types with ≥5 examples each.

6. **Replace "Exactly one row beats baseline with delta > 2σ" in plan 05 §4 Verification with
   a schema assertion** (address M3). This is the only test case in the 11 plans that will
   "fail" even when the notebook executes correctly.

7. **Add a provenance sidecar spec to plan 09** (address provenance gap in §6 Cross-Plan
   Findings). Plan 09 must write `adapters/<run>/provenance.json` with fields that plan 10's
   manifest can cite without manual reconstruction.

8. **Remove the 19/20 golden threshold relaxation from plan 10 §5.2** or formally document it
   as a policy change in `EVAL_POLICY.md` and update plans 03, 04, and 09 accordingly
   (address M8).

9. **Fix plan 06 hardware routing**: cap max_new_tokens in the inference loop to 8,192 for
   RTX 3080 runs (not 16,384), or reroute plan 06 to "Colab Pro recommended" with RTX 3080 as
   fallback with reduced token cap (address hardware routing finding).

10. **Invert teacher priority in plan 08**: make SYNTHETIC-1 subsample the primary path and
    self-hosted DeepSeek-R1 the "advanced" path with explicit HPC multi-GPU requirements
    (address M6).

11. **Add plan 10 to plan 09's consumers list** and add plans 02, 07, 09 to plan 10's
    dependency list (address incomplete dependency declarations in plans 09 and 10).

12. **Remove the `#15` (review harness) dependency from plan 07's header** if plan 07 consumes
    nothing from #15, or name the specific artifact consumed (address M5).

13. **Fix the hardcoded date in plan 06 §8 Artifacts**: change
    `data/analysis/trajectories_2026-04-20.jsonl` to `data/analysis/trajectories_<date>.jsonl`.

14. **Uncheck all pre-checked success criteria in plan 03 §6**: every `[x]` should be `[ ]`;
    these are planning artifacts, not completed work.

15. **Designate a single owner for `src/data/schema.py` schema evolution** (address schema
    fragmentation finding). Plans 04, 06, 08, and 09 all implicitly extend the schema without
    a documented change process. The owner should be the plan 02 author; all schema extensions
    require a PR review against the canonical dataclass.

---

## 8. What the Planner Got Right

The following decisions are well-founded and should be preserved through revisions:

**Golden-set design in plan 03.** The 5-inference-run, temperature=0, ≥90% consistency threshold
for golden-set selection is a rigorous operationalization of plan_v0.2's informal instruction.
The SHA-256 hash commitment, the `EVAL_POLICY.md` no-touch rule, and the explicit hash-check
guards in downstream notebooks form a coherent regression framework. This is the best-engineered
section of the 11 plans.

**Sparse grid sweep design in plan 05.** "Start with 8 configs, 3 runs each; early exit if NVIDIA
defaults dominate" is the correct experimental design for a compute-limited student. The paired
t-test criterion (delta > 2σ for statistical significance) is the right significance framework
for the validation_200 set size. The wall-clock time cap (4 GPU-hours) prevents the notebook from
becoming an unbounded experiment.

**Boxed-regex fix propagated correctly.** All 11 plans use
`r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'` (the nested-brace-aware version) rather than the lazy
`(.+?)` match that was a CRITICAL issue in plan_review.md. The fix from MJ-7 was correctly
picked up and applied consistently.

**Plan 08 verification assertions.** The four inline `assert` statements in plan 08 §4
Verification are the most concrete test assertions in any plan's MVP section. They are copy-paste
deployable and leave zero ambiguity about what the pilot run must achieve.

**Dependency wave ordering in SPRINTS.md.** The Wave A → B → C → D structure correctly enforces
that constraints are frozen before schemas are defined, schemas are frozen before training sets
are curated, and training sets are frozen before the SFT runbook runs. This respects the
leakage-prevention requirements from plan_v0.2 Phase 3.1.

**The plan 04 open-question about Mamba-2 nondeterminism.** This is a genuine architectural risk
that is not present in plan_review.md (which reviewed plan v0.1) and was correctly identified
by the planner. Temperature=0 may not guarantee determinism in SSM-based layers. Flagging it as
an open question with a mitigation path ("record nondeterminism; use fixed kernel seed if
supported") is the right conservative engineering decision.

**Plan 01 decision-log CSV column specification.** The 13-column CSV spec
(`source | approach | dataset | masking_strategy | eval_metric | outcome | adoption_decision
| downstream_notebook | risk_score | cost_hours | decision_rationale | citation_url
| citation_date`) is sufficiently detailed to be machine-parseable and auditable. It would survive
a competition post-mortem review without ambiguity.

---

*Adversarial review completed 2026-04-20. 11 plans reviewed, 4 blockers identified, 9 major
issues, 15 nits. Recommended action: fix items 1–4 in the priority list before any notebook
is opened.*
