# Notebook 00: Competition Constraints and Rules

**Parent Issue**: `#14`  
**Plan Phase**: Prerequisite gate (constraints freeze, before Phase 0)  
**Scaffold**: `notebooks/00_competition_constraints_and_rules.ipynb`  
**Status**: `planned`  
**Dependencies (upstream)**: `#13` (notebook template)  
**Consumers (downstream)**: All subsequent notebooks (01–10)

---

## 1. Objective

Verify and freeze all official Kaggle competition constraints (deadlines, submission format, base model identity, LoRA rank caps, evaluation metric, adapter size limits) by cross-referencing Kaggle's competition page, rules tab, NVIDIA's model card, and public submission demos. Produce a single constraint table with citations and provenance, plus a mismatch log identifying gaps between repo assumptions and verified facts. Treat every constraint as provisional until confirmed from authoritative sources—community write-ups and rumors do not count.

## 2. Why It Matters

- **Downstream inheritance**: All subsequent notebooks (baseline inference, training, evaluation, submission) must reference verified constraint values, not speculated ones. Any notebook that hard-codes base model ID or LoRA rank without linking to this notebook's findings introduces unnecessary risk.
- **Leaderboard stakes**: A single misunderstood deadline, submission format, or adapter size limit breaks the final submission.
- **Capstone documentation**: Must demonstrate due diligence in verifying competition terms before implementation.

## 3. Strategy — How We Aim To Accomplish It

1. **Fetch Kaggle competition page** (most authoritative): deadlines, submission rules, base model statement, scoring metric, file format requirements.
2. **Cross-reference Kaggle Rules tab** for clarifications: LoRA rank cap, adapter file limits (safetensors only?), merge/quantization restrictions.
3. **Check NVIDIA model card** (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` on HuggingFace): architecture, context window, tokenizer version, `enable_thinking` mechanism, special tokens.
4. **Review submission demo notebook** (Kaggle, Ryan Holbrook): expected adapter directory structure, merge procedure, inference setup, evaluation environment.
5. **Scan recent Kaggle forum posts** (competition page discussions): clarifications on ambiguous rules, timezone anchoring for deadlines, hidden test set composition hints.
6. **Record provenance for each fact**: URL + snapshot date (YYYY-MM-DD) to flag stale information if the page updates.
7. **Log mismatches**: For each constraint in `docs/architecture/COMPETITION.md`, note whether it is verified, provisional (rumored), or contradicted.

## 4. MVP (Minimum Viable Notebook)

### Inputs
- Browser access to Kaggle competition page
- HuggingFace model card (API call or manual inspection)
- Current `docs/architecture/COMPETITION.md` file
- Current `docs/planning/plan_v0.2.md` (Competition Summary section)

### Cells (minimal)

1. **Cell 1 (Setup)**: Import libraries, define verification log schema, set snapshot date.
2. **Cell 2 (Kaggle fetch)**: Load competition metadata (deadline, base model, submission format, scoring). Record URLs and snapshot time.
3. **Cell 3 (HF model card check)**: Verify base model parameters, context window, tokenizer, `enable_thinking` token IDs.
4. **Cell 4 (Constraint table)**: Build single authoritative table: Field, Verified Value, Source, Snapshot Date, Confidence (High/Medium/Low/Open).
5. **Cell 5 (Mismatch log)**: Compare verified facts against `docs/architecture/COMPETITION.md` and `plan_v0.2.md`. Note discrepancies.
6. **Cell 6 (Open questions)**: List fields that cannot be verified from public sources (hidden test set size, exact eval hardware, timezone offset).
7. **Cell 7 (Export)**: Write constraint table as JSON to `data/eval/competition_constraints.json` and CSV to `data/eval/competition_constraints.csv`.

### Outputs
- `data/eval/competition_constraints.json` — machine-readable constraint snapshot
- `data/eval/competition_constraints.csv` — human-readable verification table
- Updated `docs/architecture/COMPETITION.md` with links to this notebook's findings
- Open-question list in the notebook (printed and saved as `data/eval/open_questions_snapshot.txt`)

### Verification
Print and visually confirm: (1) deadline is non-null and formatted as YYYY-MM-DD; (2) base model matches one of the two candidate checkpoints; (3) LoRA rank cap has a source URL; (4) every critical field has a confidence rating (no blanks).

## 5. Test Cases

### 5.1 Primary (MVP path)

- **Setup**: Notebook runs with internet access, all imports succeed (requests, json, pandas, datetime).
- **Action**: Fetch Kaggle competition page, parse deadlines, check HF model card, cross-reference submission demo.
- **Expected**: Constraint table with all downstream-critical fields (deadline, base model id, LoRA rank cap, submission format, max adapter size) filled with non-null values and citations.

### 5.2 Alternative / Fallback

- **Setup**: Kaggle page is behind login or Hugging Face API times out.
- **Action**: Use cached snapshots from `docs/architecture/COMPETITION.md` as provisional; annotate each field with **(UNVERIFIED)** flag.
- **Expected**: Constraint table still generated with provisional values marked clearly; log generated with "needs human verification" for each flagged row. Proceed with provisional values **if** snapshots are ≤1 week old (else escalate).

### 5.3 Regression Guardrails

- **Must not break**: The notebook's output JSON schema must match the schema defined in Cell 4. If schema changes, all downstream notebooks must be re-run.
- **Golden constraint check**: If re-run on same date, constraint table output must be identical (deterministic).
- **Staleness alarm**: If notebook is re-run and Kaggle page has updated (page timestamp differs from last snapshot), flag the top 5 changed fields for human review.

## 6. Success Criteria (Done When)

- [ ] Constraint table contains all downstream-critical fields with non-null values
- [ ] Every field has a source URL and snapshot date
- [ ] Confidence ratings (High/Medium/Low/Open) assigned to all fields
- [ ] Mismatch log created and compared against `docs/architecture/COMPETITION.md`
- [ ] Open questions clearly marked (e.g., "hidden test set size: OPEN", "exact eval GPU type: UNVERIFIED")
- [ ] JSON and CSV exports created in `data/eval/`
- [ ] Notebook runs cleanly from start to finish (no errors, no manual intervention)
- [ ] Artifact(s) committed and linked in `docs/execution/NOTEBOOKS.md`

## 7. Risks & Open Questions

| Risk | Mitigation |
|------|-----------|
| **Community write-ups stale**: Competition rules page updates; mirrors lag by days or weeks. | Snapshot timestamp every fetch; set staleness alarm if page timestamp > 7 days old. Re-verify monthly. |
| **Rumored LoRA rank cap unverified**: External sources claim rank ≤ 32, but official Rules page silent. | Mark as **(UNVERIFIED)** if Kaggle Rules tab does not mention rank; document where rumor originated. |
| **Hidden test set undisclosed**: Eval dataset composition not public; only public benchmarks known. | File as OPEN; note that submission accuracy on Kaggle will be the only true metric. |
| **Timezone ambiguity on deadline**: LinkedIn announcement says "June 15" but no timezone or cutoff time. | Fetch Kaggle page (authoritative timezone); default to UTC if ambiguous; escalate if still unclear. |
| **Eval hardware mismatch**: NVIDIA docs say RTX PRO 6000; confirm Kaggle page actually specifies this. | Cross-ref competition page + submission demo notebook; if conflict, trust Kaggle page. |

| Open Question | Who Answers | Fallback |
|---------------|-----------|----------|
| Exact deadline time (HH:MM UTC)? | Kaggle competition page (Rules tab) | Assume 23:59 UTC if omitted |
| LoRA rank cap enforced? | Kaggle Rules page or submission rejection | Assume ≤ 64 (NVIDIA recommended) if omitted |
| Max adapter file size? | Submission demo notebook or Rules page | Assume ≤ 500 MB if omitted |
| Hidden test set size / categories? | Post-competition analysis only | Use public benchmarks for validation |
| Does evaluation merge LoRA automatically or expect user-merged weights? | Submission demo notebook | Assume automatic merge (common pattern) |

## 8. Artifacts & Handoff

### Produces
- `data/eval/competition_constraints.json` — Structured constraint table (dict: field → {value, source, snapshot_date, confidence})
- `data/eval/competition_constraints.csv` — Human-readable CSV export
- `data/eval/open_questions_snapshot.txt` — Dated list of unverifiable constraints
- Updated `docs/architecture/COMPETITION.md` — Link to notebook findings; note any corrections to previous assumptions

### Consumed by
- Notebook 01 (baseline inference): Uses base model ID, context window
- Notebook 04 (data curation): Uses LoRA rank cap to design training config
- Notebook 06 (SFT LoRA): Uses rank cap, max adapter size
- Notebook 08 (GRPO RL): Uses adapter size limit
- Notebook 10 (submission): Uses submission format, file naming, deadline

### External references cited
- [Kaggle Competition Page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- [Kaggle Rules Tab](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules) (if accessible)
- [Kaggle Submission Demo Notebook](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- [NVIDIA Nemotron-3-Nano-4B Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [Kaggle Discussion Forum](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) (for clarifications)

## 9. Estimated Effort

| Step | Hours | Hardware |
|------|-------|----------|
| MVP (fetch + verify + export) | 2 | Local (browser + python) |
| Alternative path (cached fallback) | 0.5 | Local |
| Full polish (forum scrape, monthly re-check) | 1 | Local |
