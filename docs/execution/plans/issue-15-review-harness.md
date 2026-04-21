# Issue 15 — Agent + Human Review Harness (Planning Only)

**Parent issue:** `#15`  
**Deliverable path:** `docs/execution/plans/issue-15-review-harness.md`  
**Dependencies:** `#13` (notebook template + workflow baseline)  
**Agent owner:** Orchestrator (this session)  
**Human reviewer:** Repo owner / PM reviewer  
**Architecture reviewer:** Not required (process-only; no schema changes)  
**Status:** `decision-complete`

## 1) Goal

Make the GitHub issue workflow for the notebook-first foundation phase decision-complete, so that:

- every child issue `#13–#25` has the same required fields
- reviewers have an unambiguous acceptance checklist
- work can proceed asynchronously by dependency waves (Wave A/B/C/D) without contract drift

## 2) Non-goals

- Do not implement any notebooks (`notebooks/*.ipynb`)
- Do not change schema contracts (`ReasoningExample`, `EvalRecord`, etc.)
- Do not create new CI pipelines
- Do not require new tools beyond GitHub Issues + Markdown

## 3) Decisions required (and gates)

| Decision | Status | Why it matters | Upstream gate |
|---|---|---|---|
| Required fields for all child issues | frozen | Prevents missing ownership/dependencies/acceptance criteria | `docs/execution/ISSUE_REVIEW_HARNESS.md` |
| Review lanes (agent vs human vs architecture) | frozen | Prevents self-approval on shared-contract changes | `docs/execution/ISSUE_REVIEW_HARNESS.md` |
| Wave labeling scheme | frozen | Enables async dependency waves | `docs/execution/SPRINTS.md` |
| Label taxonomy availability in GitHub | gated | GitHub may require manual label creation | GitHub UI / repo permissions |

## 4) Files to create / modify

### Files that should exist and be kept in sync

- `docs/execution/ISSUE_REVIEW_HARNESS.md`
- `docs/execution/SPRINTS.md`
- `docs/execution/NOTEBOOKS.md`

### GitHub issue templates (must include required fields)

- `.github/ISSUE_TEMPLATE/agent-execution.md`
- `.github/ISSUE_TEMPLATE/human-review.md`
- `.github/ISSUE_TEMPLATE/config.yml`

### This planning artifact

- `docs/execution/plans/issue-15-review-harness.md` (this file)

## 5) Interfaces / contracts referenced

- Required issue fields contract from `docs/execution/ISSUE_REVIEW_HARNESS.md`
- Wave dependency model from `docs/execution/SPRINTS.md`
- Notebook registry rules from `docs/execution/NOTEBOOKS.md`

## 6) Step-by-step tasks

### Task 1 — Verify templates match the harness contract

- [ ] Confirm `docs/execution/ISSUE_REVIEW_HARNESS.md` enumerates the canonical required fields for child issues.
- [ ] Confirm `.github/ISSUE_TEMPLATE/agent-execution.md` includes *all* required fields:
  - Parent epic
  - Deliverable path
  - Dependencies
  - Agent owner
  - Human reviewer
  - Architecture reviewer (optional, but present)
  - Acceptance checklist
  - Sources to verify
  - Risks / open questions
- [ ] Confirm `.github/ISSUE_TEMPLATE/human-review.md` includes:
  - artifact under review
  - parent epic
  - reviewer checklist aligned to the harness

### Task 2 — Freeze wave labeling semantics (A/B/C/D)

- [ ] Confirm `docs/execution/SPRINTS.md` defines the Wave A/B/C/D child issues and dependency order.
- [ ] Ensure each child issue body (GitHub) includes a “Wave: A|B|C|D” line (even if labels are missing).
- [ ] Attempt to apply GitHub labels `wave:A`…`wave:D`:
  - if label creation fails via API, record a manual step to create labels in GitHub UI, then re-apply.

### Task 3 — Freeze closure rules and review workflow

- [ ] Confirm `docs/execution/ISSUE_REVIEW_HARNESS.md` includes closure rules:
  - architecture-impact issues can’t be self-closed by agent owners
  - notebooks must include required notebook sections (`Audience and Why It Matters`, `Decision / Hypothesis`, etc.)
- [ ] Ensure `docs/execution/NOTEBOOKS.md` includes the notebook section requirements and update procedure.
- [ ] Confirm the GitHub templates’ default labels align with the review workflow (`needs-human-review`, `owner:agent`).

### Task 4 — Link plan docs from GitHub issues (Wave A/B)

- [ ] Add a “Plan doc” line to each child issue body `#14–#20` linking to:
  - `docs/execution/plans/issue-14-constraints-freeze.md`
  - `docs/execution/plans/issue-15-review-harness.md`
  - `docs/execution/plans/issue-16-external-baselines-delta.md`
  - `docs/execution/plans/issue-17-schema-and-eda.md`
  - `docs/execution/plans/issue-18-validation-and-golden-set.md`
  - `docs/execution/plans/issue-19-baseline-eval-and-normalization.md`
  - `docs/execution/plans/issue-20-submission-packaging-and-provenance.md`

## 7) Verification commands (and expected outputs)

| Command | Expected output |
|---|---|
| `ls .github/ISSUE_TEMPLATE` | Lists `agent-execution.md`, `human-review.md`, `config.yml` |
| `rg -n \"Parent epic|Deliverable path|Dependencies|Agent owner|Human reviewer|Architecture reviewer|Acceptance checklist|Sources to verify|Risks / open questions\" .github/ISSUE_TEMPLATE/agent-execution.md` | All fields are found at least once |
| `rg -n \"Reviewer checklist|Sources\" .github/ISSUE_TEMPLATE/human-review.md` | Review checklist and sources guidance are found |
| `rg -n \"Required Issue Fields\" docs/execution/ISSUE_REVIEW_HARNESS.md` | Harness clearly enumerates required issue fields |
| `git diff --check` | No whitespace errors |

Manual verification (GitHub UI):

- Create a test issue using the “Agent execution” template and confirm the fields render.
- Confirm the issue gets the intended default labels (or record missing labels as a blocker + manual fix).

## 8) Acceptance checklist (aligned to `docs/execution/ISSUE_REVIEW_HARNESS.md`)

- [ ] `docs/execution/ISSUE_REVIEW_HARNESS.md` is the canonical, complete source of required issue fields and closure rules.
- [ ] `.github/ISSUE_TEMPLATE/agent-execution.md` includes every required field.
- [ ] `.github/ISSUE_TEMPLATE/human-review.md` includes reviewer checklist aligned to harness.
- [ ] `docs/execution/SPRINTS.md` wave semantics are reflected in child issue bodies and (if possible) GitHub labels.
- [ ] Wave A/B issues link to their plan docs.
- [ ] No notebooks were created or modified as part of this work.

## 9) Risks / open questions

- GitHub label creation may require manual UI actions or elevated repo permissions.
- The “wave:*” labels might not exist yet; in that case, the issue body must still carry “Wave: …” to avoid losing the dependency-wave semantics.

## 10) Sources

- `docs/execution/ISSUE_REVIEW_HARNESS.md`
- `docs/execution/SPRINTS.md`
- `docs/execution/NOTEBOOKS.md`
- `.github/ISSUE_TEMPLATE/agent-execution.md`
- `.github/ISSUE_TEMPLATE/human-review.md`
- `.github/ISSUE_TEMPLATE/config.yml`

