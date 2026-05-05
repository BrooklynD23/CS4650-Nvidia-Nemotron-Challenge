# Issue 17 — Schema + EDA Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Track progress with checkbox (`- [ ]`) items.

**Goal:** Freeze a decision-complete schema layer for `ReasoningExample`, `EvalRecord`, `PackageManifest`, and `SFTExample`, then validate the raw dataset shape and the ingest-to-train/eval mappings so downstream work cannot drift.

**Non-Goals:**
- Do not train models, build notebooks, or tune prompts.
- Do not create new benchmark definitions or change competition scoring.
- Do not broaden the schema beyond the fields needed to keep ingest, SFT, eval, and packaging aligned.

**Architecture:** Treat `ReasoningExample` as the canonical raw row contract, derive `SFTExample` from it through an explicit prompt-template boundary, and make `EvalRecord` record-level and example-linked so every evaluation result can be traced back to a source row and decode configuration. `PackageManifest` stays a separate provenance artifact for exports; it should not be mixed into submission payloads.

**Tech Stack:** Markdown docs, repo contract docs, future `src/` dataclasses or typed dicts, and one thin schema/EDA implementation step later.

---

## Decision Summary

1. **Canonical raw schema:** `ReasoningExample` is the only persisted raw-example contract.
2. **Training schema:** `SFTExample` is the only persisted SFT/teacher-trace contract.
3. **Evaluation schema:** `EvalRecord` is per-example, per-run, and must carry config attribution.
4. **Packaging schema:** `PackageManifest` is a provenance card for exports, not a submission artifact.
5. **Alias policy:** `question`/`expected_answer` and `input`/`output` are ingest-time aliases only; they must be normalized immediately and never become new persistent schemas.

## Gates

### Gate 0 — Source Freeze
- Confirm the schema source docs are the only inputs for this issue.
- Confirm there is no separate hidden contract in notebook code.

### Gate 1 — Contract Freeze
- Freeze field names, types, and requiredness for all four schemas.
- Freeze the `prompt` meaning: raw problem text, not rendered chat output.

### Gate 2 — Mapping Freeze
- Freeze raw CSV → `ReasoningExample` mapping.
- Freeze `ReasoningExample` → `SFTExample` mapping.
- Freeze `ReasoningExample` → eval input / `EvalRecord` mapping.

### Gate 3 — Drift Guardrails
- Identify the minimum follow-up edits needed in architecture docs and future `src/` files.
- Ensure no field appears under two different names in different pipeline stages.

## Contract Definitions

### `ReasoningExample`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `id` | `str` | Yes | Stable source row identifier. |
| `category` | `str` | Yes | Puzzle family or other task category. |
| `prompt` | `str` | Yes | Raw competition prompt text. |
| `answer` | `str` | Yes | Canonical gold answer string. |
| `source` | `str` | Yes | Dataset origin, e.g. `kaggle:train.csv`, `kaggle:test.csv`, or `mirror:<name>`. |
| `split` | `str` | Yes | `train`, `val`, `golden`, `test`, `external`, or another explicitly documented split label. |
| `metadata` | `dict[str, Any]` | Yes | Extra raw columns, ingest notes, and parsing leftovers; may be empty. |

### `EvalRecord`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `run_id` | `str` | Yes | Stable identifier for one evaluation run. |
| `example_id` | `str` | Yes | Links the result back to `ReasoningExample.id`. |
| `model_id` | `str` | Yes | Model or adapter identifier used for the run. |
| `prompt_template_id` | `str` | Yes | Versioned prompt template used to render the input. |
| `normalizer_id` | `str` | Yes | Exact answer-normalization rule set used for scoring. |
| `category` | `str` | Yes | Copied from the source example. |
| `split` | `str` | Yes | Copied from the source example. |
| `gold` | `str` | Yes | Gold answer string used for comparison. |
| `prediction` | `str` | Yes | Raw model output before normalization. |
| `normalized_prediction` | `str` | Yes | Prediction after applying `normalizer_id`. |
| `correct` | `bool` | Yes | Comparison result after normalization. |
| `latency_ms` | `float` | Yes | End-to-end inference latency for this example. |
| `tokens_in` | `int` | Yes | Input token count. |
| `tokens_out` | `int` | Yes | Output token count. |
| `seed` | `int` | Yes | Decode seed for reproducibility. |
| `decode_config` | `dict[str, Any]` | Yes | Decoding knobs such as temperature, top_p, and max_new_tokens. |

### `PackageManifest`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `manifest_version` | `str` | Yes | Version of the manifest schema itself. |
| `base_model_id` | `str` | Yes | Exact base model used for the exported adapter. |
| `adapter_rank` | `int` | Yes | LoRA rank used for the adapter. |
| `dataset_version` | `str` | Yes | Frozen dataset build or release identifier. |
| `eval_sha` | `str` | Yes | Git SHA for the exact evaluation code/config state that produced the manifest. |
| `artifact_paths` | `dict[str, str]` | Yes | File paths for adapter weights, adapter config, eval summary, and any out-of-band provenance bundle. |
| `created_at` | `str` | Yes | ISO-8601 timestamp for manifest creation. |

### `SFTExample`

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `example_id` | `str` | Yes | Stable link back to the source `ReasoningExample`. |
| `category` | `str` | Yes | Copied from the source example. |
| `messages` | `list[dict[str, str]]` | Yes | Chat messages used to construct the SFT input; each entry must have `role` and `content`. |
| `completion` | `str` | Yes | Assistant target text to train on. |
| `source` | `str` | Yes | Origin dataset or generator label. |
| `split` | `str` | Yes | Training split label. |
| `provenance` | `dict[str, Any]` | Yes | Template, teacher, and transform metadata needed to reproduce the example. |

## Mapping Rules

### Raw dataset rows → `ReasoningExample`

- `id` maps directly to `ReasoningExample.id`.
- `prompt` maps directly to `ReasoningExample.prompt`.
- `answer` maps directly to `ReasoningExample.answer`.
- `category` is copied from the source row when present; otherwise it is inferred from the file/source family and recorded in `metadata`.
- `source` is derived from the dataset origin, not from the prompt text.
- `split` is derived from the file or directory the row came from.
- Any extra columns are preserved under `metadata` so ingest remains lossless.

### `ReasoningExample` → `SFTExample`

- The raw prompt stays raw in `ReasoningExample`; the rendered chat prompt belongs in `messages`, not in the raw contract.
- `messages` contains the versioned prompt template output, with the user turn built from `ReasoningExample.prompt`.
- `completion` contains the exact assistant target text.
- Use answer-only completion by default for baseline SFT examples.
- Use reasoning-plus-answer completion only when the upstream source explicitly provides a teacher trace and the prompt template is marked for reasoning traces.
- `provenance` must include the source `example_id`, `prompt_template_id`, and any teacher-model or transform metadata needed to reproduce the sample.

### `ReasoningExample` → eval inputs / `EvalRecord`

- Evaluation inputs are built from `ReasoningExample.prompt` plus `prompt_template_id`; the gold answer is never part of the model input.
- `EvalRecord.example_id` must equal `ReasoningExample.id`.
- `EvalRecord.gold` copies `ReasoningExample.answer`.
- `EvalRecord.prediction` stores raw model output; `normalized_prediction` stores the post-normalization string.
- `EvalRecord.correct` is computed only after applying the explicit `normalizer_id`.
- `EvalRecord.decode_config` must capture the exact decode settings used for that run so sweeps can be reproduced.

### `PackageManifest` mapping

- `base_model_id` and `adapter_rank` come from the trained adapter export.
- `dataset_version` must reflect the frozen raw/processed dataset build, not a vague notebook name.
- `eval_sha` is the exact git SHA of the evaluation code/config state that produced the manifest.
- `artifact_paths` should point to the adapter files and provenance outputs that live outside the Kaggle submission zip.

## Explicit Files Later

The implementation issue for this plan should touch only focused schema files and one EDA helper layer:

- Create: `src/contracts.py` — typed schema definitions for the four contracts.
- Create: `src/data/schema_mapping.py` — raw-row normalization and alias handling.
- Create: `src/data/schema_eda.py` — lightweight dataset-shape inspection helpers.
- Create: `src/evaluation/records.py` — `EvalRecord` construction and serialization.
- Create: `src/packaging/manifest.py` — `PackageManifest` construction and serialization.
- Update: `docs/architecture/ARCHITECTURE.md` — if the canonical contract text needs to be synchronized after implementation.

## Verification Commands

1. `rg -n "ReasoningExample|EvalRecord|PackageManifest|SFTExample" docs/architecture/ARCHITECTURE.md docs/analysis/PLAN_V0_2_REVIEW_PLAN.md docs/planning/plan_v0.2.md`
   - Expected: the contract names are still discoverable in the source docs, and this plan’s decisions stay aligned with them.

2. `git diff --check -- docs/execution/plans/issue-17-schema-and-eda.md docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
   - Expected: no whitespace or patch-format errors.

3. `git diff -- docs/execution/plans/issue-17-schema-and-eda.md docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`
   - Expected: only the new plan doc and the Agent 2 report block changed.

## Acceptance Checklist

- [ ] Parent epic is identified and the deliverable path is explicit.
- [ ] Dependencies are stated, including the shared-contract architecture review gate.
- [ ] `ReasoningExample`, `EvalRecord`, `PackageManifest`, and `SFTExample` each have explicit fields, types, and requiredness.
- [ ] Raw dataset column mapping is explicit and lossless.
- [ ] `ReasoningExample` → `SFTExample` and `ReasoningExample` → eval mappings are explicit.
- [ ] Minimal drift-prevention edits are enumerated with exact future files.
- [ ] Verification commands are concrete and have expected outputs.
- [ ] The plan avoids notebook implementation and avoids touching unrelated plan docs.

## Risks / Open Questions

- The competition’s final scoring normalization may still force a small adjustment to `normalizer_id` semantics.
- If raw competition CSVs expose extra columns or hidden categories, the `metadata` and `category` inference rules may need a narrow refinement.
- The exact export-path names for provenance artifacts may still be tightened when packaging is implemented.

## Issue Metadata

- **Parent epic:** `#13`
- **Deliverable path:** `docs/execution/plans/issue-17-schema-and-eda.md`
- **Dependencies:** `#14` constraints freeze and `#15` review harness
- **Agent owner:** Agent 2
- **Human reviewer:** Project owner / capstone reviewer
- **Architecture reviewer:** Required
- **Sources to verify:** `docs/architecture/ARCHITECTURE.md`, `docs/architecture/COMPETITION.md`, `docs/planning/plan_v0.2.md`, `docs/analysis/PLAN_V0_2_REVIEW_PLAN.md`, `docs/execution/ISSUE_REVIEW_HARNESS.md`
