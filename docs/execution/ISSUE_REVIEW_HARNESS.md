# Agent and Human Review Harness

This document defines how GitHub issues should be assigned, reviewed, and closed while the repo is still notebook-first.

## Purpose

The immediate goal is not full model training. The goal is to create a reliable execution harness so multiple agents and human reviewers can work in parallel without drifting on assumptions, sources, or output format.

## Required Issue Fields

Every child issue under `#13-#25` should include:

- `Parent epic`
- `Deliverable path`
- `Dependencies`
- `Agent owner`
- `Human reviewer`
- `Architecture reviewer` when the issue changes a shared contract
- `Acceptance checklist`
- `Sources to verify`
- `Risks / open questions`

## Epic vs Child Issues (Duplication Policy)

To avoid duplicated work between “Sprint (n)” issues and “Wave (A–D)” issues, we use a strict two-tier model:

- **Epics / Sprint issues (`#1–#12`)** track goals and close only when their child issues are complete.
- **Execution / Wave issues (`#13–#25`)** own deliverables (notebooks, harness artifacts) and contain the review checklist fields.

Rules (enforced by reviewers):

- Do not open a “Wave” issue that restates an existing epic’s goal; instead, link the epic to the correct child issue(s).
- Do not execute work directly “in the epic”: execution evidence (artifacts, plan docs, commands) must live in the child issue.
- Each execution issue must have a unique `Deliverable path`. If two issues point at the same path, one is a duplicate and should be closed or merged.
- If new work is discovered, create a **new child issue** (next available number, e.g. `#26+`) and link it from the relevant epic; do not create a new “Sprint” issue.

## Review Lanes

### Agent owner

The agent owner is responsible for:

- producing the artifact at the target path
- keeping citations and assumptions explicit
- linking any downstream issues that depend on the work
- leaving a concise change summary in the issue before review

### Human reviewer

The human reviewer is responsible for:

- checking whether the artifact is understandable without deep ML context
- confirming that claims are backed by cited sources
- verifying that the artifact satisfies the issue checklist

### Architecture reviewer

Use an architecture reviewer when the issue changes shared contracts or project-wide assumptions:

- competition constraints and scoring rules
- schema contracts
- eval record shape
- packaging and provenance requirements
- training masking policy

## Notebook Review Checklist

All notebooks must include these sections inside the notebook:

1. `Audience and Why It Matters`
2. `Decision / Hypothesis`
3. `Environment and Reproduction`
4. `Method and Outputs`
5. `Results / Open Risks`
6. `Sources`

Reviewers should reject the issue if any of the following are missing:

- a non-technical framing section
- explicit sources with URLs
- a statement separating verified facts from community heuristics
- a declared parent issue and dependencies

## Closure Rules

- Agent owners do not self-close architecture-impact issues.
- A notebook issue is ready to close when the notebook executes without error or, for spec-only notebooks, when all checklist items are present and the reviewer explicitly signs off.
- A harness issue is ready to close when the template or doc exists and the workflow is referenced from `docs/execution/SPRINTS.md` or `docs/execution/NOTEBOOKS.md`.

## Suggested Comment Template

Use this issue comment when requesting review:

```md
Artifact: `path/to/artifact`
Parent epic: #X
Dependencies satisfied: #Y, #Z

What changed:
- ...

What needs review:
- ...

Sources checked:
- ...

Open risks:
- ...
```
