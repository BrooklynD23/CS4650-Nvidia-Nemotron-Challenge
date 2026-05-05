---
title: Autonomous Documentation Agent System
date: 2026-04-21
status: approved
---

# Autonomous Documentation Agent System

## Context

The CS4650-Nvidia-Nemotron-Challenge repo has a well-structured documentation hierarchy but no mechanism to keep docs aligned with commits automatically. As Wave A/B notebooks landed and Wave C began, `docs/learn/` project-status pages and `docs/execution/NOTEBOOKS.md` drifted within a single working session.

This system builds five Claude Code agents — one orchestrator plus four specialists — that run autonomously after every commit and maintain four distinct documentation surfaces with a human-in-the-loop gate for critical architectural decisions.

## Architecture

```
git commit
     │
     ▼
.githooks/post-commit          /doc-sync <range>  ← manual skill
     │                               │
     └──────────┬────────────────────┘
                ▼
        doc-orchestrator
        (reads diff, classifies domains, dispatches in parallel)
                │
        ┌───────┼────────────────┐──────────────┐
        ▼       ▼                ▼               ▼
  routing-  execution-      learn-         architecture-
    sync      sync           sync              sync
```

## Agents

### doc-orchestrator

**File:** `.claude/agents/doc-orchestrator.md`  
**Model:** haiku  
**Role:** Read `git diff --name-only`, classify touched files into domains, dispatch relevant specialist agents in parallel. No writes.

**Classification rules:**

| Domain | Trigger files |
|---|---|
| ROUTING | `AGENTS.md`, `CLAUDE.md`, `docs/README.md`, new top-level dirs |
| EXECUTION | `notebooks/`, `docs/execution/`, `src/`, `configs/` |
| LEARN | Any substantive commit (agent does its own relevance check) |
| ARCHITECTURE | `docs/architecture/`, `docs/analysis/`, `docs/planning/`, system-level `src/` |

### routing-sync

**File:** `.claude/agents/routing-sync.md`  
**Model:** haiku  
**Write scope:** `AGENTS.md`, `CLAUDE.md`, `docs/README.md`  
**Updates:** routing tables, canonical doc pointers, Last Updated metadata  
**Guardrail:** navigation content only — no substantive project content

### execution-sync

**File:** `.claude/agents/execution-sync.md`  
**Model:** haiku  
**Write scope:** `docs/execution/NOTEBOOKS.md`, `docs/execution/SPRINTS.md`  
**Promotion logic:**
- `scaffolded → active`: substantive notebook body delivered
- `active → validated`: explicit test-passage evidence only — never self-certifies  
**Guardrail:** never changes issue ownership, dependency order, or wave structure

### learn-sync

**File:** `.claude/agents/learn-sync.md`  
**Model:** sonnet  
**Write scope:** `docs/learn/project/*.md`, `docs/learn/foundations/*.md`, `docs/learn/sources/citation-ledger.md`  
**Can create:** new foundations pages from `docs/learn/_templates/page_template.md`  
**Guardrail:** never collapses planned → implemented; never invents results; beginner-friendly voice; repo sources cited first

### architecture-sync

**File:** `.claude/agents/architecture-sync.md`  
**Model:** sonnet  
**Write scope (routine):** `docs/architecture/*.md`, `docs/analysis/*.md` — factual corrections applied directly  
**Write scope (critical):** `docs/analysis/PENDING_REVIEW.md` — proposed change written here, architectural docs not touched  

**Critical decision triggers:** training strategy, base model choice, evaluation contract, dataset composition, solver architecture, `src/` contradicting `ARCHITECTURE.md` assumptions

## Human Gate

When `architecture-sync` detects a critical decision it writes `docs/analysis/PENDING_REVIEW.md`:

```markdown
# Pending Architecture Review

**Triggered by:** <SHA> — <message>
**Date:** <YYYY-MM-DD>
**Agent:** architecture-sync

## Proposed Change
## Why This Is Flagged as Critical
## Recommended Action
- [ ] Accept / Reject / Defer

## Auto-applied Routine Updates
```

A resolved pending review is signalled by a follow-up commit that removes or resolves the file. The agent will not open a second review on the same topic until the first is resolved.

## Trigger Layer

**Post-commit hook** (`.githooks/post-commit`): fires `doc-orchestrator` in the background after every commit. Skipped automatically if `claude` CLI is not on PATH (safe for CI).

**Manual skill** (`.claude/commands/doc-sync.md`): `/doc-sync [range]` invokes the same orchestrator with an optional commit range. Defaults to `HEAD~1..HEAD`.

**Hook registration:** `init.sh` runs `git config core.hooksPath .githooks` so all collaborators get the hook automatically on setup.

## File Layout

```
.claude/agents/
├── doc-orchestrator.md
├── routing-sync.md
├── execution-sync.md
├── learn-sync.md
└── architecture-sync.md

.claude/commands/
└── doc-sync.md

.githooks/
└── post-commit

docs/superpowers/specs/
└── 2026-04-21-doc-agent-system-design.md  ← this file
```

## Verification Checklist

1. Make a test commit touching `notebooks/05_*.ipynb` — confirm `execution-sync` promotes status and `learn-sync` updates `implemented-today.md`
2. Make a test commit to `docs/planning/plan_v0.2.md` changing training strategy — confirm `PENDING_REVIEW.md` is created and `docs/architecture/` is NOT modified
3. Run `/doc-sync HEAD~3..HEAD` manually — confirm same behavior as the hook
4. Make a purely mechanical commit (whitespace only) — confirm no agents fire or all output "no changes needed"
