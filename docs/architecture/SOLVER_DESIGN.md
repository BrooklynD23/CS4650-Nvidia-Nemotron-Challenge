# Solver Framework Design

**Issue:** #23 / #8  
**Date:** 2026-05-04  
**Status:** Active

---

## Overview

The solver framework provides a category-aware plugin interface for deterministic
or high-confidence answer generation. It sits between prompt generation and the
LLM teacher path: if a solver can answer with sufficient confidence, the LLM call
is skipped, reducing cost and latency.

---

## Protocol Definitions

### `Solver` Protocol (`src/inference/solver.py`)

```python
class Solver(Protocol):
    def solve(self, prompt: str, category: str) -> SolverOutput: ...
```

### `Verifier` Protocol

```python
class Verifier(Protocol):
    def verify(self, pred: str, gold: str) -> bool: ...
```

### `SolverOutput`

```python
{
    "answer": str | None,     # None → fall through to LLM teacher
    "confidence": float,      # 0.0–1.0
    "metadata": dict,         # rule name, failure_mode, etc.
}
```

---

## Routing Config (`configs/solver_routing.yaml`)

Maps each category to a solver class and per-category confidence threshold.
The `default` key is the catch-all for unregistered categories.
`llm_fallback_model` names the LLM used when all solvers return `answer=None`.

---

## Fallback Policy

```
route(prompt, category)
  ├─ solver registered? No  → answer=None (LLM path)
  └─ Yes
      ├─ call solver.solve(prompt, category) → result
      ├─ confidence ≥ threshold? Yes → return result
      └─ No
          ├─ retry solver.solve(prompt, category) → result2
          ├─ confidence ≥ threshold? Yes → return result2
          └─ No → answer=None, failure_mode=LOW_CONFIDENCE (LLM path)
```

---

## Confidence Signal Design

Two signals are combined in solver implementations:

1. **Rule satisfaction rate** — fraction of in-prompt examples the inferred
   rule satisfies. A rule with 100% satisfaction on all examples gets confidence=1.0.
2. **Rule uniqueness** — when multiple rules satisfy all examples but predict
   different outputs, the majority-vote fraction becomes the confidence score.

---

## FailureMode Mapping

| FailureMode | Meaning |
|---|---|
| `NO_ANSWER` | No solver registered, or no candidate rule found |
| `FORMAT_ERROR` | Prompt did not contain parseable examples or query |
| `LOW_CONFIDENCE` | Rule(s) found but confidence below threshold after retry |
| `AMBIGUOUS` | Multiple rules found that disagree on the query output |

**Relationship to trajectory ErrorType:** `FORMAT_ERROR` ↔ `format_miss`;
`NO_ANSWER` ↔ `hallucinated_reasoning` / `refusal`; `LOW_CONFIDENCE` ↔
`arithmetic_slip`. These are separate enums — ErrorType classifies past
failures, FailureMode reports current solver state.

---

## Risk Mitigations

- **Over-engineering risk:** Default YAML has a `null` solver for unregistered
  categories, so the system gracefully degrades to the LLM teacher.
- **Category mismatch risk:** Router checks `default` fallback before returning
  `NO_ANSWER`; category taxonomy from #22 is not hard-coded into the router.
- **Confidence miscalibration:** Threshold is per-category in YAML, tunable
  without code changes.

---

## First Solver: `BitManipulationSolver` (`src/solvers/bit_manipulation.py`)

Registered for `category == "bit_manipulation"`. Enumerates shift, XOR, AND,
OR, NOT, mask, modulo, base-conversion, and numeric transforms; selects rules
that satisfy all in-prompt input→output examples; returns `\boxed{<value>}`.

---

## Open Questions

- How many categories does #22 discover? Next solver category (#26) to be
  decided from error-slice taxonomy.
- Should per-category adapters specialize, or only decode settings vary?
  Deferred to Phase 4 (SFT) results.
