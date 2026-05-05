"""Category-aware solver framework with Protocol definitions and routing.

Defines the typed contracts for solver plugins, a confidence-gated
CategoryRouter, and a FailureMode enum for solver-facing error reporting.
See docs/architecture/SOLVER_DESIGN.md for design rationale.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml


class SolverOutput(dict):
    """Typed helper for solver output dicts.

    Structural shape: ``{answer: str | None, confidence: float, metadata: dict}``

    Inherits from dict so Protocol consumers can treat it as a plain dict
    while still using attribute-style construction via the factory method.
    """

    @classmethod
    def make(
        cls,
        answer: str | None,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> "SolverOutput":
        obj = cls()
        obj["answer"] = answer
        obj["confidence"] = float(confidence)
        obj["metadata"] = metadata if metadata is not None else {}
        return obj


class FailureMode(str, Enum):
    """Solver-facing failure reasons (distinct from trajectory ErrorType).

    These categorize why a solver returned ``answer=None``, enabling the
    router to choose an appropriate fallback strategy.
    """

    NO_ANSWER = "no_answer"
    FORMAT_ERROR = "format_error"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"


@runtime_checkable
class Solver(Protocol):
    """Protocol for category-aware solver plugins."""

    def solve(self, prompt: str, category: str) -> SolverOutput:
        ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for answer verification."""

    def verify(self, pred: str, gold: str) -> bool:
        ...


_DEFAULT_THRESHOLD = 0.3
_DEFAULT_CONFIG: dict[str, Any] = {
    "categories": {},
    "llm_fallback_model": None,
}


class CategoryRouter:
    """Routes prompts to registered solvers based on category.

    Routing policy:
    1. Call registered solver for the category.
    2. If confidence < threshold → retry once.
    3. If still below threshold → return ``answer=None`` (LLM teacher path).
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._solvers: dict[str, Solver] = {}
        self._thresholds: dict[str, float] = {}
        self._fallback_model: str | None = None

        if config_path is not None and Path(config_path).exists():
            self._load_config(Path(config_path))

    def _load_config(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        categories = config.get("categories", {})
        for cat, settings in categories.items():
            if isinstance(settings, dict):
                threshold = settings.get("confidence_threshold", _DEFAULT_THRESHOLD)
                self._thresholds[cat] = float(threshold)
        self._fallback_model = config.get("llm_fallback_model")

    def register(self, category: str, solver: Solver) -> None:
        """Register a solver for a category (overrides any previous registration)."""
        self._solvers[category] = solver

    def _threshold_for(self, category: str) -> float:
        return self._thresholds.get(
            category,
            self._thresholds.get("default", _DEFAULT_THRESHOLD),
        )

    def route(self, prompt: str, category: str) -> SolverOutput:
        """Route a prompt to the appropriate solver, with one retry on low confidence.

        Returns an output with ``answer=None`` when no solver is registered or
        when confidence stays below threshold after the retry.
        """
        solver = self._solvers.get(category) or self._solvers.get("default")
        if solver is None:
            return SolverOutput.make(
                answer=None,
                confidence=0.0,
                metadata={"failure_mode": FailureMode.NO_ANSWER.value},
            )

        threshold = self._threshold_for(category)
        result = solver.solve(prompt, category)

        if result.get("confidence", 0.0) < threshold:
            # one retry
            result = solver.solve(prompt, category)

        if result.get("confidence", 0.0) < threshold:
            return SolverOutput.make(
                answer=None,
                confidence=result.get("confidence", 0.0),
                metadata={
                    "failure_mode": FailureMode.LOW_CONFIDENCE.value,
                    "original_answer": result.get("answer"),
                },
            )

        return SolverOutput.make(
            answer=result.get("answer"),
            confidence=result.get("confidence", 0.0),
            metadata=result.get("metadata", {}),
        )


__all__ = [
    "SolverOutput",
    "FailureMode",
    "Solver",
    "Verifier",
    "CategoryRouter",
]
