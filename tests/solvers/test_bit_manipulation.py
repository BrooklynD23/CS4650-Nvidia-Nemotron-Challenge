"""Tests for src/solvers/bit_manipulation.py and CategoryRouter (#23, #9)."""

from __future__ import annotations

import pytest

from src.inference.solver import CategoryRouter, FailureMode, Solver, SolverOutput
from src.solvers.bit_manipulation import BitManipulationSolver


# ---------------------------------------------------------------------------
# BitManipulationSolver
# ---------------------------------------------------------------------------

def _prompt_lshift2() -> str:
    """Prompt with a left-shift-by-2 pattern: N → N*4."""
    return (
        "Find the rule:\n"
        "1 → 4\n"
        "2 → 8\n"
        "3 → 12\n"
        "What is 5 → ?"
    )


def _prompt_add3() -> str:
    return (
        "Find the rule:\n"
        "10 → 13\n"
        "20 → 23\n"
        "30 → 33\n"
        "What is 40 → ?"
    )


def _prompt_xor4() -> str:
    return (
        "Find the rule:\n"
        "5 → 1\n"
        "9 → 13\n"
        "12 → 8\n"
        "What is 7 → ?"
    )


def test_successful_rule_inference_lshift():
    solver = BitManipulationSolver(confidence_threshold=0.7)
    result = solver.solve(_prompt_lshift2(), "bit_manipulation")
    assert result["answer"] is not None
    assert r"\boxed{" in result["answer"]
    # 5 << 2 == 20
    assert "20" in result["answer"]
    assert result["confidence"] >= 0.7


def test_successful_rule_inference_add3():
    solver = BitManipulationSolver(confidence_threshold=0.7)
    result = solver.solve(_prompt_add3(), "bit_manipulation")
    assert result["answer"] is not None
    # 40 + 3 == 43
    assert "43" in result["answer"]


def test_successful_rule_xor4():
    solver = BitManipulationSolver(confidence_threshold=0.7)
    result = solver.solve(_prompt_xor4(), "bit_manipulation")
    assert result["answer"] is not None
    # 7 ^ 4 == 3
    assert "3" in result["answer"]


def test_no_solution_returns_none():
    # Contradictory examples — no rule can satisfy both
    prompt = (
        "Find the rule:\n"
        "5 → 10\n"
        "5 → 20\n"
        "What is 7 → ?"
    )
    solver = BitManipulationSolver(confidence_threshold=0.7)
    result = solver.solve(prompt, "bit_manipulation")
    # Either no_answer or ambiguous; answer must be None
    assert result["answer"] is None


def test_format_error_no_examples():
    prompt = "What is the answer?"
    solver = BitManipulationSolver()
    result = solver.solve(prompt, "bit_manipulation")
    assert result["answer"] is None
    assert result["metadata"].get("failure_mode") == FailureMode.FORMAT_ERROR.value


def test_format_error_no_query():
    # Has examples but no "→ ?" query
    prompt = "1 → 2\n2 → 4\n3 → 6"
    solver = BitManipulationSolver()
    result = solver.solve(prompt, "bit_manipulation")
    assert result["answer"] is None


def test_answer_wrapped_in_boxed():
    solver = BitManipulationSolver(confidence_threshold=0.5)
    result = solver.solve(_prompt_lshift2(), "bit_manipulation")
    if result["answer"] is not None:
        assert result["answer"].startswith(r"\boxed{")
        assert result["answer"].endswith("}")


def test_implements_solver_protocol():
    solver = BitManipulationSolver()
    assert isinstance(solver, Solver)


# ---------------------------------------------------------------------------
# CategoryRouter
# ---------------------------------------------------------------------------

class _MockSolver:
    """Deterministic mock solver for router testing."""

    def __init__(self, answer: str | None, confidence: float) -> None:
        self._answer = answer
        self._confidence = confidence

    def solve(self, prompt: str, category: str) -> SolverOutput:  # noqa: ARG002
        return SolverOutput.make(answer=self._answer, confidence=self._confidence)


def test_router_routes_to_registered_solver():
    router = CategoryRouter()
    router.register("bit_manipulation", _MockSolver(answer=r"\boxed{5}", confidence=0.9))
    result = router.route("1 → 2\nWhat is 2 → ?", "bit_manipulation")
    assert result["answer"] == r"\boxed{5}"


def test_router_returns_none_when_no_solver_registered():
    router = CategoryRouter()
    result = router.route("some prompt", "unknown_category")
    assert result["answer"] is None
    assert result["metadata"]["failure_mode"] == FailureMode.NO_ANSWER.value


def test_router_retries_on_low_confidence():
    call_count = {"n": 0}

    class CountingSolver:
        def solve(self, prompt: str, category: str) -> SolverOutput:
            call_count["n"] += 1
            # First call: low confidence; second call: high confidence
            if call_count["n"] == 1:
                return SolverOutput.make(answer=r"\boxed{3}", confidence=0.1)
            return SolverOutput.make(answer=r"\boxed{3}", confidence=0.9)

    router = CategoryRouter()
    router.register("math", CountingSolver())
    result = router.route("prompt", "math")
    assert call_count["n"] == 2
    assert result["answer"] == r"\boxed{3}"


def test_router_returns_none_after_two_low_confidence_attempts():
    router = CategoryRouter()
    router.register("math", _MockSolver(answer=r"\boxed{1}", confidence=0.1))
    result = router.route("prompt", "math")
    assert result["answer"] is None
    assert result["metadata"]["failure_mode"] == FailureMode.LOW_CONFIDENCE.value


def test_router_loads_yaml_thresholds(tmp_path):
    config = tmp_path / "solver_routing.yaml"
    config.write_text(
        "categories:\n"
        "  math:\n"
        "    confidence_threshold: 0.8\n"
        "  default:\n"
        "    confidence_threshold: 0.3\n",
        encoding="utf-8",
    )
    router = CategoryRouter(config_path=config)
    # Confidence 0.5 is below math threshold (0.8) but above default (0.3)
    router.register("math", _MockSolver(answer=r"\boxed{1}", confidence=0.5))
    result = router.route("prompt", "math")
    # Both attempts return 0.5 < 0.8 → answer=None
    assert result["answer"] is None

    router2 = CategoryRouter(config_path=config)
    router2.register("other", _MockSolver(answer=r"\boxed{2}", confidence=0.5))
    result2 = router2.route("prompt", "other")
    # 0.5 >= default threshold (0.3) → answer returned
    assert result2["answer"] == r"\boxed{2}"


def test_router_with_real_bit_manipulation_solver():
    router = CategoryRouter()
    solver = BitManipulationSolver(confidence_threshold=0.7)
    router.register("bit_manipulation", solver)
    router._thresholds["bit_manipulation"] = 0.7

    result = router.route(_prompt_lshift2(), "bit_manipulation")
    if result["answer"] is not None:
        assert r"\boxed{" in result["answer"]


def _prompt_lshift2() -> str:
    return (
        "Find the rule:\n"
        "1 → 4\n"
        "2 → 8\n"
        "3 → 12\n"
        "What is 5 → ?"
    )
