"""Bit-manipulation solver for category-specific rule induction (#9 / #26).

Given a problem prompt that presents input→output examples of bit operations,
this solver enumerates candidate transforms and returns the one that satisfies
ALL in-prompt examples. The answer is formatted as ``\\boxed{<value>}`` to
comply with the competition evaluation contract.
"""

from __future__ import annotations

import re
from typing import Any

from src.inference.solver import FailureMode, SolverOutput


# ── pattern for extracting "N → M" style examples from a prompt ─────────────
_EXAMPLE_RE = re.compile(r"(\d+)\s*[-→>]+\s*(\d+)")
# ── pattern for extracting the query value ("? = N" or "N = ?") ─────────────
_QUERY_RE = re.compile(r"(\d+)\s*[-→>]+\s*\?|\?\s*[-→>]+\s*(\d+)")


def _extract_examples(prompt: str) -> list[tuple[int, int]]:
    """Parse all input→output integer pairs from the prompt."""
    return [
        (int(m.group(1)), int(m.group(2)))
        for m in _EXAMPLE_RE.finditer(prompt)
    ]


def _extract_query(prompt: str, examples: list[tuple[int, int]]) -> int | None:
    """Extract the query input value (the one we must predict the output for)."""
    example_inputs = {inp for inp, _ in examples}
    for m in re.finditer(r"(\d+)\s*[-→>]+\s*\?", prompt):
        v = int(m.group(1))
        if v not in example_inputs:
            return v
    return None


def _candidate_transforms() -> list[tuple[str, Any]]:
    """Return (name, transform_fn) pairs to try as candidate rules."""
    transforms = []
    for shift in range(1, 17):
        transforms.append((f"lshift_{shift}", lambda x, s=shift: x << s))
        transforms.append((f"rshift_{shift}", lambda x, s=shift: x >> s))
    for delta in list(range(-32, 33)):
        transforms.append((f"add_{delta}", lambda x, d=delta: x + d))
    for mod in [2, 4, 8, 16, 32, 64, 128, 256]:
        transforms.append((f"xor_{mod}", lambda x, m=mod: x ^ m))
        transforms.append((f"and_{mod}", lambda x, m=mod: x & m))
        transforms.append((f"or_{mod}", lambda x, m=mod: x | m))
        transforms.append((f"mod_{mod}", lambda x, m=mod: x % m))
    for base in [2, 8, 16]:
        transforms.append((
            f"to_base_{base}_digit_sum",
            lambda x, b=base: sum(int(d) for d in _to_base(x, b) if d.isdigit()),
        ))
    transforms.append(("bit_count", lambda x: bin(x).count("1")))
    transforms.append(("bit_length", lambda x: x.bit_length()))
    transforms.append(("negate_low8", lambda x: (~x) & 0xFF))
    transforms.append(("reverse_bits_8", lambda x: int(f"{x & 0xFF:08b}"[::-1], 2)))
    return transforms


def _to_base(n: int, base: int) -> str:
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(str(n % base))
        n //= base
    return "".join(reversed(digits))


def _rule_confidence(
    fn: Any,
    examples: list[tuple[int, int]],
) -> float:
    """Return fraction of examples the transform satisfies (all-or-nothing gate)."""
    if not examples:
        return 0.0
    try:
        matches = sum(1 for inp, out in examples if fn(inp) == out)
    except Exception:
        return 0.0
    return matches / len(examples)


class BitManipulationSolver:
    """Solver for bit-manipulation category problems.

    Implements the ``Solver`` Protocol (structural — no explicit base class).
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self._threshold = confidence_threshold

    def solve(self, prompt: str, category: str) -> SolverOutput:  # noqa: ARG002
        examples = _extract_examples(prompt)
        if not examples:
            return SolverOutput.make(
                answer=None,
                confidence=0.0,
                metadata={"failure_mode": FailureMode.FORMAT_ERROR.value},
            )

        query = _extract_query(prompt, examples)
        if query is None:
            return SolverOutput.make(
                answer=None,
                confidence=0.0,
                metadata={"failure_mode": FailureMode.FORMAT_ERROR.value},
            )

        # Find all candidate rules that satisfy every example
        passing: list[tuple[str, Any, int]] = []
        for name, fn in _candidate_transforms():
            conf = _rule_confidence(fn, examples)
            if conf >= 1.0:
                try:
                    prediction = fn(query)
                    passing.append((name, fn, prediction))
                except Exception:
                    continue

        if not passing:
            return SolverOutput.make(
                answer=None,
                confidence=0.0,
                metadata={"failure_mode": FailureMode.NO_ANSWER.value},
            )

        # Ambiguous: multiple rules agree on the query value → pick it; else ambiguous
        predictions = {pred for _, _, pred in passing}
        if len(predictions) > 1:
            # Check if all passing rules agree despite different names
            top_pred = max(predictions, key=lambda p: sum(1 for _, _, pr in passing if pr == p))
            top_count = sum(1 for _, _, pr in passing if pr == top_pred)
            confidence = top_count / len(passing)
            if confidence < self._threshold:
                return SolverOutput.make(
                    answer=None,
                    confidence=confidence,
                    metadata={
                        "failure_mode": FailureMode.AMBIGUOUS.value,
                        "candidates": len(passing),
                    },
                )
            answer_val = top_pred
        else:
            answer_val = passing[0][2]
            confidence = 1.0

        return SolverOutput.make(
            answer=rf"\boxed{{{answer_val}}}",
            confidence=confidence,
            metadata={
                "rule": passing[0][0],
                "candidates_passing": len(passing),
            },
        )
