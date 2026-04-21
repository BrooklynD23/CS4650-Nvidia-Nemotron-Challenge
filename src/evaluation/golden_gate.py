"""Strict golden regression gate.

Policy (see ``docs/execution/plans/issue-18-validation-and-golden-set.md``):

- A model run passes only if **every** golden example is correct under
  the frozen scoring contract.
- Any single miss blocks promotion, even if aggregate validation
  accuracy has improved.
- Nondeterminism (two records for the same golden ``example_id`` with
  different ``normalized_prediction``) is treated as a miss.

The gate consumes :class:`~src.contracts.EvalRecord` rows (the output
of the eval runner) plus the frozen :class:`SplitArtifactRow` rows
that define the golden set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.contracts import EvalRecord
from src.evaluation.splits import SplitArtifactRow


_MISS_REASON_MISSING = "missing"
_MISS_REASON_INCORRECT = "incorrect"
_MISS_REASON_NONDETERMINISTIC = "nondeterministic"


@dataclass(slots=True, frozen=True)
class GoldenGateResult:
    """Outcome of a single golden regression evaluation.

    ``misses`` is a list of small dicts (one per failing example) so it
    can be serialized straight into a report alongside an
    :class:`EvalRecord` stream without any extra schema work.
    """

    passed: bool
    total: int
    misses: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError(
                "GoldenGateResult.passed must be bool, got "
                f"{type(self.passed).__name__}"
            )
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise TypeError(
                "GoldenGateResult.total must be int, got "
                f"{type(self.total).__name__}"
            )
        if not isinstance(self.misses, list):
            raise TypeError(
                "GoldenGateResult.misses must be list, got "
                f"{type(self.misses).__name__}"
            )
        for i, miss in enumerate(self.misses):
            if not isinstance(miss, dict):
                raise TypeError(
                    f"GoldenGateResult.misses[{i}] must be dict, got "
                    f"{type(miss).__name__}"
                )


def _miss_entry(
    *,
    example_id: str,
    category: str,
    gold: str,
    normalized_prediction: str,
    reason: str,
) -> dict[str, str]:
    return {
        "example_id": example_id,
        "category": category,
        "gold": gold,
        "normalized_prediction": normalized_prediction,
        "reason": reason,
    }


def evaluate_golden_gate(
    records: Sequence[EvalRecord],
    golden: Sequence[SplitArtifactRow],
) -> GoldenGateResult:
    """Compute the strict pass/fail golden regression gate.

    Args:
        records: Flat stream of :class:`EvalRecord` rows from the run
            under evaluation. Records for non-golden examples are
            ignored.
        golden: The frozen golden split rows.

    Returns:
        A :class:`GoldenGateResult` with ``passed=False`` if any golden
        example is missing, incorrect, or has conflicting
        ``normalized_prediction`` values across records.
    """
    # Bucket records by example_id so we can detect nondeterminism cheaply.
    records_by_id: dict[str, list[EvalRecord]] = {}
    for rec in records:
        if not isinstance(rec, EvalRecord):
            raise TypeError(
                "evaluate_golden_gate: expected EvalRecord, got "
                f"{type(rec).__name__}"
            )
        records_by_id.setdefault(rec.example_id, []).append(rec)

    misses: list[dict[str, str]] = []
    for row in golden:
        if not isinstance(row, SplitArtifactRow):
            raise TypeError(
                "evaluate_golden_gate: expected SplitArtifactRow, got "
                f"{type(row).__name__}"
            )
        matching = records_by_id.get(row.example_id)
        if not matching:
            misses.append(
                _miss_entry(
                    example_id=row.example_id,
                    category=row.category,
                    gold=row.gold,
                    normalized_prediction="",
                    reason=_MISS_REASON_MISSING,
                )
            )
            continue

        # Nondeterminism: multiple records, differing normalized predictions.
        normalized_values = {rec.normalized_prediction for rec in matching}
        if len(normalized_values) > 1:
            # Pick a stable, sorted representative for the report.
            preview = " | ".join(sorted(normalized_values))
            misses.append(
                _miss_entry(
                    example_id=row.example_id,
                    category=row.category,
                    gold=row.gold,
                    normalized_prediction=preview,
                    reason=_MISS_REASON_NONDETERMINISTIC,
                )
            )
            continue

        # All records agree; check correctness. If *any* record for this
        # example id is marked incorrect, fail (strict gate).
        if not all(rec.correct for rec in matching):
            # Use the first record's normalized_prediction for the report.
            misses.append(
                _miss_entry(
                    example_id=row.example_id,
                    category=row.category,
                    gold=row.gold,
                    normalized_prediction=matching[0].normalized_prediction,
                    reason=_MISS_REASON_INCORRECT,
                )
            )

    return GoldenGateResult(
        passed=(len(misses) == 0),
        total=len(golden),
        misses=misses,
    )


def summarize_gate(result: GoldenGateResult) -> str:
    """Render a :class:`GoldenGateResult` as a human-readable summary.

    Intended for logs and PR comments. The first line is the verdict;
    subsequent lines enumerate any misses.
    """
    if not isinstance(result, GoldenGateResult):
        raise TypeError(
            "summarize_gate: expected GoldenGateResult, got "
            f"{type(result).__name__}"
        )
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"Golden gate: {verdict} "
        f"(passed={result.total - len(result.misses)}/{result.total})"
    ]
    if result.misses:
        lines.append(f"Misses: {len(result.misses)}")
        for miss in result.misses:
            lines.append(
                f"  - {miss.get('example_id', '?')} "
                f"[{miss.get('category', '?')}] "
                f"reason={miss.get('reason', '?')} "
                f"gold={miss.get('gold', '?')!r} "
                f"normalized_prediction={miss.get('normalized_prediction', '')!r}"
            )
    return "\n".join(lines)


__all__ = [
    "GoldenGateResult",
    "evaluate_golden_gate",
    "summarize_gate",
]
