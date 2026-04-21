"""Reserve and serialize held-out validation / golden split artifacts.

Issue #18 freezes a validation split and an immutable golden regression
set before any training occurs. Both splits are derived from the
canonical :class:`~src.contracts.ReasoningExample` corpus and written
to JSONL so any later eval run can re-load them deterministically on a
clean machine.

Selection policy
----------------

- Deterministic: same (examples, seed, sizes, rule) tuple yields the
  same ordered rows.
- Stratified by ``category``: rows are picked round-robin across
  categories (sorted alphabetically) so no one family dominates either
  split. Within each category, rows are shuffled with a
  seed-derived :class:`random.Random` instance.
- Disjoint: no ``example_id`` can appear in both the validation and
  golden artifacts, and neither set can overlap with ``train``.
- Provenance on every row: ``dataset_version``, ``selection_seed``,
  ``selection_rule``, and a human-readable ``selection_reason`` travel
  with each record so reviewers can trace selection back without a
  separate README.

See ``docs/execution/plans/issue-18-validation-and-golden-set.md``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.contracts import ReasoningExample


SplitLabel = Literal["val", "golden"]


@dataclass(slots=True, frozen=True)
class SplitArtifactRow:
    """One row inside a reserved split artifact.

    These rows are intentionally flatter than :class:`ReasoningExample`:
    eval runners should be able to load a split without dragging the
    raw metadata dictionary through the pipeline.
    """

    example_id: str
    category: str
    prompt: str
    gold: str
    source: str
    split: SplitLabel
    dataset_version: str
    selection_seed: int
    selection_rule: str
    selection_reason: str

    def __post_init__(self) -> None:
        for field_name in (
            "example_id",
            "category",
            "prompt",
            "gold",
            "source",
            "split",
            "dataset_version",
            "selection_rule",
            "selection_reason",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(
                    f"SplitArtifactRow.{field_name} must be str, got "
                    f"{type(value).__name__}"
                )
        if self.split not in ("val", "golden"):
            raise ValueError(
                f"SplitArtifactRow.split must be 'val' or 'golden', "
                f"got {self.split!r}"
            )
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed, int
        ):
            raise TypeError(
                "SplitArtifactRow.selection_seed must be int, got "
                f"{type(self.selection_seed).__name__}"
            )


def _validate_inputs(
    examples: Sequence[ReasoningExample],
    *,
    validation_size: int,
    golden_size: int,
) -> None:
    if validation_size < 1:
        raise ValueError(
            f"validation_size must be >= 1, got {validation_size}"
        )
    if golden_size < 1:
        raise ValueError(f"golden_size must be >= 1, got {golden_size}")
    if validation_size + golden_size > len(examples):
        raise ValueError(
            "validation_size + golden_size exceeds corpus size: "
            f"{validation_size} + {golden_size} > {len(examples)}"
        )

    seen_ids: set[str] = set()
    for ex in examples:
        if not isinstance(ex, ReasoningExample):
            raise TypeError(
                "reserve_splits: expected ReasoningExample, got "
                f"{type(ex).__name__}"
            )
        if ex.id in seen_ids:
            raise ValueError(
                f"reserve_splits: duplicate example_id detected: {ex.id!r}"
            )
        seen_ids.add(ex.id)

        # ReasoningExample already enforces types; here we check the
        # eval-specific invariants: no empty category/prompt.
        if ex.category is None or ex.category == "":
            raise ValueError(
                f"reserve_splits: row {ex.id!r} is missing 'category'"
            )
        if ex.prompt is None or ex.prompt == "":
            raise ValueError(
                f"reserve_splits: row {ex.id!r} is missing 'prompt'"
            )


def _group_by_category(
    examples: Sequence[ReasoningExample],
) -> dict[str, list[ReasoningExample]]:
    """Return a dict mapping category -> stable-ordered list of rows."""
    groups: dict[str, list[ReasoningExample]] = {}
    for ex in examples:
        groups.setdefault(ex.category, []).append(ex)
    return groups


def _shuffled(
    rows: Sequence[ReasoningExample], rng: random.Random
) -> list[ReasoningExample]:
    out = list(rows)
    rng.shuffle(out)
    return out


def _round_robin_pick(
    groups: dict[str, list[ReasoningExample]],
    count: int,
) -> list[ReasoningExample]:
    """Pick ``count`` rows round-robin across categories (sorted)."""
    if count == 0:
        return []
    # Categories in a deterministic order so the result is stable across
    # Python versions and dict orderings.
    categories = sorted(groups.keys())
    picked: list[ReasoningExample] = []
    # Use index cursors per category rather than mutating the lists
    # directly, so the caller's lists stay intact and popping is O(1).
    while len(picked) < count:
        made_progress = False
        for cat in categories:
            if len(picked) >= count:
                break
            bucket = groups[cat]
            if not bucket:
                continue
            picked.append(bucket.pop(0))
            made_progress = True
        if not made_progress:
            # Should be unreachable because _validate_inputs guarantees
            # corpus is large enough; defensive check anyway.
            raise RuntimeError(
                "reserve_splits: ran out of examples during round-robin "
                f"pick (wanted {count}, got {len(picked)})"
            )
    return picked


def _to_artifact_row(
    example: ReasoningExample,
    *,
    split: SplitLabel,
    dataset_version: str,
    selection_seed: int,
    selection_rule: str,
) -> SplitArtifactRow:
    reason = f"{split}:{selection_rule}; seed={selection_seed}"
    return SplitArtifactRow(
        example_id=example.id,
        category=example.category,
        prompt=example.prompt,
        gold=example.answer,
        source=example.source,
        split=split,
        dataset_version=dataset_version,
        selection_seed=selection_seed,
        selection_rule=selection_rule,
        selection_reason=reason,
    )


def reserve_splits(
    examples: Sequence[ReasoningExample],
    *,
    validation_size: int,
    golden_size: int,
    seed: int,
    dataset_version: str,
    selection_rule: str = "stratified-by-category",
) -> tuple[list[SplitArtifactRow], list[SplitArtifactRow]]:
    """Reserve the validation and golden rows from ``examples``.

    Returns a tuple ``(val_rows, golden_rows)``. Neither list contains
    any ``example_id`` that appears in the other, and neither touches
    the rest of the corpus (caller is responsible for treating the
    remainder as ``train``).

    Args:
        examples: The full canonical corpus (typically ``split=="train"``
            upstream rows that have not yet been reserved).
        validation_size: Exact number of rows for the validation split.
        golden_size: Exact number of rows for the golden regression set.
        seed: RNG seed; identical seeds yield identical row order.
        dataset_version: Version tag recorded on each reserved row.
        selection_rule: Human-readable rule name stamped on every row.

    Raises:
        TypeError: if any element is not a :class:`ReasoningExample`.
        ValueError: for size violations, duplicate IDs, or missing
            ``category`` / ``prompt`` fields.
    """
    _validate_inputs(
        examples,
        validation_size=validation_size,
        golden_size=golden_size,
    )

    rng = random.Random(seed)
    groups_for_golden = {
        cat: _shuffled(rows, rng)
        for cat, rows in sorted(_group_by_category(examples).items())
    }

    # Pick the golden set first so it has first claim on each category
    # (regression gate is the more restrictive artifact).
    golden_picks = _round_robin_pick(groups_for_golden, golden_size)
    golden_ids = {ex.id for ex in golden_picks}

    # Validation is drawn from the remaining rows in each category.
    remaining = [ex for ex in examples if ex.id not in golden_ids]
    groups_for_val = {
        cat: _shuffled(rows, rng)
        for cat, rows in sorted(_group_by_category(remaining).items())
    }
    val_picks = _round_robin_pick(groups_for_val, validation_size)

    val_rows = [
        _to_artifact_row(
            ex,
            split="val",
            dataset_version=dataset_version,
            selection_seed=seed,
            selection_rule=selection_rule,
        )
        for ex in val_picks
    ]
    golden_rows = [
        _to_artifact_row(
            ex,
            split="golden",
            dataset_version=dataset_version,
            selection_seed=seed,
            selection_rule=selection_rule,
        )
        for ex in golden_picks
    ]
    return val_rows, golden_rows


def write_split_jsonl(
    rows: Iterable[SplitArtifactRow],
    path: str | Path,
) -> int:
    """Write ``rows`` to ``path`` as JSONL. Returns the count written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if not isinstance(row, SplitArtifactRow):
                raise TypeError(
                    "write_split_jsonl: expected SplitArtifactRow, got "
                    f"{type(row).__name__}"
                )
            fh.write(json.dumps(asdict(row), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def read_split_jsonl(path: str | Path) -> list[SplitArtifactRow]:
    """Read a JSONL split artifact into a list of validated rows."""
    in_path = Path(path)
    rows: list[SplitArtifactRow] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"read_split_jsonl: invalid JSON at "
                    f"{in_path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"read_split_jsonl: line {line_no} is not a JSON "
                    f"object (got {type(data).__name__})"
                )
            rows.append(SplitArtifactRow(**data))
    return rows


__all__ = [
    "SplitArtifactRow",
    "reserve_splits",
    "write_split_jsonl",
    "read_split_jsonl",
]
