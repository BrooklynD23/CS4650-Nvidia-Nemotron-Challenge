"""Normalize raw dataset rows into the canonical contracts.

This module is the single place alias columns collapse onto the
canonical :class:`ReasoningExample` fields. Normalization rules for
the *content* of those fields (answer canonicalization, whitespace
policy, etc.) are out of scope here and live in issue-19.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.contracts import ReasoningExample, SFTExample


# Canonical field names expected on :class:`ReasoningExample`.
_CANONICAL_FIELDS: frozenset[str] = frozenset(
    {"id", "category", "prompt", "answer", "source", "split", "metadata"}
)


ALIAS_MAP: dict[str, str] = {
    "question": "prompt",
    "expected_answer": "answer",
}
"""Default ingest-time aliases.

These are the *only* aliases accepted at ingest; any other column is
preserved under ``metadata`` so ingest stays lossless.
"""


_REQUIRED: tuple[str, ...] = ("id", "prompt", "answer")


def _require_text_field(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"reasoning_example_from_row: field {field_name!r} must be str, "
            f"got {type(value).__name__}"
        )
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"none", "nan", "null"}:
        raise ValueError(
            f"reasoning_example_from_row: field {field_name!r} contains "
            "a missing/empty sentinel"
        )
    return value


def _require_category(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "reasoning_example_from_row: field 'category' must be str, "
            f"got {type(value).__name__}"
        )
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"none", "nan", "null"}:
        raise ValueError(
            "reasoning_example_from_row: field 'category' contains "
            "a missing/empty sentinel"
        )
    return value


def reasoning_example_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    split: str,
    alias_map: Mapping[str, str] | None = None,
) -> ReasoningExample:
    """Build a :class:`ReasoningExample` from a raw dataset row.

    Args:
        row: The raw mapping from a csv reader or equivalent.
        source: Dataset origin, e.g. ``"kaggle:train.csv"``.
        split: Split label, e.g. ``"train"``, ``"val"``, ``"golden"``.
        alias_map: Optional override for the default :data:`ALIAS_MAP`.

    Returns:
        A fully validated :class:`ReasoningExample`.

    Raises:
        ValueError: if a required canonical field is missing after
            applying aliases. The error message names the offending
            field(s).
    """
    effective_aliases: Mapping[str, str] = (
        alias_map if alias_map is not None else ALIAS_MAP
    )

    # Collapse aliases onto canonical names while preserving unknown
    # columns as metadata. We process the raw row into two buckets.
    canonical: dict[str, Any] = {}
    extra_metadata: dict[str, Any] = {}

    for key, value in row.items():
        target = effective_aliases.get(key, key)
        if target in _CANONICAL_FIELDS:
            # If both the alias and its canonical name appear, the
            # canonical name wins (don't silently overwrite).
            if target in canonical and key != target:
                # Canonical already present; drop the alias into metadata
                # to keep ingest lossless.
                extra_metadata[key] = value
            else:
                canonical[target] = value
        else:
            extra_metadata[key] = value

    # Required fields check (names in terms of the canonical schema).
    missing = [f for f in _REQUIRED if f not in canonical or canonical[f] is None]
    if missing:
        raise ValueError(
            "reasoning_example_from_row: missing required field(s): "
            f"{missing}. Provided row keys: {sorted(row.keys())}. "
            f"Active alias_map: {dict(effective_aliases)!r}"
        )

    # Derive category: prefer row column, else fall back to metadata hint
    # that a caller may have stashed for inference.
    category = canonical.get("category")
    if category is None:
        category = extra_metadata.pop("category", None)
    if category is None:
        # Last-resort fallback to keep ingest lossless; callers that
        # care should pass a category column explicitly.
        category = "unknown"
    else:
        category = _require_category(category)

    # Merge pre-existing metadata (if row had one) with the leftover
    # columns; row-provided metadata wins on direct conflicts.
    if "metadata" in canonical:
        provided_metadata = canonical["metadata"]
    else:
        provided_metadata = {}
    if not isinstance(provided_metadata, Mapping):
        raise ValueError(
            "reasoning_example_from_row: row['metadata'] must be a mapping"
            f" if provided; got {type(provided_metadata).__name__}"
        )
    merged_metadata: dict[str, Any] = dict(extra_metadata)
    merged_metadata.update(dict(provided_metadata))

    return ReasoningExample(
        id=_require_text_field(canonical["id"], "id"),
        category=category,
        prompt=_require_text_field(canonical["prompt"], "prompt"),
        answer=_require_text_field(canonical["answer"], "answer"),
        source=source,
        split=split,
        metadata=merged_metadata,
    )


def sft_example_from_reasoning(
    example: ReasoningExample,
    *,
    messages: list[dict[str, str]],
    completion: str,
    prompt_template_id: str,
    source: str | None = None,
) -> SFTExample:
    """Derive an :class:`SFTExample` from a :class:`ReasoningExample`.

    Args:
        example: The source reasoning example.
        messages: Rendered chat-template output (role/content entries).
        completion: Assistant target text to train on.
        prompt_template_id: Versioned template identifier that produced
            ``messages``. Recorded in ``provenance``.
        source: Optional override for the SFT-level source label;
            defaults to ``example.source``.

    Returns:
        A fully validated :class:`SFTExample`.
    """
    effective_source = source if source is not None else example.source
    provenance: dict[str, Any] = {
        "prompt_template_id": prompt_template_id,
        "source_example_id": example.id,
    }
    return SFTExample(
        example_id=example.id,
        category=example.category,
        messages=list(messages),
        completion=completion,
        source=effective_source,
        split=example.split,
        provenance=provenance,
    )


__all__ = [
    "ALIAS_MAP",
    "reasoning_example_from_row",
    "sft_example_from_reasoning",
]
