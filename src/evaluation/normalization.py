"""Versioned normalization / parsing rules for eval predictions.

Scoring contract policy (see ``docs/execution/plans/issue-19-...``):

- Every change in observable normalization behavior MUST land as a new
  ``normalizer_id`` (e.g. ``exact_v1`` → ``exact_v2``). We never edit an
  existing named normalizer in place; doing so would silently rewrite
  history because record-level ``normalized_prediction`` values would
  change for the same raw inputs.
- The raw prediction is preserved on every :class:`EvalRecord`, so a
  dispute can always be replayed under a different normalizer without
  re-running the model.
- ``exact_v1`` is the strictest baseline (no transformation). Permissive
  versions (``strip_v1``, ``collapse_ws_v1``, ``last_line_v1``) make
  their relaxations explicit in their IDs.

Registration is in-process and deterministic: the built-in normalizers
are registered at import time. Custom normalizers can be registered by
callers, but re-registering an existing id fails closed.
"""

from __future__ import annotations

from collections.abc import Callable

Normalizer = Callable[[str, str | None], str]
"""Signature for a normalizer.

``(raw_prediction, category) -> normalized_prediction``. ``category`` is
allowed to be ``None`` for normalizers that have no category-specific
behavior.
"""


class NormalizerNotFoundError(KeyError):
    """Raised when :func:`get_normalizer` is asked for an unknown id."""


_REGISTRY: dict[str, Normalizer] = {}


def register_normalizer(normalizer_id: str, fn: Normalizer) -> None:
    """Register ``fn`` under ``normalizer_id``.

    Fails closed on duplicate ids so a version bump cannot be disguised
    as an in-place edit.
    """
    if not isinstance(normalizer_id, str) or not normalizer_id:
        raise ValueError(
            "register_normalizer: normalizer_id must be a non-empty str"
        )
    if not callable(fn):
        raise TypeError(
            "register_normalizer: fn must be callable, got "
            f"{type(fn).__name__}"
        )
    if normalizer_id in _REGISTRY:
        raise ValueError(
            f"register_normalizer: {normalizer_id!r} is already registered; "
            "bump the version id instead of overwriting"
        )
    _REGISTRY[normalizer_id] = fn


def get_normalizer(normalizer_id: str) -> Normalizer:
    """Return the registered normalizer callable."""
    if not isinstance(normalizer_id, str) or not normalizer_id:
        raise ValueError(
            "get_normalizer: normalizer_id must be a non-empty str"
        )
    try:
        return _REGISTRY[normalizer_id]
    except KeyError as exc:
        raise NormalizerNotFoundError(
            f"Unknown normalizer_id {normalizer_id!r}; "
            f"registered={sorted(_REGISTRY)}"
        ) from exc


def available_normalizers() -> list[str]:
    """Return the sorted list of currently registered normalizer ids."""
    return sorted(_REGISTRY)


def normalize(
    raw_prediction: str,
    *,
    normalizer_id: str,
    category: str | None = None,
) -> str:
    """Apply the named normalizer to ``raw_prediction``.

    Always returns a string. Fails closed on non-string input so a
    stray ``None`` from a predictor cannot be laundered into a blank
    normalized prediction.
    """
    if not isinstance(raw_prediction, str):
        raise TypeError(
            "normalize: raw_prediction must be str, got "
            f"{type(raw_prediction).__name__}"
        )
    if category is not None and not isinstance(category, str):
        raise TypeError(
            "normalize: category must be str or None, got "
            f"{type(category).__name__}"
        )
    fn = get_normalizer(normalizer_id)
    result = fn(raw_prediction, category)
    if not isinstance(result, str):
        raise TypeError(
            f"normalizer {normalizer_id!r} returned "
            f"{type(result).__name__}, expected str"
        )
    return result


# ---------------------------------------------------------------------------
# Built-in normalizers
# ---------------------------------------------------------------------------


def _exact_v1(raw: str, _category: str | None) -> str:
    """Strictest baseline: no transformation at all."""
    return raw


def _strip_v1(raw: str, _category: str | None) -> str:
    """Strip leading/trailing whitespace only."""
    return raw.strip()


def _collapse_ws_v1(raw: str, _category: str | None) -> str:
    """Strip edges and collapse any interior whitespace run to a single space."""
    return " ".join(raw.split())


def _last_line_v1(raw: str, _category: str | None) -> str:
    """Return the last non-empty line, stripped.

    Intended for the ``reasoning + final answer`` output shape. The raw
    prediction is still preserved on the :class:`EvalRecord`, so picking
    the last line is auditable.
    """
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


BUILTIN_NORMALIZERS: frozenset[str] = frozenset(
    {"exact_v1", "strip_v1", "collapse_ws_v1", "last_line_v1"}
)
"""Frozen set of ids registered at import time."""


# Register at import time so callers never need to remember to call a
# ``bootstrap()`` helper first.
register_normalizer("exact_v1", _exact_v1)
register_normalizer("strip_v1", _strip_v1)
register_normalizer("collapse_ws_v1", _collapse_ws_v1)
register_normalizer("last_line_v1", _last_line_v1)


def _unregister_for_tests(normalizer_id: str) -> None:
    """Test-only: drop a normalizer from the registry.

    Not part of the public API — here so tests can register a custom
    normalizer and then clean up after themselves without mutating
    ``_REGISTRY`` from the outside.
    """
    _REGISTRY.pop(normalizer_id, None)


__all__ = [
    "Normalizer",
    "NormalizerNotFoundError",
    "BUILTIN_NORMALIZERS",
    "register_normalizer",
    "get_normalizer",
    "available_normalizers",
    "normalize",
]
