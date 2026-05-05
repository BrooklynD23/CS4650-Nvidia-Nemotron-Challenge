"""Canonical typed contracts for the Nemotron Challenge pipeline.

These are the four frozen dataclasses that represent the stable data
surface used by ingest, SFT construction, evaluation, and packaging.

Scope note: this module is schema-only. Normalization rules, prompt
rendering, and validation beyond presence/type shape are owned by
downstream modules (see issue-19 and issue-20).

See `docs/execution/plans/issue-17-schema-and-eda.md` for the
authoritative field tables and mapping policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MANIFEST_VERSION: str = "1.0.0"
"""Current schema version of :class:`PackageManifest` itself.

Bumped when the manifest's on-disk shape changes in a
backwards-incompatible way.
"""


def _require_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"field {field_name!r} must be str, got {type(value).__name__}"
        )


def _require_int(value: Any, field_name: str) -> None:
    # bool is a subclass of int; exclude it to catch obvious mistakes.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"field {field_name!r} must be int, got {type(value).__name__}"
        )


def _require_float(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"field {field_name!r} must be float, got {type(value).__name__}"
        )


def _require_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"field {field_name!r} must be bool, got {type(value).__name__}"
        )


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(
            f"field {field_name!r} must be dict, got {type(value).__name__}"
        )


def _require_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise TypeError(
            f"field {field_name!r} must be list, got {type(value).__name__}"
        )


@dataclass(slots=True, frozen=True)
class ReasoningExample:
    """The canonical raw-row contract for a competition example.

    Any dataset row (Kaggle csv, mirror, external) must normalize into
    this shape before any downstream stage consumes it.
    """

    id: str
    category: str
    prompt: str
    answer: str
    source: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_str(self.id, "id")
        _require_str(self.category, "category")
        _require_str(self.prompt, "prompt")
        _require_str(self.answer, "answer")
        _require_str(self.source, "source")
        _require_str(self.split, "split")
        _require_dict(self.metadata, "metadata")


@dataclass(slots=True, frozen=True)
class SFTExample:
    """A single SFT training row derived from a :class:`ReasoningExample`.

    ``messages`` holds the rendered chat template output; the raw
    problem text stays in the source :class:`ReasoningExample`.
    """

    example_id: str
    category: str
    messages: list[dict[str, str]]
    completion: str
    source: str
    split: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_str(self.example_id, "example_id")
        _require_str(self.category, "category")
        _require_list(self.messages, "messages")
        for i, msg in enumerate(self.messages):
            if not isinstance(msg, dict):
                raise TypeError(
                    f"messages[{i}] must be dict, got {type(msg).__name__}"
                )
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"messages[{i}] must contain both 'role' and 'content' keys"
                )
            _require_str(msg["role"], f"messages[{i}]['role']")
            _require_str(msg["content"], f"messages[{i}]['content']")
        _require_str(self.completion, "completion")
        _require_str(self.source, "source")
        _require_str(self.split, "split")
        _require_dict(self.provenance, "provenance")


@dataclass(slots=True, frozen=True)
class EvalRecord:
    """One evaluation result, per example and per run.

    Fields are deliberately flat so records can be serialized as JSONL.
    ``decode_config`` captures the exact knobs used so a sweep can be
    reproduced from any single record.
    """

    run_id: str
    example_id: str
    model_id: str
    prompt_template_id: str
    normalizer_id: str
    category: str
    split: str
    gold: str
    prediction: str
    normalized_prediction: str
    correct: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    seed: int
    decode_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_str(self.run_id, "run_id")
        _require_str(self.example_id, "example_id")
        _require_str(self.model_id, "model_id")
        _require_str(self.prompt_template_id, "prompt_template_id")
        _require_str(self.normalizer_id, "normalizer_id")
        _require_str(self.category, "category")
        _require_str(self.split, "split")
        _require_str(self.gold, "gold")
        _require_str(self.prediction, "prediction")
        _require_str(self.normalized_prediction, "normalized_prediction")
        _require_bool(self.correct, "correct")
        _require_float(self.latency_ms, "latency_ms")
        _require_int(self.tokens_in, "tokens_in")
        _require_int(self.tokens_out, "tokens_out")
        _require_int(self.seed, "seed")
        _require_dict(self.decode_config, "decode_config")


@dataclass(slots=True, frozen=True)
class PackageManifest:
    """Provenance card for an exported adapter/submission bundle.

    This is intentionally distinct from the submission payload itself:
    the manifest documents *what* was trained and *how* it was
    evaluated, so reviewers can trace back from any package to the
    exact code/data/eval state.
    """

    manifest_version: str
    base_model_id: str
    adapter_rank: int
    dataset_version: str
    eval_sha: str
    artifact_paths: dict[str, str]
    created_at: str

    def __post_init__(self) -> None:
        _require_str(self.manifest_version, "manifest_version")
        _require_str(self.base_model_id, "base_model_id")
        _require_int(self.adapter_rank, "adapter_rank")
        _require_str(self.dataset_version, "dataset_version")
        _require_str(self.eval_sha, "eval_sha")
        _require_dict(self.artifact_paths, "artifact_paths")
        for key, value in self.artifact_paths.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "artifact_paths must be dict[str, str]; "
                    f"offending entry: {key!r}: {value!r}"
                )
        _require_str(self.created_at, "created_at")


__all__ = [
    "MANIFEST_VERSION",
    "ReasoningExample",
    "SFTExample",
    "EvalRecord",
    "PackageManifest",
]
