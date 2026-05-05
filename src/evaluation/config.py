"""Run configuration contract for the baseline eval pipeline.

Every eval run snapshots its own :class:`EvalRunConfig` alongside the
record-level :class:`~src.contracts.EvalRecord` stream. The invariant
enforced here is that the attribution fields stored on each record MUST
match the run config that produced it; otherwise aggregate summaries
would be ambiguous and impossible to re-attribute after the fact.

See ``docs/execution/plans/issue-19-baseline-eval-and-normalization.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import EvalRecord


# Attribution fields that must agree between a record and its run config.
# Listed explicitly (rather than reflected off the dataclass) so a future
# schema addition is an intentional change, not an implicit one.
_RECORD_ATTRIBUTION_FIELDS: tuple[str, ...] = (
    "run_id",
    "model_id",
    "prompt_template_id",
    "normalizer_id",
    "seed",
)

# Fields that live on the run config but not on the record. Captured on
# the run config only so one summary can speak to dataset_version and
# split without duplicating those strings on every row.
_RUN_ONLY_FIELDS: tuple[str, ...] = (
    "dataset_version",
    "split",
    "created_at",
)


@dataclass(slots=True, frozen=True)
class EvalRunConfig:
    """Snapshot of the knobs that define a single eval run.

    The record-level attribution on each :class:`EvalRecord` MUST line
    up with the corresponding fields here. ``decode_config`` is copied
    verbatim into every record's ``decode_config`` so any single record
    can be re-run without the original config file.
    """

    run_id: str
    model_id: str
    prompt_template_id: str
    normalizer_id: str
    seed: int
    split: str
    dataset_version: str
    decode_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "model_id",
            "prompt_template_id",
            "normalizer_id",
            "split",
            "dataset_version",
            "created_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(
                    f"EvalRunConfig.{field_name} must be str, got "
                    f"{type(value).__name__}"
                )
        for required_nonempty in (
            "run_id",
            "model_id",
            "prompt_template_id",
            "normalizer_id",
            "split",
            "dataset_version",
        ):
            if getattr(self, required_nonempty) == "":
                raise ValueError(
                    f"EvalRunConfig.{required_nonempty} must be non-empty"
                )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                "EvalRunConfig.seed must be int, got "
                f"{type(self.seed).__name__}"
            )
        if not isinstance(self.decode_config, dict):
            raise TypeError(
                "EvalRunConfig.decode_config must be dict, got "
                f"{type(self.decode_config).__name__}"
            )


def make_run_config(
    *,
    run_id: str,
    model_id: str,
    prompt_template_id: str,
    normalizer_id: str,
    seed: int,
    split: str,
    dataset_version: str,
    decode_config: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> EvalRunConfig:
    """Convenience constructor that stamps ``created_at`` in UTC."""
    stamp = created_at if created_at is not None else _utcnow_iso()
    return EvalRunConfig(
        run_id=run_id,
        model_id=model_id,
        prompt_template_id=prompt_template_id,
        normalizer_id=normalizer_id,
        seed=seed,
        split=split,
        dataset_version=dataset_version,
        decode_config=dict(decode_config or {}),
        created_at=stamp,
    )


def validate_record_matches_config(
    record: EvalRecord,
    config: EvalRunConfig,
) -> None:
    """Fail closed if ``record`` attribution diverges from ``config``.

    Mismatches are reported all at once so reviewers see the full drift
    surface instead of the first differing field.
    """
    if not isinstance(record, EvalRecord):
        raise TypeError(
            "validate_record_matches_config: expected EvalRecord, got "
            f"{type(record).__name__}"
        )
    if not isinstance(config, EvalRunConfig):
        raise TypeError(
            "validate_record_matches_config: expected EvalRunConfig, got "
            f"{type(config).__name__}"
        )
    mismatches: list[str] = []
    for name in _RECORD_ATTRIBUTION_FIELDS:
        rec_val = getattr(record, name)
        cfg_val = getattr(config, name)
        if rec_val != cfg_val:
            mismatches.append(
                f"{name}: record={rec_val!r} config={cfg_val!r}"
            )
    # Split is carried on the record too (used by golden gate); guard it.
    if record.split != config.split:
        mismatches.append(
            f"split: record={record.split!r} config={config.split!r}"
        )
    if mismatches:
        raise ValueError(
            "EvalRecord attribution does not match EvalRunConfig: "
            + "; ".join(mismatches)
        )


def validate_records_against_config(
    records: Iterable[EvalRecord],
    config: EvalRunConfig,
) -> None:
    """Validate a batch of records. Raises on the first mismatch found."""
    for record in records:
        validate_record_matches_config(record, config)


def write_run_config(config: EvalRunConfig, path: str | Path) -> Path:
    """Persist ``config`` as pretty-printed JSON. Returns the path written."""
    if not isinstance(config, EvalRunConfig):
        raise TypeError(
            "write_run_config: expected EvalRunConfig, got "
            f"{type(config).__name__}"
        )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return out_path


def read_run_config(path: str | Path) -> EvalRunConfig:
    """Load an :class:`EvalRunConfig` from a JSON file written by
    :func:`write_run_config`.
    """
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"read_run_config: {in_path} did not contain a JSON object"
        )
    return EvalRunConfig(**data)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "EvalRunConfig",
    "make_run_config",
    "validate_record_matches_config",
    "validate_records_against_config",
    "write_run_config",
    "read_run_config",
]
