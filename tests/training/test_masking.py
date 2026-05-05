"""Tests for src/training/sft_trainer.py — loss mask construction."""

from __future__ import annotations

import pytest

from src.training.sft_trainer import IGNORE_INDEX, apply_loss_mask


class MockTokenizer:
    """Minimal mock tokenizer with predictable turn boundaries.

    role_start_ids maps each role to a single-token marker:
      system    → [100]
      user      → [101]
      assistant → [102]
    """

    role_start_ids: dict[str, list[int]] = {
        "system": [100],
        "user": [101],
        "assistant": [102],
    }

    def encode(self, text: str) -> list[int]:
        return [ord(c) + 200 for c in text]


_TOKENIZER = MockTokenizer()

# Shared sequence: sys(100,1,2) + user(101,3,4) + asst(102,5,6)
_SYS_TOKENS = [100, 1, 2]
_USER_TOKENS = [101, 3, 4]
_ASST_TOKENS = [102, 5, 6]
_FULL_IDS = _SYS_TOKENS + _USER_TOKENS + _ASST_TOKENS
_FULL_LABELS = list(_FULL_IDS)


def test_user_tokens_masked() -> None:
    result = apply_loss_mask(_FULL_IDS, _FULL_LABELS, _TOKENIZER)
    user_start = len(_SYS_TOKENS)
    user_end = len(_SYS_TOKENS) + len(_USER_TOKENS)
    for i in range(user_start, user_end):
        assert result[i] == IGNORE_INDEX, f"user label at position {i} should be -100"


def test_assistant_tokens_unmasked() -> None:
    result = apply_loss_mask(_FULL_IDS, _FULL_LABELS, _TOKENIZER)
    asst_start = len(_SYS_TOKENS) + len(_USER_TOKENS)
    for i in range(asst_start, len(_FULL_IDS)):
        assert result[i] == _FULL_LABELS[i], (
            f"assistant label at position {i} should be unchanged"
        )


def test_system_tokens_masked() -> None:
    # Sequence with only system + assistant (no user turn).
    sys_ids = [100, 7, 8]
    asst_ids = [102, 9, 10]
    ids = sys_ids + asst_ids
    labels = list(ids)
    result = apply_loss_mask(ids, labels, _TOKENIZER)
    for i in range(len(sys_ids)):
        assert result[i] == IGNORE_INDEX, f"system label at position {i} should be -100"
    for i in range(len(sys_ids), len(ids)):
        assert result[i] == labels[i], (
            f"assistant label at position {i} should be unchanged"
        )


def test_all_masked_raises() -> None:
    # No assistant marker in sequence → ValueError.
    no_asst_ids = [101, 1, 2, 3]
    no_asst_labels = list(no_asst_ids)
    with pytest.raises(ValueError, match="No assistant turn"):
        apply_loss_mask(no_asst_ids, no_asst_labels, _TOKENIZER)
