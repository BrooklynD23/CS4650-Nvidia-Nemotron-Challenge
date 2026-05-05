"""SFT training utilities — loss mask construction.

Token labels for non-assistant turns are set to IGNORE_INDEX (-100) so the
cross-entropy loss ignores prompt tokens during fine-tuning.
"""

from __future__ import annotations

IGNORE_INDEX: int = -100


def _find_subsequence(sequence: list[int], pattern: list[int]) -> list[int]:
    """Return all starting indices where *pattern* appears in *sequence*."""
    if not pattern:
        return []
    positions: list[int] = []
    plen = len(pattern)
    for i in range(len(sequence) - plen + 1):
        if sequence[i : i + plen] == pattern:
            positions.append(i)
    return positions


def apply_loss_mask(
    input_ids: list[int],
    labels: list[int],
    tokenizer: object,
    mask_role: str = "user",
) -> list[int]:
    """Return a new labels list with all non-assistant turn positions set to IGNORE_INDEX.

    Parameters
    ----------
    input_ids:
        Token IDs for the full conversation sequence.
    labels:
        Labels aligned positionally with *input_ids* (typically a copy).
    tokenizer:
        Must expose ``role_start_ids: dict[str, list[int]]`` — a mapping
        from role name to the token sequence that opens that role's block.
        Also needs an ``encode(text: str) -> list[int]`` method (unused at
        mask time but required by the interface contract).
    mask_role:
        The primary non-assistant role to mask; kept for API compatibility.
        All roles other than ``"assistant"`` are masked regardless of this
        value.

    Returns
    -------
    list[int]
        A new list of the same length as *labels* where every position
        belonging to a non-assistant turn has been replaced with
        ``IGNORE_INDEX``.

    Raises
    ------
    ValueError
        If *input_ids* contains no assistant turn according to the token
        boundaries reported by ``tokenizer.role_start_ids``.
    """
    role_start_ids: dict[str, list[int]] = getattr(tokenizer, "role_start_ids", {})

    # Collect (position, role) for every role-start marker found in input_ids.
    boundaries: list[tuple[int, str]] = []
    for role, marker in role_start_ids.items():
        for pos in _find_subsequence(input_ids, marker):
            boundaries.append((pos, role))

    if not any(role == "assistant" for _, role in boundaries):
        raise ValueError(
            "No assistant turn found in input_ids; cannot build SFT loss mask."
        )

    boundaries.sort(key=lambda x: x[0])

    new_labels: list[int] = list(labels)
    for idx, (start, role) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(input_ids)
        if role != "assistant":
            for j in range(start, end):
                new_labels[j] = IGNORE_INDEX

    return new_labels
