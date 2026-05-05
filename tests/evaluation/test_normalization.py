"""Drift-catching tests for :mod:`src.evaluation.normalization`.

Each test pins the observable behavior of a named normalizer. A version
bump (e.g. ``exact_v1`` → ``exact_v2``) must land as a new registered
normalizer, not as an invisible change to an existing one.
"""

from __future__ import annotations

import pytest

from src.evaluation.normalization import (
    BUILTIN_NORMALIZERS,
    NormalizerNotFoundError,
    available_normalizers,
    get_normalizer,
    normalize,
    register_normalizer,
)


class TestExactV1:
    """``exact_v1`` is the strictest normalizer: no transformation at all."""

    def test_preserves_leading_and_trailing_whitespace(self) -> None:
        # A naive parser that strips whitespace would make this pass
        # (and silently change historical scores) - the strict version
        # must keep the raw string untouched.
        assert normalize("  A  ", normalizer_id="exact_v1") == "  A  "

    def test_preserves_trailing_newline(self) -> None:
        # The plan calls this out explicitly: "A\n" -> "A" is a version
        # bump, not a silent behavior change.
        assert normalize("A\n", normalizer_id="exact_v1") == "A\n"

    def test_preserves_interior_whitespace(self) -> None:
        assert normalize("a  b", normalizer_id="exact_v1") == "a  b"

    def test_preserves_case(self) -> None:
        assert normalize("HeLLo", normalizer_id="exact_v1") == "HeLLo"

    def test_preserves_reasoning_prefix(self) -> None:
        # Under the strict normalizer, reasoning text is *not* stripped.
        raw = "let me think...\nfinal: A"
        assert normalize(raw, normalizer_id="exact_v1") == raw


class TestStripV1:
    """``strip_v1`` removes only leading/trailing whitespace."""

    def test_strips_leading_and_trailing(self) -> None:
        assert normalize("  A  ", normalizer_id="strip_v1") == "A"

    def test_strips_trailing_newline(self) -> None:
        assert normalize("A\n", normalizer_id="strip_v1") == "A"

    def test_preserves_interior_whitespace(self) -> None:
        # Interior whitespace collapse is NOT allowed in strip_v1.
        assert normalize("a  b", normalizer_id="strip_v1") == "a  b"

    def test_blank_input_normalizes_to_blank(self) -> None:
        assert normalize("   \n  ", normalizer_id="strip_v1") == ""


class TestCollapseWsV1:
    """``collapse_ws_v1`` is the permissive whitespace normalizer."""

    def test_collapses_interior_whitespace(self) -> None:
        assert normalize("a   b\tc", normalizer_id="collapse_ws_v1") == "a b c"

    def test_strips_edges_and_collapses(self) -> None:
        assert normalize("  foo   bar  ", normalizer_id="collapse_ws_v1") == "foo bar"


class TestLastLineV1:
    """``last_line_v1`` permits reasoning text preceding the final answer."""

    def test_extracts_last_nonempty_line(self) -> None:
        raw = "step 1: compute\nstep 2: check\nA"
        assert normalize(raw, normalizer_id="last_line_v1") == "A"

    def test_ignores_trailing_whitespace_lines(self) -> None:
        raw = "reasoning...\nA\n\n   \n"
        assert normalize(raw, normalizer_id="last_line_v1") == "A"

    def test_blank_input_returns_empty_string(self) -> None:
        assert normalize("   \n\n ", normalizer_id="last_line_v1") == ""


class TestDriftScenarios:
    """Scenarios that demonstrate a naive parser can silently mis-score."""

    def test_strict_vs_permissive_disagree_on_trailing_newline(self) -> None:
        raw = "A\n"
        gold = "A"
        assert normalize(raw, normalizer_id="exact_v1") != gold
        assert normalize(raw, normalizer_id="strip_v1") == gold

    def test_strict_vs_permissive_disagree_on_reasoning_prefix(self) -> None:
        raw = "thinking...\nanswer: 42\n42"
        gold = "42"
        # Strict: the whole blob is the prediction, so it fails.
        assert normalize(raw, normalizer_id="exact_v1") != gold
        # Strip alone also fails (trailing whitespace gone but prefix stays).
        assert normalize(raw, normalizer_id="strip_v1") != gold
        # Permissive last-line version agrees with gold.
        assert normalize(raw, normalizer_id="last_line_v1") == gold

    def test_interior_whitespace_only_changes_under_named_version(self) -> None:
        raw = "a  b"
        gold = "a b"
        assert normalize(raw, normalizer_id="strip_v1") != gold
        assert normalize(raw, normalizer_id="collapse_ws_v1") == gold


class TestRegistry:
    def test_unknown_id_raises(self) -> None:
        with pytest.raises(NormalizerNotFoundError, match="not_registered_v9"):
            get_normalizer("not_registered_v9")

    def test_available_includes_all_builtins(self) -> None:
        names = set(available_normalizers())
        assert BUILTIN_NORMALIZERS <= names

    def test_register_custom_normalizer(self) -> None:
        register_normalizer(
            "upper_v1",
            lambda raw, category: raw.upper(),
        )
        try:
            assert normalize("abc", normalizer_id="upper_v1") == "ABC"
            assert "upper_v1" in available_normalizers()
        finally:
            # Keep the registry clean for other tests.
            from src.evaluation.normalization import _unregister_for_tests

            _unregister_for_tests("upper_v1")

    def test_cannot_overwrite_existing_id(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_normalizer("exact_v1", lambda raw, category: raw)

    def test_normalize_rejects_non_string_input(self) -> None:
        with pytest.raises(TypeError, match="raw_prediction"):
            normalize(42, normalizer_id="exact_v1")  # type: ignore[arg-type]

    def test_normalize_rejects_empty_normalizer_id(self) -> None:
        with pytest.raises(ValueError, match="normalizer_id"):
            normalize("x", normalizer_id="")


class TestCategoryHook:
    """Category-specific behavior stays opt-in through the category arg."""

    def test_default_normalizer_ignores_category(self) -> None:
        assert (
            normalize("A", normalizer_id="exact_v1", category="binary")
            == normalize("A", normalizer_id="exact_v1", category="cipher")
        )

    def test_custom_normalizer_can_branch_on_category(self) -> None:
        def category_aware(raw: str, category: str | None) -> str:
            if category == "binary":
                return raw.replace(" ", "")
            return raw

        register_normalizer("cat_aware_v1", category_aware)
        try:
            assert (
                normalize("1 0 1", normalizer_id="cat_aware_v1", category="binary")
                == "101"
            )
            assert (
                normalize("1 0 1", normalizer_id="cat_aware_v1", category="cipher")
                == "1 0 1"
            )
        finally:
            from src.evaluation.normalization import _unregister_for_tests

            _unregister_for_tests("cat_aware_v1")
