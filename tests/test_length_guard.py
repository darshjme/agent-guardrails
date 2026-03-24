"""Tests for LengthGuard."""

import pytest
from agent_guardrails import LengthGuard, LengthGuardError
from agent_guardrails.length_guard import estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        assert estimate_tokens("hello") == 1

    def test_sentence(self):
        count = estimate_tokens("The quick brown fox jumps over the lazy dog")
        assert count == 9  # 9 words

    def test_with_punctuation(self):
        count = estimate_tokens("Hello, world!")
        assert count >= 2  # at least "Hello" and "world"


class TestLengthGuardChars:
    def test_valid_text_passes(self):
        guard = LengthGuard(min_chars=5, max_chars=100)
        result = guard.validate("hello world")
        assert result == "hello world"

    def test_too_short_raises(self):
        guard = LengthGuard(min_chars=20)
        with pytest.raises(LengthGuardError) as exc_info:
            guard.validate("hi")
        assert exc_info.value.actual_chars == 2
        assert exc_info.value.min_chars == 20

    def test_too_long_raises(self):
        guard = LengthGuard(max_chars=10)
        with pytest.raises(LengthGuardError) as exc_info:
            guard.validate("this is a very long string")
        assert exc_info.value.max_chars == 10

    def test_truncate_chars(self):
        guard = LengthGuard(max_chars=5, truncate=True)
        result = guard.validate("hello world")
        assert result == "hello"
        assert len(result) == 5

    def test_exact_boundary_passes(self):
        guard = LengthGuard(min_chars=5, max_chars=5)
        result = guard.validate("hello")
        assert result == "hello"

    def test_no_limits_passes_anything(self):
        guard = LengthGuard()
        assert guard.validate("") == ""
        assert guard.validate("x" * 10000) == "x" * 10000


class TestLengthGuardTokens:
    def test_token_too_short_raises(self):
        guard = LengthGuard(min_tokens=10)
        with pytest.raises(LengthGuardError) as exc_info:
            guard.validate("hi there")
        assert exc_info.value.min_tokens == 10

    def test_token_too_long_raises(self):
        guard = LengthGuard(max_tokens=3)
        with pytest.raises(LengthGuardError) as exc_info:
            guard.validate("one two three four five")
        assert exc_info.value.max_tokens == 3

    def test_truncate_tokens(self):
        guard = LengthGuard(max_tokens=3, truncate=True)
        result = guard.validate("one two three four five")
        token_count = estimate_tokens(result)
        assert token_count <= 3


class TestLengthGuardValidation:
    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            LengthGuard(min_chars=100, max_chars=50)

    def test_min_tokens_greater_than_max_tokens_raises(self):
        with pytest.raises(ValueError):
            LengthGuard(min_tokens=100, max_tokens=50)
