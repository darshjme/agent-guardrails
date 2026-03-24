"""Tests for GuardrailChain."""

import pytest
from pydantic import BaseModel
from agent_guardrails import (
    GuardrailChain,
    GuardrailChainError,
    SchemaGuard,
    ContentGuard,
    LengthGuard,
)


class Reply(BaseModel):
    text: str
    score: float


class TestGuardrailChainBasic:
    def test_empty_chain_returns_input(self):
        chain = GuardrailChain([])
        assert chain.validate("hello") == "hello"

    def test_single_guard_passes(self):
        guard = LengthGuard(min_chars=1, max_chars=100)
        chain = GuardrailChain([guard.validate])
        result = chain.validate("hello world")
        assert result == "hello world"

    def test_single_guard_fails(self):
        guard = LengthGuard(max_chars=3)
        chain = GuardrailChain([guard.validate])
        with pytest.raises(GuardrailChainError) as exc_info:
            chain.validate("too long string")
        assert exc_info.value.guard_index == 0
        assert isinstance(exc_info.value.cause, Exception)

    def test_multiple_guards_all_pass(self):
        length = LengthGuard(min_chars=5, max_chars=500)
        content = ContentGuard(check_pii=False, check_toxic=False, check_injection=False)
        chain = GuardrailChain([length.validate, content.validate])
        result = chain.validate("The answer is 42.")
        assert result == "The answer is 42."

    def test_schema_guard_in_chain(self):
        schema = SchemaGuard(Reply)
        chain = GuardrailChain([schema.validate])
        result = chain.validate('{"text": "hello", "score": 0.9}')
        assert isinstance(result, Reply)
        assert result.text == "hello"

    def test_chain_stops_on_first_by_default(self):
        call_log = []

        def guard1(text):
            call_log.append("guard1")
            raise ValueError("guard1 fails")

        def guard2(text):
            call_log.append("guard2")
            return text

        chain = GuardrailChain([guard1, guard2], stop_on_first=True)
        with pytest.raises(GuardrailChainError):
            chain.validate("test")
        assert "guard2" not in call_log  # should NOT be called

    def test_chain_collects_all_errors(self):
        def guard1(text):
            raise ValueError("error 1")

        def guard2(text):
            raise ValueError("error 2")

        chain = GuardrailChain([guard1, guard2], stop_on_first=False)
        with pytest.raises(GuardrailChainError) as exc_info:
            chain.validate("test")
        assert "error 1" in str(exc_info.value)
        assert "error 2" in str(exc_info.value)

    def test_callable_alias(self):
        guard = LengthGuard(min_chars=1)
        chain = GuardrailChain([guard.validate])
        result = chain("hello")
        assert result == "hello"

    def test_len_returns_guard_count(self):
        chain = GuardrailChain([lambda x: x, lambda x: x, lambda x: x])
        assert len(chain) == 3

    def test_repr(self):
        chain = GuardrailChain([], name="MyChain")
        assert "MyChain" in repr(chain)


class TestGuardrailChainTransform:
    def test_schema_guard_transforms_output(self):
        """SchemaGuard returns a Pydantic object — the chain should forward it."""
        schema = SchemaGuard(Reply)
        results = []

        def capture(output):
            results.append(output)
            return output

        chain = GuardrailChain([schema.validate, capture])
        chain.validate('{"text": "hi", "score": 0.5}')
        assert len(results) == 1
        assert isinstance(results[0], Reply)

    def test_none_return_preserves_current(self):
        """A guard returning None should not overwrite current output."""
        def passthrough(text):
            return None  # no transformation

        chain = GuardrailChain([passthrough])
        result = chain.validate("original")
        assert result == "original"
