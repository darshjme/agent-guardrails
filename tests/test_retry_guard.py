"""Tests for RetryGuard."""

import pytest
from agent_guardrails import RetryGuard, RetryGuardError, RetryExhausted
from agent_guardrails.retry_guard import _default_refiner


class TestRetryGuardBasic:
    def test_succeeds_on_first_attempt(self):
        call_count = [0]

        def llm(prompt: str) -> str:
            call_count[0] += 1
            return "success"

        guard = RetryGuard(llm, max_attempts=3)
        result = guard.run("test prompt")
        assert result == "success"
        assert call_count[0] == 1

    def test_retries_on_failure(self):
        call_count = [0]

        def llm(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not ready yet")
            return "finally succeeded"

        guard = RetryGuard(llm, max_attempts=3)
        result = guard.run("test")
        assert result == "finally succeeded"
        assert call_count[0] == 3

    def test_raises_retry_exhausted(self):
        def llm(prompt: str) -> str:
            raise ValueError("always fails")

        guard = RetryGuard(llm, max_attempts=2)
        with pytest.raises(RetryExhausted) as exc_info:
            guard.run("test")
        assert exc_info.value.attempts == 2
        assert len(exc_info.value.errors) == 2
        assert isinstance(exc_info.value.last_error, ValueError)

    def test_max_attempts_one_no_retry(self):
        call_count = [0]

        def llm(prompt: str) -> str:
            call_count[0] += 1
            raise RuntimeError("fail")

        guard = RetryGuard(llm, max_attempts=1)
        with pytest.raises(RetryExhausted):
            guard.run("test")
        assert call_count[0] == 1

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError):
            RetryGuard(lambda p: p, max_attempts=0)


class TestRetryGuardWithGuard:
    def test_run_with_guard_succeeds(self):
        def llm(prompt: str) -> str:
            return '{"value": 42}'

        def guard_fn(output: str) -> dict:
            import json
            return json.loads(output)

        retry = RetryGuard(llm, max_attempts=3)
        raw, validated = retry.run_with_guard("generate JSON", guard_fn)
        assert raw == '{"value": 42}'
        assert validated == {"value": 42}

    def test_run_with_guard_retries_on_validation_failure(self):
        call_count = [0]

        def llm(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] < 2:
                return "bad output"
            return '{"value": 99}'

        def guard_fn(output: str) -> dict:
            import json
            return json.loads(output)

        retry = RetryGuard(llm, max_attempts=3)
        raw, validated = retry.run_with_guard("generate JSON", guard_fn)
        assert validated == {"value": 99}
        assert call_count[0] == 2

    def test_on_retry_callback_called(self):
        retry_log = []

        def llm(prompt: str) -> str:
            if len(retry_log) < 1:
                raise ValueError("fail once")
            return "ok"

        def on_retry(attempt: int, error: Exception):
            retry_log.append((attempt, str(error)))

        retry = RetryGuard(llm, max_attempts=3, on_retry=on_retry)
        result = retry.run("test")
        assert result == "ok"
        assert len(retry_log) == 1


class TestDefaultRefiner:
    def test_default_refiner_appends_correction(self):
        refined = _default_refiner("original prompt", ValueError("bad output"))
        assert "original prompt" in refined
        assert "CORRECTION REQUIRED" in refined
        assert "bad output" in refined

    def test_custom_refiner_used(self):
        def my_refiner(prompt: str, error: Exception) -> str:
            return f"[RETRY] {prompt}"

        def llm(prompt: str) -> str:
            if not prompt.startswith("[RETRY]"):
                raise ValueError("needs retry")
            return "ok"

        retry = RetryGuard(llm, max_attempts=2, refiner=my_refiner)
        result = retry.run("original")
        assert result == "ok"
