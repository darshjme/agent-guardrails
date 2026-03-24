"""RetryGuard — Retry LLM calls with a refined prompt on guardrail failure.

Wraps any callable that produces LLM output and automatically retries with
an enhanced prompt when a downstream guardrail raises an exception.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence, Type


class RetryGuardError(Exception):
    """Base class for retry-related errors."""


class RetryExhausted(RetryGuardError):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made.
        last_error: The exception from the final attempt.
        errors: All exceptions raised across all attempts.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception,
        errors: list[Exception],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.errors = errors


# Callable type: takes a prompt string, returns a string response
LLMCallable = Callable[[str], str]

# Callable that refines a prompt given the original prompt and the error
PromptRefiner = Callable[[str, Exception], str]


def _default_refiner(prompt: str, error: Exception) -> str:
    """Default prompt refinement strategy.

    Appends a correction note to the original prompt describing what went wrong.

    Args:
        prompt: The original prompt that produced the failing output.
        error: The exception raised by the guardrail.

    Returns:
        A refined prompt string.
    """
    return (
        f"{prompt}\n\n"
        f"[CORRECTION REQUIRED] Your previous response was rejected for the "
        f"following reason: {error}. "
        f"Please provide a corrected response that addresses this issue."
    )


class RetryGuard:
    """Retry an LLM call with a refined prompt on guardrail failure.

    Wraps a callable (your LLM client function) and automatically retries
    up to ``max_attempts`` times when the provided guardrails raise exceptions.

    Args:
        llm_fn: A callable ``(prompt: str) -> str`` representing your LLM call.
        max_attempts: Maximum number of total attempts (1 = no retry).
            Defaults to ``3``.
        catch: Exception type(s) to catch and retry on. Any other exception
            propagates immediately. Defaults to ``(Exception,)``.
        refiner: A callable ``(prompt, error) -> refined_prompt`` that builds
            the retry prompt. Defaults to ``_default_refiner``.
        delay: Seconds to wait between attempts. Defaults to ``0.0``.
        on_retry: Optional callback ``(attempt, error)`` called before each retry.

    Example:
        >>> def mock_llm(prompt: str) -> str:
        ...     return '{"name": "Alice", "age": 30}'
        >>>
        >>> from agent_guardrails import SchemaGuard
        >>> from pydantic import BaseModel
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> guard = SchemaGuard(User)
        >>> retry = RetryGuard(mock_llm, max_attempts=3)
        >>> raw = retry.run("Generate a user JSON")
        >>> user = guard.validate(raw)
        >>> assert user.name == "Alice"
    """

    def __init__(
        self,
        llm_fn: LLMCallable,
        *,
        max_attempts: int = 3,
        catch: tuple[Type[Exception], ...] = (Exception,),
        refiner: PromptRefiner = _default_refiner,
        delay: float = 0.0,
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be ≥ 1, got {max_attempts}")
        self._llm_fn = llm_fn
        self._max_attempts = max_attempts
        self._catch = catch
        self._refiner = refiner
        self._delay = delay
        self._on_retry = on_retry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> str:
        """Execute the LLM call, retrying on failure.

        Args:
            prompt: The initial prompt to send to the LLM.

        Returns:
            The raw LLM output string from the first successful attempt.

        Raises:
            RetryExhausted: If all attempts fail.
        """
        current_prompt = prompt
        errors: list[Exception] = []

        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._llm_fn(current_prompt)
                return result
            except self._catch as exc:
                errors.append(exc)
                if attempt < self._max_attempts:
                    if self._on_retry:
                        self._on_retry(attempt, exc)
                    if self._delay > 0:
                        time.sleep(self._delay)
                    current_prompt = self._refiner(current_prompt, exc)

        raise RetryExhausted(
            f"LLM call failed after {self._max_attempts} attempt(s). "
            f"Last error: {errors[-1]}",
            attempts=self._max_attempts,
            last_error=errors[-1],
            errors=errors,
        )

    def run_with_guard(
        self,
        prompt: str,
        guard_fn: Callable[[str], Any],
    ) -> tuple[str, Any]:
        """Run the LLM call and apply *guard_fn* to validate the output.

        On guardrail failure, the prompt is refined and the call is retried.
        Both the raw output and the guard's return value are returned on success.

        Args:
            prompt: The initial prompt to send to the LLM.
            guard_fn: A callable that takes the raw LLM output and either
                returns a validated result or raises an exception.

        Returns:
            A ``(raw_output, validated_result)`` tuple on success.

        Raises:
            RetryExhausted: If all attempts fail validation.
        """
        current_prompt = prompt
        errors: list[Exception] = []

        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._llm_fn(current_prompt)
                validated = guard_fn(raw)
                return raw, validated
            except self._catch as exc:
                errors.append(exc)
                if attempt < self._max_attempts:
                    if self._on_retry:
                        self._on_retry(attempt, exc)
                    if self._delay > 0:
                        time.sleep(self._delay)
                    current_prompt = self._refiner(current_prompt, exc)

        raise RetryExhausted(
            f"Guard validation failed after {self._max_attempts} attempt(s). "
            f"Last error: {errors[-1]}",
            attempts=self._max_attempts,
            last_error=errors[-1],
            errors=errors,
        )
