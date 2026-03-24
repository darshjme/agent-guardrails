"""GuardrailChain — Compose multiple guards in sequence.

Run an ordered pipeline of guardrail checks against LLM output.
Each guard in the chain receives the output from the previous guard
(allowing transforms), or the original text if a guard returns ``None``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


class GuardrailChainError(Exception):
    """Raised when any guard in the chain fails.

    Attributes:
        guard_index: 0-based index of the failing guard in the chain.
        guard_name: Name of the failing guard (class name or label).
        cause: The original exception raised by the guard.
    """

    def __init__(
        self,
        message: str,
        *,
        guard_index: int,
        guard_name: str,
        cause: Exception,
    ) -> None:
        super().__init__(message)
        self.guard_index = guard_index
        self.guard_name = guard_name
        self.cause = cause


# A guard is any callable: (text_or_output) -> Any
# It must either:
#   - Return a value (used as input to the next guard if it's a string, otherwise ignored)
#   - Raise an exception on failure
GuardCallable = Callable[[Any], Any]


class GuardrailChain:
    """Compose an ordered sequence of guardrails into a single pipeline.

    Each guard is called in sequence with the current output. If a guard
    returns a non-``None`` value, that value becomes the input to the next
    guard. This allows transformation guards (e.g. ``SchemaGuard`` parsing
    JSON into a Pydantic model) to coexist with pure-validation guards.

    Args:
        guards: Ordered sequence of guard callables. Each callable receives
            the current output and either returns a (potentially transformed)
            value or raises an exception.
        stop_on_first: If ``True`` (default), stop the chain on the first
            failing guard. If ``False``, collect all errors and raise a
            combined ``GuardrailChainError`` at the end.
        name: Optional human-readable name for this chain (used in error messages).

    Example:
        >>> from pydantic import BaseModel
        >>> from agent_guardrails import SchemaGuard, ContentGuard, LengthGuard, GuardrailChain
        >>>
        >>> class Reply(BaseModel):
        ...     text: str
        >>>
        >>> chain = GuardrailChain([
        ...     LengthGuard(min_chars=2, max_chars=500).validate,
        ...     ContentGuard().validate,
        ...     SchemaGuard(Reply).validate,
        ... ])
        >>> result = chain.validate('{"text": "Hello!"}')
        >>> assert result.text == "Hello!"
    """

    def __init__(
        self,
        guards: Sequence[GuardCallable],
        *,
        stop_on_first: bool = True,
        name: str = "GuardrailChain",
    ) -> None:
        self._guards = list(guards)
        self._stop_on_first = stop_on_first
        self._name = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, output: Any) -> Any:
        """Run *output* through all guards in sequence.

        Args:
            output: The initial LLM output (typically a ``str``). May be
                transformed by intermediate guards.

        Returns:
            The output after passing through all guards. If any guard returns
            a non-``None`` value, that value is forwarded to subsequent guards.

        Raises:
            GuardrailChainError: If any guard raises an exception (and
                ``stop_on_first=True``), or if ``stop_on_first=False`` and
                at least one guard failed (raises with the *first* failure as
                ``cause``).
        """
        current = output
        all_errors: list[tuple[int, str, Exception]] = []

        for idx, guard in enumerate(self._guards):
            guard_name = getattr(guard, "__self__", guard).__class__.__name__
            # Handle bound methods — use the class name of the instance
            if hasattr(guard, "__self__"):
                guard_name = guard.__self__.__class__.__name__
            elif hasattr(guard, "__name__"):
                guard_name = guard.__name__

            try:
                result = guard(current)
                if result is not None:
                    current = result
            except Exception as exc:  # noqa: BLE001
                if self._stop_on_first:
                    raise GuardrailChainError(
                        f"[{self._name}] Guard #{idx} ({guard_name}) failed: {exc}",
                        guard_index=idx,
                        guard_name=guard_name,
                        cause=exc,
                    ) from exc
                all_errors.append((idx, guard_name, exc))

        if all_errors:
            first_idx, first_name, first_exc = all_errors[0]
            summary = "; ".join(
                f"#{i} ({n}): {e}" for i, n, e in all_errors
            )
            raise GuardrailChainError(
                f"[{self._name}] {len(all_errors)} guard(s) failed: {summary}",
                guard_index=first_idx,
                guard_name=first_name,
                cause=first_exc,
            ) from first_exc

        return current

    def __call__(self, output: Any) -> Any:
        """Alias for :meth:`validate` — makes the chain itself callable.

        Args:
            output: The LLM output to validate.

        Returns:
            The validated (and possibly transformed) output.
        """
        return self.validate(output)

    def __len__(self) -> int:
        """Return the number of guards in the chain."""
        return len(self._guards)

    def __repr__(self) -> str:
        guard_names = [
            getattr(g, "__self__", g).__class__.__name__
            if hasattr(g, "__self__")
            else getattr(g, "__name__", repr(g))
            for g in self._guards
        ]
        return f"{self._name}([{', '.join(guard_names)}])"
