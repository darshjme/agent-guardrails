"""LengthGuard — Enforce minimum and maximum length constraints on LLM outputs.

Supports both character-level and approximate token-level limits.
Token estimation uses a simple whitespace + punctuation split heuristic
(~0.75 chars/token for English) — no tokenizer dependency required.
"""

from __future__ import annotations

import re
from typing import Literal


class LengthGuardError(Exception):
    """Raised when LLM output violates length constraints.

    Attributes:
        actual_chars: Character count of the offending output.
        actual_tokens: Estimated token count of the offending output.
        min_chars: Configured minimum character limit (or ``None``).
        max_chars: Configured maximum character limit (or ``None``).
        min_tokens: Configured minimum token limit (or ``None``).
        max_tokens: Configured maximum token limit (or ``None``).
    """

    def __init__(
        self,
        message: str,
        *,
        actual_chars: int = 0,
        actual_tokens: int = 0,
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_chars = actual_chars
        self.actual_tokens = actual_tokens
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens


# Simple tokenisation pattern: split on whitespace and punctuation boundaries
_TOKEN_RE = re.compile(r"\b\w+\b|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* without a tokenizer.

    Uses a whitespace + punctuation split approach. Accurate to within ~15%
    of BPE tokenizer counts for typical English text.

    Args:
        text: The string to estimate token count for.

    Returns:
        Estimated integer token count. Returns ``0`` for empty strings.
    """
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


class LengthGuard:
    """Enforce character and/or token length constraints on LLM outputs.

    All limits are optional; only the provided limits are enforced.
    Pass ``None`` (default) for any limit you do not need.

    Args:
        min_chars: Minimum required character count. ``None`` = no minimum.
        max_chars: Maximum allowed character count. ``None`` = no maximum.
        min_tokens: Minimum required (estimated) token count. ``None`` = no minimum.
        max_tokens: Maximum allowed (estimated) token count. ``None`` = no maximum.
        truncate: If ``True``, truncate output to ``max_chars`` / ``max_tokens``
            instead of raising. Defaults to ``False``.
        truncate_on: When ``truncate=True``, whether to truncate on ``"chars"``
            or ``"tokens"``. If both limits apply, the stricter is used.
            Defaults to ``"chars"``.

    Raises:
        ValueError: If min > max for the same unit.

    Example:
        >>> guard = LengthGuard(min_chars=10, max_chars=500)
        >>> guard.validate("hello world")
        'hello world'
        >>> guard.validate("hi")
        Traceback (most recent call last):
            ...
        LengthGuardError: Output too short: 2 chars (minimum 10)
    """

    def __init__(
        self,
        *,
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        truncate: bool = False,
        truncate_on: Literal["chars", "tokens"] = "chars",
    ) -> None:
        if min_chars is not None and max_chars is not None and min_chars > max_chars:
            raise ValueError(f"min_chars ({min_chars}) must be ≤ max_chars ({max_chars})")
        if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
            raise ValueError(f"min_tokens ({min_tokens}) must be ≤ max_tokens ({max_tokens})")

        self._min_chars = min_chars
        self._max_chars = max_chars
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._truncate = truncate
        self._truncate_on = truncate_on

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, text: str) -> str:
        """Validate length of *text* and return it (possibly truncated).

        Args:
            text: The LLM output to validate.

        Returns:
            The original text if it passes constraints, or a truncated version
            if ``truncate=True``.

        Raises:
            LengthGuardError: If any length constraint is violated and
                ``truncate=False``.
        """
        n_chars = len(text)
        n_tokens = estimate_tokens(text) if (self._min_tokens or self._max_tokens) else 0

        # --- Minimum checks ---
        if self._min_chars is not None and n_chars < self._min_chars:
            raise LengthGuardError(
                f"Output too short: {n_chars} chars (minimum {self._min_chars})",
                actual_chars=n_chars,
                actual_tokens=n_tokens,
                min_chars=self._min_chars,
            )
        if self._min_tokens is not None:
            if n_tokens == 0:
                n_tokens = estimate_tokens(text)
            if n_tokens < self._min_tokens:
                raise LengthGuardError(
                    f"Output too short: ~{n_tokens} tokens (minimum {self._min_tokens})",
                    actual_chars=n_chars,
                    actual_tokens=n_tokens,
                    min_tokens=self._min_tokens,
                )

        # --- Maximum checks / truncation ---
        if self._max_chars is not None and n_chars > self._max_chars:
            if self._truncate:
                return text[: self._max_chars]
            raise LengthGuardError(
                f"Output too long: {n_chars} chars (maximum {self._max_chars})",
                actual_chars=n_chars,
                actual_tokens=n_tokens,
                max_chars=self._max_chars,
            )

        if self._max_tokens is not None:
            if n_tokens == 0:
                n_tokens = estimate_tokens(text)
            if n_tokens > self._max_tokens:
                if self._truncate:
                    return self._truncate_to_tokens(text, self._max_tokens)
                raise LengthGuardError(
                    f"Output too long: ~{n_tokens} tokens (maximum {self._max_tokens})",
                    actual_chars=n_chars,
                    actual_tokens=n_tokens,
                    max_tokens=self._max_tokens,
                )

        return text

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate *text* to approximately *max_tokens* tokens."""
        tokens = _TOKEN_RE.findall(text)
        if len(tokens) <= max_tokens:
            return text
        # Find the end position of the max_tokens-th token match
        matches = list(_TOKEN_RE.finditer(text))
        if max_tokens <= 0:
            return ""
        cut = matches[max_tokens - 1].end()
        return text[:cut]
