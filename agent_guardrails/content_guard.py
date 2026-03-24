"""ContentGuard — Detect PII, toxic content, and prompt injection in LLM outputs.

Uses compiled regex patterns for zero-latency, zero-dependency content scanning.
All detection is rule-based and fully auditable — no external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ViolationType(str, Enum):
    """Categories of content violations."""

    PII = "pii"
    TOXIC = "toxic"
    PROMPT_INJECTION = "prompt_injection"
    CUSTOM = "custom"


@dataclass
class ContentViolation:
    """Represents a single detected content violation.

    Attributes:
        type: The category of the violation.
        label: A short human-readable label (e.g. ``"email"``, ``"ssn"``).
        snippet: The matched text snippet (truncated for safety).
        start: Start index in the original text.
        end: End index in the original text.
    """

    type: ViolationType
    label: str
    snippet: str
    start: int
    end: int


class ContentGuardError(Exception):
    """Raised when LLM output contains one or more content violations.

    Attributes:
        violations: List of detected ``ContentViolation`` objects.
        text: The original text that triggered the error.
    """

    def __init__(self, message: str, violations: list[ContentViolation], text: str = "") -> None:
        super().__init__(message)
        self.violations = violations
        self.text = text


# ---------------------------------------------------------------------------
# Built-in pattern libraries
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str]] = [
    # Email addresses
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # US Social Security Numbers
    ("ssn", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    # US phone numbers (many formats)
    ("phone_us", r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
    # Credit card numbers (Visa, MC, Amex, etc.) — 13-19 digits with optional separators
    ("credit_card", r"\b(?:\d[ -]?){13,19}\b"),
    # IPv4 addresses
    ("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Passport-like patterns (e.g. US: letter + 8 digits)
    ("passport", r"\b[A-Z]{1,2}\d{6,9}\b"),
    # API keys / tokens — high-entropy strings common in LLM leakage
    ("api_key", r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b"),
]

_TOXIC_PATTERNS: list[tuple[str, str]] = [
    ("hate_speech", r"\b(?:kill\s+(?:all\s+)?(?:them|you|yourself|everyone)|die\s+(?:in\s+a\s+fire|now)|go\s+kill\s+yourself)\b"),
    ("slur_generic", r"\b(?:f[4a@]gg[o0]t|n[i1!]gg[ae3]r|ch[i1!]nk|sp[i1!]c)\b"),
    ("threat", r"\b(?:i(?:'ll|'m\s+going\s+to|will)\s+(?:kill|hurt|rape|murder)\s+you)\b"),
    ("self_harm", r"\b(?:how\s+to\s+(?:kill|hang|poison)\s+(?:my)?self|methods?\s+of\s+suicide)\b"),
]

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("ignore_instructions", r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions?|prompts?|context|rules?|directives?)\b"),
    ("new_instructions", r"(?i)\bnew\s+(?:instructions?|system\s+prompt|directives?)\s*[:\-]"),
    ("jailbreak_dan", r"(?i)\b(?:do\s+anything\s+now|jailbreak|pretend\s+you(?:'re|\s+are)\s+(?:an?\s+)?(?:AI|GPT|LLM)\s+without\s+restrictions?)\b"),
    ("role_override", r"(?i)\b(?:you\s+are\s+now|act\s+as|roleplay\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?(?:unrestricted|uncensored|evil|malicious|jailbroken|(?:\w+\s+)?without\s+restrictions?)\b"),
    ("system_prompt_leak", r"(?i)\brepeat\s+(?:your\s+)?(?:system\s+prompt|instructions?|initial\s+prompt)\b"),
]


class ContentGuard:
    """Scan LLM output for PII, toxic language, and prompt injection attempts.

    All scanning is performed locally using compiled regular expressions — no
    external API calls, no network latency.

    Args:
        check_pii: Enable PII detection. Defaults to ``True``.
        check_toxic: Enable toxic content detection. Defaults to ``True``.
        check_injection: Enable prompt injection detection. Defaults to ``True``.
        custom_patterns: Optional list of ``(label, regex_pattern)`` tuples for
            domain-specific detection. Classified as ``ViolationType.CUSTOM``.
        raise_on_violation: If ``True`` (default), raise ``ContentGuardError``
            when violations are found. If ``False``, return violations list
            instead of raising.

    Example:
        >>> guard = ContentGuard()
        >>> guard.validate("My email is john@example.com")
        Traceback (most recent call last):
            ...
        ContentGuardError: Detected 1 content violation(s): [pii/email]
    """

    def __init__(
        self,
        *,
        check_pii: bool = True,
        check_toxic: bool = True,
        check_injection: bool = True,
        custom_patterns: Sequence[tuple[str, str]] | None = None,
        raise_on_violation: bool = True,
    ) -> None:
        self._check_pii = check_pii
        self._check_toxic = check_toxic
        self._check_injection = check_injection
        self._raise_on_violation = raise_on_violation

        self._pii_re = self._compile(_PII_PATTERNS) if check_pii else []
        self._toxic_re = self._compile(_TOXIC_PATTERNS) if check_toxic else []
        self._injection_re = self._compile(_INJECTION_PATTERNS) if check_injection else []
        self._custom_re = self._compile(list(custom_patterns or []))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, text: str) -> list[ContentViolation]:
        """Scan *text* for content violations.

        Args:
            text: The LLM output to scan.

        Returns:
            An empty list if no violations are detected (or if
            ``raise_on_violation=False`` and violations exist, returns the
            violations list).

        Raises:
            ContentGuardError: If ``raise_on_violation=True`` and violations
                are detected.
        """
        violations: list[ContentViolation] = []
        violations.extend(self._scan(text, self._pii_re, ViolationType.PII))
        violations.extend(self._scan(text, self._toxic_re, ViolationType.TOXIC))
        violations.extend(self._scan(text, self._injection_re, ViolationType.PROMPT_INJECTION))
        violations.extend(self._scan(text, self._custom_re, ViolationType.CUSTOM))

        if violations and self._raise_on_violation:
            labels = ", ".join(f"{v.type.value}/{v.label}" for v in violations)
            raise ContentGuardError(
                f"Detected {len(violations)} content violation(s): [{labels}]",
                violations=violations,
                text=text,
            )
        # Return violations list when raise_on_violation=False, else return None
        # so that GuardrailChain preserves the current text value.
        if not self._raise_on_violation:
            return violations
        return None

    def is_safe(self, text: str) -> bool:
        """Return ``True`` if *text* passes all enabled checks.

        Args:
            text: The LLM output to scan.

        Returns:
            ``True`` if no violations detected, ``False`` otherwise.
        """
        guard = ContentGuard(
            check_pii=self._check_pii,
            check_toxic=self._check_toxic,
            check_injection=self._check_injection,
            raise_on_violation=False,
        )
        return len(guard.validate(text)) == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile(patterns: list[tuple[str, str]]) -> list[tuple[str, re.Pattern]]:
        return [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in patterns]

    @staticmethod
    def _scan(
        text: str,
        compiled: list[tuple[str, re.Pattern]],
        vtype: ViolationType,
    ) -> list[ContentViolation]:
        found: list[ContentViolation] = []
        for label, pattern in compiled:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                # Truncate snippet to avoid leaking sensitive data in errors
                safe_snippet = snippet[:8] + "…" if len(snippet) > 8 else snippet
                found.append(
                    ContentViolation(
                        type=vtype,
                        label=label,
                        snippet=safe_snippet,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return found
