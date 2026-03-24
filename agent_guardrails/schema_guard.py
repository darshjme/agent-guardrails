"""SchemaGuard — Validate LLM output against a Pydantic schema.

Ensures JSON-mode LLM outputs conform to a well-defined structure before
downstream processing. Handles both raw JSON strings and pre-parsed dicts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class SchemaGuardError(Exception):
    """Raised when LLM output fails schema validation.

    Attributes:
        raw: The original raw output that failed validation.
        errors: Pydantic validation errors, if applicable.
    """

    def __init__(self, message: str, raw: str = "", errors: list[dict] | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.errors = errors or []


class SchemaGuard(Generic[T]):
    """Validate LLM output against a Pydantic model schema.

    Strips markdown code fences, extracts JSON, and validates against the
    provided Pydantic model. Raises ``SchemaGuardError`` on any failure.

    Args:
        schema: A Pydantic ``BaseModel`` subclass to validate against.
        strict: If ``True``, disallow extra fields in the JSON output.
            Defaults to ``False``.
        extract_json: If ``True``, attempt to extract a JSON object from
            surrounding text (e.g. prose + JSON). Defaults to ``True``.

    Example:
        >>> from pydantic import BaseModel
        >>> class Reply(BaseModel):
        ...     text: str
        ...     score: float
        >>> guard = SchemaGuard(Reply)
        >>> obj = guard.validate('{"text": "hello", "score": 0.9}')
        >>> assert obj.score == 0.9
    """

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    _BARE_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)

    def __init__(
        self,
        schema: Type[T],
        *,
        strict: bool = False,
        extract_json: bool = True,
    ) -> None:
        self._schema = schema
        self._strict = strict
        self._extract_json = extract_json

    @property
    def schema(self) -> Type[T]:
        """The Pydantic model used for validation."""
        return self._schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, output: str | dict[str, Any]) -> T:
        """Validate *output* and return a parsed model instance.

        Args:
            output: Raw LLM text (JSON string, markdown-fenced JSON, or JSON
                embedded in prose) **or** a pre-parsed ``dict``.

        Returns:
            A validated instance of the configured Pydantic model.

        Raises:
            SchemaGuardError: If parsing or validation fails.
        """
        if isinstance(output, dict):
            return self._validate_dict(output)

        raw = output
        cleaned = self._clean(raw)
        data = self._parse_json(cleaned, raw)
        return self._validate_dict(data, raw=raw)

    def json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema representation of the target model.

        Useful for injecting into LLM system prompts.

        Returns:
            A JSON Schema dict for the configured Pydantic model.
        """
        return self._schema.model_json_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Strip markdown fences and surrounding whitespace."""
        match = self._JSON_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _parse_json(self, text: str, raw: str) -> dict[str, Any]:
        """Attempt JSON parsing; optionally extract embedded JSON."""
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise SchemaGuardError(
                    f"Expected a JSON object, got {type(parsed).__name__}",
                    raw=raw,
                )
            return parsed
        except json.JSONDecodeError:
            pass

        if self._extract_json:
            match = self._BARE_JSON_RE.search(text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        raise SchemaGuardError(
            "LLM output is not valid JSON and no JSON object could be extracted.",
            raw=raw,
        )

    def _validate_dict(self, data: dict[str, Any], raw: str = "") -> T:
        """Validate a parsed dict against the Pydantic schema."""
        try:
            if self._strict:
                return self._schema.model_validate(data, strict=True)
            return self._schema.model_validate(data)
        except ValidationError as exc:
            raise SchemaGuardError(
                f"Output does not conform to {self._schema.__name__}: {exc}",
                raw=raw,
                errors=exc.errors(),
            ) from exc
