"""Tests for SchemaGuard."""

import json
import pytest
from pydantic import BaseModel, Field
from typing import Optional

from agent_guardrails import SchemaGuard, SchemaGuardError


class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    title: str
    score: float
    tags: list[str] = Field(default_factory=list)
    meta: Optional[dict] = None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestSchemaGuardHappyPath:
    def test_valid_json_string(self):
        guard = SchemaGuard(SimpleModel)
        result = guard.validate('{"name": "Alice", "age": 30}')
        assert result.name == "Alice"
        assert result.age == 30

    def test_valid_dict_input(self):
        guard = SchemaGuard(SimpleModel)
        result = guard.validate({"name": "Bob", "age": 25})
        assert result.name == "Bob"

    def test_markdown_fenced_json(self):
        guard = SchemaGuard(SimpleModel)
        output = '```json\n{"name": "Carol", "age": 22}\n```'
        result = guard.validate(output)
        assert result.name == "Carol"

    def test_markdown_fenced_no_lang(self):
        guard = SchemaGuard(SimpleModel)
        output = '```\n{"name": "Dave", "age": 40}\n```'
        result = guard.validate(output)
        assert result.age == 40

    def test_json_embedded_in_prose(self):
        guard = SchemaGuard(SimpleModel)
        output = 'Here is your data: {"name": "Eve", "age": 28} — let me know if correct.'
        result = guard.validate(output)
        assert result.name == "Eve"

    def test_nested_model(self):
        guard = SchemaGuard(NestedModel)
        data = {"title": "Test", "score": 0.95, "tags": ["a", "b"]}
        result = guard.validate(json.dumps(data))
        assert result.score == 0.95
        assert result.tags == ["a", "b"]

    def test_extra_fields_allowed_by_default(self):
        guard = SchemaGuard(SimpleModel)
        result = guard.validate('{"name": "Frank", "age": 35, "extra": "ignored"}')
        assert result.name == "Frank"

    def test_json_schema_method(self):
        guard = SchemaGuard(SimpleModel)
        schema = guard.json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_schema_property(self):
        guard = SchemaGuard(SimpleModel)
        assert guard.schema is SimpleModel


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestSchemaGuardErrors:
    def test_invalid_json_raises(self):
        guard = SchemaGuard(SimpleModel)
        with pytest.raises(SchemaGuardError) as exc_info:
            guard.validate("not json at all")
        assert exc_info.value.raw == "not json at all"

    def test_wrong_type_raises(self):
        guard = SchemaGuard(SimpleModel)
        with pytest.raises(SchemaGuardError):
            guard.validate('{"name": "Alice", "age": "not_an_int"}')

    def test_missing_required_field(self):
        guard = SchemaGuard(SimpleModel)
        with pytest.raises(SchemaGuardError) as exc_info:
            guard.validate('{"name": "Alice"}')
        assert exc_info.value.errors  # Pydantic errors populated

    def test_json_array_raises(self):
        guard = SchemaGuard(SimpleModel)
        with pytest.raises(SchemaGuardError):
            guard.validate('[{"name": "Alice", "age": 30}]')

    def test_empty_string_raises(self):
        guard = SchemaGuard(SimpleModel)
        with pytest.raises(SchemaGuardError):
            guard.validate("")

    def test_error_has_raw_attribute(self):
        guard = SchemaGuard(SimpleModel)
        bad = '{"name": 1}'
        with pytest.raises(SchemaGuardError) as exc_info:
            guard.validate(bad)
        # SchemaGuardError.raw should be populated (either the input or empty)
        assert isinstance(exc_info.value.raw, str)
