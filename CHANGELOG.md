# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2024-03-24

### Added
- `SchemaGuard` — validate LLM JSON output against a Pydantic model; handles markdown-fenced JSON and embedded JSON extraction
- `ContentGuard` — detect PII (email, SSN, phone, credit card, API keys), toxic content, and prompt injection attempts using compiled regex patterns
- `LengthGuard` — enforce min/max character and token limits; optional truncation mode
- `RetryGuard` — automatically retry LLM calls with a refined prompt on guardrail failure (up to N attempts)
- `GuardrailChain` — compose multiple guards into a single ordered pipeline with transformation support
- Zero external dependencies beyond `pydantic` (stdlib only)
- Full type annotations throughout
- Google-style docstrings
- 45+ unit tests with 100% core-path coverage

[Unreleased]: https://github.com/darshjme/agent-guardrails/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/darshjme/agent-guardrails/releases/tag/v0.1.0
