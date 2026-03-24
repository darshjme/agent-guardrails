# Contributing to agent-guardrails

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/darshjme/agent-guardrails.git
cd agent-guardrails
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=agent_guardrails --cov-report=term-missing
```

## Code Style

- **Type hints** on every function (parameter + return)
- **Google-style docstrings** on every public class and method
- **Ruff** for linting: `ruff check .`
- **Mypy** for type checking: `mypy agent_guardrails/`

## Adding a New Guard

1. Create `agent_guardrails/your_guard.py`
2. Define a `YourGuardError(Exception)` with diagnostic attributes
3. Define `YourGuard` with a `validate(self, output)` method
4. Export from `agent_guardrails/__init__.py`
5. Add `tests/test_your_guard.py` with ≥ 8 test cases (happy + error paths)
6. Update `README.md` with a usage example

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-guard`
3. Make your changes with tests
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a PR with a clear description of what your guard does

## Reporting Bugs

Please open an issue with:
- Python version
- `agent-guardrails` version
- Minimal reproducible example
- Expected vs actual behaviour

## Roadmap

- [ ] `TokenGuard` — enforce exact tokenizer counts (tiktoken optional dep)
- [ ] `SemanticGuard` — cosine similarity to expected output (optional dep)
- [ ] `AsyncRetryGuard` — async-native retry with `asyncio`
- [ ] Streaming support for `LengthGuard`
