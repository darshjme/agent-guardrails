"""Agent Guardrails — Production output validation and content safety for LLM agents.

This library provides a composable set of guardrails for validating and filtering
LLM outputs in production environments. It is designed to be lightweight, fast,
and dependency-minimal (only pydantic + stdlib).

Example:
    >>> from agent_guardrails import SchemaGuard, ContentGuard, GuardrailChain
    >>> from pydantic import BaseModel
    >>>
    >>> class Answer(BaseModel):
    ...     answer: str
    ...     confidence: float
    >>>
    >>> chain = GuardrailChain([SchemaGuard(Answer), ContentGuard()])
    >>> result = chain.validate('{"answer": "42", "confidence": 0.95}')
"""

from agent_guardrails.schema_guard import SchemaGuard, SchemaGuardError
from agent_guardrails.content_guard import ContentGuard, ContentGuardError, ContentViolation
from agent_guardrails.length_guard import LengthGuard, LengthGuardError
from agent_guardrails.retry_guard import RetryGuard, RetryGuardError, RetryExhausted
from agent_guardrails.chain import GuardrailChain, GuardrailChainError

__version__ = "0.1.0"
__author__ = "Darshankumar Joshi"
__email__ = "darshjme@gmail.com"
__license__ = "MIT"

__all__ = [
    "SchemaGuard",
    "SchemaGuardError",
    "ContentGuard",
    "ContentGuardError",
    "ContentViolation",
    "LengthGuard",
    "LengthGuardError",
    "RetryGuard",
    "RetryGuardError",
    "RetryExhausted",
    "GuardrailChain",
    "GuardrailChainError",
    "__version__",
]
