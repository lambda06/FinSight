"""
Custom Exceptions — Typed error classes for the pipeline.

Each exception maps to a specific failure mode so callers can handle
errors precisely instead of catching a generic Exception.

Usage example:
    try:
        data = store.get_user_data(user_id)
    except UserNotFoundError as e:
        return {"error": str(e)}
"""


class FinSightError(Exception):
    """Base class for all FinSight errors."""


# ── Data Layer ─────────────────────────────────────────────────────────────────

class UserNotFoundError(FinSightError):
    """Raised when a user_id doesn't exist in the DataFrame."""


class EmptyDataError(FinSightError):
    """Raised when a query returns no rows (e.g. no transactions in a period)."""


# ── Guardrails ─────────────────────────────────────────────────────────────────

class GuardrailViolationError(FinSightError):
    """Raised when an input guardrail blocks a prompt."""

    def __init__(self, reason: str, flag: str):
        super().__init__(reason)
        self.flag = flag          # e.g. "prompt_injection_blocked"


class OutputGuardrailError(FinSightError):
    """Raised when an output guardrail flags the LLM's response."""

    def __init__(self, reason: str, flag: str):
        super().__init__(reason)
        self.flag = flag          # e.g. "hallucination_flagged"


# ── LLM Layer ──────────────────────────────────────────────────────────────────

class LLMUnavailableError(FinSightError):
    """Raised when the LLM is unreachable after all retries and fallbacks."""


class CircuitBreakerOpenError(FinSightError):
    """Raised when the circuit breaker is open (too many recent failures)."""
