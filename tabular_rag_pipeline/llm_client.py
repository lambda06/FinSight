"""
LLMClient — OpenRouter API integration via OpenAI-compatible SDK.

Responsibilities:
  - Send messages + tool schemas to the LLM
  - Parse tool_calls from the response (the LLM decides which charts to draw)
  - Retry on transient failures (network blip, brief 5xx)
  - Fall back through a model chain if the primary model is unavailable
  - Trip a circuit breaker after 3 consecutive failures (stops hammering a dead service)

Why openai SDK instead of raw requests?
  OpenRouter is fully OpenAI-compatible. The SDK handles JSON parsing,
  type checking, streaming, and connection pooling automatically.
  Just point it at OpenRouter's base_url.

Retry strategy (two layers):
  Layer 1 — Model fallback:  Primary → Qwen → DeepSeek (sequential, same call style)
  Layer 2 — Tenacity retry:  Each model gets up to 3 attempts for transient errors
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, NotFoundError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, RetryError

from .exceptions import LLMUnavailableError, CircuitBreakerOpenError
from . import config


# Load .env file so OPENROUTER_API_KEY is available
load_dotenv()


def _is_retryable(exc: BaseException) -> bool:
    """
    Only retry on transient infrastructure failures.
    Never retry 404 (model not found) or 401 (bad key) — they will never succeed.
    """
    if isinstance(exc, NotFoundError):
        return False
    if isinstance(exc, APIStatusError) and exc.status_code in (401, 403, 404):
        return False
    return isinstance(exc, (APIConnectionError, APIStatusError))


class LLMClient:
    """
    Wraps the OpenRouter API with retry, model fallback, and a circuit breaker.

    Usage:
        client = LLMClient()
        result = client.chat_completion(messages, tools=TOOL_SCHEMAS)
        # result = {
        #     "text":       "Your top spending category was...",
        #     "tool_calls": [{"function": "plot_category_breakdown",
        #                     "arguments": {"user_id": "usr_a1b2c3d4", "months": 1}}],
        #     "model_used": "google/gemini-2.0-flash-exp:free"
        # }
    """

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found. "
                "Copy .env.example to .env and add your key."
            )

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0   # Unix timestamp

    # ── Public Interface ───────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Send messages to the LLM and return a structured response.

        Tries each model in the fallback chain until one succeeds.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            tools:    Optional list of tool schemas (TOOL_SCHEMAS from visualizations.py)

        Returns:
            {
                "text":       str,          # LLM's text response (may be empty if only tool calls)
                "tool_calls": list[dict],   # Parsed tool calls: [{"function": "...", "arguments": {...}}]
                "model_used": str,          # Which model actually responded
            }

        Raises:
            CircuitBreakerOpenError: if too many recent failures (stops retrying)
            LLMUnavailableError:     if all models + retries exhausted
        """
        self._check_circuit_breaker()

        model_chain = [config.PRIMARY_MODEL] + config.FALLBACK_MODELS
        last_error: Exception | None = None

        for model in model_chain:
            try:
                result = self._call_with_retry(model, messages, tools)
                # Success — reset the failure counter
                self._consecutive_failures = 0
                return result

            except RetryError as e:
                # Tenacity exhausted retries — unwrap to get the original exception
                original = e.last_attempt.exception()
                last_error = original
                # 404/401 on one model → try the next; others → stop
                if isinstance(original, (NotFoundError, RateLimitError)):
                    continue
                if isinstance(original, APIStatusError) and original.status_code in (503, 429):
                    continue
                # Unrecoverable error — stop trying models
                break

            except (NotFoundError, RateLimitError) as e:
                # Model not found or rate limited — try the next one
                last_error = e
                continue

            except APIStatusError as e:
                # 402 = provider quota exhausted, 503 = unavailable, 429 = rate limit
                # All these mean "try next model"
                if e.status_code in (402, 503, 429):
                    last_error = e
                    continue
                last_error = e
                break

            except LLMUnavailableError as e:
                last_error = e
                continue

        # All models failed
        self._consecutive_failures += 1
        if self._consecutive_failures >= config.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.time() + config.CIRCUIT_BREAKER_COOLDOWN

        raise LLMUnavailableError(
            f"All models failed after full fallback chain. Last error: {last_error}"
        )

    # ── Internal Methods ───────────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> None:
        """Raise if the circuit breaker is open (too many recent failures)."""
        if time.time() < self._circuit_open_until:
            remaining = int(self._circuit_open_until - time.time())
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open. Retry in {remaining}s. "
                "The LLM service is temporarily unavailable."
            )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _call_with_retry(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict:
        """
        Make a single API call with automatic retry on transient failures.

        Tenacity handles: network timeouts, brief 5xx errors.
        Retry schedule: 1s → 2s → 4s gaps (exponential backoff).

        Does NOT retry: 401 (bad key), 404 (model not found), 429 (rate limit —
        we handle those in the caller by switching models).
        """
        kwargs: dict = {
            "model":    model,
            "messages": messages,
            "timeout":  config.LLM_TIMEOUT_SECONDS,
        }

        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APIStatusError) as e:
            raise  # Let tenacity decide whether to retry via _is_retryable
        except Exception as e:
            raise LLMUnavailableError(f"LLM call failed: {e}") from e

        return self._parse_response(response, model)

    def _parse_response(self, response, model: str) -> dict:
        """
        Extract text and tool calls from the raw OpenAI response object.

        The openai SDK gives us structured objects — we convert them to
        plain dicts so the rest of the pipeline doesn't import openai types.
        """
        message = response.choices[0].message

        # Text content (may be None if the model only returned tool calls)
        text = message.content or ""

        # Parse tool calls into plain dicts
        tool_calls: list[dict] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id":        tc.id,
                    "function":  tc.function.name,
                    "arguments": args,
                })

        return {
            "text":       text,
            "tool_calls": tool_calls,
            "model_used": getattr(response, "model", model),
        }
