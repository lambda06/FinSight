"""
LLMClient — Amazon Bedrock API integration via boto3 Converse API.

Responsibilities:
  - Send messages + tool schemas to the LLM via Amazon Bedrock
  - Parse tool_calls from the response
  - Retry on transient failures (network blip, brief 5xx)
  - Fall back through a model chain if the primary model is unavailable
  - Trip a circuit breaker after 3 consecutive failures
"""

import json
import os
import time
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, RetryError

from .exceptions import LLMUnavailableError, CircuitBreakerOpenError
from . import config

load_dotenv()


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on transient infrastructure failures."""
    if isinstance(exc, ClientError):
        error_code = exc.response.get('Error', {}).get('Code', 'Unknown')
        # ThrottlingException, InternalServerException, ServiceUnavailableException
        if error_code in ['ThrottlingException', 'InternalServerException', 'ServiceUnavailableException']:
            return True
        return False
    return isinstance(exc, BotoCoreError)


class LLMClient:
    """
    Wraps the Amazon Bedrock API with retry, model fallback, and a circuit breaker.
    """

    def __init__(self):
        # boto3 automatically picks up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION from env
        # No need to explicitly pass them if they are set, otherwise it will use default credential provider chain
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        self._check_circuit_breaker()

        model_chain = [config.PRIMARY_MODEL] + config.FALLBACK_MODELS
        last_error: Exception | None = None

        for model in model_chain:
            try:
                result = self._call_with_retry(model, messages, tools)
                self._consecutive_failures = 0
                return result
            except RetryError as e:
                original = e.last_attempt.exception()
                last_error = original
                if isinstance(original, ClientError):
                    error_code = original.response.get('Error', {}).get('Code')
                    if error_code in ['AccessDeniedException', 'ResourceNotFoundException']:
                        continue
                break
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code in ['AccessDeniedException', 'ResourceNotFoundException', 'ThrottlingException']:
                    last_error = e
                    continue
                last_error = e
                break
            except LLMUnavailableError as e:
                last_error = e
                continue

        self._consecutive_failures += 1
        if self._consecutive_failures >= config.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.time() + config.CIRCUIT_BREAKER_COOLDOWN

        raise LLMUnavailableError(
            f"All models failed after full fallback chain. Last error: {last_error}"
        )

    def _check_circuit_breaker(self) -> None:
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
        
        # Convert OpenAI messages to Bedrock Converse format
        system_prompts = []
        bedrock_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompts.append({"text": msg["content"]})
            elif msg["role"] == "assistant":
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "toolUse": {
                                "toolUseId": tc["id"],
                                "name": tc["function"],
                                "input": tc["arguments"]
                            }
                        })
                if content_blocks:
                    bedrock_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
            elif msg["role"] == "tool":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg["tool_call_id"],
                            "content": [{"text": msg.get("content", "Success")}]
                        }
                    }]
                })
            else:
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": msg["content"]}]
                })

        kwargs = {
            "modelId": model,
            "messages": bedrock_messages,
        }
        
        if system_prompts:
            kwargs["system"] = system_prompts

        if tools:
            # Convert OpenAI JSON schema tools to Bedrock tool format
            bedrock_tools = []
            for t in tools:
                # tools are given as {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    bedrock_tools.append({
                        "toolSpec": {
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "inputSchema": {
                                "json": fn.get("parameters", {})
                            }
                        }
                    })
            if bedrock_tools:
                kwargs["toolConfig"] = {
                    "tools": bedrock_tools,
                    "toolChoice": {"auto": {}}
                }

        try:
            response = self._client.converse(**kwargs)
        except Exception as e:
            raise

        return self._parse_response(response, model)

    def _parse_response(self, response: dict, model: str) -> dict:
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        text = ""
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append({
                    "id": tool_use.get("toolUseId"),
                    "function": tool_use.get("name"),
                    "arguments": tool_use.get("input", {}),
                })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "model_used": model,
        }
