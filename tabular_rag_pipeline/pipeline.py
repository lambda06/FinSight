"""
TransactionRAGPipeline — The master orchestrator.

Every user query passes through exactly 10 stages:
  1.  Validate user       — does this user_id exist?
  2.  Cache check         — load or compute profile
  3.  Input guardrails    — block injection / cross-user / off-topic
  4.  Build prompt        — assemble system + user messages with real data
  5.  LLM call            — send to OpenRouter, get text + tool_calls back
  6.  Execute tool calls  — draw charts (the AI decided which ones)
  7.  Output guardrails   — flag hallucinations, toxicity, low confidence
  8.  Update cache        — store Q&A pair for future few-shot examples
  9.  Audit log           — write one JSONL line (no raw PII)
  10. Return result       — clean dict with response, charts, metadata

If a guardrail blocks at step 3, the pipeline returns a polite refusal
immediately without ever calling the LLM (fast and cheap).

If the LLM is unavailable at step 5, the pipeline returns a graceful
error response with whatever cached data is available.
"""

import time

from .data_store import DataStore
from .cache_manager import CacheManager
from .guardrails import GuardrailEngine
from .audit_logger import AuditLogger
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient
from .visualizations import VisualizationEngine, TOOL_SCHEMAS
from .exceptions import (
    UserNotFoundError,
    GuardrailViolationError,
    LLMUnavailableError,
    CircuitBreakerOpenError,
)
from . import config


class TransactionRAGPipeline:
    """
    End-to-end pipeline: user query → AI response + charts.

    All modules are initialised once at startup and reused across queries.
    This means the DataFrame is loaded once, the cache persists across queries,
    and the LLM connection is pooled.

    Usage:
        pipeline = TransactionRAGPipeline()
        result = pipeline.query("usr_a1b2c3d4", "What did I spend most on?")
    """

    def __init__(self):
        # Initialise all modules once — heavy lifting happens here, not per query
        self.store   = DataStore()
        self.cache   = CacheManager()
        self.guard   = GuardrailEngine()
        self.builder = PromptBuilder(self.store)
        self.viz     = VisualizationEngine(self.store)
        self.llm     = LLMClient()
        self.logger  = AuditLogger()

    def query(self, user_id: str, prompt: str) -> dict:
        """
        Process a natural language query for a specific user.

        Args:
            user_id: The user whose transactions to analyse
            prompt:  Natural language question

        Returns:
            {
                "user_name":       str,         # e.g. "Jose BazBaz"
                "response":        str,         # AI's text answer
                "visualizations":  list[str],   # file paths of generated charts
                "cache_hit":       bool,         # was the profile already cached?
                "latency_ms":      float,        # total wall-clock time
                "guardrail_flags": list[str],    # any flags raised
                "model_used":      str,          # which model responded
                "status":          str,          # "success" | "guardrail_blocked" | "llm_error"
            }
        """
        start_time = time.time()

        # ── Stage 1: Validate user ────────────────────────────────────────────
        try:
            self.store.validate_user(user_id)
        except UserNotFoundError as e:
            return self._error_result(str(e), "user_not_found", start_time)

        # ── Stage 2: Cache — load or compute profile ──────────────────────────
        cache_hit = self.cache.has_profile(user_id)
        if cache_hit:
            profile = self.cache.get_profile(user_id)
        else:
            profile = self.store.compute_user_profile(user_id)
            self.cache.set_profile(user_id, profile)

        user_name = profile["user_name"]

        # ── Stage 3: Input guardrails ─────────────────────────────────────────
        try:
            clean_prompt, input_flags = self.guard.check_input(user_id, prompt)
        except GuardrailViolationError as e:
            # Return polite refusal — LLM never called
            latency = self._ms(start_time)
            refusal = self.guard.get_refusal_message(e.flag)
            self.logger.log(
                user_id=user_id, prompt=prompt, response=refusal,
                latency_ms=latency, cache_hit=cache_hit,
                guardrail_flags=[e.flag], tool_calls=[],
                model_used="none", status="guardrail_blocked",
            )
            return {
                "user_name":       user_name,
                "response":        refusal,
                "visualizations":  [],
                "cache_hit":       cache_hit,
                "latency_ms":      latency,
                "guardrail_flags": [e.flag],
                "model_used":      "none",
                "status":          "guardrail_blocked",
            }

        # ── Stage 4: Build prompt ─────────────────────────────────────────────
        query_history  = self.cache.get_query_history(user_id)
        messages       = self.builder.build(user_id, clean_prompt, profile, query_history)
        data_summaries = self.builder.get_data_summaries(user_id)

        # ── Stage 5 & 6: LLM call and Tool Execution (Multi-Turn Loop) ────────
        import re
        viz_paths:  list[str] = []
        tool_names: list[str] = []
        viz_errors: list[str] = []
        model_used = "none"
        final_response_text = ""
        
        MAX_TURNS = 3
        turn_count = 0
        
        while turn_count < MAX_TURNS:
            turn_count += 1
            
            try:
                llm_result = self.llm.chat_completion(messages, tools=TOOL_SCHEMAS)
            except (LLMUnavailableError, CircuitBreakerOpenError) as e:
                print(f"LLM Error Details: {e}")
                latency = self._ms(start_time)
                msg = (
                    "I'm temporarily unable to reach the AI service. "
                    "Please try again in a moment."
                )
                self.logger.log(
                    user_id=user_id, prompt=clean_prompt, response=msg,
                    latency_ms=latency, cache_hit=cache_hit,
                    guardrail_flags=input_flags, tool_calls=[],
                    model_used="none", status="llm_error",
                )
                return self._error_result(msg, "llm_error", start_time,
                                          user_name=user_name, cache_hit=cache_hit,
                                          flags=input_flags)
            
            model_used = llm_result["model_used"]
            response_text = llm_result["text"]
            
            # Append the assistant's response (and tool calls) to the history
            if llm_result["tool_calls"] or response_text:
                assistant_msg = {"role": "assistant", "content": response_text}
                if llm_result["tool_calls"]:
                    assistant_msg["tool_calls"] = llm_result["tool_calls"]
                messages.append(assistant_msg)
                
            # If no tool calls were requested, the AI has provided its final answer!
            if not llm_result["tool_calls"]:
                final_response_text = response_text
                break
                
            # Otherwise, execute the requested tools and loop again
            for tc in llm_result["tool_calls"]:
                args = dict(tc["arguments"])
                args["user_id"] = user_id
                try:
                    path = self.viz.execute_tool_call(tc["function"], args)
                    viz_paths.append(path)
                    tool_names.append(tc["function"])
                    tool_result = f"Successfully generated chart at {path}"
                except Exception as e:
                    viz_errors.append(tc["function"])
                    tool_result = f"Error generating chart: {e}"
                    self.logger.log(
                        user_id=user_id, prompt=clean_prompt,
                        response=f"[chart_error] {tc['function']}: {e}",
                        latency_ms=self._ms(start_time), cache_hit=cache_hit,
                        guardrail_flags=["visualization_failed"], tool_calls=[tc["function"]],
                        model_used=model_used, status="visualization_failed",
                    )
                
                # Append tool result so the LLM can read it on the next turn
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"],
                    "content": tool_result
                })

        # Strip out any XML <thinking> tags from the final text
        response_text = re.sub(r'</?thinking>', '', final_response_text, flags=re.IGNORECASE).strip()
        
        # If we hit max turns or the text is still empty, add a fallback
        if not response_text and viz_paths:
            response_text = "I've generated the charts below based on your question."

        # ── Stage 7: Output guardrails ────────────────────────────────────────
        response_text, output_flags = self.guard.check_output(
            response_text, data_summaries
        )
        all_flags = input_flags + output_flags
        if viz_errors:
            all_flags.append("visualization_failed")

        # ── Stage 8: Update cache ─────────────────────────────────────────────
        self.cache.append_query_history(user_id, {
            "prompt":           clean_prompt,
            "response_summary": response_text[:200],  # Truncated — enough for few-shot context
        })
        if viz_paths:
            self.cache.set_viz_state(user_id, {
                "last_charts": viz_paths,
                "tool_names":  tool_names,
            })

        # ── Stage 9: Audit log ────────────────────────────────────────────────
        latency = self._ms(start_time)
        self.logger.log(
            user_id=user_id, prompt=clean_prompt, response=response_text,
            latency_ms=latency, cache_hit=cache_hit,
            guardrail_flags=all_flags, tool_calls=tool_names,
            model_used=model_used, status="success",
        )

        # ── Stage 10: Return ──────────────────────────────────────────────────
        return {
            "user_name":       user_name,
            "response":        response_text,
            "visualizations":  viz_paths,
            "cache_hit":       cache_hit,
            "latency_ms":      latency,
            "guardrail_flags": all_flags,
            "model_used":      model_used,
            "status":          "success",
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ms(self, start: float) -> float:
        """Wall-clock milliseconds since start."""
        return round((time.time() - start) * 1000, 1)

    def _error_result(
        self,
        message: str,
        status: str,
        start: float,
        user_name: str = "Unknown",
        cache_hit: bool = False,
        flags: list[str] | None = None,
    ) -> dict:
        """Return a structured error result dict."""
        return {
            "user_name":       user_name,
            "response":        message,
            "visualizations":  [],
            "cache_hit":       cache_hit,
            "latency_ms":      self._ms(start),
            "guardrail_flags": flags or [],
            "model_used":      "none",
            "status":          status,
        }
