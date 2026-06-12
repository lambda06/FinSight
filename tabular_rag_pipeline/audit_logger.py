"""
AuditLogger — Logs every pipeline run to a JSON-lines file.

Why log?
  In production you need to know: who asked what, how fast it was,
  whether guardrails fired, which model was used. This is your audit trail.

Privacy design:
  - user_id is HASHED (SHA-256 first 16 chars) — never logged raw
  - The prompt is NEVER logged — only its length (char count)
  - Response is NEVER logged — only its length

Each log entry is one JSON object per line (JSONL format).
This makes it easy to grep, tail, and parse with any tool.

Example entry:
  {
    "timestamp":     "2025-12-15T10:30:00Z",
    "user_id_hash":  "a1b2c3d4e5f6g7h8",
    "prompt_chars":  42,
    "response_chars": 312,
    "latency_ms":    820,
    "cache_hit":     true,
    "guardrail_flags": [],
    "tool_calls":    ["plot_category_breakdown"],
    "model_used":    "google/gemini-2.0-flash-exp:free",
    "status":        "success"
  }
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config


class AuditLogger:
    """
    Writes one JSON line per pipeline run to logs/audit.jsonl.

    Usage:
        logger = AuditLogger()
        logger.log(
            user_id="usr_a1b2c3d4",
            prompt="What did I spend on?",
            response="Your top category was...",
            latency_ms=820,
            cache_hit=True,
            guardrail_flags=[],
            tool_calls=["plot_category_breakdown"],
            model_used="google/gemini-2.0-flash-exp:free",
            status="success",
        )
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or config.AUDIT_LOG_FILE
        # Create the logs directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        user_id: str,
        prompt: str,
        response: str,
        latency_ms: float,
        cache_hit: bool,
        guardrail_flags: list[str],
        tool_calls: list[str],
        model_used: str,
        status: str,          # "success", "guardrail_blocked", "llm_error", "fallback"
    ) -> None:
        """Write one audit log entry."""
        entry = {
            "timestamp":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user_id_hash":     self._hash(user_id),
            "prompt_chars":     len(prompt),       # length only — never the content
            "response_chars":   len(response),
            "latency_ms":       round(latency_ms, 1),
            "cache_hit":        cache_hit,
            "guardrail_flags":  guardrail_flags,
            "tool_calls":       tool_calls,
            "model_used":       model_used,
            "status":           status,
        }
        self._write(entry)

    def _hash(self, value: str) -> str:
        """SHA-256 hash, truncated to 16 hex chars — enough to be unique, not reversible."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _write(self, entry: dict) -> None:
        """Append one JSON line to the log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_recent(self, n: int = 10) -> list[dict]:
        """
        Return the last N log entries (useful for debugging / monitoring).
        Returns empty list if the log file doesn't exist yet.
        """
        if not self.log_file.exists():
            return []
        with open(self.log_file, encoding="utf-8") as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent if line.strip()]
