"""
CacheManager — User-specific key-value cache layer.

Stores three things per user:
  1. Profile      — name, spending summary, top categories (computed once, reused always)
  2. Query history — last N (question → answer) pairs (used as few-shot examples for LLM)
  3. Viz state    — last chart type used (keeps charts consistent across queries)

Implementation: plain Python dict (in-memory).
In production: swap this for Redis — the interface stays identical.
"""

from typing import Any, Optional
from . import config


class CacheManager:
    """
    In-memory key-value store with user-specific helper methods.

    All data lives in a single dict:
        {
            "user:usr_a1b2c3d4:profile":       { ... },
            "user:usr_a1b2c3d4:query_history":  [ ... ],
            "user:usr_a1b2c3d4:viz_state":      { ... },
        }

    The key naming convention makes it easy to see what's cached at a glance,
    and mirrors how Redis keys are typically structured in production.
    """

    def __init__(self, max_history: int = config.MAX_QUERY_HISTORY):
        self._store: dict[str, Any] = {}
        self.max_history = max_history              

    # ── Low-level get / set ────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return the value for key, or None if not found."""
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value under key."""
        self._store[key] = value

    def delete(self, key: str) -> None:
        """Remove a key from the cache (used in tests and cache invalidation)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Wipe the entire cache (useful for testing)."""
        self._store.clear()

    # ── User Profile ───────────────────────────────────────────────────────────

    def _profile_key(self, user_id: str) -> str:
        return f"user:{user_id}:profile"

    def has_profile(self, user_id: str) -> bool:
        """Return True if a profile has already been computed for this user."""
        return self._profile_key(user_id) in self._store

    def get_profile(self, user_id: str) -> Optional[dict]:
        """Return the cached profile dict, or None on cache miss."""
        return self.get(self._profile_key(user_id))

    def set_profile(self, user_id: str, profile: dict) -> None:
        """Cache the user's profile dict."""
        self.set(self._profile_key(user_id), profile)

    # ── Query History (few-shot examples) ─────────────────────────────────────

    def _history_key(self, user_id: str) -> str:
        return f"user:{user_id}:query_history"

    def get_query_history(self, user_id: str) -> list[dict]:
        """
        Return the list of past Q&A pairs for this user.

        Each entry is:
            {"prompt": "...", "response_summary": "..."}
        """
        return self.get(self._history_key(user_id)) or []

    def append_query_history(self, user_id: str, entry: dict) -> None:
        """
        Add a new Q&A pair to this user's history.

        Automatically trims to max_history entries (oldest removed first),
        so the cache doesn't grow forever.
        """
        history = self.get_query_history(user_id)
        history.append(entry)
        # Keep only the most recent N entries
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        self.set(self._history_key(user_id), history)

    # ── Visualization State ────────────────────────────────────────────────────

    def _viz_key(self, user_id: str) -> str:
        return f"user:{user_id}:viz_state"

    def get_viz_state(self, user_id: str) -> Optional[dict]:
        """
        Return the last chart parameters for this user, or None.

        Used to keep chart style/axes consistent across queries.
        """
        return self.get(self._viz_key(user_id))

    def set_viz_state(self, user_id: str, state: dict) -> None:
        """Update the viz state after generating a chart."""
        self.set(self._viz_key(user_id), state)

    # ── Utility ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a simple summary of what's in the cache (useful for debugging)."""
        return {
            "total_keys": len(self._store),
            "keys":       list(self._store.keys()),
        }
