"""
Unit tests for CacheManager.

Tests in-memory cache behavior: get/set, profile helpers,
query history eviction, viz state, and the stats utility.

Run:
    pytest tests/test_cache_manager.py -v
"""

import pytest
from tabular_rag_pipeline.cache_manager import CacheManager


@pytest.fixture
def cache():
    """Fresh CacheManager for each test."""
    return CacheManager(max_history=3)


# ── Low-Level Get / Set ────────────────────────────────────────────────────────

class TestLowLevelOps:
    def test_set_and_get(self, cache):
        cache.set("my_key", {"value": 42})
        assert cache.get("my_key") == {"value": 42}

    def test_get_missing_key_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_delete_removes_key(self, cache):
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_delete_nonexistent_does_not_raise(self, cache):
        cache.delete("never_existed")   # must not raise

    def test_clear_wipes_all_keys(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


# ── User Profile ───────────────────────────────────────────────────────────────

class TestUserProfile:
    def test_has_profile_false_initially(self, cache):
        assert cache.has_profile("usr_a") is False

    def test_has_profile_true_after_set(self, cache):
        cache.set_profile("usr_a", {"user_name": "Alice"})
        assert cache.has_profile("usr_a") is True

    def test_get_profile_returns_stored_value(self, cache):
        profile = {"user_name": "Alice", "avg_monthly_expense": 1500}
        cache.set_profile("usr_a", profile)
        assert cache.get_profile("usr_a") == profile

    def test_get_profile_miss_returns_none(self, cache):
        assert cache.get_profile("usr_unknown") is None

    def test_different_users_isolated(self, cache):
        cache.set_profile("usr_a", {"name": "Alice"})
        cache.set_profile("usr_b", {"name": "Bob"})
        assert cache.get_profile("usr_a")["name"] == "Alice"
        assert cache.get_profile("usr_b")["name"] == "Bob"


# ── Query History ──────────────────────────────────────────────────────────────

class TestQueryHistory:
    def test_empty_history_returns_empty_list(self, cache):
        assert cache.get_query_history("usr_a") == []

    def test_append_single_entry(self, cache):
        cache.append_query_history("usr_a", {"prompt": "Q1", "response_summary": "A1"})
        history = cache.get_query_history("usr_a")
        assert len(history) == 1
        assert history[0]["prompt"] == "Q1"

    def test_append_preserves_order(self, cache):
        for i in range(3):
            cache.append_query_history("usr_a", {"prompt": f"Q{i}", "response_summary": f"A{i}"})
        history = cache.get_query_history("usr_a")
        prompts = [h["prompt"] for h in history]
        assert prompts == ["Q0", "Q1", "Q2"]

    def test_evicts_oldest_when_over_max(self, cache):
        """max_history=3: after 4 entries, oldest (Q0) is dropped."""
        for i in range(4):
            cache.append_query_history("usr_a", {"prompt": f"Q{i}", "response_summary": ""})
        history = cache.get_query_history("usr_a")
        assert len(history) == 3
        prompts = [h["prompt"] for h in history]
        assert "Q0" not in prompts          # evicted
        assert "Q1" in prompts
        assert "Q3" in prompts

    def test_history_isolated_per_user(self, cache):
        cache.append_query_history("usr_a", {"prompt": "A question", "response_summary": ""})
        assert cache.get_query_history("usr_b") == []


# ── Visualization State ────────────────────────────────────────────────────────

class TestVizState:
    def test_viz_state_none_initially(self, cache):
        assert cache.get_viz_state("usr_a") is None

    def test_set_and_get_viz_state(self, cache):
        state = {"last_charts": ["output/chart.png"], "tool_names": ["plot_category_breakdown"]}
        cache.set_viz_state("usr_a", state)
        assert cache.get_viz_state("usr_a") == state

    def test_viz_state_overwritten(self, cache):
        cache.set_viz_state("usr_a", {"last_charts": ["old.png"]})
        cache.set_viz_state("usr_a", {"last_charts": ["new.png"]})
        assert cache.get_viz_state("usr_a")["last_charts"] == ["new.png"]


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_empty_cache(self, cache):
        stats = cache.stats()
        assert stats["total_keys"] == 0
        assert stats["keys"] == []

    def test_stats_reflects_stored_keys(self, cache):
        cache.set_profile("usr_a", {})
        cache.append_query_history("usr_a", {"prompt": "Q", "response_summary": "A"})
        stats = cache.stats()
        assert stats["total_keys"] == 2
        assert "user:usr_a:profile" in stats["keys"]
        assert "user:usr_a:query_history" in stats["keys"]
