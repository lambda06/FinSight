"""
tabular_rag_pipeline — Financial transaction analysis via RAG + tool-calling.

Quick start:
    from tabular_rag_pipeline import TransactionRAGPipeline

    pipeline = TransactionRAGPipeline()
    result   = pipeline.query("usr_a1b2c3d4", "What did I spend most on?")
"""

# Lazy import: only pull in the full pipeline (and its heavy deps like
# matplotlib) when explicitly requested. This lets test files import
# lightweight submodules (guardrails, cache_manager, etc.) without
# triggering the entire import chain.
from __future__ import annotations

__version__ = "1.0.0"
__all__      = ["TransactionRAGPipeline"]


def __getattr__(name: str):
    if name == "TransactionRAGPipeline":
        from .pipeline import TransactionRAGPipeline
        return TransactionRAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
