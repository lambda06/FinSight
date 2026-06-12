"""Quick single-query test to verify the pipeline works end-to-end before running full demo."""
import sys, time, os
sys.path.insert(0, '.')

# Fix Windows terminal encoding (models sometimes return narrow no-break spaces)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tabular_rag_pipeline.pipeline import TransactionRAGPipeline

print("Loading pipeline...")
t0 = time.time()
pipeline = TransactionRAGPipeline()
print(f"Ready in {(time.time()-t0)*1000:.0f}ms\n")

print("Running query...")
t1 = time.time()
result = pipeline.query("usr_a1b2c3d4", "What did I spend the most on last month?")
elapsed = (time.time()-t1)*1000

print(f"Status:  {result['status']}")
print(f"Model:   {result['model_used']}")
print(f"Time:    {elapsed:.0f}ms")
print(f"Cache:   {'HIT' if result['cache_hit'] else 'MISS'}")
print(f"Charts:  {result['visualizations']}")
print(f"Flags:   {result['guardrail_flags']}")
print(f"\nResponse preview:\n{result['response'][:300]}")
