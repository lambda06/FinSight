"""
Configuration — All configurable constants and settings.

Centralizes every tunable parameter so nothing is hardcoded in business logic.
Loaded once at startup; other modules import from here.
"""

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent          # FinSight/
DATA_DIR  = BASE_DIR / "data"                     # FinSight/data/
OUTPUT_DIR = BASE_DIR / "output"                  # FinSight/output/
LOG_DIR   = BASE_DIR / "logs"                     # FinSight/logs/

DATA_FILE = DATA_DIR / "assessment_transaction_data.xlsx"

PRIMARY_MODEL  = "amazon.nova-lite-v1:0"   # New, fast, cheap Amazon Nova Lite model
FALLBACK_MODELS = [
    "amazon.nova-micro-v1:0",              # Amazon Nova Micro fallback
]

LLM_TIMEOUT_SECONDS = 30    # Gemini BYOK: 2-5s | Free tier: up to 30s
LLM_MAX_RETRIES     = 2     # Retry once on transient errors

# ── Token Budget ───────────────────────────────────────────────────────────────
# We estimate tokens as len(text) / 4 — works across all models (Gemini, Qwen, etc.)
MAX_INPUT_TOKENS  = 8_000
MAX_OUTPUT_TOKENS = 1_000

# ── Input Guardrails ───────────────────────────────────────────────────────────
MAX_PROMPT_LENGTH = 2_000     # characters; truncated with warning if exceeded

# ── Cache ──────────────────────────────────────────────────────────────────────
MAX_QUERY_HISTORY = 5         # how many past Q&A pairs to keep per user

# ── Circuit Breaker ────────────────────────────────────────────────────────────
CIRCUIT_BREAKER_THRESHOLD = 3   # consecutive failures before tripping
CIRCUIT_BREAKER_COOLDOWN  = 60  # seconds before allowing another attempt

# ── Audit Log ──────────────────────────────────────────────────────────────────
AUDIT_LOG_FILE = LOG_DIR / "audit.jsonl"

# ── Hallucination Check ────────────────────────────────────────────────────────
HALLUCINATION_TOLERANCE_PCT = 0.02   # ±2%
HALLUCINATION_TOLERANCE_ABS = 5.0    # ±$5 (whichever is larger)
