"""
GuardrailEngine — Input and Output safety checks.

INPUT guardrails (run BEFORE the LLM sees anything):
  1. Prompt injection detection  — blocks "ignore previous instructions" style attacks
  2. Cross-user leakage check    — blocks "tell me about user_xyz's data"
  3. Scope enforcement           — redirects clearly off-topic queries (permissive)
  4. Input length limit          — truncates overly long prompts

OUTPUT guardrails (run AFTER the LLM responds):
  1. Hallucination check    — flags numbers in the response that don't match our data
  2. Toxicity filter        — flags inappropriate language
  3. Confidence gating      — flags responses full of uncertain language

Design principle: SIMPLE and EXPLICIT over clever.
  - Injection check: regex patterns against a known list
  - Hallucination: extract numbers, compare to data_summary values
  - Toxicity: keyword list
  No ML models, no external API calls — fast, predictable, offline.
"""

import re
from typing import Optional

from .exceptions import GuardrailViolationError
from . import config


# ── Injection Patterns ─────────────────────────────────────────────────────────
# Regex patterns that strongly indicate prompt injection attempts.
# Each pattern is case-insensitive (re.IGNORECASE applied at check time).
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"forget\s+(your\s+|all\s+)?instructions",
    r"you\s+are\s+now\s+a",
    r"disregard\s+(all\s+)?prior",
    r"override\s+(your\s+)?",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions",
    r"print\s+(your\s+)?system\s+prompt",
    r"jailbreak",
    r"dan\s+mode",
]

# ── Financial Keywords (for scope check) ──────────────────────────────────────
# If ANY of these appear, the query is considered financial — let it through.
_FINANCIAL_KEYWORDS = [
    "spend", "spent", "spending", "expense", "expenses",
    "income", "salary", "money", "cost", "costs", "paid", "pay",
    "transaction", "transactions", "budget", "saving", "savings",
    "category", "merchant", "purchase", "purchases", "bill", "bills",
    "monthly", "bank", "account", "transfer", "refund", "cashback",
    "financial", "finance", "trend", "report",
]

# ── Off-topic Categories (for scope check) ────────────────────────────────────
# Only used if ZERO financial keywords are found.
# We only block things that are obviously unrelated (cooking, politics, coding).
_OFFTOPIC_PATTERNS = [
    r"\brecipe\b", r"\bcook(ing)?\b", r"\bingredient",
    r"\bpolitics\b", r"\belection\b", r"\bpresident\b",
    r"\bcode\b|\bprogramming\b|\bpython\b|\bjavascript\b",
    r"\bweather\b|\bforecast\b",
    r"\bsport(s)?\b|\bfootball\b|\bbasketball\b|\bsoccer\b",
]

# ── Toxicity Keywords ─────────────────────────────────────────────────────────
# Lightweight list. Extend as needed.
_TOXIC_KEYWORDS = [
    "hate", "kill", "murder", "racist", "sexist",
    "idiot", "stupid", "dumb", "moron",
]

# ── Uncertainty Phrases (for confidence gating) ───────────────────────────────
_UNCERTAINTY_PHRASES = [
    r"i('m| am) not sure",
    r"i think\b",
    r"\bpossibly\b",
    r"\bmaybe\b",
    r"\bmight be\b",
    r"\bi believe\b",
    r"\bperhaps\b",
    r"\buncertain\b",
]


class GuardrailEngine:
    """
    Runs safety checks on inputs and outputs.

    Usage:
        guard = GuardrailEngine()

        # Before LLM call
        clean_prompt, flags = guard.check_input(user_id, prompt)

        # After LLM call
        safe_response, flags = guard.check_output(response_text, data_summary)
    """

    # ── Input Guardrails ───────────────────────────────────────────────────────

    def check_input(self, user_id: str, prompt: str) -> tuple[str, list[str]]:
        """
        Run all input guardrails on the user's prompt.

        Returns:
            (clean_prompt, flags) — clean_prompt may be truncated.

        Raises:
            GuardrailViolationError — if a blocking guard fires (injection,
            cross-user, or scope). The pipeline catches this and returns a
            polite refusal without ever calling the LLM.
        """
        flags: list[str] = []

        # 1. Injection detection (blocking)
        self._check_injection(prompt)

        # 2. Cross-user leakage (blocking)
        self._check_cross_user(user_id, prompt)

        # 3. Scope enforcement (blocking, but permissive)
        self._check_scope(prompt)

        # 4. Length limit (non-blocking — truncates and adds a flag)
        prompt, length_flags = self._check_length(prompt)
        flags.extend(length_flags)

        return prompt, flags

    def _check_injection(self, prompt: str) -> None:
        """Block known prompt injection patterns."""
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise GuardrailViolationError(
                    reason="Prompt injection attempt detected.",
                    flag="prompt_injection_blocked",
                )

    def _check_cross_user(self, current_user_id: str, prompt: str) -> None:
        """
        Block attempts to access another user's data.

        Catches two cases:
          a) A user_id pattern in the prompt that's NOT the current user
          b) Generic phrases like "other user", "all users"
        """
        # Case a: explicit user_id mentioned
        mentioned_ids = re.findall(r"usr_\w+", prompt)
        for uid in mentioned_ids:
            if uid != current_user_id:
                raise GuardrailViolationError(
                    reason=f"Cross-user data access attempt (mentioned '{uid}').",
                    flag="cross_user_blocked",
                )

        # Case b: generic multi-user phrases
        cross_user_phrases = [
            r"other\s+user", r"another\s+user", r"all\s+users",
            r"every\s+user", r"other\s+people('s)?",
            # Only flag when a proper name/ID follows — not first-person pronouns
            r"tell\s+me\s+about\s+(?!my\b|your\b|our\b|their\b)\w+('s)?\s+spending",
            r"show\s+me\s+(?!my\b|your\b|our\b|their\b)\w+('s)?\s+(spending|transactions|data)",
        ]
        for pattern in cross_user_phrases:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise GuardrailViolationError(
                    reason="Attempt to access other users' data.",
                    flag="cross_user_blocked",
                )

    def _check_scope(self, prompt: str) -> None:
        """
        Block clearly off-topic queries.

        Permissive logic:
          - If the prompt contains ANY financial keyword → pass through (don't overthink it)
          - Only block if ZERO financial keywords AND positive off-topic match
          This avoids false positives like "How does the weather affect my spending?"
        """
        prompt_lower = prompt.lower()

        # If any financial keyword found → definitely in scope
        has_financial = any(kw in prompt_lower for kw in _FINANCIAL_KEYWORDS)
        if has_financial:
            return

        # No financial keywords — check for off-topic signals
        is_offtopic = any(
            re.search(p, prompt_lower) for p in _OFFTOPIC_PATTERNS
        )
        if is_offtopic:
            raise GuardrailViolationError(
                reason="Query appears to be off-topic for a financial assistant.",
                flag="off_topic_blocked",
            )

    def _check_length(self, prompt: str) -> tuple[str, list[str]]:
        """Truncate prompts that exceed MAX_PROMPT_LENGTH."""
        if len(prompt) <= config.MAX_PROMPT_LENGTH:
            return prompt, []

        truncated = prompt[: config.MAX_PROMPT_LENGTH] + "... [truncated]"
        return truncated, ["input_truncated"]

    # ── Output Guardrails ──────────────────────────────────────────────────────

    def check_output(
        self, response_text: str, data_summary: dict
    ) -> tuple[str, list[str]]:
        """
        Run all output guardrails on the LLM's response.

        Unlike input guardrails, output guards NEVER block the response —
        they only add flags. The pipeline surfaces these flags in the result dict
        so the user/operator can see that something was flagged.

        Returns:
            (response_text, flags)
        """
        flags: list[str] = []

        flags.extend(self._check_hallucination(response_text, data_summary))
        flags.extend(self._check_toxicity(response_text))
        flags.extend(self._check_confidence(response_text))

        return response_text, flags

    def _check_hallucination(
        self, response_text: str, data_summary: dict
    ) -> list[str]:
        """
        Compare numbers in the LLM's response against our actual data.

        How it works:
          1. Extract all numbers from the response (strip $, commas)
          2. Collect all numeric values from data_summary (flatten the dict)
          3. For each response number, check if it's within tolerance of any
             real data value (±2% or ±$5, whichever is larger)
          4. If a number doesn't match anything → flag it

        Tolerance is generous to account for rounding ($1,849.60 → "$1,850").
        """
        if not data_summary:
            return []

        # Extract numbers from LLM response (handles $1,234 and 1234 and 1,234.56)
        response_numbers = self._extract_numbers(response_text)
        if not response_numbers:
            return []

        # Flatten all numeric values from data_summary
        real_values = self._flatten_numeric_values(data_summary)
        if not real_values:
            return []

        # Check each response number against all real values
        unmatched = []
        for num in response_numbers:
            if not self._is_within_tolerance(num, real_values):
                unmatched.append(num)

        if unmatched:
            return ["hallucination_flagged"]
        return []

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract financial numbers from text (handles $1,234.56 format).

        Context-aware filtering:
          - Normalises Unicode whitespace first (some models use narrow no-break spaces)
          - Skips numbers immediately followed by '%' or contextual words
            like 'months', 'transactions', 'times', 'days' — these are
            counts/percentages, not dollar amounts, and cause false positives
            if compared against financial summary values.
          - Only checks numbers >= 10 (small integers are too common to flag)
        """
        # Normalise all Unicode whitespace variants to regular space
        text = text.replace('\u202f', '').replace('\u00a0', '').replace('\u2009', '')

        # Contextual suffixes that indicate a number is NOT a dollar amount.
        _CONTEXTUAL_SUFFIXES = re.compile(
            r'(\d[\d,]*(?:\.\d+)?)\s*'
            r'(%|(?:percent|months?|transactions?|times?|days?|years?|'
            r'items?|entries|purchases?|payments?|bills?|charges?)\b)',
            re.IGNORECASE
        )
        # Find all numbers that appear in a contextual (non-financial) position
        contextual_numbers: set[float] = set()
        for m in _CONTEXTUAL_SUFFIXES.finditer(text):
            try:
                contextual_numbers.add(float(m.group(1).replace(',', '')))
            except ValueError:
                pass

        # Match: optional $, digits with optional commas, optional decimal
        pattern = r'\$?([\d,]+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        numbers = []
        for m in matches:
            try:
                val = float(m.replace(',', ''))
                # Skip common calendar years (e.g. 1990 to 2100) to prevent false positive flags
                if 1990 <= val <= 2100:
                    continue
                # Skip small numbers and contextual counts/percentages
                if val >= 10 and val not in contextual_numbers:
                    numbers.append(val)
            except ValueError:
                pass
        return numbers

    def _flatten_numeric_values(self, data: any, values: Optional[list] = None) -> list[float]:
        """Recursively extract all numeric values from a nested dict/list."""
        if values is None:
            values = []
        if isinstance(data, dict):
            for v in data.values():
                self._flatten_numeric_values(v, values)
        elif isinstance(data, list):
            for item in data:
                self._flatten_numeric_values(item, values)
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            values.append(float(data))
        return values

    def _is_within_tolerance(self, num: float, real_values: list[float]) -> bool:
        """Check if num is within tolerance of any value in real_values."""
        abs_tol = config.HALLUCINATION_TOLERANCE_ABS
        pct_tol = config.HALLUCINATION_TOLERANCE_PCT
        for real in real_values:
            tolerance = max(abs_tol, abs(real) * pct_tol)
            if abs(num - real) <= tolerance:
                return True
        return False

    def _check_toxicity(self, response_text: str) -> list[str]:
        """Flag responses containing inappropriate language."""
        text_lower = response_text.lower()
        for word in _TOXIC_KEYWORDS:
            if word in text_lower:
                return ["toxicity_flagged"]
        return []

    def _check_confidence(self, response_text: str) -> list[str]:
        """Flag responses full of uncertainty language."""
        text_lower = response_text.lower()
        matches = sum(
            1 for p in _UNCERTAINTY_PHRASES
            if re.search(p, text_lower)
        )
        # Flag only if 2+ uncertainty phrases (1 is normal hedging)
        if matches >= 2:
            return ["low_confidence"]
        return []

    # ── Polite Refusal Messages ────────────────────────────────────────────────

    @staticmethod
    def get_refusal_message(flag: str) -> str:
        """Return a user-friendly refusal message for a given guardrail flag."""
        messages = {
            "prompt_injection_blocked": (
                "I can't process that request. I'm a financial analysis assistant "
                "and I follow my guidelines at all times."
            ),
            "cross_user_blocked": (
                "I can only access your own transaction data. "
                "I can't retrieve information about other users."
            ),
            "off_topic_blocked": (
                "I'm a financial analysis assistant. I can help you with questions "
                "about your transactions, spending, income, and budgeting."
            ),
        }
        return messages.get(flag, "I'm unable to process that request.")
