"""
DataStore — DataFrame loading, filtering, and analysis.

Holds the full transaction DataFrame (347 rows, loaded once at startup).
Provides user-scoped analysis functions that all other modules call.

Design principle:
  Only this module touches pandas. Everything else receives plain Python
  dicts and lists. This makes testing easy — mock this class and the rest
  of the pipeline works without any real data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from .category_parser import parse_category
from .exceptions import UserNotFoundError, EmptyDataError
from . import config


class DataStore:
    """
    Loads and manages the transaction DataFrame.

    Usage:
        store = DataStore()               # loads from config.DATA_FILE
        store = DataStore(path)           # loads from a custom path (useful for tests)
        store = DataStore(df=my_df)       # inject an existing DataFrame (useful for tests)
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        if df is not None:
            # Allow injecting a DataFrame directly (useful for unit tests)
            self._df = df.copy()
        else:
            file_path = path or config.DATA_FILE
            self._df = pd.read_excel(file_path)

        self._prepare()

    # ── Internal Setup ─────────────────────────────────────────────────────────

    def _prepare(self) -> None:
        """Clean up column types and add a human-readable category column."""
        # Ensure dates are datetime objects
        self._df["transaction_date"] = pd.to_datetime(self._df["transaction_date"])

        # Add a readable category column alongside the raw code
        # e.g. "RENT_HOUSING" → "Housing > Rent"
        self._df["category_label"] = self._df["transaction_category_detail"].apply(
            parse_category
        )

        # Cache the set of valid user IDs for fast lookup
        self._valid_users: set[str] = set(self._df["user_id"].unique())

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_user(self, user_id: str) -> None:
        """Raise UserNotFoundError if user_id doesn't exist in the data."""
        if user_id not in self._valid_users:
            raise UserNotFoundError(
                f"No transactions found for user_id '{user_id}'. "
                f"Available users: {len(self._valid_users)}"
            )

    def get_all_user_ids(self) -> list[str]:
        """Return all unique user IDs in the dataset."""
        return list(self._valid_users)

    def get_user_name(self, user_id: str) -> str:
        """
        Return the display name for a user without computing a full profile.

        Much cheaper than compute_user_profile() — reads a single cell.
        Used by the visualization engine to title charts.
        """
        self.validate_user(user_id)
        return str(self._df.loc[self._df["user_id"] == user_id, "user_name"].iloc[0])

    # ── Data Fetching ──────────────────────────────────────────────────────────

    def get_user_data(self, user_id: str) -> pd.DataFrame:
        """
        Return all rows for a given user.

        This is the base filter — every analysis function calls this first.
        """
        self.validate_user(user_id)
        return self._df[self._df["user_id"] == user_id].copy()

    def _filter_months(self, df: pd.DataFrame, months: Optional[int]) -> pd.DataFrame:
        """
        Filter DataFrame to the last N months of data.

        If months is None, return all data (most common case).
        'Last N months' is calculated relative to the latest date in the data,
        NOT today's date — so results are consistent regardless of when you run it.
        """
        if months is None:
            return df
        latest_date = df["transaction_date"].max()
        cutoff = latest_date - pd.DateOffset(months=months)
        return df[df["transaction_date"] > cutoff]

    # ── Analysis Functions ─────────────────────────────────────────────────────
    # These return plain dicts/lists — not DataFrames — so other modules
    # don't need to know anything about pandas.

    def get_spending_by_category(
        self, user_id: str, months: Optional[int] = None
    ) -> list[dict]:
        """
        Return expenses grouped by category, sorted by total (highest first).

        Only includes expenses (positive amounts) — income is excluded.

        Returns: [{"category": "Housing > Rent", "total": 1850, "count": 8}, ...]
        """
        df = self.get_user_data(user_id)
        df = self._filter_months(df, months)

        # Positive amounts = expenses
        expenses = df[df["transaction_amount"] > 0]
        if expenses.empty:
            raise EmptyDataError("No expense transactions found for this period.")

        grouped = (
            expenses.groupby("category_label")["transaction_amount"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total", "count": "count"})
            .sort_values("total", ascending=False)
        )

        return grouped.to_dict(orient="records")

    def get_monthly_totals(
        self, user_id: str, months: Optional[int] = None
    ) -> list[dict]:
        """
        Return total expenses per calendar month, sorted chronologically.

        Only includes expenses (positive amounts).

        Returns: [{"month": "2025-05", "total": 3200, "count": 18}, ...]
        """
        df = self.get_user_data(user_id)
        df = self._filter_months(df, months)

        expenses = df[df["transaction_amount"] > 0].copy()
        if expenses.empty:
            raise EmptyDataError("No expense transactions found for this period.")

        expenses["month"] = expenses["transaction_date"].dt.to_period("M").astype(str)

        grouped = (
            expenses.groupby("month")["transaction_amount"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total", "count": "count"})
            .sort_values("month")
        )

        return grouped.to_dict(orient="records")

    def get_income_vs_expense(
        self, user_id: str, months: Optional[int] = None
    ) -> list[dict]:
        """
        Return income and expenses side-by-side per month.

        Income = negative amounts (money coming in), stored as positive values.
        Expense = positive amounts (money going out).

        Returns: [{"month": "2025-05", "income": 5200, "expense": 3200, "net": 2000}, ...]
        """
        df = self.get_user_data(user_id)
        df = self._filter_months(df, months)
        df = df.copy()

        df["month"] = df["transaction_date"].dt.to_period("M").astype(str)

        # Split into income and expense rows
        income_df  = df[df["transaction_amount"] < 0].copy()
        expense_df = df[df["transaction_amount"] > 0].copy()

        # Sum per month (flip sign on income so it's a positive number)
        income_by_month  = income_df.groupby("month")["transaction_amount"].sum().abs()
        expense_by_month = expense_df.groupby("month")["transaction_amount"].sum()

        # Combine into one table, filling 0 for months with no income or expense
        all_months = sorted(set(income_by_month.index) | set(expense_by_month.index))
        result = []
        for month in all_months:
            income  = income_by_month.get(month, 0)
            expense = expense_by_month.get(month, 0)
            result.append({
                "month":   month,
                "income":  int(income),
                "expense": int(expense),
                "net":     int(income - expense),   # positive = saved money
            })

        return result

    def get_top_merchants(self, user_id: str, n: int = 10) -> list[dict]:
        """
        Return top N merchants by total spend (expenses only).

        Returns: [{"merchant": "AvalonBay", "total": 14800, "count": 8}, ...]
        """
        df = self.get_user_data(user_id)
        expenses = df[df["transaction_amount"] > 0]

        grouped = (
            expenses.groupby("merchant_name")["transaction_amount"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"merchant_name": "merchant", "sum": "total", "count": "count"})
            .sort_values("total", ascending=False)
            .head(n)
        )

        return grouped.to_dict(orient="records")

    def get_largest_transactions(self, user_id: str, n: int = 5) -> list[dict]:
        """
        Return the N largest single expense transactions.

        Returns: [{"date": "2025-12-01", "amount": 1850, "merchant": "AvalonBay",
                   "category": "Housing > Rent"}, ...]
        """
        df = self.get_user_data(user_id)
        expenses = df[df["transaction_amount"] > 0].copy()

        top = expenses.nlargest(n, "transaction_amount")[
            ["transaction_date", "transaction_amount", "merchant_name", "category_label"]
        ].rename(columns={
            "transaction_date":   "date",
            "transaction_amount": "amount",
            "merchant_name":      "merchant",
            "category_label":     "category",
        })

        top["date"] = top["date"].dt.strftime("%Y-%m-%d")
        return top.to_dict(orient="records")

    def compute_user_profile(self, user_id: str) -> dict:
        """
        Build a summary profile for a user — this is what gets cached.

        Computed once on first query, stored in cache, reused on every
        subsequent query. Gives the LLM context about who it's talking to.

        Returns: {
            "user_name": "Jose BazBaz",
            "date_range": {"start": "2025-05-01", "end": "2025-12-31"},
            "total_transactions": 117,
            "months_of_data": 8,
            "avg_monthly_expense": 3241,
            "top_categories": [...],   # top 5 by total spend
            "income_sources": [...],   # unique income category labels
        }
        """
        df = self.get_user_data(user_id)

        expenses = df[df["transaction_amount"] > 0]
        income   = df[df["transaction_amount"] < 0]

        # Date range
        start = df["transaction_date"].min().strftime("%Y-%m-%d")
        end   = df["transaction_date"].max().strftime("%Y-%m-%d")

        # Months of data
        months = df["transaction_date"].dt.to_period("M").nunique()

        # Average monthly expense
        total_expense = int(expenses["transaction_amount"].sum())
        avg_monthly   = round(total_expense / months) if months else 0

        # Top 5 spending categories
        top_cats = self.get_spending_by_category(user_id)[:5]

        # Income sources (unique labels)
        income_sources = income["category_label"].unique().tolist()

        return {
            "user_name":          df["user_name"].iloc[0],
            "date_range":         {"start": start, "end": end},
            "total_transactions": len(df),
            "months_of_data":     months,
            "avg_monthly_expense": avg_monthly,
            "top_categories":     top_cats,
            "income_sources":     income_sources,
        }
