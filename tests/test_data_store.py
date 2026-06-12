"""
Unit tests for DataStore.

All tests use an injected synthetic DataFrame — no Excel file needed,
no API calls, no environment variables.

Run:
    pytest tests/test_data_store.py -v
"""

import pytest
import pandas as pd
from tabular_rag_pipeline.data_store import DataStore
from tabular_rag_pipeline.exceptions import UserNotFoundError, EmptyDataError


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a test DataFrame with correct column types."""
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@pytest.fixture
def sample_df():
    """
    Minimal synthetic DataFrame with 2 users and mixed transaction types.

    User A (usr_a):
      - 2025-01: $500 groceries, $1,000 rent, $3,000 salary (income)
      - 2025-02: $200 coffee, $1,000 rent, $3,000 salary (income)

    User B (usr_b):
      - 2025-01: $800 groceries, $-2,500 salary
    """
    return _make_df([
        # User A — January
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-01-10", "transaction_amount": 500,
         "merchant_name": "Whole Foods", "transaction_category_detail": "GROCERIES_FOOD"},
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-01-01", "transaction_amount": 1000,
         "merchant_name": "LandlordCo", "transaction_category_detail": "RENT_HOUSING"},
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-01-31", "transaction_amount": -3000,
         "merchant_name": "ACME Corp", "transaction_category_detail": "SALARY_INCOME"},
        # User A — February
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-02-05", "transaction_amount": 200,
         "merchant_name": "Starbucks", "transaction_category_detail": "COFFEE_FOOD"},
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-02-01", "transaction_amount": 1000,
         "merchant_name": "LandlordCo", "transaction_category_detail": "RENT_HOUSING"},
        {"user_id": "usr_a", "user_name": "Alice Test",
         "transaction_date": "2025-02-28", "transaction_amount": -3000,
         "merchant_name": "ACME Corp", "transaction_category_detail": "SALARY_INCOME"},
        # User B — January only
        {"user_id": "usr_b", "user_name": "Bob Test",
         "transaction_date": "2025-01-15", "transaction_amount": 800,
         "merchant_name": "Trader Joe's", "transaction_category_detail": "GROCERIES_FOOD"},
        {"user_id": "usr_b", "user_name": "Bob Test",
         "transaction_date": "2025-01-31", "transaction_amount": -2500,
         "merchant_name": "BigCorp", "transaction_category_detail": "SALARY_INCOME"},
    ])


@pytest.fixture
def store(sample_df):
    """DataStore backed by the synthetic DataFrame."""
    return DataStore(df=sample_df)


# ── Validation ─────────────────────────────────────────────────────────────────

class TestUserValidation:
    def test_valid_user_passes(self, store):
        store.validate_user("usr_a")   # should not raise

    def test_invalid_user_raises(self, store):
        with pytest.raises(UserNotFoundError):
            store.validate_user("usr_does_not_exist")

    def test_get_all_user_ids_returns_both(self, store):
        ids = store.get_all_user_ids()
        assert "usr_a" in ids
        assert "usr_b" in ids
        assert len(ids) == 2

    def test_get_user_name(self, store):
        assert store.get_user_name("usr_a") == "Alice Test"
        assert store.get_user_name("usr_b") == "Bob Test"

    def test_get_user_name_invalid_user_raises(self, store):
        with pytest.raises(UserNotFoundError):
            store.get_user_name("usr_unknown")


# ── Data Fetching ──────────────────────────────────────────────────────────────

class TestGetUserData:
    def test_returns_only_requested_user(self, store):
        df = store.get_user_data("usr_a")
        assert (df["user_id"] == "usr_a").all()

    def test_user_b_data_isolated(self, store):
        df = store.get_user_data("usr_b")
        assert (df["user_id"] == "usr_b").all()
        assert len(df) == 2   # only 2 rows for usr_b


# ── Spending by Category ───────────────────────────────────────────────────────

class TestSpendingByCategory:
    def test_returns_expenses_only(self, store):
        """Negative amounts (income) must not appear in spending."""
        result = store.get_spending_by_category("usr_a")
        # Salary_income rows have negative amounts — must be excluded
        category_labels = [r["category_label"] for r in result]
        assert "Income > Salary" not in category_labels

    def test_sorted_descending(self, store):
        """Highest total must be first."""
        result = store.get_spending_by_category("usr_a")
        totals = [r["total"] for r in result]
        assert totals == sorted(totals, reverse=True)

    def test_category_totals_correct(self, store):
        """usr_a has $2,000 rent (2 months × $1,000) and $700 food ($500 + $200)."""
        result = store.get_spending_by_category("usr_a")
        by_label = {r["category_label"]: r["total"] for r in result}
        assert by_label["Housing > Rent"] == 2000
        assert by_label["Food > Groceries"] == 500
        assert by_label["Food > Coffee"] == 200

    def test_months_filter_limits_scope(self, store):
        """With months=1 (last 1 month = Feb), only Feb transactions included."""
        result = store.get_spending_by_category("usr_a", months=1)
        by_label = {r["category_label"]: r["total"] for r in result}
        # February only: $1,000 rent + $200 coffee
        assert by_label.get("Housing > Rent", 0) == 1000
        assert "Food > Groceries" not in by_label   # Jan only

    def test_empty_period_raises(self, store):
        """Requesting months=0 effectively means no data → EmptyDataError."""
        # A months filter so tight it excludes everything
        with pytest.raises(EmptyDataError):
            store.get_spending_by_category("usr_b", months=0)


# ── Monthly Totals ─────────────────────────────────────────────────────────────

class TestMonthlyTotals:
    def test_returns_two_months_for_usr_a(self, store):
        result = store.get_monthly_totals("usr_a")
        months = [r["month"] for r in result]
        assert "2025-01" in months
        assert "2025-02" in months

    def test_sorted_chronologically(self, store):
        result = store.get_monthly_totals("usr_a")
        months = [r["month"] for r in result]
        assert months == sorted(months)

    def test_january_total_correct(self, store):
        """Jan: $500 groceries + $1,000 rent = $1,500 (income excluded)."""
        result = store.get_monthly_totals("usr_a")
        jan = next(r for r in result if r["month"] == "2025-01")
        assert jan["total"] == 1500

    def test_february_total_correct(self, store):
        """Feb: $200 coffee + $1,000 rent = $1,200."""
        result = store.get_monthly_totals("usr_a")
        feb = next(r for r in result if r["month"] == "2025-02")
        assert feb["total"] == 1200


# ── Income vs Expense ──────────────────────────────────────────────────────────

class TestIncomeVsExpense:
    def test_income_sign_is_positive(self, store):
        """Income amounts are stored negative in the data but returned positive."""
        result = store.get_income_vs_expense("usr_a")
        for row in result:
            assert row["income"] >= 0

    def test_net_savings_is_income_minus_expense(self, store):
        result = store.get_income_vs_expense("usr_a")
        for row in result:
            assert row["net"] == row["income"] - row["expense"]

    def test_january_values(self, store):
        """Jan: $3,000 income, $1,500 expense → $1,500 net."""
        result = store.get_income_vs_expense("usr_a")
        jan = next(r for r in result if r["month"] == "2025-01")
        assert jan["income"] == 3000
        assert jan["expense"] == 1500
        assert jan["net"] == 1500


# ── Top Merchants ──────────────────────────────────────────────────────────────

class TestTopMerchants:
    def test_landlord_is_top_merchant(self, store):
        """LandlordCo appears twice at $1,000 each = $2,000 total."""
        result = store.get_top_merchants("usr_a")
        top = result[0]
        assert top["merchant"] == "LandlordCo"
        assert top["total"] == 2000

    def test_income_merchant_not_in_results(self, store):
        """ACME Corp is income (negative amounts) — must not appear in merchant spend."""
        result = store.get_top_merchants("usr_a")
        merchants = [r["merchant"] for r in result]
        assert "ACME Corp" not in merchants

    def test_n_limit_respected(self, store):
        result = store.get_top_merchants("usr_a", n=2)
        assert len(result) <= 2


# ── User Profile ───────────────────────────────────────────────────────────────

class TestComputeUserProfile:
    def test_profile_has_required_keys(self, store):
        profile = store.compute_user_profile("usr_a")
        required = {
            "user_name", "date_range", "total_transactions",
            "months_of_data", "avg_monthly_expense", "top_categories", "income_sources"
        }
        assert required.issubset(profile.keys())

    def test_user_name_correct(self, store):
        profile = store.compute_user_profile("usr_a")
        assert profile["user_name"] == "Alice Test"

    def test_total_transactions_correct(self, store):
        profile = store.compute_user_profile("usr_a")
        assert profile["total_transactions"] == 6   # 3 Jan + 3 Feb

    def test_income_sources_populated(self, store):
        profile = store.compute_user_profile("usr_a")
        assert "Income > Salary" in profile["income_sources"]

    def test_avg_monthly_expense_correct(self, store):
        """Total expense = $2,700. Over 2 months = $1,350 avg."""
        profile = store.compute_user_profile("usr_a")
        assert profile["avg_monthly_expense"] == 1350
