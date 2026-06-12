"""
Unit tests for CategoryParser.

Tests the CATEGORY_MAP completeness and the parse_category / get_main_category functions.

Run:
    pytest tests/test_category_parser.py -v
"""

import pytest
from tabular_rag_pipeline.category_parser import (
    parse_category,
    get_main_category,
    CATEGORY_MAP,
)


# ── Map Completeness ───────────────────────────────────────────────────────────

class TestCategoryMapCompleteness:
    def test_map_has_all_known_categories(self):
        """Every raw code in the dataset must have a readable label."""
        known_codes = {
            # Income
            "SALARY_INCOME", "FREELANCE_INCOME", "CASHBACK_INCOME", "REFUND_INCOME",
            # Housing
            "RENT_HOUSING", "UTILITIES_HOUSING", "INTERNET_HOUSING",
            # Food
            "GROCERIES_FOOD", "RESTAURANT_FOOD", "COFFEE_FOOD", "FASTFOOD_FOOD",
            # Transport
            "FUEL_TRANSPORT", "RIDESHARE_TRANSPORT",
            # Health
            "GYM_HEALTH", "DOCTOR_HEALTH", "PHARMACY_HEALTH",
            # Finance
            "INSURANCE_FINANCE", "SUBSCRIPTION_FINANCE",
            # Shopping
            "CLOTHING_SHOPPING", "ELECTRONICS_SHOPPING", "GENERAL_SHOPPING",
            # Entertainment
            "STREAMING_ENTERTAINMENT", "MOVIES_ENTERTAINMENT",
            # Travel
            "FLIGHTS_TRAVEL", "HOTELS_TRAVEL",
            # Education
            "COURSES_EDUCATION",
            # Pets
            "SUPPLIES_PETS",
        }
        assert known_codes.issubset(CATEGORY_MAP.keys()), (
            f"Missing from CATEGORY_MAP: {known_codes - CATEGORY_MAP.keys()}"
        )

    def test_no_empty_labels(self):
        """Every label must be a non-empty string."""
        for code, label in CATEGORY_MAP.items():
            assert isinstance(label, str) and label.strip(), f"Empty label for {code}"

    def test_all_labels_have_hierarchy_separator(self):
        """Every label should use 'Main > Sub' format."""
        for code, label in CATEGORY_MAP.items():
            assert " > " in label, f"Label '{label}' for {code} missing ' > ' separator"


# ── parse_category ─────────────────────────────────────────────────────────────

class TestParseCategory:
    @pytest.mark.parametrize("code,expected", [
        ("RENT_HOUSING",             "Housing > Rent"),
        ("SALARY_INCOME",            "Income > Salary"),
        ("GROCERIES_FOOD",           "Food > Groceries"),
        ("COFFEE_FOOD",              "Food > Coffee"),
        ("FASTFOOD_FOOD",            "Food > Fast Food"),
        ("GYM_HEALTH",               "Health > Gym"),
        ("STREAMING_ENTERTAINMENT",  "Entertainment > Streaming"),
        ("COURSES_EDUCATION",        "Education > Courses"),
        ("SUPPLIES_PETS",            "Pets > Supplies"),
    ])
    def test_known_codes_map_correctly(self, code, expected):
        assert parse_category(code) == expected

    def test_unknown_code_returns_original(self):
        """Unknown codes must pass through unchanged — don't crash on new categories."""
        unknown = "FUTURE_CATEGORY_XYZ"
        assert parse_category(unknown) == unknown

    def test_empty_string_returns_empty(self):
        assert parse_category("") == ""


# ── get_main_category ──────────────────────────────────────────────────────────

class TestGetMainCategory:
    @pytest.mark.parametrize("code,expected_main", [
        ("RENT_HOUSING",         "Housing"),
        ("SALARY_INCOME",        "Income"),
        ("GROCERIES_FOOD",       "Food"),
        ("COFFEE_FOOD",          "Food"),
        ("FUEL_TRANSPORT",       "Transport"),
        ("GYM_HEALTH",           "Health"),
        ("INSURANCE_FINANCE",    "Finance"),
        ("CLOTHING_SHOPPING",    "Shopping"),
        ("FLIGHTS_TRAVEL",       "Travel"),
        ("COURSES_EDUCATION",    "Education"),
        ("SUPPLIES_PETS",        "Pets"),
    ])
    def test_extracts_main_category(self, code, expected_main):
        assert get_main_category(code) == expected_main

    def test_unknown_code_returns_itself(self):
        """Unknown codes: no separator → whole string returned as main category."""
        assert get_main_category("UNKNOWN_CODE") == "UNKNOWN_CODE"

    def test_groups_same_main_category(self):
        """Food > Groceries and Food > Coffee should both map to 'Food'."""
        assert get_main_category("GROCERIES_FOOD") == get_main_category("COFFEE_FOOD") == "Food"
