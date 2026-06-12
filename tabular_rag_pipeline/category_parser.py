"""
Category Parser — Converts flat category codes to hierarchical labels.

The raw data uses codes like RENT_HOUSING, COFFEE_FOOD, SALARY_INCOME.
This module maps all 27 of them to a human-readable format:
    RENT_HOUSING   → Housing > Rent
    COFFEE_FOOD    → Food > Coffee
    SALARY_INCOME  → Income > Salary

Why hardcode all 27 instead of auto-parsing?
  Auto-parsing works for simple cases (RENT → Rent) but breaks on
  multi-word subcategories (FASTFOOD → "Fastfood" instead of "Fast Food").
  An explicit map is readable, testable, and handles every edge case.
"""

# Complete map of all 27 category codes found in the dataset.
# Format: "CODE": "Main Category > Subcategory"
CATEGORY_MAP: dict[str, str] = {
    # Income
    "SALARY_INCOME":     "Income > Salary",
    "FREELANCE_INCOME":  "Income > Freelance",
    "CASHBACK_INCOME":   "Income > Cashback",
    "REFUND_INCOME":     "Income > Refund",

    # Housing
    "RENT_HOUSING":      "Housing > Rent",
    "UTILITIES_HOUSING": "Housing > Utilities",
    "INTERNET_HOUSING":  "Housing > Internet",

    # Food
    "GROCERIES_FOOD":    "Food > Groceries",
    "RESTAURANT_FOOD":   "Food > Restaurant",
    "COFFEE_FOOD":       "Food > Coffee",
    "FASTFOOD_FOOD":     "Food > Fast Food",

    # Transport
    "FUEL_TRANSPORT":      "Transport > Fuel",
    "RIDESHARE_TRANSPORT": "Transport > Rideshare",

    # Health
    "GYM_HEALTH":      "Health > Gym",
    "DOCTOR_HEALTH":   "Health > Doctor",
    "PHARMACY_HEALTH": "Health > Pharmacy",

    # Finance
    "INSURANCE_FINANCE":   "Finance > Insurance",
    "SUBSCRIPTION_FINANCE": "Finance > Subscription",

    # Shopping
    "CLOTHING_SHOPPING":    "Shopping > Clothing",
    "ELECTRONICS_SHOPPING": "Shopping > Electronics",
    "GENERAL_SHOPPING":     "Shopping > General",

    # Entertainment
    "STREAMING_ENTERTAINMENT": "Entertainment > Streaming",
    "MOVIES_ENTERTAINMENT":    "Entertainment > Movies",

    # Travel
    "FLIGHTS_TRAVEL": "Travel > Flights",
    "HOTELS_TRAVEL":  "Travel > Hotels",

    # Education
    "COURSES_EDUCATION": "Education > Courses",

    # Pets
    "SUPPLIES_PETS": "Pets > Supplies",
}


def parse_category(code: str) -> str:
    """
    Convert a raw category code to a human-readable label.

    Args:
        code: Raw code from the DataFrame, e.g. "RENT_HOUSING"

    Returns:
        Human-readable label, e.g. "Housing > Rent".
        Returns the original code unchanged if not found in the map
        (so unknown future categories don't crash the system).
    """
    return CATEGORY_MAP.get(code, code)


def get_main_category(code: str) -> str:
    """
    Extract just the main category from a code.

    "RENT_HOUSING" → "Housing"
    Useful for grouping by top-level category.
    """
    label = parse_category(code)
    return label.split(" > ")[0]
