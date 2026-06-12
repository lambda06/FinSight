"""
VisualizationEngine — Tool-callable chart generation functions.

Three chart types the LLM can call as tools:
  1. plot_monthly_spending_trend  — Line chart: spending over time + rolling average
  2. plot_category_breakdown      — Donut chart: where money went by category
  3. plot_income_vs_expense       — Grouped bar chart: income vs expense + net savings line

How the tool-calling flow works:
  Pipeline sends tool schemas to LLM → LLM decides which chart fits the query
  → LLM returns a tool_call with function name + arguments
  → Pipeline calls viz_engine.execute_tool_call(name, args)
  → Chart is saved as PNG → file path returned → included in final response

Design principles:
  - Each chart function is independent — testable without the LLM
  - All charts share a consistent visual style (same colors, fonts, DPI)
  - Functions return a file path string — that's the only contract
"""

import json
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from .data_store import DataStore
from . import config


# ── Color Palette ──────────────────────────────────────────────────────────────
# Consistent across all charts so the output looks like one product.
COLORS = {
    "primary":    "#6366f1",   # indigo — main line / primary bars
    "income":     "#22c55e",   # green  — income bars
    "expense":    "#ef4444",   # red    — expense bars
    "net":        "#f59e0b",   # amber  — net savings line
    "rolling":    "#94a3b8",   # slate  — rolling average line
    "background": "#f8fafc",   # near-white background
}

# Qualitative palette for the donut chart (up to 8 + Other)
DONUT_COLORS = [
    "#6366f1", "#22c55e", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
    "#94a3b8",  # "Other" is always slate
]


def _apply_style() -> None:
    """Apply a consistent visual theme to all charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.facecolor":     COLORS["background"],
        "figure.facecolor":   "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
    })


def _save_chart(fig: plt.Figure, user_id: str, chart_type: str) -> str:
    """
    Save a chart to the output directory and return its path.

    Filename: output/{user_id}_{chart_type}.png
    Overwrites previous chart of the same type for the same user
    (avoids accumulating stale files).
    """
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{user_id}_{chart_type}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


class VisualizationEngine:
    """
    Generates charts from transaction data.

    Usage:
        viz = VisualizationEngine(data_store)
        path = viz.plot_category_breakdown(user_id="usr_a1b2c3d4", months=3)
        # → "output/usr_a1b2c3d4_category_breakdown.png"
    """

    def __init__(self, data_store: DataStore):
        self.store = data_store
        _apply_style()

    # ── Chart 1: Monthly Spending Trend ───────────────────────────────────────

    def plot_monthly_spending_trend(
        self,
        user_id: str,
        months: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> str:
        """
        Line chart showing monthly expense totals with a rolling average overlay.

        When to use: "Show my spending trend", "How has my spending changed over time?"

        Args:
            user_id:         Target user
            months:          Look-back window (None = all available data)
            category_filter: Optional category label to filter to (e.g. "Food > Coffee")

        Returns:
            File path to the saved PNG.
        """
        # Get data
        data = self.store.get_monthly_totals(user_id, months=months)
        user_name = self.store.get_user_name(user_id)

        df = pd.DataFrame(data)

        # Apply category filter if requested (requires raw data)
        title_suffix = ""
        if category_filter:
            user_df = self.store.get_user_data(user_id)
            if months:
                user_df = self.store._filter_months(user_df, months)
            filtered = user_df[user_df["category_label"] == category_filter]
            filtered = filtered[filtered["transaction_amount"] > 0].copy()
            if not filtered.empty:
                filtered["month"] = filtered["transaction_date"].dt.to_period("M").astype(str)
                df = (
                    filtered.groupby("month")["transaction_amount"]
                    .sum()
                    .reset_index()
                    .rename(columns={"transaction_amount": "total"})
                )
                title_suffix = f" — {category_filter}"

        fig, ax = plt.subplots(figsize=(10, 5))

        x = range(len(df))
        totals = df["total"].tolist()
        months_labels = df["month"].tolist()

        # Main spending line
        ax.plot(x, totals, color=COLORS["primary"], linewidth=2.5,
                marker="o", markersize=7, label="Monthly Spend", zorder=3)

        # Fill under line for visual weight
        ax.fill_between(x, totals, alpha=0.12, color=COLORS["primary"])

        # Rolling average (3-month window) — only if enough data points
        if len(df) >= 3:
            rolling = pd.Series(totals).rolling(window=3, min_periods=1).mean()
            ax.plot(x, rolling, color=COLORS["rolling"], linewidth=1.5,
                    linestyle="--", label="3-Month Avg", zorder=2)

        # Formatting
        ax.set_xticks(list(x))
        ax.set_xticklabels(months_labels, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_title(f"Monthly Spending Trend — {user_name}{title_suffix}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Expenses ($)")
        ax.legend(frameon=False)

        # Annotate each point with its value
        for xi, yi in zip(x, totals):
            ax.annotate(f"${yi:,}", (xi, yi),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=COLORS["primary"])

        fig.tight_layout()
        return _save_chart(fig, user_id, "monthly_trend")

    # ── Chart 2: Category Breakdown (Donut) ───────────────────────────────────

    def plot_category_breakdown(
        self,
        user_id: str,
        months: Optional[int] = 3,
        top_n: int = 7,
    ) -> str:
        """
        Donut chart showing proportional spending by category.

        Top N categories are shown individually; the rest are grouped as "Other".
        Total spend is displayed in the center of the donut.

        When to use: "What did I spend most on?", "Show my spending breakdown"

        Args:
            user_id: Target user
            months:  Look-back window (default: last 3 months)
            top_n:   Number of top categories to show before grouping as "Other"

        Returns:
            File path to the saved PNG.
        """
        data = self.store.get_spending_by_category(user_id, months=months)
        user_name = self.store.get_user_name(user_id)

        df = pd.DataFrame(data)

        # Group everything beyond top_n into "Other"
        if len(df) > top_n:
            top    = df.head(top_n).copy()
            other_total = df.iloc[top_n:]["total"].sum()
            other_row = pd.DataFrame([{
                "category_label": "Other",
                "total": other_total,
                "count": df.iloc[top_n:]["count"].sum(),
            }])
            df = pd.concat([top, other_row], ignore_index=True)

        labels = df["category_label"].tolist()
        sizes  = df["total"].tolist()
        total  = sum(sizes)
        colors = DONUT_COLORS[:len(labels)]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Draw donut (wedgeprops creates the hole in the middle)
        wedges, _ = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        )

        # Total spend label in the center
        ax.text(0, 0, f"${total:,}\ntotal", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#1e293b")

        # Legend outside the chart
        period_label = f"Last {months} months" if months else "All time"
        legend_labels = [
            f"{lab}  ${amt:,}  ({amt/total*100:.1f}%)"
            for lab, amt in zip(labels, sizes)
        ]
        ax.legend(wedges, legend_labels, loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=9)

        ax.set_title(f"Spending by Category — {user_name}\n({period_label})", pad=20)

        fig.tight_layout()
        return _save_chart(fig, user_id, "category_breakdown")

    # ── Chart 3: Income vs Expense ────────────────────────────────────────────

    def plot_income_vs_expense(
        self,
        user_id: str,
        months: Optional[int] = None,
        show_net_line: bool = True,
    ) -> str:
        """
        Grouped bar chart: green income bars + red expense bars, with an
        optional amber net-savings line overlay.

        When to use: "Am I saving money?", "Show income vs spending",
                     "Am I living within my means?"

        Args:
            user_id:       Target user
            months:        Look-back window (None = all available data)
            show_net_line: Overlay a line showing net savings per month

        Returns:
            File path to the saved PNG.
        """
        data = self.store.get_income_vs_expense(user_id, months=months)
        user_name = self.store.get_user_name(user_id)

        df = pd.DataFrame(data)
        n = len(df)

        # Bar positions — income and expense bars sit side by side
        bar_width = 0.35
        x = range(n)
        x_income  = [i - bar_width / 2 for i in x]
        x_expense = [i + bar_width / 2 for i in x]

        fig, ax = plt.subplots(figsize=(11, 6))

        # Income bars (green)
        ax.bar(x_income, df["income"], width=bar_width,
               color=COLORS["income"], alpha=0.85, label="Income", zorder=3)

        # Expense bars (red)
        ax.bar(x_expense, df["expense"], width=bar_width,
               color=COLORS["expense"], alpha=0.85, label="Expense", zorder=3)

        # Net savings line (amber)
        if show_net_line:
            ax_right = ax.twinx()   # second y-axis for net line
            ax_right.plot(list(x), df["net"], color=COLORS["net"],
                          linewidth=2.5, marker="D", markersize=6,
                          label="Net Savings", zorder=4)
            ax_right.axhline(0, color=COLORS["net"], linewidth=0.8,
                             linestyle="--", alpha=0.5)
            ax_right.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
            )
            ax_right.set_ylabel("Net Savings ($)", color=COLORS["net"])
            ax_right.tick_params(axis="y", colors=COLORS["net"])

        # Formatting
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["month"].tolist(), rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_title(f"Income vs Expense — {user_name}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount ($)")

        # Combined legend
        handles, labels_ax = ax.get_legend_handles_labels()
        if show_net_line:
            h2, l2 = ax_right.get_legend_handles_labels()
            handles += h2
            labels_ax += l2
        ax.legend(handles, labels_ax, frameon=False, loc="upper left")

        fig.tight_layout()
        return _save_chart(fig, user_id, "income_vs_expense")

    # ── Tool Dispatcher ────────────────────────────────────────────────────────

    def execute_tool_call(self, function_name: str, arguments: dict) -> str:
        """
        Dispatch an LLM tool call to the right chart function.

        The pipeline calls this after parsing the LLM's tool_call response.

        Args:
            function_name: e.g. "plot_category_breakdown"
            arguments:     dict of parameters the LLM chose (already parsed from JSON)

        Returns:
            File path to the generated chart PNG.

        Raises:
            ValueError if function_name doesn't match any known chart.
        """
        dispatch = {
            "plot_monthly_spending_trend": self.plot_monthly_spending_trend,
            "plot_category_breakdown":     self.plot_category_breakdown,
            "plot_income_vs_expense":      self.plot_income_vs_expense,
        }
        if function_name not in dispatch:
            raise ValueError(f"Unknown visualization tool: '{function_name}'")

        # user_id is always required — pipeline injects it if LLM didn't include it
        fn = dispatch[function_name]
        return fn(**arguments)


# ── Tool Schemas ───────────────────────────────────────────────────────────────
# These JSON schemas are sent to the LLM in the `tools` parameter.
# The LLM reads the `description` fields to decide which chart to call and when.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "plot_monthly_spending_trend",
            "description": (
                "Generate a line chart showing monthly spending totals over time, "
                "with a rolling average overlay. "
                "Use when the user asks about spending trends, how spending has changed, "
                "monthly patterns, or time-based analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The target user's ID.",
                    },
                    "months": {
                        "type": "integer",
                        "description": (
                            "Number of recent months to include. "
                            "Omit to show all available data (recommended for trend queries)."
                        ),
                    },
                    "category_filter": {
                        "type": "string",
                        "description": (
                            "Optional: filter to a specific category label, "
                            "e.g. 'Food > Coffee'. Only use when the user asks "
                            "about trends for a specific category."
                        ),
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_category_breakdown",
            "description": (
                "Generate a donut chart showing proportional spending by category, "
                "with the total spend in the center. "
                "Use when the user asks what they spent most on, wants a spending breakdown, "
                "or asks about categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The target user's ID.",
                    },
                    "months": {
                        "type": "integer",
                        "description": (
                            "Number of recent months to include. "
                            "Default is 3. Use 1 for 'last month' queries."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": (
                            "Number of top categories to show individually "
                            "(rest grouped as 'Other'). Default is 7."
                        ),
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_income_vs_expense",
            "description": (
                "Generate a grouped bar chart comparing monthly income vs expenses, "
                "with a net savings line overlay. "
                "Use when the user asks if they are saving money, wants to compare "
                "income and spending, or asks about their financial health."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The target user's ID.",
                    },
                    "months": {
                        "type": "integer",
                        "description": (
                            "Number of recent months to include. "
                            "Omit to show all available data."
                        ),
                    },
                    "show_net_line": {
                        "type": "boolean",
                        "description": (
                            "Whether to overlay the net savings line. "
                            "Default is true — almost always keep this on."
                        ),
                    },
                },
                "required": ["user_id"],
            },
        },
    },
]
