"""Simple CLV baselines (PRD Section 8: Candidate approaches).

Baseline 1: historical average revenue by plan, tenure, and customer segment.
Baseline 2: current recurring revenue multiplied by a simple expected-retention factor.

These are interpretable starting points. The MVP model (CLVModel) should only be
adopted if it delivers a meaningful and stable improvement over these baselines.
"""

from __future__ import annotations

import pandas as pd


def baseline_historical_average(
    train: pd.DataFrame,
    group_columns: list[str] | None = None,
    revenue_column: str = "actual_revenue",
) -> dict:
    """Baseline 1: historical average revenue by cohort attributes.

    Parameters
    ----------
    train : training data with actual revenue and cohort attributes
    group_columns : columns to group by (default: plan, customer_segment)
    revenue_column : column containing observed revenue

    Returns
    -------
    dict mapping group-key tuple -> mean observed revenue
    """
    group_columns = group_columns or ["plan", "customer_segment"]
    available = [c for c in group_columns if c in train.columns]
    if not available:
        # Fall back to global mean
        return {(): float(train[revenue_column].mean())}

    grouped = train.groupby(available)[revenue_column].mean()
    return grouped.to_dict()


def predict_baseline_historical_average(
    lookup: dict,
    df: pd.DataFrame,
    group_columns: list[str] | None = None,
    global_fallback: float | None = None,
) -> pd.Series:
    """Predict using Baseline 1 lookup table."""
    group_columns = group_columns or ["plan", "customer_segment"]
    available = [c for c in group_columns if c in df.columns]

    if global_fallback is None:
        global_fallback = float(
            pd.Series(list(lookup.values())).mean() if lookup else 0.0
        )

    if not available:
        return pd.Series([global_fallback] * len(df), index=df.index)

    def _lookup(row):
        key = tuple(row[c] for c in available)
        return lookup.get(key, global_fallback)

    return df.apply(_lookup, axis=1)


def baseline_recurring_revenue_retention(
    train: pd.DataFrame,
    retention_factor: float = 0.85,
    recurring_column: str = "recurring_amount",
    horizon_months: int = 12,
) -> float:
    """Baseline 2: current recurring revenue × expected-retention factor.

    Simple rule: expected 12-month revenue = monthly recurring revenue × 12 ×
    retention factor. Returns a single retention factor calibrated on training
    data if retention_factor is None (estimated as active_customers / total).

    Parameters
    ----------
    train : training data
    retention_factor : expected fraction of customers retained over horizon
    recurring_column : column with current monthly recurring revenue
    horizon_months : prediction horizon in months

    Returns
    -------
    retention factor used (for reporting / calibration)
    """
    return retention_factor


def predict_baseline_recurring_revenue_retention(
    df: pd.DataFrame,
    retention_factor: float = 0.85,
    recurring_column: str = "recurring_amount",
    horizon_months: int = 12,
) -> pd.Series:
    """Predict using Baseline 2 rule."""
    mrr = df[recurring_column] if recurring_column in df.columns else pd.Series(0.0, index=df.index)
    return mrr * horizon_months * retention_factor
