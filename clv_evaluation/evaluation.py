"""Core CLV evaluation metrics.

Implements the metrics named in PRD Section 9:
- MAE (mean absolute error, currency units)
- RMSE (root mean squared error)
- WAPE / aggregate error (total predicted vs actual revenue)
- Calibration by decile (predicted-value bands vs observed outcomes)
- Business lift (highest predicted-value group vs baseline group)

All functions operate on pandas DataFrames / Series and expect:
- `y_true`: observed net revenue over the horizon (currency)
- `y_pred`: predicted net revenue over the horizon (currency)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricResult:
    """Container for a single evaluation metric."""

    name: str
    value: float
    unit: str
    description: str


def mae(y_true: pd.Series, y_pred: pd.Series) -> MetricResult:
    """Mean Absolute Error in currency units.

    Easy to explain; recommended for customer-level estimates (PRD Section 9).
    """
    y_true, y_pred = _align(y_true, y_pred)
    value = float(np.mean(np.abs(y_true - y_pred)))
    return MetricResult(
        name="MAE",
        value=value,
        unit="currency",
        description=(
            "Average absolute error between predicted and observed net revenue. "
            "Lower is better; directly interpretable in currency units."
        ),
    )


def rmse(y_true: pd.Series, y_pred: pd.Series) -> MetricResult:
    """Root Mean Squared Error.

    Penalises large misses more heavily than MAE. Useful when large errors are
    especially costly (PRD Section 9).
    """
    y_true, y_pred = _align(y_true, y_pred)
    value = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return MetricResult(
        name="RMSE",
        value=value,
        unit="currency",
        description=(
            "Root mean squared error between predicted and observed net revenue. "
            "Penalises large misses more heavily than MAE."
        ),
    )


def wape(y_true: pd.Series, y_pred: pd.Series) -> MetricResult:
    """Weighted Absolute Percentage Error (aggregate error).

    Checks total predicted revenue against actual revenue. Important for finance
    and cohort forecasting (PRD Section 9). Returns a fraction (0 = perfect).
    """
    y_true, y_pred = _align(y_true, y_pred)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return MetricResult(
            name="WAPE",
            value=float("nan"),
            unit="fraction",
            description="WAPE undefined when total observed revenue is zero.",
        )
    value = float(np.sum(np.abs(y_true - y_pred)) / denom)
    return MetricResult(
        name="WAPE",
        value=value,
        unit="fraction",
        description=(
            "Weighted Absolute Percentage Error: |actual - predicted| summed "
            "over all customers divided by total actual revenue. Checks whether "
            "aggregate predicted revenue matches aggregate actual revenue."
        ),
    )


def calibration_by_decile(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration by predicted-value decile.

    Sorts customers into predicted-value bands and compares observed outcomes
    within each band. Well-calibrated predictions show monotonic growth in
    observed revenue as predicted bands increase (PRD Section 9).

    Returns a DataFrame with one row per decile:
    - decile: 1..n_bins (1 = lowest predicted value)
    - n_customers: number of customers in the band
    - mean_predicted_revenue: mean y_pred in the band
    - mean_actual_revenue: mean y_true in the band
    - sum_predicted_revenue: sum y_pred in the band
    - sum_actual_revenue: sum y_true in the band
    - calibration_ratio: sum_actual / sum_predicted (1.0 = perfectly calibrated)
    """
    y_true, y_pred = _align(y_true, y_pred)
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values})

    # Rank customers by predicted value and assign deciles (1 = lowest)
    df["decile"] = pd.qcut(
        df["y_pred"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1

    grouped = (
        df.groupby("decile", sort=True)
        .agg(
            n_customers=("y_true", "size"),
            mean_predicted_revenue=("y_pred", "mean"),
            mean_actual_revenue=("y_true", "mean"),
            sum_predicted_revenue=("y_pred", "sum"),
            sum_actual_revenue=("y_true", "sum"),
        )
        .reset_index()
    )
    grouped["calibration_ratio"] = (
        grouped["sum_actual_revenue"] / grouped["sum_predicted_revenue"]
    ).replace([np.inf, -np.inf], np.nan)

    return grouped


def business_lift(
    y_true: pd.Series,
    y_pred: pd.Series,
    top_fraction: float = 0.2,
    baseline_group: str = "all",
) -> MetricResult:
    """Business lift: value of the highest predicted-value group vs baseline.

    Measures whether the highest predicted-value group is more valuable than a
    baseline group, evaluated on a holdout period (PRD Section 9). Avoids
    confusing correlation with causal impact.

    Parameters
    ----------
    y_true : observed revenue
    y_pred : predicted revenue
    top_fraction : share of customers in the top predicted-value group (default 0.2)
    baseline_group : "all" (all customers) or "random" (same-sized random sample)

    Returns MetricResult with value = mean actual revenue of top group / mean
    actual revenue of baseline group.
    """
    y_true, y_pred = _align(y_true, y_pred)
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values})
    n = len(df)
    if n == 0:
        return MetricResult(
            name="BusinessLift",
            value=float("nan"),
            unit="ratio",
            description="Cannot compute lift on an empty dataset.",
        )

    n_top = max(1, int(round(top_fraction * n)))
    top = df.nlargest(n_top, "y_pred")
    top_mean = float(top["y_true"].mean())

    if baseline_group == "random":
        rng = np.random.default_rng(42)
        baseline_idx = rng.choice(n, size=n_top, replace=False)
        baseline_mean = float(df.iloc[baseline_idx]["y_true"].mean())
        baseline_desc = "same-sized random sample"
    else:
        baseline_mean = float(df["y_true"].mean())
        baseline_desc = "all customers"

    if baseline_mean == 0:
        value = float("nan") if top_mean == 0 else float("inf")
    else:
        value = top_mean / baseline_mean

    return MetricResult(
        name="BusinessLift",
        value=value,
        unit="ratio",
        description=(
            f"Mean actual revenue of the top {top_fraction:.0%} predicted-value "
            f"customers divided by mean actual revenue of the {baseline_desc}. "
            f"A value > 1 indicates the model's top-ranked customers are more "
            f"valuable than baseline."
        ),
    )


def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    cohort: Optional[pd.Series] = None,
    n_bins: int = 10,
    top_fraction: float = 0.2,
) -> dict:
    """Run the full PRD Section 9 evaluation suite.

    Returns a dict with:
    - overall: dict of MetricResult objects (MAE, RMSE, WAPE, BusinessLift)
    - calibration: DataFrame from calibration_by_decile
    - by_cohort: dict mapping cohort name -> dict of MAE / RMSE / WAPE, if
      `cohort` is provided
    """
    y_true, y_pred = _align(y_true, y_pred)

    results: dict = {
        "overall": {
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "WAPE": wape(y_true, y_pred),
            "BusinessLift": business_lift(y_true, y_pred, top_fraction=top_fraction),
        },
        "calibration": calibration_by_decile(y_true, y_pred, n_bins=n_bins),
        "by_cohort": {},
    }

    if cohort is not None:
        _, cohort = _align(y_true, cohort)
        for name, group in pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "cohort": cohort}
        ).groupby("cohort", sort=True):
            gt = group["y_true"].reset_index(drop=True)
            gp = group["y_pred"].reset_index(drop=True)
            results["by_cohort"][str(name)] = {
                "n_customers": len(gt),
                "MAE": mae(gt, gp),
                "RMSE": rmse(gt, gp),
                "WAPE": wape(gt, gp),
            }

    return results


def _align(*series: pd.Series) -> tuple:
    """Align multiple Series on index and drop rows with any missing values."""
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    return tuple(aligned.iloc[:, i] for i in range(aligned.shape[1]))
