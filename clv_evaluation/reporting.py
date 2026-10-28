"""Prediction output records and reporting utilities.

Implements PRD Section 10 (User Experience and Output): each prediction record
includes customer ID, prediction date and horizon, predicted revenue amount and
currency, optional uncertainty range, model version, data-quality status, and
top explanatory factors. Also provides cohort aggregation (FR-5) and export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PredictionRecord:
    """A single customer CLV prediction output record.

    Fields match PRD Section 10 requirements. Display estimates as forecasts,
    not guaranteed revenue, and show horizon and last refresh date prominently.
    """

    customer_id: str
    prediction_date: str  # ISO-8601 timestamp (FR-2: as-of timestamp)
    horizon_months: int
    predicted_revenue: float
    currency: str
    model_version: str
    data_quality_status: str  # e.g. "ok", "stale_features", "missing_values"
    uncertainty_lower: Optional[float] = None
    uncertainty_upper: Optional[float] = None
    top_factors: list[str] = field(default_factory=list)
    analytics_view_url: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Nullable floats serialise as None -> null in JSON, which is fine.
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_prediction_record(
    customer_id: str,
    prediction_date: datetime,
    horizon_months: int,
    predicted_revenue: float,
    currency: str,
    model_version: str,
    data_quality_status: str = "ok",
    uncertainty_lower: Optional[float] = None,
    uncertainty_upper: Optional[float] = None,
    top_factors: Optional[list[str]] = None,
    analytics_view_url: Optional[str] = None,
) -> PredictionRecord:
    """Factory for a PredictionRecord with an ISO-8601 as-of timestamp (FR-2)."""
    return PredictionRecord(
        customer_id=str(customer_id),
        prediction_date=prediction_date.astimezone(timezone.utc).isoformat(),
        horizon_months=int(horizon_months),
        predicted_revenue=float(predicted_revenue),
        currency=currency,
        model_version=model_version,
        data_quality_status=data_quality_status,
        uncertainty_lower=uncertainty_lower,
        uncertainty_upper=uncertainty_upper,
        top_factors=top_factors or [],
        analytics_view_url=analytics_view_url,
    )


def cohort_aggregate(
    predictions: pd.DataFrame,
    group_by: list[str],
    revenue_column: str = "predicted_revenue",
    customer_column: str = "customer_id",
) -> pd.DataFrame:
    """Aggregate predictions by cohort, plan, channel, geography, or tenure.

    Supports FR-5 (aggregate views). Requires `group_by` columns to exist in the
    predictions DataFrame. Returns one row per cohort with count, total and mean
    predicted revenue.
    """
    missing = [c for c in group_by + [revenue_column, customer_column] if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing columns required for aggregation: {missing}")

    agg = (
        predictions.groupby(group_by, sort=True)
        .agg(
            n_customers=(customer_column, "count"),
            total_predicted_revenue=(revenue_column, "sum"),
            mean_predicted_revenue=(revenue_column, "mean"),
            min_predicted_revenue=(revenue_column, "min"),
            max_predicted_revenue=(revenue_column, "max"),
        )
        .reset_index()
    )
    return agg


def export_predictions(
    records: list[PredictionRecord],
    path: str,
    format: str = "json",
) -> str:
    """Export prediction records to JSON or CSV.

    Parameters
    ----------
    records : list of PredictionRecord
    path : output file path
    format : "json" (default) or "csv"

    Returns the written path.
    """
    if format == "json":
        payload = [r.to_dict() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    elif format == "csv":
        df = pd.DataFrame([r.to_dict() for r in records])
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {format!r}. Use 'json' or 'csv'.")
    return path
