"""Batch scoring job for CLV predictions (PRD FR-4, FR-8, Section 10).

Produces one prediction record per eligible customer for the selected horizon,
with horizon, timestamp, model version, currency, and optional uncertainty.
Safe retries supported: each run is idempotent given the same prediction date
and model version.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone

from clv_evaluation.reporting import build_prediction_record, export_predictions
from features.pipeline import build_point_in_time_features
from features.quality_checks import run_data_quality_checks
from models.clv_model import CLVModel


def run_scoring_job(
    model: CLVModel,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    billing: pd.DataFrame,
    engagement: pd.DataFrame,
    prediction_date: datetime,
    horizon_months: int = 12,
    currency: str = "USD",
    output_path: str = "predictions.json",
    feature_columns: list[str] | None = None,
    customer_id_column: str = "customer_id",
) -> list:
    """Run a batch scoring job and export prediction records.

    Parameters
    ----------
    model : fitted CLVModel
    customers / subscriptions / billing / engagement : raw source tables
    prediction_date : as-of date (ISO-8601 stored on every record, FR-2)
    horizon_months : prediction horizon (default 12)
    currency : revenue currency (displayed on every record, Section 10)
    output_path : where to write the exported predictions
    feature_columns : feature columns used by the model
    customer_id_column : name of the customer ID column

    Returns
    -------
    list of PredictionRecord objects (also exported to output_path)
    """
    # 1. Build point-in-time features (no future-data leakage, FR-2)
    features = build_point_in_time_features(
        customers=customers,
        subscriptions=subscriptions,
        billing=billing,
        engagement=engagement,
        prediction_date=prediction_date,
        horizon_months=horizon_months,
        customer_id_column=customer_id_column,
    )

    if features.empty:
        raise ValueError("No eligible customers found for scoring.")

    # 2. Data quality checks (FR-6) -> data_quality_status on each record
    dq_report = run_data_quality_checks(
        features,
        id_column=customer_id_column,
        date_columns=["prediction_date"],
        reference_date=pd.Timestamp(prediction_date),
    )

    # 3. Score
    if feature_columns is None:
        feature_columns = [
            c for c in features.columns
            if c not in {"customer_id", "prediction_date", "horizon_months"}
        ]
    X = features[feature_columns].copy()
    predictions = model.predict(X)

    # 4. Build prediction records (Section 10 output shape)
    records = []
    for i, row in features.iterrows():
        rec = build_prediction_record(
            customer_id=row[customer_id_column],
            prediction_date=prediction_date,
            horizon_months=horizon_months,
            predicted_revenue=float(predictions[i]),
            currency=currency,
            model_version=model.model_version,
            data_quality_status=dq_report.status,
            uncertainty_lower=float(predictions[i] * 0.8),
            uncertainty_upper=float(predictions[i] * 1.2),
            top_factors=["plan", "tenure_months", "recurring_amount", "region"],
        )
        records.append(rec)

    # 5. Export (idempotent given same inputs)
    export_predictions(records, output_path, format="json")

    return records


def score_customers_from_features(
    model: CLVModel,
    features: pd.DataFrame,
    prediction_date: datetime,
    horizon_months: int = 12,
    currency: str = "USD",
    customer_id_column: str = "customer_id",
) -> list:
    """Score an already-built feature DataFrame (useful for testing/pipelines)."""
    if features.empty:
        return []

    X = features.copy()
    predictions = model.predict(X)

    records = []
    for i, row in X.iterrows():
        cid = row.get(customer_id_column, f"unknown_{i}")
        rec = build_prediction_record(
            customer_id=str(cid),
            prediction_date=prediction_date,
            horizon_months=horizon_months,
            predicted_revenue=float(predictions[i]),
            currency=currency,
            model_version=model.model_version,
            data_quality_status="ok",
        )
        records.append(rec)
    return records
