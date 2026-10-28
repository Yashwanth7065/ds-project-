"""End-to-end example of CLV evaluation & reporting.

Uses synthetic customer data to demonstrate:
1. Chronological train/validation/test split (no future-data leakage)
2. Overall metrics: MAE, RMSE, WAPE, Business Lift
3. Calibration by decile
4. Cohort-level error analysis (by plan and region)
5. Prediction output records with horizon/timestamp/model version/currency
6. Cohort aggregation and export

Run:  python -m clv_evaluation.example_usage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clv_evaluation import (
    chronological_split,
    evaluate,
    build_prediction_record,
    cohort_aggregate,
    export_predictions,
)


def make_synthetic_data(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic subscription customers with a prediction date and revenue."""
    rng = np.random.default_rng(seed)

    # Prediction dates spread over two years (chronological structure)
    pred_dates = pd.date_range("2024-01-01", "2025-12-31", periods=n_customers)

    plans = rng.choice(["Starter", "Professional", "Enterprise"], size=n_customers, p=[0.5, 0.35, 0.15])
    regions = rng.choice(["NA", "EMEA", "APAC"], size=n_customers)
    tenure_months = rng.integers(1, 60, size=n_customers)

    # Base monthly revenue by plan; true revenue adds tenure + noise
    plan_base = pd.Series(plans).map({"Starter": 50, "Professional": 150, "Enterprise": 500}).to_numpy()
    # Observed 12-month net revenue (the target)
    actual_revenue = plan_base * 12 * (1 + 0.01 * tenure_months) + rng.normal(0, 200, n_customers)
    actual_revenue = np.clip(actual_revenue, 0, None)

    # Model predictions: a deliberately imperfect signal correlated with actuals
    signal = plan_base * 12 * (1 + 0.008 * tenure_months)
    predicted_revenue = signal + rng.normal(0, 300, n_customers)
    predicted_revenue = np.clip(predicted_revenue, 0, None)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(n_customers)],
            "prediction_date": pred_dates,
            "plan": plans,
            "region": regions,
            "tenure_months": tenure_months,
            "actual_revenue": actual_revenue,
            "predicted_revenue": predicted_revenue,
        }
    )


def main() -> None:
    df = make_synthetic_data()

    # ------------------------------------------------------------------
    # 1. Chronological split (PRD Section 9; FR-2 as-of timestamp)
    # ------------------------------------------------------------------
    fold = chronological_split(
        df,
        date_column="prediction_date",
        train_end="2024-12-31",
        validation_end="2025-06-30",
        test_end="2025-12-31",
    )
    print(f"Fold: {fold.fold_name}")
    print(f"  train={len(fold.train)}, val={len(fold.validation)}, test={len(fold.test)}")

    # Evaluate on the chronological test holdout
    y_true = fold.test["actual_revenue"].reset_index(drop=True)
    y_pred = fold.test["predicted_revenue"].reset_index(drop=True)
    cohort = fold.test["plan"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Overall metrics (PRD Section 9)
    # ------------------------------------------------------------------
    results = evaluate(y_true, y_pred, cohort=cohort, n_bins=10, top_fraction=0.2)

    print("\n=== Overall metrics (test holdout) ===")
    for name, metric in results["overall"].items():
        print(f"  {name}: {metric.value:.4f} {metric.unit} — {metric.description.split('.')[0]}")

    print("\n=== Calibration by decile ===")
    print(results["calibration"].round(2).to_string(index=False))

    print("\n=== Cohort-level error (by plan) ===")
    for plan, m in results["by_cohort"].items():
        print(
            f"  {plan}: n={m['n_customers']}, "
            f"MAE={m['MAE'].value:.2f}, RMSE={m['RMSE'].value:.2f}, WAPE={m['WAPE'].value:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Prediction output records (PRD Section 10)
    # ------------------------------------------------------------------
    from datetime import datetime, timezone

    prediction_date = datetime(2026, 1, 15, tzinfo=timezone.utc)
    records = []
    for _, row in fold.test.head(5).iterrows():
        rec = build_prediction_record(
            customer_id=row["customer_id"],
            prediction_date=prediction_date,
            horizon_months=12,
            predicted_revenue=row["predicted_revenue"],
            currency="USD",
            model_version="clv-gbt-v1.0",
            data_quality_status="ok",
            uncertainty_lower=row["predicted_revenue"] * 0.8,
            uncertainty_upper=row["predicted_revenue"] * 1.2,
            top_factors=["plan", "tenure_months", "region"],
            analytics_view_url="https://analytics.example.com/clv/CUST_00001",
        )
        records.append(rec)
        print("\n=== Sample prediction record ===")
        print(rec.to_json())

    # ------------------------------------------------------------------
    # 4. Cohort aggregation + export (FR-5)
    # ------------------------------------------------------------------
    pred_df = fold.test[["customer_id", "plan", "region", "predicted_revenue"]].copy()
    agg = cohort_aggregate(pred_df, group_by=["plan", "region"])
    print("\n=== Cohort aggregation (plan x region) ===")
    print(agg.round(2).to_string(index=False))

    export_predictions(records, "sample_predictions.json", format="json")
    print("\nExported sample predictions to sample_predictions.json")


if __name__ == "__main__":
    main()
