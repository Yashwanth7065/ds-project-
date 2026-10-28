"""End-to-end CLV prediction pipeline (PRD rollout phases 2-5).

Orchestrates:
1. Load config
2. Build point-in-time features + data quality checks
3. Train baselines + MVP model on chronological splits
4. Evaluate (MAE, RMSE, WAPE, calibration, business lift, cohort errors)
5. Select model (only if meaningfully better than baseline)
6. Score and export prediction records

Run:  python run_pipeline.py
"""

from __future__ import annotations

import pandas as pd
import yaml

from clv_evaluation import evaluate, cohort_aggregate, export_predictions
from features.pipeline import build_point_in_time_features
from features.quality_checks import run_data_quality_checks
from models.train import train_and_select


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_synthetic_data(n_customers: int = 2000, seed: int = 42):
    """Generate synthetic subscription customers for demonstration.

    Customers have prediction dates spread across 2024-2025 so that
    chronological train/validation/test splits are meaningful.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Prediction dates spread over two years (chronological structure)
    pred_dates = pd.date_range("2024-01-15", "2025-12-15", periods=n_customers)
    plans = rng.choice(["Starter", "Professional", "Enterprise"], size=n_customers, p=[0.5, 0.35, 0.15])
    regions = rng.choice(["NA", "EMEA", "APAC"], size=n_customers)
    tenure_months = rng.integers(1, 60, size=n_customers)
    plan_base = pd.Series(plans).map({"Starter": 50, "Professional": 150, "Enterprise": 500}).to_numpy()

    customers = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(n_customers)],
            "prediction_date": pred_dates,
            "plan": plans,
            "region": regions,
            "tenure_months": tenure_months,
            "billing_interval": rng.choice(["monthly", "annual"], size=n_customers),
            "acquisition_channel": rng.choice(["organic", "paid", "referral"], size=n_customers),
            "customer_segment": rng.choice(["SMB", "Mid-Market", "Enterprise"], size=n_customers),
            "is_active": True,
            "onboarding_completed": rng.choice([True, False], size=n_customers, p=[0.7, 0.3]),
        }
    )

    # Subscriptions: one per customer, started `tenure_months` before prediction date
    subs = []
    for i in range(n_customers):
        start = pred_dates[i] - pd.DateOffset(months=int(tenure_months[i]))
        subs.append(
            {
                "customer_id": f"CUST_{i:05d}",
                "start_date": start,
                "cancellation_date": pd.NaT,
                "plan": plans[i],
                "plan_changes": int(rng.integers(0, 3)),
                "pause_resume_count": int(rng.integers(0, 2)),
                "renewal_date": pred_dates[i],
            }
        )
    subscriptions = pd.DataFrame(subs)

    # Billing: 12 invoices per customer trailing their prediction date
    bills = []
    for i in range(n_customers):
        for m in range(12):
            inv_date = pred_dates[i] - pd.DateOffset(months=11 - m)
            bills.append(
                {
                    "customer_id": f"CUST_{i:05d}",
                    "invoice_date": inv_date,
                    "amount": float(plan_base[i]) * rng.uniform(0.9, 1.1),
                    "status": rng.choice(["success", "failed"], p=[0.95, 0.05]),
                    "discounts": float(rng.uniform(0, 20)),
                    "refunds": float(rng.uniform(0, 10)),
                    "add_ons": float(rng.uniform(0, 50)),
                }
            )
    billing = pd.DataFrame(bills)

    # Engagement: 30 days of events per customer trailing their prediction date
    eng = []
    for i in range(n_customers):
        for d in range(30):
            event_date = pred_dates[i] - pd.DateOffset(days=29 - d)
            eng.append(
                {
                    "customer_id": f"CUST_{i:05d}",
                    "event_date": event_date,
                    "login": int(rng.choice([0, 1], p=[0.3, 0.7])),
                    "feature_usage": float(rng.uniform(0, 10)),
                    "support_interactions": int(rng.choice([0, 1], p=[0.9, 0.1])),
                }
            )
    engagement = pd.DataFrame(eng)

    return customers, subscriptions, billing, engagement


def build_features_across_dates(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    billing: pd.DataFrame,
    engagement: pd.DataFrame,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Build point-in-time features for each customer at their own prediction date.

    Groups customers by prediction_date, builds features per date (so every
    feature is computed from data on/before that date), and concatenates.
    This yields a labeled dataset with varied prediction dates suitable for
    chronological splitting.
    """
    feature_frames = []
    for pdate, group in customers.groupby("prediction_date"):
        feats = build_point_in_time_features(
            customers=group,
            subscriptions=subscriptions,
            billing=billing,
            engagement=engagement,
            prediction_date=pdate,
            horizon_months=horizon_months,
        )
        feature_frames.append(feats)

    if not feature_frames:
        return pd.DataFrame()

    return pd.concat(feature_frames, ignore_index=True)


def main() -> None:
    print("=== CLV Prediction Pipeline ===")
    config = load_config()
    print(f"Loaded config: horizon={config['target']['horizon_months']} months, "
          f"currency={config['target']['currency']}")

    # ---- Data (synthetic for demo; replace with real source-of-truth tables) ----
    customers, subscriptions, billing, engagement = make_synthetic_data()
    print(f"Loaded {len(customers)} customers")

    # ---- Point-in-time features (FR-2) ----
    horizon = config["target"]["horizon_months"]
    features = build_features_across_dates(
        customers=customers,
        subscriptions=subscriptions,
        billing=billing,
        engagement=engagement,
        horizon_months=horizon,
    )
    print(f"Built features for {len(features)} eligible customers across "
          f"{features['prediction_date'].nunique()} prediction dates")

    # ---- Data quality checks (FR-6) ----
    dq = run_data_quality_checks(features, id_column="customer_id")
    print(f"Data quality status: {dq.status} (passed={dq.passed})")

    # ---- Build labeled dataset for training ----
    # In production, actual_revenue = observed net revenue over (t, t+12m] from
    # billing history. Here we synthesize a target correlated with features.
    import numpy as np

    rng = np.random.default_rng(42)
    features["actual_revenue"] = (
        features["recurring_amount"].fillna(100) * 12
        * (1 + 0.01 * features["tenure_months"].fillna(12))
        + rng.normal(0, 200, len(features))
    )
    features["actual_revenue"] = features["actual_revenue"].clip(lower=0)

    # ---- Train / validate / select model (chronological, Section 9) ----
    result = train_and_select(
        df=features,
        date_column="prediction_date",
        target_column="actual_revenue",
        train_end=config["validation"]["train_end"],
        validation_end=config["validation"]["validation_end"],
        test_end=config["validation"]["test_end"],
    )

    print(f"\nSelected model: {result['selected_model']}")
    print(f"Validation MAE by candidate: {result['candidates_mae']}")

    # ---- Report test metrics (Section 9) ----
    fold = result["fold"]
    y_test = fold.test["actual_revenue"].reset_index(drop=True)

    print(f"\nTrain={len(fold.train)}, Val={len(fold.validation)}, Test={len(fold.test)}")

    if result["best_model"] is not None:
        y_pred = result["best_model"].predict(
            fold.test[result["feature_columns"]].reset_index(drop=True)
        )
        test_metrics = evaluate(
            y_test,
            pd.Series(y_pred),
            cohort=fold.test["plan"].reset_index(drop=True),
        )

        print("\n=== Test holdout metrics ===")
        for name, m in test_metrics["overall"].items():
            print(f"  {name}: {m.value:.4f} {m.unit}")
        print("\nCalibration by decile:")
        print(test_metrics["calibration"].round(2).to_string(index=False))
        print("\nCohort error by plan:")
        for plan_name, m in test_metrics["by_cohort"].items():
            print(f"  {plan_name}: n={m['n_customers']}, "
                  f"MAE={m['MAE'].value:.2f}, WAPE={m['WAPE'].value:.4f}")
    else:
        print("\nNo ML model selected (baseline chosen).")

    # ---- Score & export predictions (FR-4, Section 10) ----
    best_model = result["best_model"]
    if best_model is None:
        # Fallback: train a quick model on all data for export demo
        from models.clv_model import CLVModel

        best_model = CLVModel(model_version="clv-gbt-v1.0.0")
        X_all = features[result["feature_columns"]].reset_index(drop=True)
        y_all = features["actual_revenue"].reset_index(drop=True)
        best_model.fit(X_all, y_all)

    from deployment.scoring_job import score_customers_from_features

    # Score at the latest prediction date in the data
    scoring_date = features["prediction_date"].max()
    scoring_customers = customers[customers["prediction_date"] == scoring_date]

    # Rebuild features for the scoring date
    scoring_features = build_point_in_time_features(
        customers=scoring_customers,
        subscriptions=subscriptions,
        billing=billing,
        engagement=engagement,
        prediction_date=scoring_date,
        horizon_months=horizon,
    )

    records = score_customers_from_features(
        model=best_model,
        features=scoring_features,
        prediction_date=scoring_date.to_pydatetime(),
        horizon_months=horizon,
        currency=config["target"]["currency"],
    )

    export_predictions(records, "clv_predictions.json", format="json")
    print(f"\nExported {len(records)} prediction records to clv_predictions.json")

    # ---- Cohort aggregation (FR-5) ----
    pred_df = pd.DataFrame([r.to_dict() for r in records])
    pred_df["plan"] = scoring_features["plan"].values
    pred_df["region"] = scoring_features["region"].values
    agg = cohort_aggregate(pred_df, group_by=["plan", "region"])
    print("\n=== Cohort aggregation (plan x region) ===")
    print(agg.round(2).to_string(index=False))

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
