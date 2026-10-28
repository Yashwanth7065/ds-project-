"""Model training with chronological validation (PRD Section 9).

Trains candidate models (baselines + CLVModel) on chronological train/val/test
splits and selects the best based on validation MAE/WAPE, with business
usefulness as the deciding criterion (not training accuracy alone).
"""

from __future__ import annotations

import pandas as pd

from clv_evaluation import chronological_split, evaluate
from models.baselines import (
    baseline_historical_average,
    predict_baseline_historical_average,
    predict_baseline_recurring_revenue_retention,
)
from models.clv_model import CLVModel


def train_and_select(
    df: pd.DataFrame,
    date_column: str,
    target_column: str = "actual_revenue",
    train_end: str = "2024-12-31",
    validation_end: str = "2025-06-30",
    test_end: str = "2025-12-31",
    feature_columns: list[str] | None = None,
) -> dict:
    """Run chronological training, validation, and model selection.

    Parameters
    ----------
    df : full dataset with features, target, and prediction dates
    date_column : column holding prediction / as-of date
    target_column : observed revenue column
    train_end / validation_end / test_end : chronological cutoffs
    feature_columns : feature columns to use (auto-detected if None)

    Returns
    -------
    dict with:
      - fold: the chronological split
      - baseline1_val / baseline2_val / clv_val: validation metrics
      - baseline1_test / baseline2_test / clv_test: test metrics
      - selected_model: name of selected model
      - best_model: fitted CLVModel instance (if selected)
      - baseline_lookup: Baseline 1 lookup dict
    """
    if feature_columns is None:
        exclude = {date_column, target_column, "customer_id"}
        feature_columns = [c for c in df.columns if c not in exclude]

    fold = chronological_split(df, date_column, train_end, validation_end, test_end)

    X_train = fold.train[feature_columns].reset_index(drop=True)
    y_train = fold.train[target_column].reset_index(drop=True)
    X_val = fold.validation[feature_columns].reset_index(drop=True)
    y_val = fold.validation[target_column].reset_index(drop=True)
    X_test = fold.test[feature_columns].reset_index(drop=True)
    y_test = fold.test[target_column].reset_index(drop=True)

    # ---- Baseline 1: historical average by cohort ----
    baseline_lookup = baseline_historical_average(fold.train)
    b1_val_pred = predict_baseline_historical_average(baseline_lookup, fold.validation)
    b1_val = evaluate(y_val, b1_val_pred)
    b1_test_pred = predict_baseline_historical_average(baseline_lookup, fold.test)
    b1_test = evaluate(y_test, b1_test_pred)

    # ---- Baseline 2: recurring revenue × retention ----
    b2_val_pred = predict_baseline_recurring_revenue_retention(fold.validation)
    b2_val = evaluate(y_val, b2_val_pred)
    b2_test_pred = predict_baseline_recurring_revenue_retention(fold.test)
    b2_test = evaluate(y_test, b2_test_pred)

    # ---- MVP model: gradient-boosted trees ----
    clv = CLVModel(algorithm="gradient_boosted_trees", model_version="clv-gbt-v1.0.0")
    clv.fit(X_train, y_train)
    clv_val_pred = clv.predict(X_val)
    clv_val = evaluate(y_val, pd.Series(clv_val_pred))
    clv_test_pred = clv.predict(X_test)
    clv_test = evaluate(y_test, pd.Series(clv_test_pred))

    # ---- Model selection: pick best validation MAE, require improvement over baselines ----
    candidates = {
        "baseline_historical_average": b1_val["overall"]["MAE"].value,
        "baseline_recurring_retention": b2_val["overall"]["MAE"].value,
        "clv_gbt": clv_val["overall"]["MAE"].value,
    }
    selected_model = min(candidates, key=candidates.get)

    # Only adopt the ML model if it meaningfully beats the best baseline
    best_baseline_mae = min(
        b1_val["overall"]["MAE"].value, b2_val["overall"]["MAE"].value
    )
    if selected_model == "clv_gbt" and clv_val["overall"]["MAE"].value >= best_baseline_mae * 0.95:
        # ML model not meaningfully better (< 5% improvement); keep baseline
        selected_model = (
            "baseline_historical_average"
            if candidates["baseline_historical_average"] <= candidates["baseline_recurring_retention"]
            else "baseline_recurring_retention"
        )

    return {
        "fold": fold,
        "feature_columns": feature_columns,
        "baseline_lookup": baseline_lookup,
        "validation": {
            "baseline_historical_average": b1_val,
            "baseline_recurring_retention": b2_val,
            "clv_gbt": clv_val,
        },
        "test": {
            "baseline_historical_average": b1_test,
            "baseline_recurring_retention": b2_test,
            "clv_gbt": clv_test,
        },
        "selected_model": selected_model,
        "best_model": clv if selected_model == "clv_gbt" else None,
        "candidates_mae": candidates,
    }
