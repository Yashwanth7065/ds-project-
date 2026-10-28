"""MVP supervised regression model for CLV prediction.

Compares regularized linear regression with gradient-boosted trees (PRD Section 8:
MVP model). Model selection is based on time-based validation and business
usefulness, not training accuracy alone.

The model predicts expected net revenue over the horizon using point-in-time
features only.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CLVModel:
    """CLV regression model with optional algorithm selection.

    Parameters
    ----------
    algorithm : "gradient_boosted_trees" (default) or "regularized_linear"
    model_version : version string for auditability (PRD FR-8)
    horizon_months : prediction horizon
    """

    NUMERIC_FEATURES = [
        "tenure_months",
        "recurring_amount",
        "successful_payments_12m",
        "failed_payments_12m",
        "total_discounts_12m",
        "total_refunds_12m",
        "add_on_revenue_12m",
        "payment_frequency",
        "login_frequency_30d",
        "feature_usage_score",
        "usage_recency_days",
        "support_interactions_12m",
        "days_since_last_renewal",
        "plan_changes",
        "pause_resume_count",
    ]

    CATEGORICAL_FEATURES = [
        "plan",
        "billing_interval",
        "acquisition_channel",
        "customer_segment",
        "region",
    ]

    def __init__(
        self,
        algorithm: str = "gradient_boosted_trees",
        model_version: str = "clv-model-v1.0.0",
        horizon_months: int = 12,
    ):
        self.algorithm = algorithm
        self.model_version = model_version
        self.horizon_months = horizon_months
        self.pipeline_: Pipeline | None = None
        self.feature_names_in_: list[str] | None = None
        self.is_fitted_ = False

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, [c for c in self.NUMERIC_FEATURES]),
                ("cat", categorical_transformer, [c for c in self.CATEGORICAL_FEATURES]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        if self.algorithm == "gradient_boosted_trees":
            estimator = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )
        elif self.algorithm == "regularized_linear":
            estimator = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CLVModel":
        """Fit the model on point-in-time features and observed revenue."""
        self.feature_names_in_ = list(X.columns)
        X_fit = X[self._available_features(X)].copy()
        y_fit = y.reset_index(drop=True)

        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X_fit, y_fit)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected net revenue over the horizon."""
        if not self.is_fitted_ or self.pipeline_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        X_pred = X[self._available_features(X)].copy()
        preds = self.pipeline_.predict(X_pred)
        return np.clip(preds, 0, None)  # revenue cannot be negative

    def _available_features(self, X: pd.DataFrame) -> list[str]:
        """Return features that exist in X (union of numeric + categorical)."""
        available = []
        for col in self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES:
            if col in X.columns:
                available.append(col)
        return available

    def feature_importance(self) -> pd.DataFrame | None:
        """Return feature importances if the underlying estimator supports them."""
        if not self.is_fitted_ or self.pipeline_ is None:
            return None
        regressor = self.pipeline_.named_steps["regressor"]
        if not hasattr(regressor, "feature_importances_"):
            return None
        preprocessor = self.pipeline_.named_steps["preprocessor"]
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(len(regressor.feature_importances_))]
        imp = pd.DataFrame(
            {"feature": feature_names, "importance": regressor.feature_importances_}
        ).sort_values("importance", ascending=False)
        return imp

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CLVModel":
        return joblib.load(path)
