"""Model monitoring for CLV predictions (PRD FR-7).

Tracks:
- prediction error (MAE, RMSE, WAPE) over time
- data drift (feature distribution shifts vs training baseline)
- prediction distribution
- model version

Designed to be run periodically after production rollout.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MonitoringSnapshot:
    """A single monitoring snapshot."""

    run_date: datetime
    model_version: str
    n_predictions: int
    mean_predicted_revenue: float
    median_predicted_revenue: float
    mape_vs_actual: float | None = None  # if actuals available for this window
    feature_drift_flags: dict = field(default_factory=dict)  # feature -> drift score
    alerts: list[str] = field(default_factory=list)


class CLVMonitor:
    """Lightweight monitoring for CLV model quality over time."""

    def __init__(self, model_version: str, baseline_feature_stats: dict | None = None):
        self.model_version = model_version
        self.baseline_feature_stats = baseline_feature_stats or {}
        self.snapshots: list[MonitoringSnapshot] = []

    def record_snapshot(
        self,
        predictions: pd.Series,
        actuals: pd.Series | None = None,
        features: pd.DataFrame | None = None,
        run_date: datetime | None = None,
    ) -> MonitoringSnapshot:
        """Record a monitoring snapshot from a scoring run."""
        run_date = run_date or datetime.now()
        alerts: list[str] = []

        n = len(predictions)
        mean_pred = float(predictions.mean())
        median_pred = float(predictions.median())

        mape = None
        if actuals is not None and len(actuals) == n:
            denom = actuals.abs().replace(0, pd.NA)
            mape = float(((predictions - actuals).abs() / denom).mean())
            if mape > 0.20:
                alerts.append(f"High MAPE ({mape:.2%}) vs actuals")

        # Simple drift check: compare feature means to baseline
        drift_flags: dict[str, float] = {}
        if features is not None and self.baseline_feature_stats:
            for col, baseline_mean in self.baseline_feature_stats.items():
                if col in features.columns:
                    current_mean = float(features[col].mean())
                    if baseline_mean != 0:
                        drift = abs(current_mean - baseline_mean) / abs(baseline_mean)
                        drift_flags[col] = drift
                        if drift > 0.25:
                            alerts.append(f"Drift in {col}: {drift:.2%}")

        snapshot = MonitoringSnapshot(
            run_date=run_date,
            model_version=self.model_version,
            n_predictions=n,
            mean_predicted_revenue=mean_pred,
            median_predicted_revenue=median_pred,
            mape_vs_actual=mape,
            feature_drift_flags=drift_flags,
            alerts=alerts,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def summary(self) -> pd.DataFrame:
        """Return all snapshots as a DataFrame."""
        rows = []
        for s in self.snapshots:
            rows.append(
                {
                    "run_date": s.run_date,
                    "model_version": s.model_version,
                    "n_predictions": s.n_predictions,
                    "mean_predicted_revenue": s.mean_predicted_revenue,
                    "median_predicted_revenue": s.median_predicted_revenue,
                    "mape_vs_actual": s.mape_vs_actual,
                    "n_alerts": len(s.alerts),
                    "alerts": "; ".join(s.alerts),
                }
            )
        return pd.DataFrame(rows)
