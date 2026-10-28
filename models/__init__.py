"""Model package: baselines, MVP model, and training (PRD Section 8)."""

from .baselines import baseline_historical_average, baseline_recurring_revenue_retention
from .clv_model import CLVModel

__all__ = [
    "baseline_historical_average",
    "baseline_recurring_revenue_retention",
    "CLVModel",
]
