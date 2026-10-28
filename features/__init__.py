"""Feature engineering package for CLV prediction.

Builds point-in-time customer features as of a prediction date, with no
future-data leakage (PRD FR-2).
"""

from .pipeline import build_point_in_time_features, FeatureSnapshot
from .quality_checks import run_data_quality_checks, DataQualityReport

__all__ = [
    "build_point_in_time_features",
    "FeatureSnapshot",
    "run_data_quality_checks",
    "DataQualityReport",
]
