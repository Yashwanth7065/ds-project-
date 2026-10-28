"""CLV evaluation and reporting package.

Provides metrics, chronological validation utilities, and prediction output
record formatting aligned to the CLV Prediction PRD (Section 9: Evaluation and
Success Metrics; Section 10: User Experience and Output).
"""

from .evaluation import (
    mae,
    rmse,
    wape,
    calibration_by_decile,
    business_lift,
    evaluate,
    MetricResult,
)
from .validation import chronological_split, PredictionFold
from .reporting import (
    PredictionRecord,
    build_prediction_record,
    cohort_aggregate,
    export_predictions,
)

__all__ = [
    "mae",
    "rmse",
    "wape",
    "calibration_by_decile",
    "business_lift",
    "evaluate",
    "MetricResult",
    "chronological_split",
    "PredictionFold",
    "PredictionRecord",
    "build_prediction_record",
    "cohort_aggregate",
    "export_predictions",
]
