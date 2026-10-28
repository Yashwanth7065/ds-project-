"""Data quality checks for CLV inputs and features (PRD FR-6).

Flags missing, stale, duplicated, or inconsistent input data. Returns a
DataQualityReport that can be attached to prediction records (data_quality_status).
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataQualityReport:
    """Results of data quality checks."""

    total_rows: int
    missing_counts: dict = field(default_factory=dict)
    missing_fraction: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    stale_columns: dict = field(default_factory=dict)  # column -> max age in days
    inconsistent_flags: list[str] = field(default_factory=list)
    passed: bool = True

    @property
    def status(self) -> str:
        """Overall status for prediction record data_quality_status."""
        if not self.passed:
            return "failed"
        if self.duplicate_rows > 0 or any(v > 0 for v in self.missing_counts.values()):
            return "warn"
        if any(v > 30 for v in self.stale_columns.values()):
            return "stale_features"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "missing_counts": self.missing_counts,
            "missing_fraction": self.missing_fraction,
            "duplicate_rows": self.duplicate_rows,
            "stale_columns": self.stale_columns,
            "inconsistent_flags": self.inconsistent_flags,
            "passed": self.passed,
            "status": self.status,
        }


def run_data_quality_checks(
    df: pd.DataFrame,
    id_column: str = "customer_id",
    date_columns: Optional[list[str]] = None,
    reference_date: Optional[pd.Timestamp] = None,
    max_missing_fraction: float = 0.5,
    stale_threshold_days: int = 30,
) -> DataQualityReport:
    """Run FR-6 data quality checks on a feature DataFrame.

    Checks:
    - missing values per column (flag if fraction > max_missing_fraction)
    - duplicate customer IDs
    - staleness of date columns (days since reference_date)
    - inconsistent values (negative revenue, tenure < 0, etc.)

    Returns a DataQualityReport with an overall status.
    """
    reference_date = reference_date or pd.Timestamp.now()
    date_columns = date_columns or []

    report = DataQualityReport(total_rows=len(df))

    # Missing values
    for col in df.columns:
        missing = int(df[col].isna().sum())
        report.missing_counts[col] = missing
        report.missing_fraction[col] = float(missing / len(df)) if len(df) else 0.0

    # Duplicates
    if id_column in df.columns:
        report.duplicate_rows = int(df[id_column].duplicated().sum())

    # Staleness
    for col in date_columns:
        if col in df.columns:
            dates = pd.to_datetime(df[col], errors="coerce")
            valid = dates.dropna()
            if not valid.empty:
                max_age_days = float((reference_date - valid.max()).days)
                report.stale_columns[col] = max_age_days

    # Inconsistency flags
    inconsistencies: list[str] = []
    for col in ["tenure_months", "recurring_amount", "successful_payments_12m", "failed_payments_12m"]:
        if col in df.columns:
            negatives = int((df[col] < 0).sum())
            if negatives > 0:
                inconsistencies.append(f"{col}: {negatives} negative values")
    if "tenure_months" in df.columns and "is_active" in df.columns:
        active_zero_tenure = int(((df["is_active"] == True) & (df["tenure_months"] == 0)).sum())
        if active_zero_tenure > 0:
            inconsistencies.append(f"is_active=True with tenure_months=0: {active_zero_tenure} rows")

    report.inconsistent_flags = inconsistencies

    # Overall pass/fail
    missing_fail = any(v > max_missing_fraction for v in report.missing_fraction.values())
    report.passed = not missing_fail and report.duplicate_rows == 0 and len(inconsistencies) == 0

    return report
