"""Chronological train/validation/test split utilities.

PRD Section 9 requires chronological splits so that each example's features are
available at its prediction date and its full outcome window is observable.
Future-data leakage must be avoided (PRD FR-2).
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PredictionFold:
    """A single chronological train/validation/test split."""

    fold_name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_cutoff: pd.Timestamp
    validation_cutoff: pd.Timestamp
    test_cutoff: pd.Timestamp


def chronological_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_end: Optional[str | pd.Timestamp] = None,
) -> PredictionFold:
    """Split a dataset chronologically into train / validation / test.

    All rows with prediction date <= train_end go to train; rows in
    (train_end, validation_end] go to validation; rows in
    (validation_end, test_end] (or to the max date if test_end is None) go to
    test.

    This guarantees that model training never sees outcomes that occur after the
    prediction date used for features, satisfying FR-2 (as-of timestamp) and the
    PRD's leakage-prevention requirement.

    Parameters
    ----------
    df : input DataFrame containing customer-level rows with a prediction date
    date_column : name of the column holding the prediction / as-of date
    train_end : last prediction date included in training (inclusive)
    validation_end : last prediction date included in validation (inclusive)
    test_end : last prediction date included in test (inclusive). If None, uses
        the maximum date in the data.

    Returns
    -------
    PredictionFold
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    test_end = pd.Timestamp(test_end) if test_end is not None else df[date_column].max()

    train_mask = df[date_column] <= train_end
    validation_mask = (df[date_column] > train_end) & (df[date_column] <= validation_end)
    test_mask = (df[date_column] > validation_end) & (df[date_column] <= test_end)

    train = df.loc[train_mask].reset_index(drop=True)
    validation = df.loc[validation_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)

    return PredictionFold(
        fold_name=f"train_{train_end.date()}_val_{validation_end.date()}_test_{test_end.date()}",
        train=train,
        validation=validation,
        test=test,
        train_cutoff=train_end,
        validation_cutoff=validation_end,
        test_cutoff=test_end,
    )


def rolling_chronological_folds(
    df: pd.DataFrame,
    date_column: str,
    n_folds: int = 3,
    train_months: int = 12,
    validation_months: int = 3,
    test_months: int = 3,
    gap_months: int = 0,
) -> list[PredictionFold]:
    """Generate multiple rolling chronological folds for backtesting.

    Each fold shifts the train/validation/test windows forward in time, mimicking
    production retraining on progressively newer data while always evaluating on
    a future holdout window. Useful for the PRD's "monitor model quality over
    time" requirement (FR-7).
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    folds: list[PredictionFold] = []
    month_offset = pd.DateOffset(months=1)

    for i in range(n_folds):
        # Shift each window forward by i * validation_months
        shift = i * validation_months
        train_start = min_date
        train_end = min_date + pd.DateOffset(months=train_months) + shift * month_offset
        validation_end = train_end + pd.DateOffset(months=validation_months)
        test_end = validation_end + pd.DateOffset(months=test_months)

        if train_end > max_date:
            break

        fold = chronological_split(
            df,
            date_column,
            train_end=train_end,
            validation_end=validation_end,
            test_end=min(test_end, max_date),
        )
        folds.append(fold)

    return folds
