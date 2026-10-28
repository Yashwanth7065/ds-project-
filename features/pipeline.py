"""Point-in-time feature pipeline for CLV prediction.

Builds customer-level features as of a prediction date. All features are
computed from data available on or before the prediction date — no future-data
leakage (PRD FR-2, Section 7: Data controls).

Expected raw inputs (subject to availability per PRD Section 7):
- subscriptions: customer_id, start_date, plan, billing_interval, cancellation_status, ...
- billing: customer_id, invoice_date, amount, status, discounts, refunds, add_ons, ...
- engagement: customer_id, event_date, login, feature_usage, support_interactions, ...
- customers: customer_id, acquisition_channel, campaign, segment, region, ...
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class FeatureSnapshot:
    """A point-in-time feature snapshot for one customer at one prediction date.

    Attributes mirror PRD Section 7 feature groups.
    """

    customer_id: str
    prediction_date: pd.Timestamp
    horizon_months: int

    # Subscription features
    tenure_months: Optional[float] = None
    plan: Optional[str] = None
    billing_interval: Optional[str] = None
    plan_changes: Optional[int] = None
    pause_resume_count: Optional[int] = None
    days_since_last_renewal: Optional[float] = None
    is_active: Optional[bool] = None

    # Billing features
    recurring_amount: Optional[float] = None
    successful_payments_12m: Optional[int] = None
    failed_payments_12m: Optional[int] = None
    total_discounts_12m: Optional[float] = None
    total_refunds_12m: Optional[float] = None
    add_on_revenue_12m: Optional[float] = None
    payment_frequency: Optional[float] = None

    # Engagement features
    login_frequency_30d: Optional[float] = None
    feature_usage_score: Optional[float] = None
    usage_recency_days: Optional[float] = None
    support_interactions_12m: Optional[int] = None
    onboarding_completed: Optional[bool] = None

    # Acquisition / context
    acquisition_channel: Optional[str] = None
    campaign: Optional[str] = None
    customer_segment: Optional[str] = None
    region: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "prediction_date": self.prediction_date.isoformat(),
            "horizon_months": self.horizon_months,
            "tenure_months": self.tenure_months,
            "plan": self.plan,
            "billing_interval": self.billing_interval,
            "plan_changes": self.plan_changes,
            "pause_resume_count": self.pause_resume_count,
            "days_since_last_renewal": self.days_since_last_renewal,
            "is_active": self.is_active,
            "recurring_amount": self.recurring_amount,
            "successful_payments_12m": self.successful_payments_12m,
            "failed_payments_12m": self.failed_payments_12m,
            "total_discounts_12m": self.total_discounts_12m,
            "total_refunds_12m": self.total_refunds_12m,
            "add_on_revenue_12m": self.add_on_revenue_12m,
            "payment_frequency": self.payment_frequency,
            "login_frequency_30d": self.login_frequency_30d,
            "feature_usage_score": self.feature_usage_score,
            "usage_recency_days": self.usage_recency_days,
            "support_interactions_12m": self.support_interactions_12m,
            "onboarding_completed": self.onboarding_completed,
            "acquisition_channel": self.acquisition_channel,
            "campaign": self.campaign,
            "customer_segment": self.customer_segment,
            "region": self.region,
        }


def build_point_in_time_features(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    billing: pd.DataFrame,
    engagement: pd.DataFrame,
    prediction_date: datetime,
    horizon_months: int = 12,
    customer_id_column: str = "customer_id",
    prediction_date_column: str = "prediction_date",
) -> pd.DataFrame:
    """Build point-in-time features for all eligible customers as of prediction_date.

    Parameters
    ----------
    customers : master customer table with acquisition/context attributes
    subscriptions : subscription history with start/cancellation dates
    billing : billing history with invoice dates, amounts, statuses
    engagement : engagement events (logins, feature usage, support tickets)
    prediction_date : as-of date; only data on or before this date is used
    horizon_months : CLV prediction horizon (default 12)
    customer_id_column : name of the customer ID column
    prediction_date_column : name of the prediction date column in customers

    Returns
    -------
    pd.DataFrame with one row per eligible customer and all feature columns.
    """
    prediction_date = pd.Timestamp(prediction_date)
    horizon_end = prediction_date + pd.DateOffset(months=horizon_months)

    # ---- Eligible customers: active as of prediction date ----
    active_customers = _eligible_customers(
        customers, subscriptions, prediction_date, customer_id_column
    )

    rows: list[FeatureSnapshot] = []

    for _, cust in active_customers.iterrows():
        cid = cust[customer_id_column]

        # Subscription features computed from history on/before prediction date
        sub_hist = subscriptions[
            subscriptions[customer_id_column] == cid
        ].copy()
        sub_hist = sub_hist[sub_hist["start_date"] <= prediction_date]

        tenure_months = None
        if not sub_hist.empty:
            first_start = pd.to_datetime(sub_hist["start_date"]).min()
            tenure_months = (prediction_date - first_start).days / 30.44

        plan_changes = int(sub_hist["plan_changes"].sum()) if "plan_changes" in sub_hist.columns else 0
        pause_resume_count = int(sub_hist["pause_resume_count"].sum()) if "pause_resume_count" in sub_hist.columns else 0

        last_renewal = None
        if "renewal_date" in sub_hist.columns:
            renewals = pd.to_datetime(sub_hist["renewal_date"], errors="coerce")
            renewals = renewals[renewals <= prediction_date]
            if not renewals.empty:
                last_renewal = (prediction_date - renewals.max()).days

        is_active = bool(cust.get("is_active", True))

        # Billing features: trailing 12 months before prediction date
        billing_hist = billing[billing[customer_id_column] == cid].copy()
        billing_hist = billing_hist[billing_hist["invoice_date"] <= prediction_date]
        billing_window = billing_hist[
            billing_hist["invoice_date"] >= prediction_date - pd.DateOffset(months=12)
        ]

        recurring_amount = float(billing_hist["amount"].iloc[-1]) if "amount" in billing_hist.columns and not billing_hist.empty else None
        successful_payments_12m = int((billing_window["status"] == "success").sum()) if "status" in billing_window.columns else 0
        failed_payments_12m = int((billing_window["status"] == "failed").sum()) if "status" in billing_window.columns else 0
        total_discounts_12m = float(billing_window["discounts"].sum()) if "discounts" in billing_window.columns else 0.0
        total_refunds_12m = float(billing_window["refunds"].sum()) if "refunds" in billing_window.columns else 0.0
        add_on_revenue_12m = float(billing_window["add_ons"].sum()) if "add_ons" in billing_window.columns else 0.0
        payment_frequency = float(billing_window.shape[0] / 12.0) if not billing_window.empty else 0.0

        # Engagement features: trailing 30 days and 12 months
        eng_hist = engagement[engagement[customer_id_column] == cid].copy()
        eng_hist = eng_hist[eng_hist["event_date"] <= prediction_date]
        eng_30d = eng_hist[eng_hist["event_date"] >= prediction_date - pd.DateOffset(days=30)]
        eng_12m = eng_hist[eng_hist["event_date"] >= prediction_date - pd.DateOffset(months=12)]

        login_frequency_30d = float(eng_30d["login"].sum()) if "login" in eng_30d.columns else 0.0
        feature_usage_score = float(eng_30d["feature_usage"].mean()) if "feature_usage" in eng_30d.columns and not eng_30d.empty else 0.0
        usage_recency_days = None
        if "event_date" in eng_hist.columns and not eng_hist.empty:
            last_event = pd.to_datetime(eng_hist["event_date"]).max()
            usage_recency_days = float((prediction_date - last_event).days)
        support_interactions_12m = int(eng_12m["support_interactions"].sum()) if "support_interactions" in eng_12m.columns else 0
        onboarding_completed = bool(cust.get("onboarding_completed", False))

        rows.append(
            FeatureSnapshot(
                customer_id=str(cid),
                prediction_date=prediction_date,
                horizon_months=horizon_months,
                tenure_months=tenure_months,
                plan=cust.get("plan"),
                billing_interval=cust.get("billing_interval"),
                plan_changes=plan_changes,
                pause_resume_count=pause_resume_count,
                days_since_last_renewal=last_renewal,
                is_active=is_active,
                recurring_amount=recurring_amount,
                successful_payments_12m=successful_payments_12m,
                failed_payments_12m=failed_payments_12m,
                total_discounts_12m=total_discounts_12m,
                total_refunds_12m=total_refunds_12m,
                add_on_revenue_12m=add_on_revenue_12m,
                payment_frequency=payment_frequency,
                login_frequency_30d=login_frequency_30d,
                feature_usage_score=feature_usage_score,
                usage_recency_days=usage_recency_days,
                support_interactions_12m=support_interactions_12m,
                onboarding_completed=onboarding_completed,
                acquisition_channel=cust.get("acquisition_channel"),
                campaign=cust.get("campaign"),
                customer_segment=cust.get("customer_segment"),
                region=cust.get("region"),
            )
        )

    df = pd.DataFrame([r.to_dict() for r in rows])
    # Keep prediction_date as a Timestamp (not ISO string) so chronological
    # splits can parse it reliably (PRD FR-2).
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df


def _eligible_customers(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    prediction_date: pd.Timestamp,
    customer_id_column: str,
) -> pd.DataFrame:
    """Return customers with an active subscription as of prediction_date."""
    # A customer is eligible if they have a subscription started on/before the
    # prediction date and not cancelled before it.
    sub = subscriptions.copy()
    sub["start_date"] = pd.to_datetime(sub["start_date"])
    sub["cancellation_date"] = pd.to_datetime(sub.get("cancellation_date", pd.NaT), errors="coerce")

    active_subs = sub[
        (sub["start_date"] <= prediction_date)
        & (sub["cancellation_date"].isna() | (sub["cancellation_date"] > prediction_date))
    ]

    eligible_ids = set(active_subs[customer_id_column].unique())
    return customers[customers[customer_id_column].isin(eligible_ids)].copy()
