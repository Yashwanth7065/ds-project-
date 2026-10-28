# CLV Prediction — Customer Lifetime Value

Estimate the total net revenue a subscription customer will generate over a defined horizon (MVP: next 12 months).

Based on the CLV Prediction PRD (v1.0 draft, 28 October 2026).

## Project structure

```
clv_prediction/
├── data/                     # Raw and processed data (point-in-time snapshots)
│   ├── raw/                  # Source-of-truth extracts (subscription, billing, engagement)
│   ├── processed/            # Cleaned, point-in-time feature datasets
│   └── targets/              # Target variable (CLV_12m) per customer/prediction date
├── features/                 # Feature engineering
│   ├── __init__.py
│   ├── pipeline.py           # Point-in-time feature builder
│   └── quality_checks.py     # FR-6: missing, stale, duplicated, inconsistent flags
├── models/                   # Model training & scoring
│   ├── __init__.py
│   ├── baselines.py          # Baseline 1 & 2 (PRD Section 8)
│   ├── clv_model.py          # MVP supervised regression (GBM / regularized linear)
│   └── train.py              # Chronological train/val/test, model selection
├── evaluation/               # Metrics, validation, reporting (PRD Sections 9-10)
│   ├── __init__.py
│   ├── evaluation.py         # MAE, RMSE, WAPE, calibration, business lift
│   ├── validation.py         # Chronological splits, rolling folds
│   └── reporting.py          # PredictionRecord, cohort aggregation, export
├── monitoring/               # FR-7: drift, error, distribution, model version
│   └── monitor.py
├── deployment/               # Batch scoring, retraining procedures
│   └── scoring_job.py
├── docs/                     # PRD, target definition sign-off, runbooks
│   └── PRD_CLV_v1.0.md
├── tests/                    # Unit tests for metrics, splits, features
│   └── __init__.py
├── config.yaml               # Horizon, currency, model version, thresholds
├── requirements.txt
└── run_pipeline.py           # Orchestrates discovery -> baseline -> model -> evaluation
```

## Rollout phases (PRD Section 13)

1. **Discovery** — confirm revenue definition, horizon, data sources, privacy constraints
2. **Data & baseline** — point-in-time dataset, quality checks, simple baselines
3. **Model evaluation** — candidate models, chronological backtests, cohort error review
4. **Pilot** — read-only predictions to a limited group, feedback collection
5. **Production** — automated scoring, monitoring, version tracking, retraining

## Acceptance criteria (PRD Section 15)

- [ ] Signed-off target definition and horizon documented
- [ ] Reproducible training dataset with point-in-time features + quality checks
- [ ] At least one baseline and one ML model on chronological holdout
- [ ] MAE and aggregate revenue error reported overall and by cohort
- [ ] Model meets thresholds, no unexplained critical cohort failures
- [ ] Outputs include horizon, timestamp, model version, currency
- [ ] Monitoring, access control, ownership, retraining documented

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (discovery -> baseline -> model -> evaluation)
python run_pipeline.py

# Run evaluation demo
python -m clv_evaluation.example_usage
```

## Key design principles

- **Point-in-time features only** — no future-data leakage (FR-2)
- **Chronological validation** — train/val/test splits by prediction date (Section 9)
- **Finance-signed target** — net revenue definition agreed before training (Section 7)
- **Interpretable first** — baselines before complex models (Section 8)
- **Forecasts, not guarantees** — outputs labelled with horizon and refresh date (Section 10)
