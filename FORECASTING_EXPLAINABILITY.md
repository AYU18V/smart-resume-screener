# Forecasting Explainability Notes

## Model Purpose

The forecasting system predicts future monthly workforce demand for technical skills. It is designed to be stable, explainable, and market-aware rather than a simple one-column LSTM.

## Dataset Used

Primary dataset:

```text
backend/data/processed/enterprise_workforce_dataset.csv
```

The dataset is built from:

```text
jobs_dataset.csv
future_jobs_dataset.csv
job_trend.csv
indian-job-market-dataset-2025.xlsx
```

Each row represents one skill in one month with demand, market, salary, remote-work, economic, momentum, volatility, and seasonality features. The training pipeline also writes:

```text
backend/data/processed/clean_workforce_forecasting_dataset.csv
```

This clean modeling dataset converts incompatible raw job/trend counts into a per-skill `0-100` workforce demand index, fills true internal time gaps, suppresses source-scale spikes, and removes sparse skills before sequence generation.

## Input Features

The model uses a rolling 18-month multivariate window. Main features include:

```text
jobs_count
job_openings
avg_salary_usd
remote_pct
python_pct
aws_pct
gdp_growth
vc_funding_tech_bn
emerging_flag
experience_level
location_demand
industry_growth
semantic_skill_score
hiring_velocity
demand_momentum
trend_volatility
skill_growth_rate
technology_adoption_rate
demand_acceleration
trend_slope_6
month_sin
month_cos
```

## Architecture

Saved model:

```text
backend/models/forecasting/modelnew.h5
```

Architecture:

```text
Bidirectional LSTM(64)
Dropout(0.3)
GRU(32)
Batch normalization
Dense(16)
AdamW optimizer with gradient clipping
```

The model is intentionally compact because the usable time-series sample count is small/medium: 372 training sequences, 80 validation sequences, and 80 test sequences. The previous heavy model was not appropriate for this dataset size, so the current version avoids transformer blocks, multi-head attention, and deep residual stacks.

The model is trained with time-ordered splits, early stopping, validation monitoring, learning-rate reduction, separate feature and target scalers, and `log1p` target normalization.

## Forecasting Strategy

API flow:

```text
GET /forecast-skills
  -> forecast_service.forecast_skills()
  -> load modelnew.h5, metadata.json, scaler.pkl
  -> load latest 18-month feature sequence per skill
  -> run deep forecast
  -> blend with trend/seasonal and ARIMA-style signals using validation-selected weights
  -> apply drift smoothing
  -> compute confidence band and explanation metadata
```

The serving ensemble uses:

```text
deep_learning: 0%
trend/seasonal signal: 55%
ARIMA-style signal: 45%
```

The neural model is still trained and saved, but current validation shows it overfits this small dataset. The ensemble therefore assigns it zero serving weight until future data makes its holdout performance competitive.

These weights are saved in:

```text
backend/models/forecasting/ensemble_weights.json
```

## Stability Controls

The forecast service applies:

- Recursive feature updates
- Drift correction
- Volatility-aware smoothing
- Confidence calibration
- Confidence bands
- Trend continuity preservation

This reduces random spikes and keeps predictions closer to realistic historical market behavior.

Latest holdout metrics:

```text
MAE: 6.29 demand-index points
RMSE: 7.05
MAPE/SMAPE: 7.26%
Weighted MAPE: 6.58%
R2: 0.706
Forecast stability: 99.65
Prediction smoothness: 99.62
Confidence calibration: 91.71
```

## Explainability Fields

Every prediction explanation can include:

```text
skill
forecast
confidence
confidence_level
confidence_band
forecast_stability
trend_strength
growth_direction
growth_percentage
prediction_reason
top_influencing_features
historical_pattern
economic_signal_summary
market_signal_summary
volatility_assessment
seasonality_pattern
```

Confidence is a practical reliability score based on history coverage, volatility, residual calibration, model error, and forecast smoothness. It is not a guaranteed probability.

## Why A Prediction Happens

The dashboard explains predictions in plain language. Example:

```text
Python has positive demand momentum and the hybrid model found the recent workforce signal strong enough to continue growth while smoothing sudden spikes.
```

The explanation also shows:

- recent historical demand
- growth direction
- demand volatility
- confidence range
- top influencing features
- market and economic signals

## Metrics And Artifacts

Saved evaluation files:

```text
metrics.json
training_history.json
feature_importance.json
confidence_calibration.json
prediction_examples.json
forecast_stability.json
```

Saved visual outputs:

```text
loss_graph.png
training_loss_graph.png
validation_loss_graph.png
actual_vs_predicted.png
forecast_stability.png
drift_analysis.png
confidence_band_graph.png
trend_smoothness_graph.png
```

## UI Flow

```text
ForecastPanel.jsx
  -> fetchForecast()
  -> predictDemand()
  -> renders selected skill trend, stability, confidence, and reasoning

ForecastExplainabilityPanel.jsx
  -> explains dataset, model architecture, metrics, features, confidence bands, and decision logic
```
