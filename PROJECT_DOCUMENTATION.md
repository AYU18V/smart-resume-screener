# Computational Workforce Intelligence & Skill Demand Forecasting Framework

## Current Architecture

```text
skill-demand-forecasting/
  backend/
    api/main.py
    analysis/skill_analysis.py
    preprocessing/
      build_skill_forecast_dataset.py
      enterprise_workforce_preprocessing.py
    services/
      config.py
      forecast_service.py
      model_loader.py
      recommendation_service.py
      resume_service.py
      skill_extraction_service.py
    models/
      lstm_model.py
      forecasting/
        modelnew.h5
        scaler.pkl
        metadata.json
        metrics.json
        feature_importance.json
        confidence_calibration.json
        ensemble_weights.json
        forecast_stability.json
        training_history.json
        prediction_examples.json
        loss_graph.png
        actual_vs_predicted.png
        drift_analysis.png
        confidence_band_graph.png
        trend_smoothness_graph.png
        smoothness_analysis.png
      huggingface/
        dslim-bert-base-NER/
        all-MiniLM-L6-v2/
    data/
      raw/
      processed/
        enterprise_workforce_dataset.csv
        clean_workforce_forecasting_dataset.csv
        final_skill_forecast_dataset.csv
  frontend/skill-dashboard/
    src/components/
    src/pages/
    src/services/apiService.js
```

The frontend is React/Vite. The backend is FastAPI. Forecasting, NLP, recommendation, resume analysis, and model loading are isolated in reusable backend services.

## Datasets Used

The enterprise preprocessing pipeline uses all four raw datasets together:

```text
backend/data/raw/jobs_dataset.csv
backend/data/raw/future_jobs_dataset.csv
backend/data/raw/job_trend.csv
backend/data/raw/indian-job-market-dataset-2025.xlsx
```

Detected signal groups:

- Dates: `date_posted`, `posting_date`, `year/month`, `jobUploaded`
- Skills: `skills_list`, `skills_required`, `tagsAndSkills`, `python_pct`, `aws_pct`
- Titles: `job_title`, `job_role`, `title`
- Market fields: company, location, salary, experience, remote flag, economic indicators

## Unified Preprocessing

Main script:

```bash
python backend/preprocessing/build_skill_forecast_dataset.py
```

Enterprise implementation:

```text
backend/preprocessing/enterprise_workforce_preprocessing.py
```

It deep-cleans the raw data, merges skill synonyms, normalizes salary/location/experience/date fields, detects emerging skills, and generates multivariate monthly workforce features.

Primary output:

```text
backend/data/processed/enterprise_workforce_dataset.csv
```

Compatibility output:

```text
backend/data/processed/final_skill_forecast_dataset.csv
```

Current enterprise output contains 2,417 monthly skill rows across 1,437 skills.

## Forecasting Model

Training script:

```bash
python backend/models/lstm_model.py
```

Architecture:

- Small-data Bidirectional LSTM(64)
- GRU(32)
- Dense(16) regression head
- Batch normalization
- Dropout and recurrent dropout
- AdamW optimizer with gradient clipping
- Time-series train/validation/test split
- Early stopping and ReduceLROnPlateau
- Validation-weighted ensemble with trend/seasonal and ARIMA-style forecasts

The current dataset has enough usable continuity for only a small/medium time-series model: 20 forecastable skills, 18-month windows, 372 training sequences, 80 validation sequences, and 80 test sequences. The trainer therefore avoids transformer blocks, multi-head attention, and deep residual stacks.

The raw datasets mix incompatible demand scales, so the forecasting target is now a per-skill `0-100` workforce demand index. This prevents national trend counts from overwhelming job-record counts and makes the dashboard predictions more stable.

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

The saved ensemble currently gives the neural model `0%` serving weight and the statistical trend components `100%` combined weight because the validation/test comparison showed the neural model overfitting the small dataset. The neural model remains saved for retraining and future use, but the production forecast path follows validation-weighted model selection.
- Weighted ensemble serving with deep learning, trend/seasonal, and ARIMA-style forecasts

Input features include job openings, salary, remote percentage, Python/AWS trend percentages, GDP growth, VC funding, emerging flag, experience level, location demand, industry growth, semantic skill score, hiring velocity, demand momentum, volatility, skill growth, technology adoption, demand acceleration, trend slope, and seasonality.

Saved model artifacts:

```text
backend/models/forecasting/modelnew.h5
backend/models/forecasting/scaler.pkl
backend/models/forecasting/metadata.json
backend/models/forecasting/metrics.json
backend/models/forecasting/feature_importance.json
backend/models/forecasting/confidence_calibration.json
backend/models/forecasting/ensemble_weights.json
backend/models/forecasting/forecast_stability.json
```

Saved graphs:

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

## Backend APIs

Base URL:

```text
http://127.0.0.1:8000
```

Endpoints:

```text
GET  /
GET  /skills
GET  /top-jobs
GET  /skill-trends
GET  /live-jobs
GET  /forecast-skills?months=6&skills=Python,AWS
GET  /workforce-analytics
GET  /dataset-insights
GET  /model-explainability
POST /predict-demand
POST /extract-skills
POST /analyze-resume
POST /match-skills
```

Forecast responses include:

```text
predicted_jobs_count
deep_learning_forecast
prophet_forecast
arima_forecast
ensemble_forecast
confidence
confidence_band
prediction_reason
trend_strength
growth_direction
forecast_stability
top_influencing_features
market_signal_summary
economic_signal_summary
```

## Resume Intelligence

The resume analyzer uses:

- Hugging Face `dslim/bert-base-NER`
- Sentence Transformers `all-MiniLM-L6-v2`
- Rule-backed technical skill extraction
- Market-aware recommendations from the enterprise workforce dataset

Returned scores:

- ATS score
- Resume strength score
- Future readiness score
- Career growth score
- Industry alignment score
- Market competitiveness score

## Frontend Dashboard

The React dashboard displays:

- Workforce forecasting
- Historical vs predicted demand
- Confidence bands and forecast stability
- Explainable AI reasoning
- Dataset insights
- Model architecture and metrics
- Top future skills
- Resume intelligence scores
- Career recommendations
- Market-aware skill matching

Key files:

```text
frontend/skill-dashboard/src/components/ForecastPanel.jsx
frontend/skill-dashboard/src/components/ForecastExplainabilityPanel.jsx
frontend/skill-dashboard/src/components/ResumeAnalyzer.jsx
frontend/skill-dashboard/src/services/apiService.js
```

## Environment

Root `.env`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
MODEL_DIR=backend/models
HF_MODEL_DIR=backend/models/huggingface
FORECAST_MODEL_PATH=backend/models/forecasting/modelnew.h5
```

Optional production variable:

```text
BACKEND_CORS_ORIGINS=https://your-frontend-domain.com
```

## Setup

Install backend dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

Run preprocessing:

```bash
python backend/preprocessing/build_skill_forecast_dataset.py
```

Train the enterprise model:

```bash
python backend/models/lstm_model.py
```

Run backend:

```bash
cd backend
uvicorn api.main:app --reload
```

Run frontend:

```bash
cd frontend/skill-dashboard
npm install
npm run dev
```

## Validation

Commands used:

```bash
python -m py_compile backend/services/forecast_service.py backend/services/recommendation_service.py backend/services/resume_service.py backend/api/main.py backend/preprocessing/enterprise_workforce_preprocessing.py backend/models/lstm_model.py
python -m pip check
npm run build -- --outDir "$env:TEMP\skill-dashboard-build-check" --emptyOutDir
```

FastAPI smoke tests passed for:

```text
/dataset-insights
/model-explainability
/forecast-skills
/predict-demand
/match-skills
```

## Deployment Notes

- Persist `backend/models/forecasting` and `backend/models/huggingface`.
- Build the frontend with `VITE_API_BASE_URL` pointing to the deployed FastAPI backend.
- Set `BACKEND_CORS_ORIGINS` to the deployed frontend origin.
- Run preprocessing and training before release, or ship the generated artifacts.
- Use a production ASGI server for FastAPI.
- Do not commit secrets in `.env`.
