# Dataset Guide

This project uses multiple workforce and job-market datasets. Small CSV files are kept in the repository so another developer can run the app in demo mode after installing dependencies.

## Kept in Git

- `backend/data/raw/jobs_dataset.csv`
- `backend/data/raw/future_jobs_dataset.csv`
- `backend/data/raw/job_trend.csv`
- `backend/data/processed/jobs_with_skills.csv`
- `backend/data/processed/final_skill_forecast_dataset.csv`
- `backend/data/processed/enterprise_workforce_dataset.csv`
- `backend/data/processed/clean_workforce_forecasting_dataset.csv`

These files are small enough for GitHub and are referenced by the backend fallback and analytics services.

## Excluded from Git

- `backend/data/raw/indian-job-market-dataset-2025.xlsx`
- `backend/data/processed/unified_workforce_dataset.csv`

The Excel file is a bulky raw dataset and the unified CSV is a regenerated processed dataset. Keep them outside Git and place them back at the same paths only when retraining or rebuilding the full processed dataset.

## Rebuilding Processed Data

From the backend directory:

```bash
python preprocessing/build_skill_forecast_dataset.py
python preprocessing/enterprise_workforce_preprocessing.py
python preprocessing/unified_workforce_preprocessing.py
```

If a script expects a missing raw file, download or restore that file locally first, then rerun the preprocessing step.
