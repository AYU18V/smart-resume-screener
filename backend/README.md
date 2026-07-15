# Backend

FastAPI backend for the Computational Workforce Intelligence & Skill Demand Forecasting Framework.

## Workflow

```bash
python preprocessing/build_skill_forecast_dataset.py
python models/lstm_model.py
uvicorn api.main:app --reload
```

## Key Paths

```text
api/main.py
preprocessing/build_skill_forecast_dataset.py
services/
models/lstm_model.py
models/forecasting/modelnew.h5
models/forecasting/scaler.pkl
models/forecasting/metadata.json
models/huggingface/
data/processed/final_skill_forecast_dataset.csv
```
