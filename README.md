# Skill Demand Forecasting

Computational workforce intelligence and skill demand forecasting project with a FastAPI backend, Streamlit dashboard utilities, and a Vite React frontend.

## Repository Structure

```text
backend/
  api/                  FastAPI entry point
  analysis/             Dataset and skill analysis scripts
  dashboard/            Streamlit dashboard
  data/                 Small demo/reference CSV datasets
  models/               Training code and empty artifact folders
  nlp/                  Skill extraction code
  preprocessing/        Dataset preparation pipelines
  services/             Backend business logic
frontend/skill-dashboard/
  src/                  React application source
  package.json          Frontend dependencies and scripts
```

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cd frontend/skill-dashboard
npm install
```

Copy `.env.example` to `.env` and fill values for your local machine. For normal local development:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
BACKEND_CORS_ORIGINS=http://localhost:5173
MODEL_DIR=backend/models
HF_MODEL_DIR=backend/models/huggingface
FORECAST_MODEL_PATH=backend/models/forecasting/modelnew.h5
```

## Run

Backend API:

```bash
cd backend
uvicorn api.main:app --reload
```

Frontend:

```bash
cd frontend/skill-dashboard
npm run dev
```

Optional Streamlit dashboard:

```bash
cd backend
streamlit run dashboard/app.py
```

## Data and Models

This repository keeps only small CSV files needed for development and demos. Large raw datasets, trained model weights, generated model reports, and local Hugging Face caches are intentionally excluded from Git.

See `DATASET.md` for dataset guidance and `MODELS.md` for model regeneration/download guidance.

## GitHub Readiness

The repository has been cleaned for GitHub by excluding virtual environments, `node_modules`, build artifacts, bytecode caches, local `.env` files, large datasets, archives, trained weights, and generated forecasting artifacts. See `REPORT.md` for the cleanup audit.
