from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


def _load_env_file():
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def resolve_path(value, default):
    raw_path = Path(os.getenv(value, default))
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_DIR / raw_path


MODEL_DIR = resolve_path("MODEL_DIR", "backend/models")
HF_MODEL_DIR = resolve_path("HF_MODEL_DIR", "backend/models/huggingface")
FORECAST_MODEL_PATH = resolve_path(
    "FORECAST_MODEL_PATH",
    "backend/models/forecasting/modelnew.h5",
)
RAW_DATA_PATH = BACKEND_DIR / "data" / "raw" / "jobs_dataset.csv"
PROCESSED_DATA_PATH = BACKEND_DIR / "data" / "processed" / "jobs_with_skills.csv"
FINAL_FORECAST_DATASET_PATH = BACKEND_DIR / "data" / "processed" / "final_skill_forecast_dataset.csv"
ENTERPRISE_WORKFORCE_DATASET_PATH = BACKEND_DIR / "data" / "processed" / "enterprise_workforce_dataset.csv"
CLEAN_FORECAST_DATASET_PATH = BACKEND_DIR / "data" / "processed" / "clean_workforce_forecasting_dataset.csv"
FORECAST_MODEL_DIR = MODEL_DIR / "forecasting"
FORECAST_SCALER_PATH = FORECAST_MODEL_DIR / "scaler.pkl"
FORECAST_METADATA_PATH = FORECAST_MODEL_DIR / "metadata.json"
FORECAST_METRICS_PATH = FORECAST_MODEL_DIR / "metrics.json"
FORECAST_LOSS_PLOT_PATH = FORECAST_MODEL_DIR / "loss_graph.png"
FORECAST_FEATURE_IMPORTANCE_PATH = FORECAST_MODEL_DIR / "feature_importance.json"
FORECAST_CONFIDENCE_CALIBRATION_PATH = FORECAST_MODEL_DIR / "confidence_calibration.json"
FORECAST_TRAINING_HISTORY_PATH = FORECAST_MODEL_DIR / "training_history.json"
FORECAST_PREDICTION_EXAMPLES_PATH = FORECAST_MODEL_DIR / "prediction_examples.json"
FORECAST_ENSEMBLE_WEIGHTS_PATH = FORECAST_MODEL_DIR / "ensemble_weights.json"
FORECAST_STABILITY_PATH = FORECAST_MODEL_DIR / "forecast_stability.json"
FORECAST_ACTUAL_VS_PREDICTED_PATH = FORECAST_MODEL_DIR / "actual_vs_predicted.png"
FORECAST_DRIFT_ANALYSIS_PATH = FORECAST_MODEL_DIR / "drift_analysis.png"
FORECAST_CONFIDENCE_BAND_PATH = FORECAST_MODEL_DIR / "confidence_band_graph.png"
FORECAST_TREND_SMOOTHNESS_PATH = FORECAST_MODEL_DIR / "trend_smoothness_graph.png"

NER_MODEL_NAME = "dslim/bert-base-NER"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL_PATH = HF_MODEL_DIR / "dslim-bert-base-NER"
SEMANTIC_MODEL_PATH = HF_MODEL_DIR / "all-MiniLM-L6-v2"

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_TIMEOUT = float(os.getenv("GEMINI_API_TIMEOUT", "30"))
