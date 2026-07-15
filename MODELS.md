# Model Artifact Guide

Trained model files and local model caches are intentionally excluded from GitHub.

## Excluded Artifacts

- `backend/models/forecasting/modelnew.h5`
- `backend/models/forecasting/scaler.pkl`
- `backend/models/forecasting/*.json`
- `backend/models/forecasting/*.png`
- `backend/models/huggingface/`
- `*.h5`, `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, `*.safetensors`, `*.pkl`, `*.joblib`

These files are generated outputs or downloaded caches. They can be large and can cause slow clones or GitHub push failures.

## Regenerate Forecasting Artifacts

From the backend directory:

```bash
python models/lstm_model.py
```

This recreates the forecasting model, scaler, metadata, metrics, graphs, and explainability artifacts under `backend/models/forecasting/`.

## Hugging Face Models

The backend loads these model names when local caches are absent:

- `dslim/bert-base-NER`
- `sentence-transformers/all-MiniLM-L6-v2`

On first use, the application may download and cache them under `backend/models/huggingface/`. Keep that folder local and untracked.
