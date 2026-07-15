# Repository Cleanup Report

## Size Summary

- Original archive: `skill-demand-forecasting.zip` was 1,386,970,370 bytes, about 1.29 GB.
- Extracted working copy before cleanup: 42.76 MB across 102 files. This excluded the virtual environment, `node_modules`, Hugging Face caches, and the `.h5` model during extraction.
- Repository after cleanup: 2.70 MB across 69 files.

## Files Removed or Excluded

| Path or pattern | Action | Reason |
| --- | --- | --- |
| `.env` | Removed and ignored | Local environment file; may contain machine-specific paths or secrets. |
| `frontend/skill-dashboard/.env` | Removed and ignored | Local frontend environment file; replaced by `.env.example`. |
| `backend/**/__pycache__/` | Removed and ignored | Python bytecode cache; regenerated automatically. |
| `*.pyc` | Removed and ignored | Python compiled bytecode; not source code. |
| `frontend/skill-dashboard/dist/` | Removed and ignored | Vite production build output; regenerated with `npm run build`. |
| `backend/models/forecasting/*` except `.gitkeep` | Removed and ignored | Generated model weights, scaler, metrics, graphs, and metadata. |
| `backend/models/huggingface/` | Excluded during extraction and ignored | Downloaded model cache; can be re-downloaded by the app. |
| `backend/venv/` | Excluded during extraction and ignored | Local Python virtual environment; recreated with `python -m venv`. |
| `frontend/skill-dashboard/node_modules/` | Excluded during extraction and ignored | Node dependency folder; recreated with `npm install`. |
| `backend/data/raw/indian-job-market-dataset-2025.xlsx` | Removed and ignored | Bulky raw dataset, 30.24 MB. Keep outside Git. |
| `backend/data/processed/unified_workforce_dataset.csv` | Removed and ignored | Generated processed dataset, 8.3 MB. Rebuild locally when needed. |
| Archives such as `*.zip`, `*.rar`, `*.7z` | Ignored | Prevents committing exported project bundles. |
| Model weights such as `*.h5`, `*.pt`, `*.onnx`, `*.ckpt`, `*.safetensors` | Ignored | Prevents oversized ML artifact commits. |

## Large Files Detected

Before cleanup, the largest files in the extracted copy were:

- `backend/data/raw/indian-job-market-dataset-2025.xlsx` - 30.24 MB
- `backend/data/processed/unified_workforce_dataset.csv` - 8.3 MB
- `backend/data/raw/future_jobs_dataset.csv` - 0.95 MB
- `backend/data/processed/enterprise_workforce_dataset.csv` - 0.61 MB
- `backend/models/forecasting/metadata.json` - 0.66 MB

After cleanup, the largest remaining files are small demo/reference datasets:

- `backend/data/raw/future_jobs_dataset.csv` - 0.95 MB
- `backend/data/processed/enterprise_workforce_dataset.csv` - 0.61 MB
- `backend/data/processed/clean_workforce_forecasting_dataset.csv` - 0.33 MB
- `backend/data/processed/jobs_with_skills.csv` - 0.19 MB
- `backend/data/raw/jobs_dataset.csv` - 0.14 MB

## Kept

- Backend source code under `backend/`
- Frontend source code under `frontend/skill-dashboard/src/`
- Dependency manifests: `backend/requirements.txt`, `frontend/skill-dashboard/package.json`, and `frontend/skill-dashboard/package-lock.json`
- Vite, Tailwind, PostCSS, and ESLint configuration
- Small datasets required for demo functionality
- Existing project documentation
- `.gitignore`, `.env.example`, `DATASET.md`, `MODELS.md`, and root `README.md`

## Git Notes

This extracted working copy is not currently initialized as a Git repository. If you initialize Git now, ignored files will not be added.

If you run this cleanup on an already initialized repository where ignored files were previously tracked, use:

```bash
git rm -r --cached backend/venv frontend/skill-dashboard/node_modules frontend/skill-dashboard/dist backend/models/huggingface backend/models/forecasting
git rm --cached .env frontend/skill-dashboard/.env backend/data/raw/indian-job-market-dataset-2025.xlsx backend/data/processed/unified_workforce_dataset.csv
```

Then review with:

```bash
git status --short
```

## Recommendations

- Add screenshots or a short GIF to the README after the UI is polished.
- Add a `LICENSE` file before publishing if this is intended to be open source.
- Pin Python dependency versions in `backend/requirements.txt` for reproducible installs.
- Add a small smoke test for the FastAPI app and one frontend build check in GitHub Actions.
- Consider Docker or `docker-compose.yml` to make recruiter setup easier.
