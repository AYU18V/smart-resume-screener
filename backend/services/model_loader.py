from functools import lru_cache

from services.config import (
    FORECAST_METADATA_PATH,
    FORECAST_MODEL_PATH,
    FORECAST_SCALER_PATH,
    NER_MODEL_NAME,
    NER_MODEL_PATH,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_MODEL_PATH,
)


def _is_local_model_ready(path):
    return path.exists() and any(path.iterdir())


@lru_cache(maxsize=1)
def load_ner_pipeline():
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    model_source = str(NER_MODEL_PATH) if _is_local_model_ready(NER_MODEL_PATH) else NER_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForTokenClassification.from_pretrained(model_source)

    if not _is_local_model_ready(NER_MODEL_PATH):
        NER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(NER_MODEL_PATH)
        model.save_pretrained(NER_MODEL_PATH)

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


@lru_cache(maxsize=1)
def load_sentence_model():
    from sentence_transformers import SentenceTransformer

    model_source = str(SEMANTIC_MODEL_PATH) if _is_local_model_ready(SEMANTIC_MODEL_PATH) else SEMANTIC_MODEL_NAME
    model = SentenceTransformer(model_source)

    if not _is_local_model_ready(SEMANTIC_MODEL_PATH):
        SEMANTIC_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        model.save(str(SEMANTIC_MODEL_PATH))

    return model


@lru_cache(maxsize=1)
def load_forecasting_model():
    if not FORECAST_MODEL_PATH.exists():
        return None

    from tensorflow.keras.models import load_model

    return load_model(FORECAST_MODEL_PATH, compile=False)


@lru_cache(maxsize=1)
def load_forecasting_artifacts():
    if not FORECAST_METADATA_PATH.exists() or not FORECAST_SCALER_PATH.exists():
        return None, None

    import json
    import joblib

    metadata = json.loads(FORECAST_METADATA_PATH.read_text())
    scaler = joblib.load(FORECAST_SCALER_PATH)
    return metadata, scaler
