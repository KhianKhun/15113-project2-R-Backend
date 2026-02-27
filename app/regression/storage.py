from __future__ import annotations

import uuid
from threading import Lock

from .types import StoredPrediction, StoredRegressionModel

_MODELS: dict[str, StoredRegressionModel] = {}
_MODEL_LOCK = Lock()


def put_model(model: StoredRegressionModel) -> str:
    model_id = str(uuid.uuid4())
    with _MODEL_LOCK:
        _MODELS[model_id] = model
    return model_id


def get_model(model_id: str) -> StoredRegressionModel:
    with _MODEL_LOCK:
        model = _MODELS.get(model_id)
    if model is None:
        raise KeyError(f"Model '{model_id}' not found.")
    return model


def set_model_prediction(model_id: str, prediction: StoredPrediction) -> None:
    with _MODEL_LOCK:
        model = _MODELS.get(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' not found.")
        model.latest_prediction = prediction
