from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder


def fit_logistic_regression(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[LogisticRegression, dict[str, float], np.ndarray, list[str]]:
    x = frame[feature_cols].to_numpy()
    y_raw = frame[y_col].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = [str(name) for name in encoder.classes_.tolist()]
    if len(classes) != 2:
        raise HTTPException(
            status_code=400,
            detail="Logistic regression currently supports binary target only (2 unique classes).",
        )

    c_value = float(params.get("c", 1.0))
    c_value = max(c_value, 1e-8)
    max_iter = int(params.get("max_iter", 1000))
    max_iter = max(max_iter, 100)

    model = LogisticRegression(C=c_value, max_iter=max_iter)
    model.fit(x, y)

    pred_class = model.predict(x)
    pred_prob = model.predict_proba(x)
    metrics = { "accuracy": float(accuracy_score(y, pred_class)),
                "log_loss": float(log_loss(y, pred_prob)) }
    return model, metrics, y.astype(float), classes
