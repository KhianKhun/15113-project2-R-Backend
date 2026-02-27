from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def fit_polynomial(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    x = frame[feature_cols].to_numpy()
    y = frame[y_col].to_numpy()

    degree = int(params.get("degree", 2))
    degree = min(max(degree, 1), 10)

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(x, y)
    pred = model.predict(x)

    metrics = {
        "r2": float(r2_score(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
    }
    return model, metrics, y.astype(float)
