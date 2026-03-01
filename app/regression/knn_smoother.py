from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


def fit_knn_smoother(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[KNeighborsRegressor, dict[str, float], np.ndarray]:
    x = frame[feature_cols].to_numpy()
    y = frame[y_col].to_numpy()

    n_neighbors = int(params.get("n_neighbors", 15))
    n_neighbors = max(1, min(n_neighbors, len(frame)))

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    model.fit(x, y)
    pred = model.predict(x)

    metrics = { "r2": float(r2_score(y, pred)),
                "mse": float(mean_squared_error(y, pred)) }
    return model, metrics, y.astype(float)
