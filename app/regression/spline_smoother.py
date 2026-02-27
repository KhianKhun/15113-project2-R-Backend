from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error, r2_score


def fit_spline_smoother(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[UnivariateSpline, dict[str, float], np.ndarray]:
    x = frame[feature_cols[0]].to_numpy()
    y = frame[y_col].to_numpy()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    smooth_factor = params.get("s")
    smooth_factor = None if smooth_factor in (None, "") else float(smooth_factor)

    k = int(params.get("k", 3))
    k = max(1, min(k, 5))
    k = min(k, max(1, len(x_sorted) - 1))

    model = UnivariateSpline(x_sorted, y_sorted, s=smooth_factor, k=k)
    pred = model(x)

    metrics = {
        "r2": float(r2_score(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
    }
    return model, metrics, y.astype(float)
