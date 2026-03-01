from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score


def fit_kernel_smoother(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[KernelRidge, dict[str, float], np.ndarray]:
    x = frame[feature_cols].to_numpy()
    y = frame[y_col].to_numpy()

    bandwidth = float(params.get("bandwidth", 1.0))
    bandwidth = max(bandwidth, 1e-6)
    alpha = float(params.get("alpha", 1e-2))
    alpha = max(alpha, 1e-9)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
    model.fit(x, y)
    pred = model.predict(x)

    metrics = { "r2": float(r2_score(y, pred)),
                "mse": float(mean_squared_error(y, pred)) }
    return model, metrics, y.astype(float)
