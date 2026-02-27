from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


def fit_additive_model(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    x = frame[feature_cols].to_numpy()
    y = frame[y_col].to_numpy()

    n_knots = int(params.get("n_knots", 6))
    n_knots = max(3, min(n_knots, 20))
    spline_degree = int(params.get("spline_degree", 3))
    spline_degree = max(1, min(spline_degree, 5))
    ridge_alpha = float(params.get("ridge_alpha", 1.0))
    ridge_alpha = max(ridge_alpha, 1e-8)

    model = Pipeline(
        [
            (
                "splines",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=spline_degree,
                    include_bias=False,
                ),
            ),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    model.fit(x, y)
    pred = model.predict(x)

    metrics = {
        "r2": float(r2_score(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
    }
    return model, metrics, y.astype(float)
