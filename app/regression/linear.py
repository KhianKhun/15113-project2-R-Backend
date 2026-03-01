from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def fit_linear(
    frame: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    params: dict,
) -> tuple[LinearRegression, dict[str, float], np.ndarray, dict[str, float]]:
    x = frame[feature_cols].to_numpy()
    y = frame[y_col].to_numpy()

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    pred = model.predict(x)
    metrics = { "r2": float(r2_score(y, pred)),
                "mse": float(mean_squared_error(y, pred)) }

    x_df = frame[feature_cols]
    ols_x = sm.add_constant(x_df, has_constant="add")
    ols_fit = sm.OLS(y, ols_x).fit()
    p_values = {
        col: float(ols_fit.pvalues.get(col, np.nan))
        for col in feature_cols
    }
    return model, metrics, y.astype(float), p_values
