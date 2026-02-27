from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt

from ..models import RegressionFitRequest, RegressionFitResponse
from ..plotting.utils import fig_to_png_bytes
from .additive_model import fit_additive_model
from .kernel_smoother import fit_kernel_smoother
from .knn_smoother import fit_knn_smoother
from .linear import fit_linear
from .logistic_regression import fit_logistic_regression
from .polynomial import fit_polynomial
from .spline_smoother import fit_spline_smoother
from .storage import get_model, put_model
from .types import StoredRegressionModel
from .utils import pick_plot_x, prepare_training_frame, require_numeric_columns

_SINGLE_X_ONLY = {"polynomial", "kernel_smoother", "knn_smoother", "spline_smoother"}


def fit_and_store_regression(dataset_id: str, df: pd.DataFrame, payload: RegressionFitRequest) -> RegressionFitResponse:
    frame = prepare_training_frame(df, payload.y, payload.x)
    require_numeric_columns(frame, payload.x)

    if payload.model_type in _SINGLE_X_ONLY and len(payload.x) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{payload.model_type}' requires exactly one x feature.",
        )

    if payload.model_type != "logistic_regression":
        require_numeric_columns(frame, [payload.y])

    plot_x = pick_plot_x(payload.x, payload.plot_x)
    feature_frame = frame[payload.x].copy()
    classes: list[str] | None = None
    task = "regression"

    try:
        if payload.model_type == "linear":
            estimator, metrics, y_for_plot, p_values = fit_linear(
                frame, payload.y, payload.x, payload.params
            )
            line_p_value = p_values.get(plot_x)
            if line_p_value is not None and not np.isnan(line_p_value):
                metrics["line_p_value"] = float(line_p_value)
        elif payload.model_type == "polynomial":
            estimator, metrics, y_for_plot = fit_polynomial(frame, payload.y, payload.x, payload.params)
        elif payload.model_type == "kernel_smoother":
            estimator, metrics, y_for_plot = fit_kernel_smoother(frame, payload.y, payload.x, payload.params)
        elif payload.model_type == "knn_smoother":
            estimator, metrics, y_for_plot = fit_knn_smoother(frame, payload.y, payload.x, payload.params)
        elif payload.model_type == "spline_smoother":
            estimator, metrics, y_for_plot = fit_spline_smoother(frame, payload.y, payload.x, payload.params)
        elif payload.model_type == "additive_model":
            estimator, metrics, y_for_plot = fit_additive_model(frame, payload.y, payload.x, payload.params)
        elif payload.model_type == "logistic_regression":
            estimator, metrics, y_for_plot, classes = fit_logistic_regression(
                frame,
                payload.y,
                payload.x,
                payload.params,
            )
            task = "classification"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {payload.model_type}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fitting failed: {exc}") from exc

    stored = StoredRegressionModel(
        dataset_id=dataset_id,
        model_type=payload.model_type,
        estimator=estimator,
        y_col=payload.y,
        feature_cols=payload.x,
        plot_x=plot_x,
        task=task,
        metrics=metrics,
        x_for_plot=feature_frame,
        y_for_plot=y_for_plot,
        class_labels=classes,
    )
    model_id = put_model(stored)

    return RegressionFitResponse(
        model_id=model_id,
        dataset_id=dataset_id,
        model_type=payload.model_type,
        y=payload.y,
        x=payload.x,
        plot_x=plot_x,
        metrics=metrics,
        message="Regression model fitted and stored. Click 'Draw Fitted Curve' to render the fitted curve.",
    )


def render_stored_curve_png(model_id: str) -> bytes:
    try:
        model = get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    x_values = model.x_for_plot[model.plot_x].to_numpy(dtype=float)
    if x_values.size == 0:
        raise HTTPException(status_code=400, detail="Stored model has no training rows.")

    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    grid = np.linspace(x_min, x_max, 220)
    x_grid = _build_feature_grid(model, grid)
    try:
        y_curve = _predict_curve(model, grid, x_grid)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to render fitted curve: {exc}") from exc

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x_values, model.y_for_plot, s=16, alpha=0.32, label="observed")
        ax.plot(grid, y_curve, color="#d62728", linewidth=2.2, label="fitted curve")
        ax.set_xlabel(model.plot_x)
        if model.task == "classification":
            positive_label = model.class_labels[1] if model.class_labels and len(model.class_labels) > 1 else "1"
            ax.set_ylabel(f"P({model.y_col}={positive_label})")
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel(model.y_col)
        ax.set_title(f"{model.model_type} fit curve")
        if model.latest_prediction and model.plot_x in model.latest_prediction.feature_values:
            x_pred = float(model.latest_prediction.feature_values[model.plot_x])
            y_pred = float(model.latest_prediction.prediction_value)
            ax.scatter([x_pred], [y_pred], color="green", s=56, zorder=8, label="prediction")
            ax.axvline(x_pred, color="green", linestyle="--", linewidth=1.2, alpha=0.85)
            ax.axhline(y_pred, color="green", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.legend()
        return fig_to_png_bytes(fig)
    finally:
        if fig is not None:
            plt.close(fig)


def _build_feature_grid(model: StoredRegressionModel, grid: np.ndarray) -> np.ndarray:
    train_matrix = model.x_for_plot[model.feature_cols].to_numpy(dtype=float)
    if train_matrix.ndim != 2 or train_matrix.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Stored model features are invalid.")

    defaults = np.nanmedian(train_matrix, axis=0)
    x_grid = np.tile(defaults, (len(grid), 1))
    plot_idx = model.feature_cols.index(model.plot_x)
    x_grid[:, plot_idx] = grid
    return x_grid


def _predict_curve(model: StoredRegressionModel, grid: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if model.model_type == "spline_smoother":
        return np.asarray(model.estimator(grid), dtype=float)
    if model.model_type == "logistic_regression":
        probs = model.estimator.predict_proba(x_grid)
        return probs[:, 1].astype(float)
    return np.asarray(model.estimator.predict(x_grid), dtype=float)
