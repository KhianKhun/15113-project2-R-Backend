from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..models import RegressionType


@dataclass
class StoredPrediction:
    feature_values: dict[str, float]
    prediction_value: float
    predicted_class: str | None = None
    positive_class_probability: float | None = None


@dataclass
class StoredRegressionModel:
    dataset_id: str
    model_type: RegressionType
    estimator: Any
    y_col: str
    feature_cols: list[str]
    plot_x: str
    task: Literal["regression", "classification"]
    metrics: dict[str, float] = field(default_factory=dict)
    x_for_plot: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_for_plot: np.ndarray = field(default_factory=lambda: np.array([]))
    class_labels: list[str] | None = None
    latest_prediction: StoredPrediction | None = None
