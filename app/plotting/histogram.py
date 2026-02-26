from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt

from .utils import dataframe_for_columns, require_numeric_columns


def render_histogram(df: pd.DataFrame, columns: list[str], params: dict) -> plt.Figure:
    data = dataframe_for_columns(df, columns)
    require_numeric_columns(data, columns)

    alpha = float(params.get("alpha", 0.45))
    alpha = min(max(alpha, 0.05), 1.0)
    binwidth = params.get("binwidth")

    fig, ax = plt.subplots(figsize=(9, 5))

    if binwidth is not None:
        try:
            width = float(binwidth)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="binwidth must be a positive number.") from exc
        if width <= 0:
            raise HTTPException(status_code=400, detail="binwidth must be a positive number.")
        all_values = np.concatenate([data[col].to_numpy() for col in columns])
        low, high = float(np.min(all_values)), float(np.max(all_values))
        if low == high:
            bins = 10
        else:
            bins = np.arange(low, high + width, width)
    else:
        bins = int(params.get("bins", 30))
        bins = max(5, min(bins, 200))

    for col in columns:
        ax.hist(data[col].to_numpy(), bins=bins, alpha=alpha, label=col, edgecolor="white")

    ax.set_title("Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    return fig
