from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt
from pandas.api import types as ptypes

from .utils import dataframe_for_columns, ensure_columns_exist, require_categorical_like, require_numeric_columns

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]


def render_scatter(
    df: pd.DataFrame,
    x: str | None,
    y: str | None,
    color_by: str | None,
    shape_by: str | None,
    params: dict,
) -> plt.Figure:
    if not x or not y:
        raise HTTPException(status_code=400, detail="Scatter plot requires x and y variables.")

    needed = [x, y]
    if color_by:
        needed.append(color_by)
    if shape_by:
        needed.append(shape_by)

    ensure_columns_exist(df, needed)
    data = dataframe_for_columns(df, needed)
    require_numeric_columns(data, [x, y])

    if shape_by:
        require_categorical_like(data, shape_by)
        shape_levels = data[shape_by].astype(str).unique().tolist()
        if len(shape_levels) > len(MARKERS):
            raise HTTPException(
                status_code=400,
                detail=f"shape_by supports at most {len(MARKERS)} unique categories.",
            )

    alpha = float(params.get("alpha", 0.75))
    alpha = min(max(alpha, 0.05), 1.0)
    size = float(params.get("size", 26))
    size = min(max(size, 4.0), 120.0)

    fig, ax = plt.subplots(figsize=(9, 5))

    if color_by and ptypes.is_numeric_dtype(data[color_by]) and not shape_by:
        scat = ax.scatter(data[x], data[y], c=data[color_by], cmap="viridis", alpha=alpha, s=size)
        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label(color_by)
    elif color_by and not ptypes.is_numeric_dtype(data[color_by]):
        _scatter_by_groups(ax, data, x, y, color_by, shape_by, alpha, size)
    elif shape_by:
        _scatter_by_groups(ax, data, x, y, None, shape_by, alpha, size)
    else:
        ax.scatter(data[x], data[y], alpha=alpha, s=size)

    ax.set_title("Scatter Plot")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig


def _scatter_by_groups(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    color_by: str | None,
    shape_by: str | None,
    alpha: float,
    size: float,
) -> None:
    group_cols = [col for col in [color_by, shape_by] if col]
    grouped = data.groupby(group_cols, dropna=False)
    cmap = plt.get_cmap("tab10")
    for idx, (key, chunk) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        marker = "o"
        if shape_by:
            shape_key = key[-1]
            marker = MARKERS[hash(str(shape_key)) % len(MARKERS)]
        color = cmap(idx % 10)
        label = ", ".join(f"{group_cols[i]}={key[i]}" for i in range(len(group_cols)))
        ax.scatter(chunk[x], chunk[y], alpha=alpha, s=size, marker=marker, color=color, label=label)
    if len(grouped) <= 20:
        ax.legend(fontsize=8)
