from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt

from .utils import dataframe_for_columns, ensure_columns_exist, require_categorical_like, require_numeric_columns


def render_boxplot(
    df: pd.DataFrame,
    columns: list[str],
    x: str | None,
    y: str | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    if x and y:
        ensure_columns_exist(df, [x, y])
        data = dataframe_for_columns(df, [x, y])
        require_categorical_like(data, x)
        require_numeric_columns(data, [y])
        groups = []
        labels = []
        for label, chunk in data.groupby(x):
            groups.append(chunk[y].to_numpy())
            labels.append(str(label))
        if not groups:
            raise HTTPException(status_code=400, detail="No data available for boxplot.")
        ax.boxplot(groups, labels=labels, patch_artist=True)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Boxplot of {y} by {x}")
        ax.tick_params(axis="x", labelrotation=35)
        return fig

    data = dataframe_for_columns(df, columns)
    require_numeric_columns(data, columns)
    arrays = [data[col].to_numpy() for col in columns]
    ax.boxplot(arrays, labels=columns, patch_artist=True)
    ax.set_title("Boxplot")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", labelrotation=25)
    return fig
