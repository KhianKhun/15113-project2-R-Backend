from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt

from .utils import dataframe_for_columns, ensure_columns_exist, require_numeric_columns


def render_lineplot(df: pd.DataFrame, columns: list[str], x: str | None, params: dict) -> plt.Figure:
    if not columns:
        raise HTTPException(status_code=400, detail="Line plot requires at least one y variable.")

    use_columns = list(columns)
    if x:
        use_columns = [x] + use_columns
    ensure_columns_exist(df, use_columns)
    data = dataframe_for_columns(df, use_columns)
    require_numeric_columns(data, columns)
    if x:
        require_numeric_columns(data, [x])

    alpha = float(params.get("alpha", 0.9))
    alpha = min(max(alpha, 0.05), 1.0)
    linewidth = float(params.get("linewidth", 1.8))
    linewidth = min(max(linewidth, 0.4), 6.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_axis = data.index.to_numpy() if not x else data[x].to_numpy()
    for col in columns:
        ax.plot(x_axis, data[col].to_numpy(), label=col, alpha=alpha, linewidth=linewidth)

    ax.set_title("Line Plot")
    ax.set_xlabel(x or "row_index")
    ax.set_ylabel("Value")
    if len(columns) <= 20:
        ax.legend(fontsize=8)
    return fig
