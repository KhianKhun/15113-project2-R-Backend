from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from matplotlib import pyplot as plt
from pandas.api import types as ptypes

from .utils import dataframe_for_columns, ensure_columns_exist


def render_barplot(
    df: pd.DataFrame,
    x: str | None,
    y: str | None,
    color_by: str | None,
    params: dict,
) -> plt.Figure:
    if not x:
        raise HTTPException(status_code=400, detail="Bar plot requires x variable.")

    needed = [x]
    if y:
        needed.append(y)
    if color_by:
        needed.append(color_by)

    ensure_columns_exist(df, needed)
    data = dataframe_for_columns(df, needed)

    agg = str(params.get("agg", "count")).lower()
    if agg not in {"count", "mean", "sum"}:
        raise HTTPException(status_code=400, detail="agg must be one of: count, mean, sum.")

    fig, ax = plt.subplots(figsize=(10, 5))

    if color_by:
        grouped = _grouped_bar_data(data, x, y, color_by, agg)
        grouped.plot(kind="bar", ax=ax)
        ax.legend(title=color_by, fontsize=8)
    else:
        series = _single_bar_data(data, x, y, agg)
        series.plot(kind="bar", ax=ax, color="#3b82f6")

    ax.set_title("Bar Plot")
    ax.set_xlabel(x)
    ax.set_ylabel("count" if not y else f"{agg}({y})")
    ax.tick_params(axis="x", labelrotation=35)
    return fig


def _single_bar_data(data: pd.DataFrame, x: str, y: str | None, agg: str) -> pd.Series:
    if y:
        if not ptypes.is_numeric_dtype(data[y]):
            raise HTTPException(status_code=400, detail=f"Numeric y variable required: '{y}'.")
        if agg == "mean":
            return data.groupby(x)[y].mean()
        if agg == "sum":
            return data.groupby(x)[y].sum()
        return data.groupby(x)[y].count()
    return data[x].value_counts()


def _grouped_bar_data(
    data: pd.DataFrame,
    x: str,
    y: str | None,
    color_by: str,
    agg: str,
) -> pd.DataFrame:
    if y:
        if not ptypes.is_numeric_dtype(data[y]):
            raise HTTPException(status_code=400, detail=f"Numeric y variable required: '{y}'.")
        grouped = data.groupby([x, color_by])[y]
        if agg == "mean":
            return grouped.mean().unstack(fill_value=0)
        if agg == "sum":
            return grouped.sum().unstack(fill_value=0)
        return grouped.count().unstack(fill_value=0)
    return data.groupby([x, color_by]).size().unstack(fill_value=0)
