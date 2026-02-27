from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from pandas.api import types as ptypes


def ensure_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Column(s) not found: {', '.join(missing)}")


def prepare_training_frame(df: pd.DataFrame, y_col: str, feature_cols: list[str]) -> pd.DataFrame:
    if not feature_cols:
        raise HTTPException(status_code=400, detail="Please select at least one feature variable.")
    ensure_columns_exist(df, [y_col] + feature_cols)
    clean = df[[y_col] + feature_cols].dropna()
    if clean.empty:
        raise HTTPException(
            status_code=400,
            detail="No rows remain after dropping NA for selected regression variables.",
        )
    return clean


def require_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    bad = [col for col in columns if not ptypes.is_numeric_dtype(df[col])]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Numeric column required for: {', '.join(bad)}",
        )


def pick_plot_x(feature_cols: list[str], plot_x: str | None) -> str:
    chosen = plot_x or feature_cols[0]
    if chosen not in feature_cols:
        raise HTTPException(status_code=400, detail="plot_x must be one of selected x features.")
    return chosen
