from __future__ import annotations

from io import BytesIO

import matplotlib
import pandas as pd
from fastapi import HTTPException
from matplotlib.figure import Figure
from pandas.api import types as ptypes

matplotlib.use("Agg")


def ensure_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Column(s) not found: {', '.join(missing)}")


def require_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    bad = [col for col in columns if not ptypes.is_numeric_dtype(df[col])]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Numeric column required for: {', '.join(bad)}",
        )


def require_categorical_like(df: pd.DataFrame, column: str) -> None:
    if ptypes.is_numeric_dtype(df[column]):
        raise HTTPException(
            status_code=400,
            detail=f"Categorical/factor-like column required for '{column}'.",
        )


def dataframe_for_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        raise HTTPException(status_code=400, detail="Please select at least one variable.")
    ensure_columns_exist(df, columns)
    cleaned = df[columns].dropna()
    if cleaned.empty:
        raise HTTPException(status_code=400, detail="No rows remain after ignoring NA for selected variables.")
    return cleaned


def fig_to_png_bytes(fig: Figure) -> bytes:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()
