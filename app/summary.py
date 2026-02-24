from __future__ import annotations

import json

import pandas as pd
from pandas.api import types as ptypes

from .models import DatasetStateResponse, DatasetSummary, SchemaColumn


def infer_column_type(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_integer_dtype(series):
        return "integer"
    if ptypes.is_float_dtype(series):
        return "float"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "string"


def build_schema(df: pd.DataFrame) -> list[SchemaColumn]:
    return [
        SchemaColumn(name=str(col_name), type=infer_column_type(df[col_name]))
        for col_name in df.columns
    ]


def build_summary(df: pd.DataFrame) -> DatasetSummary:
    na_counts = df.isna().sum().to_dict()
    normalized = {str(col): int(na_counts[col]) for col in df.columns}
    return DatasetSummary(rows=int(df.shape[0]), cols=int(df.shape[1]), na_count=normalized)


def build_preview(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    if limit <= 0:
        return []
    clipped = df.head(limit)
    # to_json ensures NaN/NaT become null and numpy values are JSON-safe.
    return json.loads(clipped.to_json(orient="records", date_format="iso"))


def build_dataset_response(
    dataset_id: str, df: pd.DataFrame, limit: int = 50
) -> DatasetStateResponse:
    return DatasetStateResponse(
        dataset_id=dataset_id,
        preview=build_preview(df, limit),
        schema=build_schema(df),
        summary=build_summary(df),
    )
