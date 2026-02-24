from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException
from pandas.api import types as ptypes
from pydantic import ValidationError

from .models import DropNaArgs, FilterArgs, Operation

SUPPORTED_OPERATIONS = {"drop_na_rows", "filter_rows"}


def apply_pipeline(df: pd.DataFrame, operations: list[Operation]) -> pd.DataFrame:
    result = df.copy(deep=True)
    for idx, operation in enumerate(operations):
        try:
            result = apply_operation(result, operation)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Operation #{idx + 1} failed: {exc}",
            ) from exc
    return result


def apply_operation(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    if operation.op == "drop_na_rows":
        return _drop_na_rows(df, operation.args)
    if operation.op == "filter_rows":
        return _filter_rows(df, operation.args)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported operation '{operation.op}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_OPERATIONS))}."
        ),
    )


def _drop_na_rows(df: pd.DataFrame, args: dict[str, Any]) -> pd.DataFrame:
    try:
        parsed = DropNaArgs(**args)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid args for drop_na_rows: {exc.errors()}",
        ) from exc

    if parsed.subset:
        _ensure_columns_exist(df, parsed.subset)
        return df.dropna(subset=parsed.subset).reset_index(drop=True)
    return df.dropna().reset_index(drop=True)


def _filter_rows(df: pd.DataFrame, args: dict[str, Any]) -> pd.DataFrame:
    try:
        parsed = FilterArgs(**args)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid args for filter_rows: {exc.errors()}",
        ) from exc

    _ensure_columns_exist(df, [clause.col for clause in parsed.clauses])

    masks = [_build_clause_mask(df, clause.col, clause.op, clause.value) for clause in parsed.clauses]
    current = masks[0]
    for next_mask in masks[1:]:
        if parsed.logic == "AND":
            current = current & next_mask
        else:
            current = current | next_mask

    return df.loc[current].reset_index(drop=True)


def _ensure_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Column(s) not found: {', '.join(missing)}",
        )


def _build_clause_mask(df: pd.DataFrame, col: str, op: str, value: Any) -> pd.Series:
    series = df[col]

    if op == "==":
        return series == value
    if op == "!=":
        return series != value

    if op in {"<", "<=", ">", ">="}:
        return _numeric_compare(series, col, op, value)

    if op in {"contains", "startswith", "endswith"}:
        return _string_compare(series, col, op, value)

    if op == "is_in":
        if not isinstance(value, list):
            raise HTTPException(
                status_code=400,
                detail="Type mismatch: 'is_in' requires a list value.",
            )
        return series.isin(value)

    raise HTTPException(status_code=400, detail=f"Unsupported filter operator '{op}'.")


def _numeric_compare(series: pd.Series, col: str, op: str, value: Any) -> pd.Series:
    if not ptypes.is_numeric_dtype(series):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: operator '{op}' requires numeric column '{col}'.",
        )

    if not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: value for operator '{op}' on '{col}' must be numeric.",
        )

    if op == "<":
        return series < value
    if op == "<=":
        return series <= value
    if op == ">":
        return series > value
    return series >= value


def _string_compare(series: pd.Series, col: str, op: str, value: Any) -> pd.Series:
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: operator '{op}' requires a string value.",
        )

    if not (ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series)):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: operator '{op}' requires string-like column '{col}'.",
        )

    text = series.astype("string")
    if op == "contains":
        return text.str.contains(value, na=False)
    if op == "startswith":
        return text.str.startswith(value, na=False)
    return text.str.endswith(value, na=False)
