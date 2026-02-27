from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pandas.api import types as ptypes
from pydantic import ValidationError

from .models import DropNaArgs, FilterArgs, Operation

SUPPORTED_OPERATIONS = {"drop_na_rows", "filter_rows"}
MATH_OPERATORS = {"exp", "log", "^", "+"}


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

    if any(clause.op in MATH_OPERATORS for clause in parsed.clauses):
        if len(parsed.clauses) != 1:
            raise HTTPException(
                status_code=400,
                detail="Math operators (exp, log, ^, +) only support one clause.",
            )
        clause = parsed.clauses[0]
        return _apply_math_operator(
            df=df,
            col=clause.col,
            op=clause.op,
            value=clause.value,
            create_new_column=parsed.create_new_column,
        )

    masks = [_build_clause_mask(df, clause.col, clause.op, clause.value) for clause in parsed.clauses]
    current = masks[0]
    for next_mask in masks[1:]:
        current = current & next_mask

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

    raise HTTPException(status_code=400, detail=f"Unsupported filter operator '{op}'.")


def _apply_math_operator(
    df: pd.DataFrame,
    col: str,
    op: str,
    value: Any,
    create_new_column: bool,
) -> pd.DataFrame:
    series = df[col]

    if not ptypes.is_numeric_dtype(series):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: operator '{op}' requires numeric column '{col}'.",
        )

    numeric_series = pd.to_numeric(series, errors="coerce")

    if op == "exp":
        transformed = np.exp(numeric_series)
    elif op == "log":
        non_na = numeric_series.dropna()
        if (non_na <= 0).any():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input: operator 'log' requires column '{col}' values > 0.",
            )
        transformed = np.log(numeric_series)
    elif op == "^":
        exponent = _coerce_number(value, op, col)
        transformed = np.power(numeric_series, exponent)
    elif op == "+":
        increment = _coerce_number(value, op, col)
        transformed = numeric_series + increment
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported numeric operator '{op}'.")

    result = df.copy(deep=True)
    if create_new_column:
        target_col = _next_column_name(result, _default_new_column_name(col, op))
    else:
        target_col = col
    result[target_col] = transformed
    return result


def _coerce_number(value: Any, op: str, col: str) -> float:
    if isinstance(value, bool):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: value for operator '{op}' on '{col}' must be numeric.",
        )

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() == "NA":
            raise HTTPException(
                status_code=400,
                detail=f"Type mismatch: value for operator '{op}' on '{col}' must be numeric.",
            )
        try:
            number = float(stripped)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Type mismatch: value for operator '{op}' on '{col}' must be numeric.",
            ) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: value for operator '{op}' on '{col}' must be numeric.",
        )

    if not np.isfinite(number):
        raise HTTPException(
            status_code=400,
            detail=f"Type mismatch: value for operator '{op}' on '{col}' must be finite.",
        )
    return number


def _default_new_column_name(col: str, op: str) -> str:
    suffix = {
        "exp": "exp",
        "log": "ln",
        "^": "pow",
        "+": "plus",
    }.get(op, "calc")
    return f"{col}_{suffix}"


def _next_column_name(df: pd.DataFrame, base_name: str) -> str:
    if base_name not in df.columns:
        return base_name

    counter = 1
    while True:
        candidate = f"{base_name}_{counter}"
        if candidate not in df.columns:
            return candidate
        counter += 1


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
