from __future__ import annotations

import os
import uuid
from threading import Lock

import pandas as pd


MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(5 * 1024 * 1024)))
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "200000"))
MAX_DATASET_COLS = int(os.getenv("MAX_DATASET_COLS", "200"))

_DATASETS: dict[str, pd.DataFrame] = {}
_DATA_LOCK = Lock()


def validate_file_size(size_bytes: int) -> None:
    if size_bytes <= 0:
        raise ValueError("Uploaded file is empty.")
    if size_bytes > MAX_FILE_BYTES:
        raise ValueError(
            f"Uploaded file is too large ({size_bytes} bytes). Limit is {MAX_FILE_BYTES} bytes."
        )


def validate_dataframe_shape(df: pd.DataFrame, allow_empty: bool = False) -> None:
    if (not allow_empty) and df.empty:
        raise ValueError("CSV has no rows.")
    if df.shape[0] > MAX_DATASET_ROWS:
        raise ValueError(
            f"Dataset has too many rows ({df.shape[0]}). Limit is {MAX_DATASET_ROWS}."
        )
    if df.shape[1] > MAX_DATASET_COLS:
        raise ValueError(
            f"Dataset has too many columns ({df.shape[1]}). Limit is {MAX_DATASET_COLS}."
        )


def put_dataset(df: pd.DataFrame) -> str:
    dataset_id = str(uuid.uuid4())
    with _DATA_LOCK:
        _DATASETS[dataset_id] = df.copy(deep=True)
    return dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    with _DATA_LOCK:
        df = _DATASETS.get(dataset_id)
    if df is None:
        raise KeyError(f"Dataset '{dataset_id}' not found.")
    return df.copy(deep=True)
