from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from ..models import ApiError, DatasetStateResponse
from ..storage import get_dataset, put_dataset, validate_dataframe_shape, validate_file_size
from ..summary import build_dataset_response

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetStateResponse,
    responses={400: {"model": ApiError}},
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetStateResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        validate_file_size(len(contents))
        dataframe = pd.read_csv(BytesIO(contents))
        validate_dataframe_shape(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    dataset_id = put_dataset(dataframe)
    return build_dataset_response(dataset_id, dataframe)


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetStateResponse,
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def get_preview(
    dataset_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> DatasetStateResponse:
    try:
        dataframe = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    return build_dataset_response(dataset_id, dataframe, limit=limit)


@router.get(
    "/{dataset_id}/download",
    responses={404: {"model": ApiError}},
)
def download_dataset_csv(dataset_id: str) -> StreamingResponse:
    try:
        dataframe = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    csv_payload = dataframe.to_csv(index=False)
    filename = f"dataset_{dataset_id[:8]}.csv"

    return StreamingResponse(
        iter([csv_payload]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
