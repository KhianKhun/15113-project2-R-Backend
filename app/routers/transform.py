from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..models import ApiError, DatasetStateResponse, TransformRequest
from ..storage import get_dataset, put_dataset, validate_dataframe_shape
from ..summary import build_dataset_response
from ..transforms import apply_pipeline

router = APIRouter(prefix="/api/datasets", tags=["transform"])


@router.post(
    "/{dataset_id}/transform",
    response_model=DatasetStateResponse,
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def transform_dataset(
    dataset_id: str,
    payload: TransformRequest,
    limit: int = Query(default=50, ge=1, le=200),
) -> DatasetStateResponse:
    try:
        dataframe = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    transformed = apply_pipeline(dataframe, payload.operations)

    try:
        validate_dataframe_shape(transformed, allow_empty=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    new_dataset_id = put_dataset(transformed)
    return build_dataset_response(new_dataset_id, transformed, limit=limit)
