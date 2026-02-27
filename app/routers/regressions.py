from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import (
    ApiError,
    RegressionFitRequest,
    RegressionFitResponse,
    RegressionPredictRequest,
    RegressionPredictResponse,
)
from ..regression import fit_and_store_regression, predict_with_stored_model, render_stored_curve_png
from ..storage import get_dataset

router = APIRouter(prefix="/api", tags=["regressions"])


@router.post(
    "/datasets/{dataset_id}/regressions/fit",
    response_model=RegressionFitResponse,
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def fit_regression(dataset_id: str, payload: RegressionFitRequest) -> RegressionFitResponse:
    try:
        dataframe = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
    return fit_and_store_regression(dataset_id, dataframe, payload)


@router.get(
    "/regressions/{model_id}/curve",
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def plot_regression_curve(model_id: str) -> StreamingResponse:
    png_bytes = render_stored_curve_png(model_id)
    filename = f"regression_curve_{model_id[:8]}.png"
    return StreamingResponse(
        iter([png_bytes]),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post(
    "/regressions/{model_id}/predict",
    response_model=RegressionPredictResponse,
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def predict_regression(
    model_id: str, payload: RegressionPredictRequest
) -> RegressionPredictResponse:
    return predict_with_stored_model(model_id, payload)
