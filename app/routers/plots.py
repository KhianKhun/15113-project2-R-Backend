from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import ApiError, PlotRequest
from ..plotting import render_plot_png
from ..storage import get_dataset

router = APIRouter(prefix="/api/datasets", tags=["plots"])


@router.post(
    "/{dataset_id}/plots/render",
    responses={400: {"model": ApiError}, 404: {"model": ApiError}},
)
def render_plot(dataset_id: str, payload: PlotRequest) -> StreamingResponse:
    try:
        dataframe = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    png_bytes = render_plot_png(dataframe, payload)
    filename = f"plot_{payload.plot_type}_{dataset_id[:8]}.png"
    return StreamingResponse(
        iter([png_bytes]),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
