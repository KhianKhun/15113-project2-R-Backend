from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from ..models import PlotRequest
from .barplot import render_barplot
from .boxplot import render_boxplot
from .histogram import render_histogram
from .lineplot import render_lineplot
from .scatter import render_scatter
from .utils import fig_to_png_bytes


def render_plot_png(df: pd.DataFrame, payload: PlotRequest) -> bytes:
    fig = None
    try:
        if payload.plot_type == "histogram":
            fig = render_histogram(df, payload.columns, payload.params)
        elif payload.plot_type == "scatter":
            fig = render_scatter(df, payload.x, payload.y, payload.color_by, payload.shape_by, payload.params)
        elif payload.plot_type == "boxplot":
            fig = render_boxplot(df, payload.columns, payload.x, payload.y)
        elif payload.plot_type == "line":
            fig = render_lineplot(df, payload.columns, payload.x, payload.params)
        elif payload.plot_type == "bar":
            fig = render_barplot(df, payload.x, payload.y, payload.color_by, payload.params)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported plot type: {payload.plot_type}")
        return fig_to_png_bytes(fig)
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
