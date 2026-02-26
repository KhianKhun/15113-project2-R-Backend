from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FilterOperator = Literal[
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "contains",
    "startswith",
    "endswith",
    "is_in",
]


class FilterClause(BaseModel):
    col: str
    op: FilterOperator
    value: Any


class FilterArgs(BaseModel):
    logic: Literal["AND", "OR"] = "AND"
    clauses: list[FilterClause] = Field(min_length=1)


class DropNaArgs(BaseModel):
    subset: list[str] | None = None


class Operation(BaseModel):
    op: str
    args: dict[str, Any] = Field(default_factory=dict)


class TransformRequest(BaseModel):
    operations: list[Operation] = Field(min_length=1)


class SchemaColumn(BaseModel):
    name: str
    type: str


class DatasetSummary(BaseModel):
    rows: int
    cols: int
    na_count: dict[str, int]


class DatasetStateResponse(BaseModel):
    dataset_id: str
    preview: list[dict[str, Any]]
    schema: list[SchemaColumn]
    summary: DatasetSummary


class ApiError(BaseModel):
    message: str
    details: list[dict[str, Any]] | None = None


PlotType = Literal["histogram", "scatter", "boxplot", "line", "bar"]


class PlotRequest(BaseModel):
    plot_type: PlotType
    columns: list[str] = Field(default_factory=list)
    x: str | None = None
    y: str | None = None
    color_by: str | None = None
    shape_by: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
