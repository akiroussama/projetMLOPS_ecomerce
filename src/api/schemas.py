from typing import Annotated, Dict, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
)


PredictionValue = Union[int, str]
StrictShortText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
]
StrictLongText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, max_length=10000),
]
StrictIdentifier = Annotated[StrictInt, Field(ge=0)]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    vectorizer_path: Optional[str] = None
    label_mapping_path: Optional[str] = None
    detail: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    designation: StrictShortText
    description: StrictLongText = ""
    productid: Optional[StrictIdentifier] = None
    imageid: Optional[StrictIdentifier] = None


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_code: PredictionValue
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_name: str
    productid: Optional[int] = None
    imageid: Optional[int] = None


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    error_type: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_predictions: int
    predictions_by_category: Dict[str, int]
    avg_inference_ms: Optional[float] = None
    min_inference_ms: Optional[float] = None
    max_inference_ms: Optional[float] = None
