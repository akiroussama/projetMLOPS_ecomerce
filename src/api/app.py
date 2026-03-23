import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import ErrorDetail, ErrorResponse, HealthResponse, PredictRequest, PredictResponse, StatsResponse
from .security import require_prediction_token
from .service import ModelNotReadyError, PredictionExecutionError, PredictionService

logger = logging.getLogger("rakuten.api")

API_TITLE = "Rakuten MLOps API"
API_DESCRIPTION = "Prediction API for the e-commerce product classifier."


class _AppStats:
    """Thread-safe in-memory prediction statistics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = defaultdict(int)
        self._inference_times: List[float] = []

    def record(self, predicted_label: str, inference_ms: float) -> None:
        with self._lock:
            self._counts[predicted_label] += 1
            self._inference_times.append(inference_ms)

    def snapshot(self) -> dict:
        with self._lock:
            times = list(self._inference_times)
            counts = dict(self._counts)
        total = sum(counts.values())
        return {
            "total_predictions": total,
            "predictions_by_category": counts,
            "avg_inference_ms": round(sum(times) / len(times), 2) if times else None,
            "min_inference_ms": round(min(times), 2) if times else None,
            "max_inference_ms": round(max(times), 2) if times else None,
        }


def _get_prediction_service(application: FastAPI) -> PredictionService:
    service = getattr(application.state, "prediction_service", None)

    if service is None:
        service = PredictionService()
        service.load()
        application.state.prediction_service = service

    return service


def _build_error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error_code=error_code,
        message=message,
        details=[ErrorDetail(**detail) for detail in details or []],
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers=headers,
    )


def _format_error_location(location: tuple) -> str | None:
    normalized_location = [str(part) for part in location if part != "body"]
    return ".".join(normalized_location) or None


def _validation_details(exc: RequestValidationError) -> list[dict]:
    return [
        {
            "field": _format_error_location(error["loc"]),
            "message": error["msg"],
            "error_type": error["type"],
        }
        for error in exc.errors()
    ]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = PredictionService()
        service.load()
        app.state.prediction_service = service
        app.state.stats = _AppStats()
        yield

    application = FastAPI(
        title=API_TITLE,
        version="1.0.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _build_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="Request payload validation failed.",
            details=_validation_details(exc),
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail

        if isinstance(detail, dict):
            return _build_error_response(
                status_code=exc.status_code,
                error_code=detail.get("error_code", "HTTP_ERROR"),
                message=detail.get("message", "HTTP error."),
                details=detail.get("details", []),
                headers=exc.headers,
            )

        error_code_map = {
            status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
            status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
            status.HTTP_403_FORBIDDEN: "FORBIDDEN",
            status.HTTP_404_NOT_FOUND: "NOT_FOUND",
            status.HTTP_422_UNPROCESSABLE_ENTITY: "VALIDATION_ERROR",
            status.HTTP_503_SERVICE_UNAVAILABLE: "MODEL_NOT_READY",
        }
        return _build_error_response(
            status_code=exc.status_code,
            error_code=error_code_map.get(exc.status_code, "HTTP_ERROR"),
            message=str(detail),
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return _build_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message="Unexpected error during request processing.",
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["system"],
    )
    def health(request: Request) -> HealthResponse:
        service = _get_prediction_service(request.app)
        return HealthResponse(**service.health_snapshot())

    @application.post(
        "/predict",
        response_model=PredictResponse,
        status_code=status.HTTP_200_OK,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    def predict(
        payload: PredictRequest,
        request: Request,
        _: None = Depends(require_prediction_token),
    ) -> PredictResponse:
        service = _get_prediction_service(request.app)

        t0 = time.perf_counter()
        try:
            prediction = service.predict(payload)
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "MODEL_NOT_READY",
                    "message": str(exc),
                },
            ) from exc
        except PredictionExecutionError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "PREDICTION_FAILED",
                    "message": str(exc),
                },
            ) from exc

        inference_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "prediction: label=%s confidence=%.4f time=%.1fms",
            prediction["predicted_label"],
            prediction.get("confidence", 0),
            inference_ms,
        )
        app_stats: _AppStats = getattr(request.app.state, "stats", None)
        if app_stats is not None:
            app_stats.record(str(prediction["predicted_label"]), inference_ms)

        return PredictResponse(**prediction)

    @application.get(
        "/stats",
        response_model=StatsResponse,
        tags=["system"],
        summary="Live prediction statistics (in-memory, resets on restart)",
    )
    def stats(request: Request) -> StatsResponse:
        app_stats: _AppStats = getattr(request.app.state, "stats", None)
        if app_stats is None:
            return StatsResponse(total_predictions=0, predictions_by_category={})
        return StatsResponse(**app_stats.snapshot())

    Instrumentator().instrument(application).expose(application)

    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
