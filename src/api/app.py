from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import ErrorDetail, ErrorResponse, HealthResponse, PredictRequest, PredictResponse
from .security import require_prediction_token
from .service import ModelNotReadyError, PredictionExecutionError, PredictionService


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
        yield

    application = FastAPI(
        title="Rakuten MLOps API",
        version="0.1.0",
        description="Prediction API for the e-commerce product classifier.",
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

        return PredictResponse(**prediction)

    Instrumentator().instrument(application).expose(application)

    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
