import pickle
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.app import create_app
from src.api.schemas import PredictRequest
from src.api.service import PredictionExecutionError, PredictionService


class PipelineStyleModel:
    def predict(self, values):
        return [1]


class BrokenModel:
    def predict(self, values):
        raise ValueError("boom")


def test_prediction_service_supports_pipeline_model_without_vectorizer(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.pkl"
    mapping_path = tmp_path / "mapping.pkl"

    with model_path.open("wb") as model_file:
        pickle.dump(PipelineStyleModel(), model_file)

    with mapping_path.open("wb") as mapping_file:
        pickle.dump({1: "fashion"}, mapping_file)

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_FILE", str(model_path))
    monkeypatch.setenv("LABEL_MAPPING_FILE", str(mapping_path))
    monkeypatch.delenv("VECTORIZER_FILE", raising=False)

    service = PredictionService()

    assert service.load() is True
    prediction = service.predict(PredictRequest(designation="robe ete"))

    assert prediction["predicted_code"] == 1
    assert prediction["predicted_label"] == "fashion"
    assert prediction["confidence"] is None
    assert prediction["model_name"] == "PipelineStyleModel"


def test_prediction_service_raises_prediction_execution_error_on_model_failure():
    service = PredictionService(project_root=Path("."))
    service.model = BrokenModel()
    service.load_error = None
    service.vectorizer = None
    service.label_mapping = None

    with pytest.raises(PredictionExecutionError):
        service.predict(PredictRequest(designation="robe ete"))


def test_require_prediction_token_rejects_invalid_auth_scheme(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
            headers={"Authorization": "Basic abc123"},
        )

    assert response.status_code == 401
    body = response.json()
    assert body["error_code"] == "INVALID_AUTH_SCHEME"
    assert response.headers["www-authenticate"] == "Bearer"


def test_get_prediction_service_bootstraps_when_state_is_missing():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.get("/health")
    client.close()

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_http_exception_handler_maps_plain_http_exception():
    from fastapi.testclient import TestClient

    application = create_app()

    @application.get("/not-found")
    def not_found():
        raise HTTPException(status_code=404, detail="missing resource")

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/not-found")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "NOT_FOUND"
    assert body["message"] == "missing resource"


def test_unhandled_exception_handler_returns_internal_server_error():
    from fastapi.testclient import TestClient

    application = create_app()

    @application.get("/explode")
    def explode():
        raise RuntimeError("unexpected")

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/explode")

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_SERVER_ERROR"
    assert body["message"] == "Unexpected error during request processing."
