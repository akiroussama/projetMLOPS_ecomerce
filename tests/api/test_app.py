import json
import pickle
from importlib import import_module

from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.api.app import create_app
from src.api.service import PredictionExecutionError


AUTH_TOKEN = "test-token"


def _write_artifacts(target_dir):
    texts = [
        "robe rouge femme elegante",
        "jupe longue coton",
        "jeu puzzle enfant bois",
        "voiture miniature collection",
    ]
    labels = [10, 10, 20, 20]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=200)
    classifier.fit(features, labels)

    with (target_dir / "tfidf_vectorizer.pkl").open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    with (target_dir / "baseline_model.pkl").open("wb") as classifier_file:
        pickle.dump(classifier, classifier_file)

    with (target_dir / "label_mapping.json").open("w", encoding="utf-8") as mapping_file:
        json.dump({"10": "fashion", "20": "toys"}, mapping_file)


def _set_auth_token(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", AUTH_TOKEN)


def _auth_headers(token: str = AUTH_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_health_reports_ready_when_model_is_available(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "LogisticRegression"


def test_predict_returns_prediction_payload(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    payload = {
        "designation": "robe ete chic",
        "description": "robe rouge legere pour femme",
        "productid": 123,
        "imageid": 456,
    }

    with TestClient(create_app()) as client:
        response = client.post("/predict", json=payload, headers=_auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "fashion"
    assert body["predicted_code"] == 10
    assert body["model_name"] == "LogisticRegression"
    assert body["productid"] == 123
    assert body["imageid"] == 456
    assert isinstance(body["confidence"], float)


def test_predict_returns_503_when_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
            headers=_auth_headers(),
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "MODEL_NOT_READY"
    assert "No serialized model artifact was found" in body["message"]


def test_predict_rejects_missing_designation(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"description": "robe rouge"},
            headers=_auth_headers(),
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["message"] == "Request payload validation failed."
    assert any(detail["field"] == "designation" for detail in body["details"])


def test_predict_rejects_extra_fields(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={
                "designation": "robe ete chic",
                "description": "robe rouge legere pour femme",
                "unexpected": "value",
            },
            headers=_auth_headers(),
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert any(detail["field"] == "unexpected" for detail in body["details"])


def test_predict_rejects_blank_designation(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "   ", "description": "robe rouge"},
            headers=_auth_headers(),
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert any(detail["field"] == "designation" for detail in body["details"])


def test_predict_rejects_non_integer_productid(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={
                "designation": "robe ete chic",
                "description": "robe rouge legere pour femme",
                "productid": "123",
            },
            headers=_auth_headers(),
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert any(detail["field"] == "productid" for detail in body["details"])


def test_predict_returns_500_with_stable_error_schema(monkeypatch):
    _set_auth_token(monkeypatch)

    class FailingService:
        def predict(self, payload):
            raise PredictionExecutionError("Prediction pipeline failed.")

    api_app_module = import_module("src.api.app")
    monkeypatch.setattr(
        api_app_module,
        "_get_prediction_service",
        lambda application: FailingService(),
    )

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
            headers=_auth_headers(),
        )

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "PREDICTION_FAILED"
    assert body["message"] == "Prediction pipeline failed."


def test_health_is_public_even_when_auth_is_configured(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_bearer_token(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
        )

    assert response.status_code == 401
    body = response.json()
    assert body["error_code"] == "AUTHENTICATION_REQUIRED"
    assert response.headers["www-authenticate"] == "Bearer"


def test_predict_rejects_invalid_token(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    _set_auth_token(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
            headers=_auth_headers("wrong-token"),
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "INVALID_TOKEN"
    assert body["message"] == "Provided API token is invalid."


def test_predict_returns_503_when_auth_is_not_configured(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)

    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={"designation": "robe", "description": "ete"},
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "AUTH_NOT_CONFIGURED"
    assert "API_AUTH_TOKEN" in body["message"]
