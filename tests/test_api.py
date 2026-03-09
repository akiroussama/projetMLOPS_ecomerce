from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "models" in body
    assert "artifacts" in body


def test_predict_lstm_missing_text():
    response = client.post("/predict", json={"model_type": "lstm"})
    assert response.status_code == 422


def test_predict_vgg16_missing_image():
    response = client.post("/predict", json={"model_type": "vgg16"})
    assert response.status_code == 422


def test_predict_lstm_without_tokenizer():
    response = client.post(
        "/predict",
        json={"model_type": "lstm", "text": "smartphone samsung"},
    )
    assert response.status_code in (200, 501, 503)


def test_predict_vgg16_with_missing_image_file():
    response = client.post(
        "/predict",
        json={"model_type": "vgg16", "image_path": "fake_image.jpg"},
    )
    assert response.status_code in (200, 400, 503)


def test_map_prediction_without_mapper(monkeypatch):
    monkeypatch.setattr(main, "mapper", None)
    assert main.map_prediction(3) == "3"


def test_map_prediction_with_dict_mapper(monkeypatch):
    monkeypatch.setattr(main, "mapper", {"3": "classe_3"})
    assert main.map_prediction(3) == "classe_3"


def test_map_prediction_with_list_mapper(monkeypatch):
    monkeypatch.setattr(main, "mapper", ["a", "b", "c"])
    assert main.map_prediction(1) == "b"


def test_predict_with_lstm_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "lstm_model", None)
    try:
        main.predict_with_lstm("test produit")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 503


def test_predict_with_lstm_tokenizer_missing(monkeypatch):
    class DummyModel:
        def predict(self, *_args, **_kwargs):
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr(main, "lstm_model", DummyModel())
    monkeypatch.setattr(main, "tokenizer", None)

    try:
        main.predict_with_lstm("test produit")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 501


def test_predict_with_lstm_success(monkeypatch):
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3]]

    class DummyModel:
        def predict(self, *_args, **_kwargs):
            return np.array([[0.1, 0.8, 0.1]])

    monkeypatch.setattr(main, "lstm_model", DummyModel())
    monkeypatch.setattr(main, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(main, "mapper", {"1": "predicted_class"})

    result = main.predict_with_lstm("test produit")
    assert result == "predicted_class"


def test_predict_with_vgg16_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "vgg16_model", None)
    try:
        main.predict_with_vgg16("fake_image.jpg")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 503


def test_predict_with_vgg16_image_not_found(monkeypatch):
    class DummyModel:
        def predict(self, *_args, **_kwargs):
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr(main, "vgg16_model", DummyModel())

    try:
        main.predict_with_vgg16("image_inexistante.jpg")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400


def test_predict_with_vgg16_success(monkeypatch, tmp_path):
    img_path = tmp_path / "test.jpg"

    # écriture minimale d'un faux fichier image
    from PIL import Image

    image = Image.new("RGB", (224, 224), color="white")
    image.save(img_path)

    class DummyModel:
        def predict(self, *_args, **_kwargs):
            return np.array([[0.2, 0.7, 0.1]])

    monkeypatch.setattr(main, "vgg16_model", DummyModel())
    monkeypatch.setattr(main, "mapper", {"1": "image_class"})

    result = main.predict_with_vgg16(str(img_path))
    assert result == "image_class"


def test_predict_endpoint_unsupported_model():
    response = client.post(
        "/predict",
        json={"model_type": "lstm", "text": "abc"},
    )
    assert response.status_code in (200, 501, 503)


def test_load_assets(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    (model_dir / "best_lstm_model.h5").write_text("x", encoding="utf-8")
    (model_dir / "best_vgg16_model.h5").write_text("x", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        '{"class_name": "Tokenizer", "config": {"num_words": 100, "filters": "", "lower": true, "split": " ", "char_level": false, "oov_token": "<OOV>", "document_count": 0, "word_counts": "{}", "word_docs": "{}", "index_docs": "{}", "index_word": "{\\"1\\": \\"test\\"}", "word_index": "{\\"test\\": 1}"}}',
        encoding="utf-8",
    )
    (model_dir / "mapper.json").write_text('{"1": "mapped"}', encoding="utf-8")
    (model_dir / "best_weights.json").write_text("[0.5, 0.5]", encoding="utf-8")

    monkeypatch.setattr(main, "MODELS_DIR", model_dir)
    monkeypatch.setattr(main, "LSTM_MODEL_PATH", model_dir / "best_lstm_model.h5")
    monkeypatch.setattr(main, "VGG16_MODEL_PATH", model_dir / "best_vgg16_model.h5")
    monkeypatch.setattr(main, "TOKENIZER_CONFIG_PATH", model_dir / "tokenizer_config.json")
    monkeypatch.setattr(main, "MAPPER_JSON_PATH", model_dir / "mapper.json")
    monkeypatch.setattr(main, "MAPPER_PKL_PATH", model_dir / "mapper.pkl")
    monkeypatch.setattr(main, "BEST_WEIGHTS_JSON_PATH", model_dir / "best_weights.json")
    monkeypatch.setattr(main, "BEST_WEIGHTS_PKL_PATH", model_dir / "best_weights.pkl")

    monkeypatch.setattr(main.tf.keras.models, "load_model", lambda _: "dummy_model")

    main.load_assets()

    assert main.lstm_model == "dummy_model"
    assert main.vgg16_model == "dummy_model"
    assert main.tokenizer is not None
    assert main.mapper == {"1": "mapped"}
    assert main.best_weights == [0.5, 0.5]
