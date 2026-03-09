from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = FastAPI(title="projetMLOPS_ecomerce API", version="0.3.0")

# Dossier des modèles
MODELS_DIR = Path("models")

# Modèles
LSTM_MODEL_PATH = MODELS_DIR / "best_lstm_model.h5"
VGG16_MODEL_PATH = MODELS_DIR / "best_vgg16_model.h5"

# Artefacts projet
TOKENIZER_CONFIG_PATH = MODELS_DIR / "tokenizer_config.json"
MAPPER_JSON_PATH = MODELS_DIR / "mapper.json"
MAPPER_PKL_PATH = MODELS_DIR / "mapper.pkl"
BEST_WEIGHTS_JSON_PATH = MODELS_DIR / "best_weights.json"
BEST_WEIGHTS_PKL_PATH = MODELS_DIR / "best_weights.pkl"

# Ressources chargées au démarrage
lstm_model = None
vgg16_model = None
tokenizer = None
mapper = None
best_weights = None

# Dans src/predict.py, le maxlen utilisé est 10
MAXLEN = 10


class PredictRequest(BaseModel):
    model_type: Literal["lstm", "vgg16"] = Field(
        ..., description="Choix du modèle à utiliser"
    )
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=5000,
        description="Texte produit pour le modèle LSTM",
    )
    image_path: Optional[str] = Field(
        None,
        description="Chemin local vers une image pour le modèle VGG16",
    )


class PredictResponse(BaseModel):
    model_used: str
    prediction: str
    input_text: Optional[str] = None
    input_image_path: Optional[str] = None


@app.on_event("startup")
def load_assets() -> None:
    global lstm_model, vgg16_model, tokenizer, mapper, best_weights

    if LSTM_MODEL_PATH.exists():
        try:
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        except ValueError as e:
            if "time_major" in str(e):
                print(f"Warning: LSTM model loading failed due to TensorFlow version incompatibility: {e}")
                lstm_model = None
            else:
                raise

    if VGG16_MODEL_PATH.exists():
        vgg16_model = tf.keras.models.load_model(VGG16_MODEL_PATH)

    if TOKENIZER_CONFIG_PATH.exists():
        tokenizer_config = TOKENIZER_CONFIG_PATH.read_text(encoding="utf-8")
        tokenizer = tokenizer_from_json(tokenizer_config)

    if MAPPER_JSON_PATH.exists():
        mapper = json.loads(MAPPER_JSON_PATH.read_text(encoding="utf-8"))
    elif MAPPER_PKL_PATH.exists():
        with open(MAPPER_PKL_PATH, "rb") as f:
            mapper = pickle.load(f)

    if BEST_WEIGHTS_JSON_PATH.exists():
        best_weights = json.loads(BEST_WEIGHTS_JSON_PATH.read_text(encoding="utf-8"))
    elif BEST_WEIGHTS_PKL_PATH.exists():
        with open(BEST_WEIGHTS_PKL_PATH, "rb") as f:
            best_weights = pickle.load(f)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "lstm_loaded": lstm_model is not None,
            "vgg16_loaded": vgg16_model is not None,
        },
        "artifacts": {
            "tokenizer_loaded": tokenizer is not None,
            "classes_loaded": mapper is not None,
            "best_weights_loaded": best_weights is not None,
            "maxlen": MAXLEN,
        },
    }


def map_prediction(pred_idx: int) -> str:
    if mapper is None:
        return str(pred_idx)

    if isinstance(mapper, dict):
        return str(mapper.get(str(pred_idx), mapper.get(pred_idx, pred_idx)))

    if isinstance(mapper, list) and pred_idx < len(mapper):
        return str(mapper[pred_idx])

    return str(pred_idx)


def predict_with_lstm(text: str) -> str:
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")

    if tokenizer is None:
        raise HTTPException(
            status_code=501,
            detail="Tokenizer missing. Add models/tokenizer_config.json to enable real LSTM inference.",
        )

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAXLEN,
        padding="post",
        truncating="post",
    )

    preds = lstm_model.predict([padded_sequences], verbose=0)
    pred_idx = int(np.argmax(preds, axis=1)[0])

    return map_prediction(pred_idx)


def predict_with_vgg16(image_path: str) -> str:
    if vgg16_model is None:
        raise HTTPException(status_code=503, detail="VGG16 model not loaded")

    img_path = Path(image_path)
    if not img_path.exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = vgg16_model.predict([img_array], verbose=0)
    pred_idx = int(np.argmax(preds, axis=1)[0])

    return map_prediction(pred_idx)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if payload.model_type == "lstm":
        if not payload.text:
            raise HTTPException(
                status_code=422,
                detail="Field 'text' is required when model_type='lstm'",
            )

        prediction = predict_with_lstm(payload.text)
        return PredictResponse(
            model_used="lstm",
            prediction=prediction,
            input_text=payload.text,
        )

    if payload.model_type == "vgg16":
        if not payload.image_path:
            raise HTTPException(
                status_code=422,
                detail="Field 'image_path' is required when model_type='vgg16'",
            )

        prediction = predict_with_vgg16(payload.image_path)
        return PredictResponse(
            model_used="vgg16",
            prediction=prediction,
            input_image_path=payload.image_path,
        )

    raise HTTPException(status_code=400, detail="Unsupported model_type")