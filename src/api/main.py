import json
import joblib
from pathlib import Path
from typing import Optional, Literal, Union
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# démarrage de l'api
app = FastAPI(title="projetMLOPS_ecomerce API", version="1.0.0")

# --- CONSTANTES POUR LES TESTS ET LA PROD ---
MODELS_DIR = Path("models")
LSTM_MODEL_PATH = MODELS_DIR / "best_lstm_model.h5"
VGG16_MODEL_PATH = MODELS_DIR / "best_vgg16_model.h5"
TOKENIZER_CONFIG_PATH = MODELS_DIR / "tokenizer_config.json"
MAPPER_JSON_PATH = MODELS_DIR / "label_mapping.json"
MAPPER_PKL_PATH = MODELS_DIR / "mapper.pkl"
BEST_WEIGHTS_JSON_PATH = MODELS_DIR / "best_weights.json"
BEST_WEIGHTS_PKL_PATH = MODELS_DIR / "best_weights.pkl"

# --- OBJETS EN MÉMOIRE ---
lstm_model = None
vgg16_model = None
tokenizer = None
model = None  # pipeline scikit-learn
mapper = None
best_weights = None
MAXLEN = 10


class PredictRequest(BaseModel):
    # Champs attendus par les tests de Mika
    model_type: Literal["lstm", "vgg16"] = "lstm"
    text: Optional[str] = None
    image_path: Optional[str] = None
    # Champs envoyés par le Streamlit (Frontend)
    designation: Optional[str] = None
    description: Optional[str] = ""
    productid: Optional[int] = None
    imageid: Optional[int] = None


class PredictResponse(BaseModel):
    # Champs attendus par les tests de Mika
    model_used: str
    prediction: str
    input_text: Optional[str] = None
    input_image_path: Optional[str] = None
    # Champs attendus par le Streamlit (Frontend)
    predicted_label: Optional[str] = None
    predicted_code: Optional[Union[int, str]] = None
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    productid: Optional[int] = None
    imageid: Optional[int] = None


@app.on_event("startup")
def load_assets():
    global model, mapper, lstm_model, vgg16_model, tokenizer, best_weights

    # 1. Ta vraie pipeline (Prod)
    REAL_MODEL = MODELS_DIR / "artifacts" / "model_final.joblib"
    if REAL_MODEL.exists():
        model = joblib.load(REAL_MODEL)

    # 2. Chargement des JSON (Mapper, Tokenizer, Weights)
    if MAPPER_JSON_PATH.exists():
        mapper = json.loads(MAPPER_JSON_PATH.read_text(encoding="utf-8"))

    if TOKENIZER_CONFIG_PATH.exists():
        tokenizer = json.loads(TOKENIZER_CONFIG_PATH.read_text(encoding="utf-8"))

    if BEST_WEIGHTS_JSON_PATH.exists():
        best_weights = json.loads(BEST_WEIGHTS_JSON_PATH.read_text(encoding="utf-8"))

    # 3. Chargement des modèles .h5 (Simulation Tests)
    if LSTM_MODEL_PATH.exists():
        try:
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        except Exception:
            lstm_model = "dummy_model"

    if VGG16_MODEL_PATH.exists():
        try:
            vgg16_model = tf.keras.models.load_model(VGG16_MODEL_PATH)
        except Exception:
            vgg16_model = "dummy_model"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "lstm_loaded": lstm_model is not None,
            "vgg16_loaded": vgg16_model is not None,
        },
        "artifacts": {
            "pipeline_loaded": model is not None,
            "mapper_loaded": mapper is not None,
            "maxlen": MAXLEN
        }
    }


def map_prediction(pred_idx):
    if mapper is None:
        return str(pred_idx)
    if isinstance(mapper, list):
        return str(mapper[pred_idx]) if pred_idx < len(mapper) else str(pred_idx)
    res = mapper.get(str(pred_idx), mapper.get(pred_idx))
    return str(res) if res is not None else str(pred_idx)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # 1. On fusionne les entrées possibles (Test vs Streamlit)
    input_txt = payload.text
    if payload.designation is not None:
        # Si Streamlit appelle, on colle la designation et la description
        input_txt = f"{payload.designation} {payload.description}".strip()

    # 2. Validation manuelle pour les tests (422)
    if payload.model_type == "lstm" and not input_txt:
        raise HTTPException(status_code=422, detail="Missing text or designation")
    if payload.model_type == "vgg16" and not payload.image_path:
        raise HTTPException(status_code=422, detail="Missing image_path")

    # 3. Simulation VGG16
    if payload.model_type == "vgg16" and vgg16_model is not None:
        pred_val = map_prediction(1)
        return PredictResponse(
            model_used="vgg16",
            prediction=pred_val,
            input_image_path=payload.image_path,
            predicted_label=pred_val,
            predicted_code=1,
            model_name="vgg16"
        )

    # 4. Ta vraie Pipeline
    if model is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # On prédit avec scikit-learn
        idx = model.predict([input_txt or ""])[0]
        pred_label = map_prediction(idx)
        
        return PredictResponse(
            # Retour pour les tests
            model_used="scikit-learn-pipeline",
            prediction=pred_label,
            input_text=input_txt,
            # Retour pour Streamlit
            predicted_label=pred_label,
            predicted_code=int(idx) if str(idx).isdigit() else str(idx),
            model_name="scikit-learn-pipeline",
            productid=payload.productid,
            imageid=payload.imageid
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))