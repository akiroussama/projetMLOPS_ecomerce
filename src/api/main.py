import json
import joblib
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# je lance lappli
app = FastAPI(title="projetMLOPS_ecomerce API", version="0.5.0")

# mes chemins (on garde les memes)
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "artifacts" / "model_final.joblib"
MAPPER_PATH = MODELS_DIR / "label_mapping.json"

# ressources en memoire
model = None
mapper = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    input_text: str

@app.on_event("startup")
def load_assets():
    global model, mapper
    
    # je charge la pipeline complete (contient deja le vectorizer)
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("✅ pipeline complete chargee")
        
    # je charge le mapping
    if MAPPER_PATH.exists():
        with open(MAPPER_PATH, "r", encoding="utf-8") as f:
            mapper = json.load(f)
        print("✅ mapping charge")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts": {
            "pipeline_loaded": model is not None,
            "mapper_loaded": mapper is not None
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="pipeline non chargee")

    try:
        # 1. je donne le texte direct a la pipeline (pas besoin de vectorizer manuel !)
        # on met le texte dans une liste [text] car le modele veut un iterable
        pred_idx = model.predict([payload.text])[0]
        
        # 2. je mappe le resultat
        prediction_label = str(pred_idx)
        if mapper:
            prediction_label = mapper.get(str(pred_idx), mapper.get(pred_idx, str(pred_idx)))

        return PredictResponse(
            prediction=prediction_label,
            input_text=payload.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erreur: {str(e)}")