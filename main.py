# main.py
from pathlib import Path
from typing import Dict, List, Optional
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Carga de artefactos ---
ARTIFACTS_DIR = Path("model")
MODEL_PATH = ARTIFACTS_DIR / "wine_nb_pipeline.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_order.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise RuntimeError("No se encontraron los artefactos del modelo en ./model")

pipe = joblib.load(MODEL_PATH)
feature_order: List[str] = json.loads(FEATURES_PATH.read_text())
metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

# --- Esquema de entrada/salida ---
class WineInput(BaseModel):
    # 13 features con validaciones mínimas:
    alcohol: float = Field(..., gt=0)
    malic_acid: float = Field(..., gt=0)
    ash: float = Field(..., gt=0)
    alcalinity_of_ash: float = Field(..., gt=0)
    magnesium: float = Field(..., gt=0)
    total_phenols: float = Field(..., gt=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., gt=0)
    od280_od315_of_diluted_wines: float = Field(..., gt=0)
    proline: float = Field(..., gt=0)

class PredictResponse(BaseModel):
    class_id: int
    class_name: Optional[str] = None
    proba: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, str]] = None

# --- App FastAPI ---
app = FastAPI(
    title="Wine Classifier API",
    version="1.0.0",
    description="API de ejemplo para TAREA 2: clasifica vinos (dataset Wine de scikit-learn).",
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.name,
        "n_features": len(feature_order),
        "features": feature_order,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: WineInput):
    # 1) Armar vector en el orden correcto
    try:
        x = np.array([getattr(payload, f) for f in feature_order], dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error armando vector de entrada: {e}")

    # 2) Inferencia
    try:
        y_pred = int(pipe.predict(x)[0])
        proba = None
        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(x)[0]
            proba = {f"class_{i}": float(p_i) for i, p_i in enumerate(p)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

    # 3) Enriquecer respuesta
    class_name = None
    if metadata and "target_names" in metadata and y_pred < len(metadata["target_names"]):
        class_name = str(metadata["target_names"][y_pred])

    return PredictResponse(
        class_id=y_pred,
        class_name=class_name,
        proba=proba,
        metadata={"model_name": metadata.get("model_name", "wine_nb_pipeline")},
    )
