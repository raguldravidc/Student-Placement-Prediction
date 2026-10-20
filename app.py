from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "placement_model.joblib"

app = FastAPI(
    title="Student Placement Prediction API",
    description="Educational API that predicts placement status from student academic and skill features.",
    version="1.0.0",
)

class StudentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    CGPA: float = Field(ge=0, le=10, examples=[8.2])
    Aptitude_Score: float = Field(ge=0, le=100, examples=[75])
    Technical_Score: float = Field(ge=0, le=100, examples=[80])
    Communication_Score: float = Field(ge=0, le=100, examples=[78])
    Internships: int = Field(ge=0, le=20, examples=[1])
    Projects: int = Field(ge=0, le=50, examples=[2])
    Backlogs: int = Field(ge=0, le=50, examples=[0])

_model_bundle = None

@app.on_event("startup")
def load_model():
    global _model_bundle
    if MODEL_PATH.exists():
        _model_bundle = joblib.load(MODEL_PATH)
    else:
        _model_bundle = None

@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model_bundle is not None
    }

@app.post("/predict")
def predict(student: StudentInput):
    if _model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run train_model.py first to create placement_model.joblib."
        )
    features = _model_bundle["features"]
    row = pd.DataFrame([student.model_dump()])[features]
    pipeline = _model_bundle["pipeline"]
    predicted_class = int(pipeline.predict(row)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0][1])

    return {
        "prediction": "Placed" if predicted_class == 1 else "Not Placed",
        "placed": predicted_class,
        "placement_probability": round(probability, 4) if probability is not None else None,
        "notice": "This is an educational estimate, not a guarantee of an actual placement outcome."
    }
