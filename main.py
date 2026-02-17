from __future__ import annotations

"""
FastAPI application exposing the trained housing price model as a REST API.

This file lives at the project root so it can be started easily with:

    uvicorn main:app --host 127.0.0.1 --port 8000

The actual model loading / prediction logic is implemented in `src/model.py`.
"""

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import DEFAULT_MODEL_INFO_PATH, build_default_service


# Create the FastAPI application instance
app = FastAPI(title="California Housing Model API", version="1.0.0")

# Enable CORS so a separate frontend (different port/origin) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """
    Request body for /predict.

    All fields correspond to the features used by model2 in model_info.json.
    """

    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictResponse(BaseModel):
    """Response body for /predict."""

    prediction: float
    units: str = "target (California housing dataset units)"
    features_used: Dict[str, float]


def _load_model_info() -> Dict[str, Any]:
    """Read metrics and feature list from model_info.json."""
    with DEFAULT_MODEL_INFO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# Construct a ModelService instance once when the application starts
service = build_default_service()


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health-check endpoint for monitoring."""
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """Return basic metadata about the loaded model (features + metrics)."""
    info = _load_model_info()
    return {
        "model": "model2_linear_regression.pkl",
        "features": service.features,
        "metrics": {
            "r2": info.get("model2_r2"),
            "rmse": info.get("model2_rmse"),
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, debug: Optional[bool] = False) -> PredictResponse:
    """
    Predict house price from the provided features.

    Set `debug=true` in query params during development to surface the
    underlying Python error instead of a generic HTTP 400.
    """
    try:
        # Convert Pydantic model into a plain dict
        payload = req.model_dump()
        # Delegate actual prediction to the ModelService
        pred = service.predict_one(payload)
        return PredictResponse(prediction=pred, features_used=payload)
    except Exception as e:
        if debug:
            # In debug mode, re-raise to see full stack trace in logs
            raise
        # For normal clients, return a clean 400 error
        raise HTTPException(status_code=400, detail=str(e))

