from __future__ import annotations

"""
Core model utilities for the California housing project.

Responsibilities of this module:
- Resolve the correct path to the saved sklearn model (.pkl) and model_info.json
- Load the model from disk
- Provide a small `ModelService` class with a convenient `predict_one()` method
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


# Project root = one level above `src/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary locations under the tracked `models/` directory
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model2_linear_regression.pkl"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"

# Alternative locations under notebooks (in case you re-run and save from Jupyter)
ALT_MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "model2_linear_regression.pkl"
ALT_MODEL_INFO_PATH = PROJECT_ROOT / "notebooks" / "models" / "model_info.json"


def resolve_existing_path(*candidates: Path) -> Path:
    """
    Return the first existing path from the list of candidates.

    This allows the service to work whether the model was saved under
    `models/` or `notebooks/models/`. If none exist, the first candidate
    is returned so the caller still gets a clear FileNotFoundError.
    """
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_model_info(path: Path = DEFAULT_MODEL_INFO_PATH) -> Dict[str, Any]:
    """Load model metadata (features and metrics) from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a sklearn model object that was serialized with joblib."""
    return joblib.load(path)


@dataclass(frozen=True)
class ModelService:
    """
    Thin wrapper around the sklearn model to standardize input / output.

    Attributes
    ----------
    model:
        The loaded sklearn model (LinearRegression in this project).
    features:
        Ordered list of feature names expected by the model.
    """

    model: Any
    features: List[str]

    def predict_one(self, payload: Dict[str, float]) -> float:
        """
        Run prediction for a single record represented as a dict.

        Parameters
        ----------
        payload:
            Mapping from feature name to numeric value. All required features
            in `self.features` must be present.
        """
        # Check that all required features are provided
        missing = [f for f in self.features if f not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Keep only known features and coerce values to float
        row = {f: float(payload[f]) for f in self.features}

        # Build a single-row DataFrame with the correct column order
        X = pd.DataFrame([row], columns=self.features)

        # Call the underlying sklearn model and unwrap the scalar result
        y_pred = self.model.predict(X)
        return float(y_pred[0])


def build_default_service(
    model_path: Path = resolve_existing_path(DEFAULT_MODEL_PATH, ALT_MODEL_PATH),
    model_info_path: Path = resolve_existing_path(DEFAULT_MODEL_INFO_PATH, ALT_MODEL_INFO_PATH),
) -> ModelService:
    """
    Factory that constructs a ModelService using the project's default files.

    It reads `model_info.json` to determine which features belong to model2,
    then loads the corresponding .pkl model from disk.
    """
    info = load_model_info(model_info_path)

    # We specifically use the feature list for "model2"
    features = info.get("model2_features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError("model_info.json missing valid 'model2_features' list")

    model = load_model(model_path)
    return ModelService(model=model, features=list(features))

