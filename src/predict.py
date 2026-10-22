"""
Inference and Early Warning Prediction Service.

Provides input validation, model inference, risk tier categorization,
key driver factor identification, and educational guidance generation.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.preprocess import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    validate_input_data
)
from src.evaluate import RISK_TIERS, assign_risk_tier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "flood_risk_pipeline.joblib")
DEFAULT_METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

# Cached model instance
_CACHED_PIPELINE = None


def load_trained_pipeline(model_path: Optional[str] = None):
    """Load or retrieve cached trained scikit-learn pipeline."""
    global _CACHED_PIPELINE
    path = model_path or DEFAULT_MODEL_PATH

    if _CACHED_PIPELINE is not None and model_path is None:
        return _CACHED_PIPELINE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Please run 'python src/train.py' first."
        )

    pipeline = joblib.load(path)
    if model_path is None:
        _CACHED_PIPELINE = pipeline
    return pipeline


def load_evaluation_metrics(metrics_path: Optional[str] = None) -> Dict[str, Any]:
    """Load pre-computed model evaluation metrics."""
    path = metrics_path or DEFAULT_METRICS_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_contributing_factors(input_dict: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Identify key meteorological and terrain factors contributing toward higher or lower risk,
    comparing user inputs against climatological thresholds.
    """
    drivers = []

    # High rainfall check (> 300 mm monthly)
    rainfall = float(input_dict.get("Rainfall", 0))
    if rainfall >= 450:
        drivers.append({
            "feature": "Rainfall",
            "impact": "High Positive",
            "message": f"Extreme precipitation ({rainfall:.1f} mm) is a primary hydrologic driver of surface runoff and channel overflow."
        })
    elif rainfall >= 250:
        drivers.append({
            "feature": "Rainfall",
            "impact": "Moderate Positive",
            "message": f"Substantial precipitation ({rainfall:.1f} mm) elevates soil moisture and stream discharge."
        })
    elif rainfall <= 50:
        drivers.append({
            "feature": "Rainfall",
            "impact": "Negative (Protective)",
            "message": f"Minimal rainfall ({rainfall:.1f} mm) strongly suppresses flood likelihood."
        })

    # Relative humidity check
    humidity = float(input_dict.get("Relative_Humidity", 50))
    if humidity >= 85:
        drivers.append({
            "feature": "Relative Humidity",
            "impact": "Moderate Positive",
            "message": f"High atmospheric moisture ({humidity:.1f}%) signals saturated air and sustained rainy spells."
        })

    # Cloud coverage (oktas)
    cloud = float(input_dict.get("Cloud_Coverage", 0))
    if cloud >= 6.0:
        drivers.append({
            "feature": "Cloud Coverage",
            "impact": "Moderate Positive",
            "message": f"Dense overcast sky ({cloud:.1f} oktas) correlates with persistent monsoon depressions."
        })

    # Sunshine hours (low sunshine = active monsoon)
    sun = float(input_dict.get("Bright_Sunshine", 6))
    if sun <= 3.0:
        drivers.append({
            "feature": "Bright Sunshine",
            "impact": "Moderate Positive",
            "message": f"Low sunshine ({sun:.1f} hrs/day) indicates sustained cloud cover and active rainstorms."
        })

    # Altitude / Elevation
    alt = float(input_dict.get("ALT", 15))
    if alt <= 8.0:
        drivers.append({
            "feature": "Altitude",
            "impact": "Moderate Positive",
            "message": f"Low-lying elevation ({alt:.1f} m above sea level) leaves terrain vulnerable to poor drainage and tidal/fluvial backflow."
        })
    elif alt >= 30.0:
        drivers.append({
            "feature": "Altitude",
            "impact": "Negative (Protective)",
            "message": f"Elevated topography ({alt:.1f} m) facilitates gravity drainage."
        })

    # Monsoon month
    month = int(input_dict.get("Month", 1))
    if month in [6, 7, 8, 9]:
        drivers.append({
            "feature": "Month / Season",
            "impact": "Contextual",
            "message": f"Month {month} falls inside the peak monsoon window when major regional river systems carry maximum discharge."
        })

    return drivers


def predict_flood_risk(input_data: Dict[str, Any], pipeline=None) -> Dict[str, Any]:
    """
    Validate input, execute inference, and return risk categorization with educational guidance.

    Args:
        input_data: Mapping of feature names to numeric values.
        pipeline: Optional scikit-learn pipeline instance.

    Returns:
        Structured prediction result dictionary.
    """
    is_valid, errors = validate_input_data(input_data)
    if not is_valid:
        return {
            "is_valid": False,
            "errors": errors,
            "risk_tier": "Invalid Input",
            "flood_probability": None,
            "binary_prediction": None,
            "guidance": "Please correct the input parameters to match valid physical ranges."
        }

    # Load model
    if pipeline is None:
        pipeline = load_trained_pipeline()

    # Create ordered single-row DataFrame
    ordered_row = {col: [float(input_data[col])] for col in FEATURE_COLUMNS}
    df_input = pd.DataFrame(ordered_row)

    # Predict probability and class
    prob_array = pipeline.predict_proba(df_input)[0]
    p_flood = float(prob_array[1])
    binary_pred = int(pipeline.predict(df_input)[0])

    # Assign risk tier
    tier = assign_risk_tier(p_flood)
    tier_info = RISK_TIERS[tier]

    # Generate guidance and action recommendations
    if tier == "Low":
        guidance = (
            "Standard baseline conditions. Historical precipitation and atmospheric parameters "
            "indicate minimal likelihood of meteorological inundation. Normal water management practices apply."
        )
        recommendation = "Continue standard seasonal monitoring. No special flood mitigation required."
    elif tier == "Medium":
        guidance = (
            "Elevated flood risk watch. Weather conditions show heightened moisture, sustained cloud cover, "
            "or above-average rainfall. Localized pooling or drainage saturation may occur."
        )
        recommendation = "Inspect local storm drains, monitor community advisories, and prepare contingency flood response plans."
    else:  # High
        guidance = (
            "Severe flood threat detected. Environmental indicators match historical peak inundation episodes "
            "characterized by heavy precipitation surges, low sunshine, and terrain susceptibility."
        )
        recommendation = "Review emergency evacuation routes, secure critical assets to higher ground, and monitor official disaster warning bulletins immediately."

    # Identify driving factors
    drivers = analyze_contributing_factors(input_data)

    return {
        "is_valid": True,
        "errors": [],
        "risk_tier": tier,
        "risk_badge": tier_info["badge"],
        "risk_color": tier_info["color"],
        "flood_probability": round(p_flood, 4),
        "flood_probability_pct": round(p_flood * 100, 2),
        "binary_prediction": binary_pred,
        "binary_label": "Flood Inundation Likely" if binary_pred == 1 else "No Significant Inundation",
        "guidance": guidance,
        "recommendation": recommendation,
        "contributing_factors": drivers,
        "disclaimer": (
            "EDUCATIONAL NOTICE: This prediction is a statistical estimate produced by a machine learning "
            "model trained on historical meteorological records. It is NOT an official hydrological warning. "
            "Always consult national disaster management and meteorological authorities for life-safety advisories."
        )
    }


def predict_batch(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """
    Run inference on a batch DataFrame, appending prediction results.
    """
    if pipeline is None:
        pipeline = load_trained_pipeline()

    # Ensure required columns are present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Batch DataFrame missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS]
    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    tiers = [assign_risk_tier(p) for p in probs]

    df_out = df.copy()
    df_out["Predicted_Flood_Probability"] = np.round(probs, 4)
    df_out["Predicted_Flood_Prob_Pct"] = np.round(probs * 100, 2)
    df_out["Predicted_Class"] = preds
    df_out["Risk_Tier"] = tiers
    return df_out
