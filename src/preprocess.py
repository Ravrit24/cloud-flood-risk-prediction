"""
Data Preprocessing and Pipeline Definition Module.

This module provides data cleaning, schema validation, and scikit-learn
preprocessing components for the Flood Risk Prediction System.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Feature definition schema: name, label, unit, min, max, default
FEATURE_SPECS: Dict[str, Dict[str, Any]] = {
    "Month": {
        "label": "Month of Observation",
        "unit": "Month (1-12)",
        "min": 1,
        "max": 12,
        "default": 7,
        "step": 1,
        "description": "Calendar month (1 = Jan, ..., 7 = July monsoon peak, 12 = Dec)."
    },
    "Rainfall": {
        "label": "Monthly Precipitation",
        "unit": "mm",
        "min": 0.0,
        "max": 2500.0,
        "default": 350.0,
        "step": 5.0,
        "description": "Total monthly rainfall recorded at the meteorological station."
    },
    "Relative_Humidity": {
        "label": "Relative Humidity",
        "unit": "%",
        "min": 10.0,
        "max": 100.0,
        "default": 82.0,
        "step": 1.0,
        "description": "Average relative atmospheric humidity percentage."
    },
    "Cloud_Coverage": {
        "label": "Cloud Coverage",
        "unit": "Oktas (0-8)",
        "min": 0.0,
        "max": 8.0,
        "default": 5.5,
        "step": 0.1,
        "description": "Cloud cover in eighths of the celestial dome (0 = clear sky, 8 = completely overcast)."
    },
    "Bright_Sunshine": {
        "label": "Bright Sunshine Duration",
        "unit": "Hours/day",
        "min": 0.0,
        "max": 14.0,
        "default": 4.5,
        "step": 0.1,
        "description": "Daily average sunshine hours (decreases sharply during stormy or monsoon conditions)."
    },
    "ALT": {
        "label": "Station Altitude / Elevation",
        "unit": "Meters",
        "min": 0.0,
        "max": 500.0,
        "default": 10.0,
        "step": 1.0,
        "description": "Topographical elevation above sea level (low-lying flatlands face higher inundation risk)."
    },
    "Wind_Speed": {
        "label": "Average Wind Speed",
        "unit": "m/s",
        "min": 0.0,
        "max": 35.0,
        "default": 2.5,
        "step": 0.1,
        "description": "Average wind speed recorded at the monitoring station."
    },
    "Max_Temp": {
        "label": "Maximum Monthly Temperature",
        "unit": "°C",
        "min": 15.0,
        "max": 50.0,
        "default": 33.5,
        "step": 0.5,
        "description": "Peak monthly recorded ambient temperature."
    },
    "Min_Temp": {
        "label": "Minimum Monthly Temperature",
        "unit": "°C",
        "min": 0.0,
        "max": 35.0,
        "default": 24.5,
        "step": 0.5,
        "description": "Lowest monthly recorded ambient temperature."
    }
}

FEATURE_COLUMNS: List[str] = list(FEATURE_SPECS.keys())
TARGET_COLUMN: str = "Flood?"
BINARY_TARGET_COLUMN: str = "Flood_Binary"


def get_feature_columns() -> List[str]:
    """Return the ordered list of model input features."""
    return FEATURE_COLUMNS.copy()


def get_feature_specs() -> Dict[str, Dict[str, Any]]:
    """Return the feature specifications and bounds dictionary."""
    return {k: v.copy() for k, v in FEATURE_SPECS.items()}


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load raw historical CSV dataset and apply initial cleaning.

    Args:
        csv_path: Absolute or relative path to raw CSV file.

    Returns:
        Cleaned pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Run data/download_data.py first.")

    df = pd.read_csv(csv_path)

    # Validate columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column '{TARGET_COLUMN}'")

    # In historical flood dataset, 1 indicates recorded inundation, blank/NaN indicates no flood event.
    df[BINARY_TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer for numerical feature preprocessing.
    Applies median imputation followed by standard scaling.

    Returns:
        Configured ColumnTransformer instance.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, FEATURE_COLUMNS)
        ],
        remainder="drop"
    )
    return preprocessor


def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate user input against physical and domain bounds.

    Args:
        input_data: Dictionary of feature name to value.

    Returns:
        Tuple of (is_valid: bool, error_messages: List[str]).
    """
    errors = []

    for feat, specs in FEATURE_SPECS.items():
        if feat not in input_data:
            errors.append(f"Missing required parameter: {specs['label']} ('{feat}')")
            continue

        val = input_data[feat]
        try:
            val_num = float(val)
        except (ValueError, TypeError):
            errors.append(f"{specs['label']} must be a valid number, got {val}.")
            continue

        if val_num < specs["min"] or val_num > specs["max"]:
            errors.append(
                f"{specs['label']} ({val_num} {specs['unit']}) is out of allowable range "
                f"[{specs['min']}, {specs['max']}]."
            )

    # Physical consistency check: Max Temp >= Min Temp
    if "Max_Temp" in input_data and "Min_Temp" in input_data:
        try:
            max_t = float(input_data["Max_Temp"])
            min_t = float(input_data["Min_Temp"])
            if min_t > max_t:
                errors.append(
                    f"Physical inconsistency: Minimum temperature ({min_t}°C) "
                    f"cannot exceed Maximum temperature ({max_t}°C)."
                )
        except (ValueError, TypeError):
            pass

    return len(errors) == 0, errors
