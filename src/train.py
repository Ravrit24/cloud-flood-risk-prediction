"""
Model Training and Artifact Serialization Script.

Trains a scikit-learn Machine Learning Pipeline for Flood Risk Classification,
computes evaluation metrics on a held-out test set, and exports production artifacts.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import (
    load_raw_dataset,
    create_preprocessor,
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    BINARY_TARGET_COLUMN
)
from src.evaluate import evaluate_model_performance, RISK_TIERS
from data.download_data import download_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "FloodPrediction.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PIPELINE_OUT_PATH = os.path.join(MODELS_DIR, "flood_risk_pipeline.joblib")
METRICS_OUT_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
METADATA_OUT_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")


def train_and_export_model(
    data_path: str = DATA_PATH,
    n_estimators: int = 120,
    random_state: int = 42
) -> Pipeline:
    """
    Train flood risk classification pipeline and serialize model artifacts.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Ensure dataset is downloaded
    if not os.path.exists(data_path):
        print(f"[INFO] Dataset not found at {data_path}. Downloading...")
        download_dataset()

    print(f"[INFO] Loading dataset from: {data_path}")
    df = load_raw_dataset(data_path)
    print(f"       Loaded {len(df):,} total records. Flood incidence rate: {df[BINARY_TARGET_COLUMN].mean():.2%}")

    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET_COLUMN]

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"[INFO] Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Define Preprocessing & Classifier Pipeline
    preprocessor = create_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    print("[INFO] Fitting Random Forest Classification Pipeline...")
    full_pipeline.fit(X_train, y_train)

    # Predict on Test Set
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]

    # Feature importances from the classifier
    rf_model = full_pipeline.named_steps["classifier"]
    feature_importances = rf_model.feature_importances_

    # Compute Comprehensive Evaluation Metrics
    metrics = evaluate_model_performance(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=FEATURE_COLUMNS,
        feature_importances=feature_importances
    )

    # Export Pipeline Artifact
    joblib.dump(full_pipeline, PIPELINE_OUT_PATH)
    print(f"[SUCCESS] Saved model pipeline artifact to: {PIPELINE_OUT_PATH}")

    # Export Evaluation Metrics
    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Saved evaluation metrics to: {METRICS_OUT_PATH}")

    # Export Metadata Schema
    metadata = {
        "features": FEATURE_SPECS,
        "feature_order": FEATURE_COLUMNS,
        "risk_tiers": RISK_TIERS,
        "model_architecture": "Pipeline(ColumnTransformer(SimpleImputer, StandardScaler), RandomForestClassifier)",
        "dataset_name": "Historical Flood & Meteorological Records (Gauhar et al., IEEE 2021)",
        "disclaimer": "Educational statistical model for demonstration and coursework. Not certified for operational life-safety flood forecasting."
    }
    with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Saved metadata schema to: {METADATA_OUT_PATH}")

    # Display Metrics Summary
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION SUMMARY (TEST SET)           ")
    print("=" * 60)
    summary = metrics["summary"]
    print(f"Accuracy:        {summary['accuracy'] * 100:.2f}%")
    print(f"Precision (Flood): {summary['precision_flood'] * 100:.2f}%")
    print(f"Recall (Flood):    {summary['recall_flood'] * 100:.2f}%")
    print(f"F1-Score (Flood):  {summary['f1_flood'] * 100:.2f}%")
    print(f"Macro F1-Score:    {summary['macro_f1'] * 100:.2f}%")
    print(f"ROC-AUC Score:     {summary['roc_auc']:.4f}")
    print(f"Brier Loss:        {summary['brier_score']:.4f}")

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN: {cm['true_negative']} | FP: {cm['false_positive']}")
    print(f"  FN: {cm['false_negative']} | TP: {cm['true_positive']}")

    print("\nRisk Tier Calibration:")
    for tier, data in metrics["risk_tier_calibration"].items():
        print(f"  [{tier.upper():6s}] {data['count']:4d} samples ({data['percentage']:5.1f}%) | "
              f"Mean Prob: {data['mean_predicted_prob']*100:5.1f}% | "
              f"Actual Historical Flood Rate: {data['empirical_flood_rate']*100:5.1f}%")
    print("=" * 60 + "\n")

    return full_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flood Risk Prediction Model")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to raw CSV dataset")
    parser.add_argument("--n-estimators", type=int, default=120, help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_and_export_model(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        random_state=args.seed
    )
