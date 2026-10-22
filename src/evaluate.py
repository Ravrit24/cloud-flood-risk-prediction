"""
Model Evaluation and Metrics Module.

Computes comprehensive classification metrics, confusion matrix, ROC-AUC,
risk tier calibration statistics, and feature importances.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

# Scientifically calibrated risk tier thresholds based on empirical historical base rates:
# Base rate ~20.1% flood events in historical records.
RISK_TIERS = {
    "Low": {"min_prob": 0.0, "max_prob": 0.25, "color": "#10B981", "badge": "LOW RISK (NORMAL)"},
    "Medium": {"min_prob": 0.25, "max_prob": 0.60, "color": "#F59E0B", "badge": "MEDIUM RISK (ADVISORY)"},
    "High": {"min_prob": 0.60, "max_prob": 1.0, "color": "#EF4444", "badge": "HIGH RISK (WARNING)"}
}


def assign_risk_tier(probability: float) -> str:
    """Assign a categorical risk tier from calibrated predicted probability."""
    p = float(np.clip(probability, 0.0, 1.0))
    if p < RISK_TIERS["Low"]["max_prob"]:
        return "Low"
    elif p < RISK_TIERS["Medium"]["max_prob"]:
        return "Medium"
    else:
        return "High"


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: List[str] = None,
    feature_importances: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics dictionary suitable for JSON serialization and app display.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Risk tier empirical calibration breakdown
    tiers = [assign_risk_tier(p) for p in y_prob]
    df_eval = pd.DataFrame({"y_true": y_true, "prob": y_prob, "tier": tiers})

    tier_stats = {}
    for tier in ["Low", "Medium", "High"]:
        subset = df_eval[df_eval["tier"] == tier]
        count = len(subset)
        actual_flood_rate = float(subset["y_true"].mean()) if count > 0 else 0.0
        mean_predicted_prob = float(subset["prob"].mean()) if count > 0 else 0.0
        tier_stats[tier] = {
            "count": count,
            "percentage": float((count / len(df_eval)) * 100) if len(df_eval) > 0 else 0.0,
            "empirical_flood_rate": round(actual_flood_rate, 4),
            "mean_predicted_prob": round(mean_predicted_prob, 4),
            "threshold_range": [RISK_TIERS[tier]["min_prob"], RISK_TIERS[tier]["max_prob"]],
            "color": RISK_TIERS[tier]["color"]
        }

    # Feature importances formatted as list of dicts
    importance_list = []
    if feature_names is not None and feature_importances is not None:
        for name, imp in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
            importance_list.append({
                "feature": name,
                "importance": round(float(imp), 4),
                "importance_pct": round(float(imp) * 100, 2)
            })

    metrics = {
        "summary": {
            "test_samples": int(len(y_true)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_flood": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall_flood": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_flood": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["No Flood (0)", "Flood Inundation (1)"],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
            "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
        },
        "classification_report": report,
        "risk_tier_calibration": tier_stats,
        "feature_importances": importance_list
    }

    return metrics
