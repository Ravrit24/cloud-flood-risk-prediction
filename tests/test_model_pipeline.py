"""
Unit and Integration Tests for Flood Risk Prediction Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocess import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    validate_input_data,
    load_raw_dataset,
    create_preprocessor
)
from src.evaluate import assign_risk_tier, RISK_TIERS
from src.predict import (
    load_trained_pipeline,
    predict_flood_risk,
    predict_batch
)


class TestFloodRiskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_data_path = os.path.join(ROOT_DIR, "data", "raw", "FloodPrediction.csv")
        cls.model_path = os.path.join(ROOT_DIR, "models", "flood_risk_pipeline.joblib")
        cls.pipeline = load_trained_pipeline(cls.model_path)

        cls.valid_sample = {
            "Month": 7,
            "Rainfall": 450.0,
            "Relative_Humidity": 85.0,
            "Cloud_Coverage": 6.2,
            "Bright_Sunshine": 3.1,
            "ALT": 8.0,
            "Wind_Speed": 3.2,
            "Max_Temp": 33.0,
            "Min_Temp": 25.0
        }

    def test_feature_specs_completeness(self):
        """Ensure all required features have valid metadata."""
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, FEATURE_SPECS)
            spec = FEATURE_SPECS[feat]
            self.assertIn("min", spec)
            self.assertIn("max", spec)
            self.assertIn("unit", spec)
            self.assertLess(spec["min"], spec["max"])
            self.assertTrue(spec["min"] <= spec["default"] <= spec["max"])

    def test_input_validation_valid_input(self):
        """Valid inputs must pass validation cleanly."""
        is_valid, errors = validate_input_data(self.valid_sample)
        self.assertTrue(is_valid, f"Expected valid, got errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_input_validation_missing_feature(self):
        """Missing required feature should fail validation."""
        incomplete = self.valid_sample.copy()
        del incomplete["Rainfall"]
        is_valid, errors = validate_input_data(incomplete)
        self.assertFalse(is_valid)
        self.assertTrue(any("Rainfall" in e for e in errors))

    def test_input_validation_out_of_bounds(self):
        """Out of bounds values (e.g. negative rainfall, humidity > 100) must fail."""
        invalid_cases = [
            ("Rainfall", -10.0),
            ("Relative_Humidity", 120.0),
            ("Month", 13),
            ("Cloud_Coverage", 9.5),
            ("Bright_Sunshine", 25.0)
        ]
        for key, bad_val in invalid_cases:
            bad_input = self.valid_sample.copy()
            bad_input[key] = bad_val
            is_valid, errors = validate_input_data(bad_input)
            self.assertFalse(is_valid, f"Expected failure for {key}={bad_val}")
            self.assertTrue(len(errors) > 0)

    def test_input_validation_physical_temperature_inconsistency(self):
        """Physical check: Min Temp cannot exceed Max Temp."""
        bad_temp = self.valid_sample.copy()
        bad_temp["Min_Temp"] = 35.0
        bad_temp["Max_Temp"] = 25.0
        is_valid, errors = validate_input_data(bad_temp)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot exceed" in e.lower() for e in errors))

    def test_risk_tier_categorization(self):
        """Test probability categorization into Low, Medium, and High tiers."""
        self.assertEqual(assign_risk_tier(0.05), "Low")
        self.assertEqual(assign_risk_tier(0.249), "Low")
        self.assertEqual(assign_risk_tier(0.25), "Medium")
        self.assertEqual(assign_risk_tier(0.45), "Medium")
        self.assertEqual(assign_risk_tier(0.599), "Medium")
        self.assertEqual(assign_risk_tier(0.60), "High")
        self.assertEqual(assign_risk_tier(0.95), "High")

    def test_end_to_end_prediction_structure(self):
        """Verify full output dictionary returned by predict_flood_risk."""
        result = predict_flood_risk(self.valid_sample, pipeline=self.pipeline)
        self.assertTrue(result["is_valid"])
        self.assertIn(result["risk_tier"], ["Low", "Medium", "High"])
        self.assertIn("flood_probability", result)
        self.assertTrue(0.0 <= result["flood_probability"] <= 1.0)
        self.assertIn("risk_badge", result)
        self.assertIn("guidance", result)
        self.assertIn("recommendation", result)
        self.assertIn("disclaimer", result)
        self.assertIn("EDUCATIONAL NOTICE", result["disclaimer"])
        self.assertIsInstance(result["contributing_factors"], list)

    def test_graceful_handling_of_invalid_prediction_input(self):
        """Verify that invalid inputs return an error dictionary without raising unhandled exceptions."""
        bad_input = self.valid_sample.copy()
        bad_input["Rainfall"] = -50.0
        result = predict_flood_risk(bad_input, pipeline=self.pipeline)
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["risk_tier"], "Invalid Input")
        self.assertIsNone(result["flood_probability"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_batch_prediction(self):
        """Verify batch dataframe prediction functionality."""
        df_test = pd.DataFrame([
            self.valid_sample,
            {**self.valid_sample, "Rainfall": 10.0, "Relative_Humidity": 55.0, "Month": 1}
        ])
        df_out = predict_batch(df_test, pipeline=self.pipeline)
        self.assertEqual(len(df_out), 2)
        self.assertIn("Predicted_Flood_Probability", df_out.columns)
        self.assertIn("Risk_Tier", df_out.columns)
        self.assertIn("Predicted_Class", df_out.columns)


if __name__ == "__main__":
    unittest.main()
