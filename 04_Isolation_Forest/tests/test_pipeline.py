"""Unit tests for schema validation and prediction quality flags."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.isolation_forest_utils import FEATURES_B6B, load_bundle  # noqa: E402
from src.live_input import SIGNED_LOG1P_FEATURES, prepare_raw_live_input  # noqa: E402
from inference.predict import score_dataframe  # noqa: E402


class PredictionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(ROOT / "artifacts" / "isolation_forest_b6b.joblib")
        base = {feature: cls.bundle["medians"][feature] for feature in FEATURES_B6B}
        cls.sample = pd.DataFrame([base, base.copy()])
        cls.sample["wallet_address"] = ["wallet_a", "wallet_b"]

    def test_schema_and_wallet_passthrough(self):
        result = score_dataframe(self.sample, self.bundle)
        self.assertEqual(len(result), 2)
        self.assertIn("wallet_address", result.columns)
        self.assertEqual(result["wallet_address"].tolist(), ["wallet_a", "wallet_b"])
        self.assertTrue(np.isfinite(result["anomaly_score"]).all())
        self.assertTrue(result["anomaly_percentile"].between(0, 100).all())
        self.assertTrue(result["if_mu"].between(0, 1).all())
        self.assertTrue(np.isfinite(result["if_sigma_squared"]).all())
        self.assertTrue((result["if_sigma_squared"] >= 0).all())
        self.assertTrue((result["if_sigma"] >= 0).all())
        self.assertIn("if_mu_semantics", result.columns)
        self.assertIn("if_prediction", result.columns)
        self.assertIn("input_quality_status", result.columns)
        self.assertEqual(result["input_quality_status"].tolist(), ["PASS", "PASS"])


    def test_raw_live_format_applies_verified_transform(self):
        raw_like = self.sample.copy()
        for feature in SIGNED_LOG1P_FEATURES:
            values = raw_like[feature].to_numpy(dtype=float)
            raw_like[feature] = np.sign(values) * np.expm1(np.abs(values))
        raw_like = raw_like.rename(columns={FEATURES_B6B[0]: f" {FEATURES_B6B[0]} "})
        raw_like["Address"] = ["0xabc", "0xdef"]
        prepared, audit = prepare_raw_live_input(raw_like)
        self.assertIn(FEATURES_B6B[0], prepared.columns)
        self.assertEqual(audit["input_format"], "raw-live")
        self.assertEqual(audit["transformed_feature_count"], len(SIGNED_LOG1P_FEATURES))
        result = score_dataframe(raw_like, self.bundle, input_format="raw-live")
        self.assertEqual(result["Address"].tolist(), ["0xabc", "0xdef"])
        baseline = score_dataframe(self.sample, self.bundle)
        np.testing.assert_allclose(result["if_mu"], baseline["if_mu"], rtol=1e-6, atol=1e-6)

    def test_all_features_missing_is_unusable_not_scored(self):
        data = self.sample.iloc[[0]].copy()
        data.loc[:, FEATURES_B6B] = np.nan
        result = score_dataframe(data, self.bundle)
        self.assertEqual(result.loc[0, "input_quality_status"], "UNUSABLE")
        self.assertEqual(result.loc[0, "if_prediction"], "NOT_SCORED_INPUT_UNUSABLE")
        self.assertTrue(np.isnan(result.loc[0, "if_mu"]))
        self.assertEqual(result.loc[0, "missing_feature_count"], len(FEATURES_B6B))

    def test_missing_feature_ratio_is_reported(self):
        data = self.sample.copy()
        data.loc[0, FEATURES_B6B[:2]] = np.nan
        result = score_dataframe(data, self.bundle)
        self.assertAlmostEqual(result.loc[0, "missing_feature_ratio"], 2 / len(FEATURES_B6B))
        self.assertEqual(result.loc[0, "input_quality_status"], "WARN")

    def test_missing_required_feature_rejected(self):
        bad = self.sample.drop(columns=[FEATURES_B6B[0]])
        with self.assertRaisesRegex(ValueError, "missing required B6b features"):
            score_dataframe(bad, self.bundle)

    def test_missing_value_imputed_and_flagged(self):
        data = self.sample.copy()
        data.loc[0, FEATURES_B6B[0]] = np.nan
        result = score_dataframe(data, self.bundle)
        self.assertIn("MISSING_VALUES_IMPUTED", result.loc[0, "data_quality_flag"])
        self.assertEqual(result.loc[0, "missing_feature_count"], 1)

    def test_out_of_training_range_flagged(self):
        data = self.sample.copy()
        feature = FEATURES_B6B[0]
        data.loc[0, feature] = self.bundle["feature_max"][feature] + 1e6
        result = score_dataframe(data, self.bundle)
        self.assertIn("OUTSIDE_TRAINING_RANGE", result.loc[0, "data_quality_flag"])
        self.assertGreater(result.loc[0, "outside_training_minmax_count"], 0)

    def test_invalid_non_numeric_value_rejected(self):
        data = self.sample.copy()
        data[FEATURES_B6B[0]] = data[FEATURES_B6B[0]].astype(object)
        data.loc[0, FEATURES_B6B[0]] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "Non-numeric values"):
            score_dataframe(data, self.bundle)

    def test_infinite_value_rejected(self):
        data = self.sample.copy()
        data.loc[0, FEATURES_B6B[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            score_dataframe(data, self.bundle)


if __name__ == "__main__":
    unittest.main()
