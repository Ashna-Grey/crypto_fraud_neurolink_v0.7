"""Fast smoke test for the bundled B6b model artifact."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.isolation_forest_utils import FEATURES_B6B, load_bundle  # noqa: E402
from inference.predict import score_dataframe  # noqa: E402


def main():
    bundle = load_bundle(ROOT / "artifacts" / "isolation_forest_b6b.joblib")
    assert len(bundle["feature_columns"]) == 20
    assert bundle["model"].n_features_in_ == 20
    assert np.all(np.diff(bundle["score_reference_sorted"]) >= 0)
    assert set(bundle["feature_reference_sorted"]) == set(FEATURES_B6B)
    assert len(bundle["ensemble_members"]) >= 2

    row = {feature: bundle["medians"][feature] for feature in FEATURES_B6B}
    sample = pd.DataFrame([row, row.copy()])
    sample["wallet_address"] = ["example_wallet_1", "example_wallet_2"]
    result = score_dataframe(sample, bundle)

    assert len(result) == 2
    assert "wallet_address" in result.columns
    assert result["anomaly_score"].notna().all()
    assert result["anomaly_percentile"].between(0, 100).all()
    assert result["if_mu"].between(0, 1).all()
    assert result["if_sigma_squared"].ge(0).all()
    assert result["if_sigma"].ge(0).all()
    assert result["data_quality_flag"].eq("OK").all()
    assert result["input_quality_status"].eq("PASS").all()
    assert result["if_prediction"].eq(result["anomaly_priority_band"]).all()

    print("Smoke test passed:")
    print(" - model loads")
    print(" - 20-feature schema matches")
    print(" - OOF score reference is sorted")
    print(" - feature references exist")
    print(" - inference preserves wallet ID passthrough fields")
    print(" - sample rows scored with valid percentiles")
    print(" - if_mu in [0,1] and if_sigma_squared is non-negative")
    print(" - bootstrap disagreement outputs are present")
    print(" - explicit anomaly prediction and input-quality status are present")


if __name__ == "__main__":
    main()
