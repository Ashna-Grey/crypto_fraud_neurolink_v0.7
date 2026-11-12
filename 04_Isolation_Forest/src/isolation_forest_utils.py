"""Schema, validation, scoring-reference, and explanation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The original dataset order, retained for schema auditing.
FEATURES_23 = [
    "Time Diff between first and last (Mins)",
    "Total ERC20 tnxs",
    "Unique Received From Addresses",
    "ERC20 uniq rec addr",
    "ERC20 uniq rec token name",
    "total ether received",
    "avg val received",
    "Avg min between received tnx",
    "total transactions (including tnx to create contract)",
    "max value received",
    "ERC20 min val rec",
    "Received Tnx",
    "min value received",
    "total Ether sent",
    "ERC20 avg val rec",
    "ERC20 max val rec",
    "Sent tnx",
    "Avg min between sent tnx",
    "max val sent",
    "avg val sent",
    "Unique Sent To Addresses",
    "ERC20 uniq sent token name",
    "current_eth_balance",
]

EXCLUDED_B6B = [
    "Sent tnx",
    "Received Tnx",
    "total transactions (including tnx to create contract)",
]
FEATURES_B6B = [column for column in FEATURES_23 if column not in EXCLUDED_B6B]

MODEL_CONFIG = {
    "model_name": "IsolationForest_B6b",
    "algorithm": "sklearn.ensemble.IsolationForest",
    "n_estimators": 300,
    "contamination": "auto",
    "random_state": 42,
    "n_jobs": -1,
    "selected_feature_count": len(FEATURES_B6B),
    "excluded_features": EXCLUDED_B6B,
    "score_definition": "anomaly_score = -model.score_samples(X); higher means more anomalous",
    "score_percentile_reference": "member-specific out-of-bag anomaly scores from bootstrap Isolation Forest ensemble",
    "ensemble_size": 25,
    "mu_definition": "mean member-specific empirical anomaly percentile on [0, 1]; this is anomaly severity, not fraud probability",
    "sigma_squared_definition": "sample variance (ddof=1) of member-specific empirical anomaly percentiles; between-model disagreement, not calibrated predictive variance",
    "percentile_bands": {
        "ROUTINE": "below P90",
        "ELEVATED_ANOMALY": "P90 to below P95",
        "HIGH_ANOMALY": "P95 to below P99",
        "EXTREME_ANOMALY": "P99 or above",
    },
    "fraud_probability": False,
}


def read_and_validate_training_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read required features and preserve missing values for fold-safe imputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Training CSV contains no data rows.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    missing = [column for column in FEATURES_B6B if column not in frame.columns]
    if missing:
        raise ValueError("Input CSV is missing required B6b features: " + ", ".join(missing))

    numeric = frame[FEATURES_B6B].copy()
    for column in FEATURES_B6B:
        raw = numeric[column]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid_parse = raw.notna() & converted.isna()
        if invalid_parse.any():
            examples = raw[invalid_parse].head(3).tolist()
            raise ValueError(
                f"Non-numeric values in feature {column!r}; examples: {examples}"
            )
        numeric[column] = converted

    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Infinite values were found in the feature matrix.")

    missing_by_feature = numeric.isna().sum().to_dict()
    all_missing = [column for column in FEATURES_B6B if numeric[column].notna().sum() == 0]
    if all_missing:
        raise ValueError("Cannot impute entirely missing features: " + ", ".join(all_missing))

    # This audit is descriptive. Imputation for each OOF fold is performed
    # separately inside train.py to avoid fitting fold imputers on validation rows.
    audit_fill = numeric.fillna(numeric.median())
    audit = {
        "input_path_name": path.name,
        "rows": int(len(frame)),
        "input_column_count": int(frame.shape[1]),
        "feature_count_before_selection": int(sum(column in frame.columns for column in FEATURES_23)),
        "model_feature_count": len(FEATURES_B6B),
        "missing_values_before_imputation": int(numeric.isna().sum().sum()),
        "missing_values_by_feature": {key: int(value) for key, value in missing_by_feature.items()},
        "exact_duplicate_feature_rows_after_median_fill": int(audit_fill.duplicated(keep=False).sum()),
        "extra_duplicate_feature_copies_after_median_fill": int(audit_fill.duplicated(keep="first").sum()),
        "fold_imputation_is_fold_local": True,
    }
    return numeric, {"audit": audit}


def empirical_percentile(value: float, sorted_reference: np.ndarray) -> float:
    """Return a smoothed empirical CDF percentile in (0, 100)."""
    values = np.asarray(sorted_reference, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Score reference must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Score reference contains non-finite values.")
    if np.any(np.diff(values) < 0):
        raise ValueError("Score reference must be sorted ascending.")
    index = np.searchsorted(values, float(value), side="right")
    return float(100.0 * (index + 0.5) / (len(values) + 1.0))


def anomaly_band(percentile: float) -> str:
    """Map a reference percentile to an anomaly-priority band, not fraud risk."""
    if percentile >= 99:
        return "EXTREME_ANOMALY"
    if percentile >= 95:
        return "HIGH_ANOMALY"
    if percentile >= 90:
        return "ELEVATED_ANOMALY"
    return "ROUTINE"


def create_feature_explanation(
    row: pd.Series,
    feature_reference_sorted: dict[str, np.ndarray],
    top_n: int = 3,
) -> str:
    """Return marginal feature-tail observations, not tree-level attribution."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    items = []
    for feature, value in row.items():
        percentile = empirical_percentile(float(value), feature_reference_sorted[feature])
        tail = min(percentile, 100.0 - percentile)
        direction = "LOW" if percentile < 50.0 else "HIGH"
        items.append((tail, feature, float(value), percentile, direction))

    items.sort(key=lambda item: item[0])
    descriptions = []
    for _, feature, value, percentile, direction in items[:top_n]:
        descriptions.append(
            f"{feature}={value:.6g} ({direction}; reference percentile {percentile:.1f})"
        )
    return " | ".join(descriptions)


def load_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load and validate the saved model bundle."""
    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    bundle = joblib.load(path)
    required = {
        "model", "ensemble_members", "feature_columns", "medians", "score_reference_sorted",
        "feature_reference_sorted", "feature_min", "feature_max", "metadata"
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is incomplete; missing keys: {sorted(missing)}")
    if list(bundle["feature_columns"]) != FEATURES_B6B:
        raise ValueError("Model feature schema does not match package schema.")
    if int(bundle["model"].n_features_in_) != len(FEATURES_B6B):
        raise ValueError("Trained model feature count does not match the B6b schema.")
    members = bundle["ensemble_members"]
    if not isinstance(members, (list, tuple)) or len(members) < 2:
        raise ValueError(
            "Model artifact must contain at least two bootstrap ensemble members "
            "to estimate if_sigma_squared. Retrain with the current training.py."
        )
    for member_index, member in enumerate(members):
        member_required = {"model", "medians", "score_reference_sorted", "seed", "oob_rows"}
        member_missing = member_required - set(member)
        if member_missing:
            raise ValueError(
                f"Ensemble member {member_index} is missing keys: {sorted(member_missing)}"
            )
        member_reference = np.asarray(member["score_reference_sorted"], dtype=float)
        if member_reference.ndim != 1 or member_reference.size == 0 or not np.all(np.isfinite(member_reference)):
            raise ValueError(f"Ensemble member {member_index} has an invalid OOB score reference.")
        if np.any(np.diff(member_reference) < 0):
            raise ValueError(f"Ensemble member {member_index} OOB score reference is not sorted.")
        if int(member["model"].n_features_in_) != len(FEATURES_B6B):
            raise ValueError(f"Ensemble member {member_index} feature count mismatch.")
        if set(member["medians"]) != set(FEATURES_B6B):
            raise ValueError(f"Ensemble member {member_index} median schema mismatch.")

    reference = np.asarray(bundle["score_reference_sorted"], dtype=float)
    if reference.ndim != 1 or reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("Model score reference is empty or invalid.")
    if np.any(np.diff(reference) < 0):
        raise ValueError("Model score reference is not sorted ascending.")
    if set(bundle["feature_reference_sorted"]) != set(FEATURES_B6B):
        raise ValueError("Feature reference does not match the B6b schema.")
    for column in FEATURES_B6B:
        if not np.isfinite(float(bundle["medians"][column])):
            raise ValueError(f"Invalid stored median for feature {column!r}.")
        if column not in bundle["feature_min"] or column not in bundle["feature_max"]:
            raise ValueError(f"Missing training range for feature {column!r}.")
    return bundle


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON object, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
