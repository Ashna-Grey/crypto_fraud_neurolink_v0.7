"""Live-wallet feature normalization and preprocessing for the B6b model.

The transformation map in this module was reconstructed and checked against the
bundled 10,000-row raw/preprocessed dataset pair. It must not be applied to data
that is already in the preprocessed B6b feature space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.isolation_forest_utils import FEATURES_B6B

# Verified transform from D2_live_wallet_dataset.csv to
# D2_IsolationForest_preprocessed_23_features.csv: signed log1p(x).
SIGNED_LOG1P_FEATURES = [
    "total ether received",
    "avg val received",
    "Avg min between received tnx",
    "max value received",
    "ERC20 min val rec",
    "min value received",
    "total Ether sent",
    "ERC20 avg val rec",
    "ERC20 max val rec",
    "Avg min between sent tnx",
    "max val sent",
    "avg val sent",
    "ERC20 uniq sent token name",
    "current_eth_balance",
]


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from names and reject collisions."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise ValueError(
            "Input has duplicate column names after trimming whitespace: "
            + ", ".join(map(repr, duplicates))
        )
    return normalized


def prepare_raw_live_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert raw live feature rows into the exact B6b model feature space.

    Extra columns (for example Address and is_contract) are preserved. Missing
    numeric values stay missing so the ensemble member's stored training median
    is used during scoring. Invalid text and infinite values are rejected.
    """
    normalized = normalize_column_names(frame)
    missing = [feature for feature in FEATURES_B6B if feature not in normalized.columns]
    if missing:
        raise ValueError(
            "Raw-live input is missing required B6b features: " + ", ".join(missing)
        )

    numeric = normalized.copy()
    for feature in FEATURES_B6B:
        raw = normalized[feature]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid_parse = raw.notna() & converted.isna()
        if invalid_parse.any():
            examples = raw.loc[invalid_parse, feature].head(3).tolist()
            raise ValueError(
                f"Non-numeric values in raw-live feature {feature!r}; examples: {examples}"
            )
        numeric[feature] = converted

    feature_values = numeric[FEATURES_B6B].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(feature_values).any():
        raise ValueError("Raw-live input contains infinite feature values.")

    for feature in SIGNED_LOG1P_FEATURES:
        values = numeric[feature].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values)
        transformed = values.copy()
        transformed[valid] = np.sign(values[valid]) * np.log1p(np.abs(values[valid]))
        numeric[feature] = transformed

    audit = {
        "input_format": "raw-live",
        "column_names_trimmed": bool(any(str(c) != str(c).strip() for c in frame.columns)),
        "required_feature_count": len(FEATURES_B6B),
        "transformed_feature_count": len(SIGNED_LOG1P_FEATURES),
        "transformed_features": SIGNED_LOG1P_FEATURES.copy(),
        "missing_required_feature_columns": [],
        "feature_value_transform": "sign(x) * log(1 + abs(x)) for listed features; unchanged for other B6b features",
        "missing_value_handling": "Preserve NaN for member-specific training-median imputation at scoring time",
    }
    return numeric, audit
