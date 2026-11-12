# ============================================================
# fit_calibrator.py
# Ethereum Fraud Detection - Random Forest Probability
# Calibration
#
# Purpose:
#   1. Load the already-trained 20-feature Random Forest
#   2. Load the calibration split created by train.py
#   3. Compare:
#        - Raw RF probability
#        - Sigmoid calibration
#        - Isotonic calibration
#   4. Select the better calibration method using Brier score
#      on an INTERNAL calibration-validation split
#   5. Refit the selected calibrator on the FULL calibration set
#   6. Save the calibrator for inference
#
# IMPORTANT:
#   - The RF itself is NOT retrained here.
#   - The final test set is NEVER used here.
#   - The RF training data is NOT used to fit the calibrator.
# ============================================================


import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Directory containing this file:
# 03_Random_Forest/calibration/
CALIBRATION_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# 03_Random_Forest/
BASE_DIR = os.path.dirname(
    CALIBRATION_DIR
)

# 03_Random_Forest/artifacts/
MODEL_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

# Your dataset:
# Adjust ONLY if your repository layout is different.
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_PATH = "C:/Users/ASHNA/Desktop/Ashna/crypto-fraud-intelligence/github_files/Crypto-fraud-intelligence-main/01_Dataset/processed/ethereum_fraud_25_features.csv"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features.joblib"
)

IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features_imputer.joblib"
)

FEATURE_SCHEMA_PATH = os.path.join(
    MODEL_DIR,
    "feature_schema.json"
)

SPLIT_PATH = os.path.join(
    MODEL_DIR,
    "rf_data_split_indices.joblib"
)

OUTPUT_CALIBRATOR_PATH = os.path.join(
    MODEL_DIR,
    "rf_calibrator.joblib"
)

OUTPUT_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "rf_calibration_config.json"
)

OUTPUT_METRICS_PATH = os.path.join(
    BASE_DIR,
    "analysis",
    "calibration_metrics.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Internal split ONLY for choosing sigmoid vs isotonic.
# This is NOT the final test set.
CALIBRATION_VALIDATION_SIZE = 0.30

TARGET = "FLAG"


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_METRICS_PATH),
    exist_ok=True
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("=" * 80)
print("LOADING RANDOM FOREST")
print("=" * 80)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"RF model not found:\n{MODEL_PATH}"
    )

rf = joblib.load(MODEL_PATH)

print("Loaded:", MODEL_PATH)
print("Model:", type(rf).__name__)
print("Number of trees:", len(rf.estimators_))


# ============================================================
# 5. LOAD IMPUTER
# ============================================================

if not os.path.exists(IMPUTER_PATH):
    raise FileNotFoundError(
        f"Imputer not found:\n{IMPUTER_PATH}"
    )

imputer = joblib.load(IMPUTER_PATH)

print("Loaded:", IMPUTER_PATH)


# ============================================================
# 6. LOAD FEATURE SCHEMA
# ============================================================

if not os.path.exists(FEATURE_SCHEMA_PATH):
    raise FileNotFoundError(
        f"Feature schema not found:\n{FEATURE_SCHEMA_PATH}"
    )

with open(
    FEATURE_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_schema = json.load(f)

FEATURES = feature_schema["feature_order"]

print("\nNumber of final features:", len(FEATURES))

print("\nFinal feature order:")
for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 7. LOAD DATA SPLIT INDICES
# ============================================================

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        "\nSplit-index artifact not found:\n"
        f"{SPLIT_PATH}\n\n"
        "Run the modified train.py first so that it creates "
        "rf_data_split_indices.joblib."
    )

split_data = joblib.load(
    SPLIT_PATH
)

required_keys = [
    "train_indices",
    "calibration_indices",
    "conformal_indices",
    "test_indices"
]

for key in required_keys:

    if key not in split_data:
        raise KeyError(
            f"Missing '{key}' in {SPLIT_PATH}"
        )

calibration_indices = np.asarray(
    split_data["calibration_indices"]
)

print(
    "\nCalibration samples:",
    len(calibration_indices)
)


# ============================================================
# 8. LOAD ORIGINAL DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(
    DATASET_PATH
)

print("Dataset shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

if not set(FEATURES).issubset(df.columns):
    missing = [
        f for f in FEATURES
        if f not in df.columns
    ]

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing)
    )


# ============================================================
# 9. BUILD CALIBRATION DATASET
# ============================================================

X_all = df[FEATURES].copy()
y_all = df[TARGET].astype(int).copy()

# IMPORTANT:
# These are original dataframe indices generated by train_test_split.
X_cal = X_all.loc[
    calibration_indices
].copy()

y_cal = y_all.loc[
    calibration_indices
].copy()

print("\nCalibration dataset:")
print("X_cal:", X_cal.shape)
print("y_cal:", y_cal.shape)

print("\nCalibration FLAG distribution:")
print(
    y_cal.value_counts()
)


# ============================================================
# 10. IMPUTE CALIBRATION DATA
# ============================================================
#
# IMPORTANT:
# The imputer was fitted on the RF training data.
# We ONLY transform calibration data here.
# ============================================================

X_cal_imp = imputer.transform(
    X_cal
)

print(
    "\nImputed calibration shape:",
    X_cal_imp.shape
)


# ============================================================
# 11. RAW RF PROBABILITY
# ============================================================

fraud_class_index = list(
    rf.classes_
).index(1)

raw_prob = rf.predict_proba(
    X_cal_imp
)[:, fraud_class_index]


print("\nRaw RF probability statistics:")
print(
    f"Mean : {raw_prob.mean():.6f}"
)

print(
    f"Min  : {raw_prob.min():.6f}"
)

print(
    f"Max  : {raw_prob.max():.6f}"
)


# ============================================================
# 12. INTERNAL SPLIT FOR METHOD SELECTION
# ============================================================
#
# We do NOT choose isotonic/sigmoid based on the same data
# on which the calibrator was fitted.
#
# 70% -> calibrator-selection fitting
# 30% -> calibrator-selection validation
#
# After selecting the method, the selected calibrator is
# refitted on the FULL calibration dataset.
# ============================================================

(
    X_cal_fit,
    X_cal_valid,
    y_cal_fit,
    y_cal_valid
) = train_test_split(
    X_cal_imp,
    y_cal,
    test_size=CALIBRATION_VALIDATION_SIZE,
    stratify=y_cal,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 80)
print("INTERNAL CALIBRATION METHOD SELECTION")
print("=" * 80)

print(
    "Calibration-fit samples:",
    X_cal_fit.shape[0]
)

print(
    "Calibration-validation samples:",
    X_cal_valid.shape[0]
)


# ============================================================
# 13. FIT SIGMOID CALIBRATION
# ============================================================

print("\nFitting SIGMOID calibration...")

sigmoid_calibrator = CalibratedClassifierCV(
    estimator=FrozenEstimator(rf),
    method="sigmoid"
)

sigmoid_calibrator.fit(
    X_cal_fit,
    y_cal_fit
)

sigmoid_prob = sigmoid_calibrator.predict_proba(
    X_cal_valid
)[:, fraud_class_index]


# ============================================================
# 14. FIT ISOTONIC CALIBRATION
# ============================================================

print("Fitting ISOTONIC calibration...")

isotonic_calibrator = CalibratedClassifierCV(
    estimator=FrozenEstimator(rf),
    method="isotonic"
)

isotonic_calibrator.fit(
    X_cal_fit,
    y_cal_fit
)

isotonic_prob = isotonic_calibrator.predict_proba(
    X_cal_valid
)[:, fraud_class_index]


# ============================================================
# 15. CALIBRATION ERROR: ECE
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):
    """
    Expected Calibration Error (ECE).

    Lower is better.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (probabilities >= lower)
                &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                &
                (probabilities < upper)
            )

        if not np.any(mask):
            continue

        bin_prob = probabilities[mask]
        bin_true = y_true[mask]

        mean_probability = np.mean(
            bin_prob
        )

        empirical_frequency = np.mean(
            bin_true
        )

        bin_weight = np.mean(
            mask
        )

        ece += (
            bin_weight
            *
            abs(
                mean_probability
                -
                empirical_frequency
            )
        )

    return float(ece)


# ============================================================
# 16. EVALUATE RAW / SIGMOID / ISOTONIC
# ============================================================

raw_prob_valid = rf.predict_proba(
    X_cal_valid
)[:, fraud_class_index]

metrics = []


# ------------------------------------------------------------
# RAW RF
# ------------------------------------------------------------

metrics.append({
    "method": "raw",
    "brier_score": brier_score_loss(
        y_cal_valid,
        raw_prob_valid
    ),
    "log_loss": log_loss(
        y_cal_valid,
        raw_prob_valid,
        labels=[0, 1]
    ),
    "ece": expected_calibration_error(
        y_cal_valid,
        raw_prob_valid
    )
})


# ------------------------------------------------------------
# SIGMOID
# ------------------------------------------------------------

metrics.append({
    "method": "sigmoid",
    "brier_score": brier_score_loss(
        y_cal_valid,
        sigmoid_prob
    ),
    "log_loss": log_loss(
        y_cal_valid,
        sigmoid_prob,
        labels=[0, 1]
    ),
    "ece": expected_calibration_error(
        y_cal_valid,
        sigmoid_prob
    )
})


# ------------------------------------------------------------
# ISOTONIC
# ------------------------------------------------------------

metrics.append({
    "method": "isotonic",
    "brier_score": brier_score_loss(
        y_cal_valid,
        isotonic_prob
    ),
    "log_loss": log_loss(
        y_cal_valid,
        isotonic_prob,
        labels=[0, 1]
    ),
    "ece": expected_calibration_error(
        y_cal_valid,
        isotonic_prob
    )
})


metrics_df = pd.DataFrame(
    metrics
)


# ============================================================
# 17. DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION COMPARISON")
print("=" * 80)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 18. SELECT BEST CALIBRATION METHOD
# ============================================================
#
# Primary criterion:
#   Brier score
#
# Secondary:
#   Log loss
#
# Lower = better.
# ============================================================

candidate_metrics = metrics_df[
    metrics_df["method"].isin(
        ["sigmoid", "isotonic"]
    )
].copy()

candidate_metrics = candidate_metrics.sort_values(
    by=[
        "brier_score",
        "log_loss"
    ],
    ascending=[
        True,
        True
    ]
).reset_index(
    drop=True
)

best_method = candidate_metrics.loc[
    0,
    "method"
]

print(
    "\nSelected calibration method:",
    best_method
)


# ============================================================
# 19. REFIT SELECTED CALIBRATOR ON FULL CALIBRATION SET
# ============================================================
#
# IMPORTANT:
# The final selected calibrator now gets ALL calibration
# samples.
#
# The final test set remains completely untouched.
# ============================================================

print("\n" + "=" * 80)
print("REFITTING SELECTED CALIBRATOR")
print("=" * 80)

if best_method == "sigmoid":

    final_calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(rf),
        method="sigmoid"
    )

elif best_method == "isotonic":

    final_calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(rf),
        method="isotonic"
    )

else:

    raise RuntimeError(
        f"Unknown calibration method: {best_method}"
    )


final_calibrator.fit(
    X_cal_imp,
    y_cal
)


# ============================================================
# 20. VERIFY CALIBRATED PROBABILITIES
# ============================================================

calibrated_prob = final_calibrator.predict_proba(
    X_cal_imp
)[:, fraud_class_index]

print(
    "\nFinal calibrated probability statistics:"
)

print(
    f"Mean : {calibrated_prob.mean():.6f}"
)

print(
    f"Min  : {calibrated_prob.min():.6f}"
)

print(
    f"Max  : {calibrated_prob.max():.6f}"
)


# ============================================================
# 21. SAVE CALIBRATOR
# ============================================================

joblib.dump(
    final_calibrator,
    OUTPUT_CALIBRATOR_PATH
)

print(
    "\nSaved calibrator:",
    OUTPUT_CALIBRATOR_PATH
)


# ============================================================
# 22. SAVE CALIBRATION CONFIGURATION
# ============================================================

best_row = candidate_metrics.iloc[0]

calibration_config = {

    "model": "RandomForestClassifier",

    "calibration_method": best_method,

    "selection_metric": "Brier score",

    "secondary_selection_metric": "Log loss",

    "candidate_methods": [
        "sigmoid",
        "isotonic"
    ],

    "n_calibration_samples": int(
        len(y_cal)
    ),

    "internal_validation_fraction": (
        CALIBRATION_VALIDATION_SIZE
    ),

    "selected_method_validation_metrics": {
        "brier_score": float(
            best_row["brier_score"]
        ),
        "log_loss": float(
            best_row["log_loss"]
        ),
        "ece": float(
            best_row["ece"]
        )
    },

    "feature_count": len(FEATURES),

    "features": FEATURES
}


with open(
    OUTPUT_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        calibration_config,
        f,
        indent=4
    )


print(
    "Saved calibration config:",
    OUTPUT_CONFIG_PATH
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

metrics_df.to_csv(
    OUTPUT_METRICS_PATH,
    index=False
)

print(
    "Saved calibration metrics:",
    OUTPUT_METRICS_PATH
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION COMPLETE")
print("=" * 80)

print(
    "Selected method:",
    best_method
)

print(
    "RF model:",
    MODEL_PATH
)

print(
    "Calibrator:",
    OUTPUT_CALIBRATOR_PATH
)

print(
    "Configuration:",
    OUTPUT_CONFIG_PATH
)

print(
    "Metrics:",
    OUTPUT_METRICS_PATH
)

print("\nFinal RF pipeline:")
print(
    "Raw RF probability"
    " -> "
    f"{best_method}"
    " -> "
    "calibrated μ_RF"
)
