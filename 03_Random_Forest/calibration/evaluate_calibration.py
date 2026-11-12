# ============================================================
# evaluate_calibration.py
# Ethereum Fraud Detection - Random Forest
# Probability Calibration Evaluation
#
# Purpose:
#   1. Load the trained 20-feature Random Forest
#   2. Load the fitted probability calibrator
#   3. Use ONLY the untouched final test set for evaluation
#   4. Compare:
#        - Raw RF probability
#        - Calibrated RF probability
#   5. Calculate:
#        - Brier Score
#        - Log Loss
#        - Expected Calibration Error (ECE)
#        - ROC-AUC
#        - PR-AUC
#   6. Save metrics
#   7. Generate a reliability/calibration diagram
#
# IMPORTANT:
#   - No model is trained here.
#   - No calibrator is fitted here.
#   - Calibration data is NOT used here.
#   - Conformal data is NOT used here.
#   - Only the final untouched test set is evaluated.
# ============================================================


import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score
)

from sklearn.calibration import calibration_curve


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# 03_Random_Forest/calibration/
CALIBRATION_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# 03_Random_Forest/
BASE_DIR = os.path.dirname(
    CALIBRATION_DIR
)

# Repository root
PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

# 03_Random_Forest/artifacts/
MODEL_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET_PATH = "C:/Users/ASHNA/Desktop/Ashna/crypto-fraud-intelligence/github_files/Crypto-fraud-intelligence-main/01_Dataset/processed/ethereum_fraud_25_features.csv"

# ------------------------------------------------------------
# Model artifacts
# ------------------------------------------------------------

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

CALIBRATOR_PATH = os.path.join(
    MODEL_DIR,
    "rf_calibrator.joblib"
)

CALIBRATION_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "rf_calibration_config.json"
)

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)

METRICS_PATH = os.path.join(
    ANALYSIS_DIR,
    "calibration_metrics.csv"
)

CURVE_DATA_PATH = os.path.join(
    ANALYSIS_DIR,
    "calibration_curve_data.csv"
)

PLOT_PATH = os.path.join(
    ANALYSIS_DIR,
    "rf_calibration_curve.png"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "FLAG"

N_BINS = 10


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# 4. FILE EXISTENCE CHECKS
# ============================================================

required_files = {
    "Dataset": DATASET_PATH,
    "RF model": MODEL_PATH,
    "Imputer": IMPUTER_PATH,
    "Feature schema": FEATURE_SCHEMA_PATH,
    "Split indices": SPLIT_PATH,
    "Calibrator": CALIBRATOR_PATH,
    "Calibration config": CALIBRATION_CONFIG_PATH
}

print("=" * 80)
print("CHECKING REQUIRED FILES")
print("=" * 80)

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"[OK] {name}: {path}")


# ============================================================
# 5. LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("LOADING RANDOM FOREST")
print("=" * 80)

rf = joblib.load(
    MODEL_PATH
)

print(
    "Model:",
    type(rf).__name__
)

print(
    "Number of trees:",
    len(rf.estimators_)
)

print(
    "Classes:",
    rf.classes_
)


# ============================================================
# 6. LOAD IMPUTER
# ============================================================

imputer = joblib.load(
    IMPUTER_PATH
)

print(
    "Loaded imputer."
)


# ============================================================
# 7. LOAD FEATURE SCHEMA
# ============================================================

with open(
    FEATURE_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_schema = json.load(f)

FEATURES = feature_schema["feature_order"]

print(
    "\nNumber of features:",
    len(FEATURES)
)


# ============================================================
# 8. LOAD SPLIT INDICES
# ============================================================

split_data = joblib.load(
    SPLIT_PATH
)

required_split_keys = [
    "train_indices",
    "calibration_indices",
    "conformal_indices",
    "test_indices"
]

for key in required_split_keys:

    if key not in split_data:

        raise KeyError(
            f"Missing split key: {key}"
        )

test_indices = np.asarray(
    split_data["test_indices"]
)

print(
    "Final test samples:",
    len(test_indices)
)


# ============================================================
# 9. LOAD CALIBRATOR
# ============================================================

calibrator = joblib.load(
    CALIBRATOR_PATH
)

print(
    "Loaded calibrator:",
    type(calibrator).__name__
)


# ============================================================
# 10. LOAD CALIBRATION CONFIG
# ============================================================

with open(
    CALIBRATION_CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    calibration_config = json.load(f)

selected_method = calibration_config.get(
    "calibration_method",
    "unknown"
)

print(
    "Calibration method:",
    selected_method
)


# ============================================================
# 11. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print(
    "\nDataset shape:",
    df.shape
)

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# 12. VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "The following required features are missing:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# 13. CREATE FINAL TEST SET
# ============================================================

X_all = df[FEATURES].copy()

y_all = df[TARGET].astype(int).copy()

X_test = X_all.loc[
    test_indices
].copy()

y_test = y_all.loc[
    test_indices
].copy()

print(
    "\nFinal test X shape:",
    X_test.shape
)

print(
    "Final test y shape:",
    y_test.shape
)

print(
    "\nFinal test FLAG distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 14. APPLY TRAINING-ONLY IMPUTER
# ============================================================

X_test_imp = imputer.transform(
    X_test
)

print(
    "\nImputed test shape:",
    X_test_imp.shape
)


# ============================================================
# 15. DETERMINE FRAUD CLASS INDEX
# ============================================================

if 1 not in rf.classes_:

    raise ValueError(
        "Fraud class '1' was not found in RF classes."
    )

rf_fraud_index = list(
    rf.classes_
).index(1)


# ============================================================
# 16. RAW RF PROBABILITY
# ============================================================

raw_prob = rf.predict_proba(
    X_test_imp
)[:, rf_fraud_index]


# ============================================================
# 17. CALIBRATED RF PROBABILITY
# ============================================================

calibrator_classes = list(
    calibrator.classes_
)

if 1 not in calibrator_classes:

    raise ValueError(
        "Fraud class '1' was not found in calibrator classes."
    )

calibrator_fraud_index = calibrator_classes.index(1)

calibrated_prob = calibrator.predict_proba(
    X_test_imp
)[:, calibrator_fraud_index]


# ============================================================
# 18. ECE FUNCTION
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):
    """
    Expected Calibration Error (ECE).

    Lower is better.

    ECE measures the weighted difference between:
        confidence
    and:
        empirical accuracy
    across probability bins.
    """

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

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

        bin_probability = probabilities[mask]

        bin_true = y_true[mask]

        mean_confidence = np.mean(
            bin_probability
        )

        empirical_accuracy = np.mean(
            bin_true
        )

        bin_weight = np.mean(
            mask
        )

        ece += (
            bin_weight
            *
            abs(
                mean_confidence
                -
                empirical_accuracy
            )
        )

    return float(ece)


# ============================================================
# 19. CALCULATE METRICS
# ============================================================

raw_brier = brier_score_loss(
    y_test,
    raw_prob
)

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_prob
)


raw_logloss = log_loss(
    y_test,
    raw_prob,
    labels=[0, 1]
)

calibrated_logloss = log_loss(
    y_test,
    calibrated_prob,
    labels=[0, 1]
)


raw_ece = expected_calibration_error(
    y_test,
    raw_prob,
    n_bins=N_BINS
)

calibrated_ece = expected_calibration_error(
    y_test,
    calibrated_prob,
    n_bins=N_BINS
)


raw_roc_auc = roc_auc_score(
    y_test,
    raw_prob
)

calibrated_roc_auc = roc_auc_score(
    y_test,
    calibrated_prob
)


raw_pr_auc = average_precision_score(
    y_test,
    raw_prob
)

calibrated_pr_auc = average_precision_score(
    y_test,
    calibrated_prob
)


# ============================================================
# 20. CREATE METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": "Random Forest",
        "probability_type": "raw",
        "calibration_method": "none",
        "brier_score": raw_brier,
        "log_loss": raw_logloss,
        "ece": raw_ece,
        "roc_auc": raw_roc_auc,
        "pr_auc": raw_pr_auc
    },
    {
        "model": "Random Forest",
        "probability_type": "calibrated",
        "calibration_method": selected_method,
        "brier_score": calibrated_brier,
        "log_loss": calibrated_logloss,
        "ece": calibrated_ece,
        "roc_auc": calibrated_roc_auc,
        "pr_auc": calibrated_pr_auc
    }
])


# ============================================================
# 21. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION EVALUATION RESULTS")
print("=" * 80)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 22. IMPROVEMENT TABLE
# ============================================================

improvement_df = pd.DataFrame({

    "metric": [
        "Brier Score",
        "Log Loss",
        "ECE",
        "ROC-AUC",
        "PR-AUC"
    ],

    "raw": [
        raw_brier,
        raw_logloss,
        raw_ece,
        raw_roc_auc,
        raw_pr_auc
    ],

    "calibrated": [
        calibrated_brier,
        calibrated_logloss,
        calibrated_ece,
        calibrated_roc_auc,
        calibrated_pr_auc
    ]
})

improvement_df["change"] = (
    improvement_df["calibrated"]
    -
    improvement_df["raw"]
)

print("\n" + "=" * 80)
print("RAW vs CALIBRATED")
print("=" * 80)

print(
    improvement_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 23. CALIBRATION CURVES
# ============================================================

fraction_positive_raw, mean_predicted_raw = (
    calibration_curve(
        y_test,
        raw_prob,
        n_bins=N_BINS,
        strategy="uniform"
    )
)

fraction_positive_cal, mean_predicted_cal = (
    calibration_curve(
        y_test,
        calibrated_prob,
        n_bins=N_BINS,
        strategy="uniform"
    )
)


# ============================================================
# 24. SAVE CALIBRATION CURVE DATA
# ============================================================

max_bins = max(
    len(mean_predicted_raw),
    len(mean_predicted_cal)
)

curve_data = pd.DataFrame({

    "raw_mean_predicted_probability": pd.Series(
        mean_predicted_raw
    ),

    "raw_fraction_positive": pd.Series(
        fraction_positive_raw
    ),

    "calibrated_mean_predicted_probability": pd.Series(
        mean_predicted_cal
    ),

    "calibrated_fraction_positive": pd.Series(
        fraction_positive_cal
    )
})

curve_data.to_csv(
    CURVE_DATA_PATH,
    index=False
)

print(
    "\nSaved calibration curve data:",
    CURVE_DATA_PATH
)


# ============================================================
# 25. SAVE METRICS
# ============================================================

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

print(
    "Saved calibration metrics:",
    METRICS_PATH
)


# ============================================================
# 26. GENERATE RELIABILITY DIAGRAM
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    mean_predicted_raw,
    fraction_positive_raw,
    marker="o",
    label="Raw RF"
)

plt.plot(
    mean_predicted_cal,
    fraction_positive_cal,
    marker="o",
    label=f"Calibrated RF ({selected_method})"
)

plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed fraction of fraud"
)

plt.title(
    "Random Forest Probability Calibration"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved calibration plot:",
    PLOT_PATH
)


# ============================================================
# 27. BASIC INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if calibrated_brier < raw_brier:
    print(
        "Brier Score improved after calibration."
    )
else:
    print(
        "Brier Score did NOT improve after calibration."
    )


if calibrated_logloss < raw_logloss:
    print(
        "Log Loss improved after calibration."
    )
else:
    print(
        "Log Loss did NOT improve after calibration."
    )


if calibrated_ece < raw_ece:
    print(
        "ECE improved after calibration."
    )
else:
    print(
        "ECE did NOT improve after calibration."
    )


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION EVALUATION COMPLETE")
print("=" * 80)

print(
    f"Calibration method : {selected_method}"
)

print(
    f"Raw Brier          : {raw_brier:.6f}"
)

print(
    f"Calibrated Brier    : {calibrated_brier:.6f}"
)

print(
    f"Raw Log Loss       : {raw_logloss:.6f}"
)

print(
    f"Calibrated Log Loss : {calibrated_logloss:.6f}"
)

print(
    f"Raw ECE            : {raw_ece:.6f}"
)

print(
    f"Calibrated ECE      : {calibrated_ece:.6f}"
)

print(
    f"Raw ROC-AUC        : {raw_roc_auc:.6f}"
)

print(
    f"Calibrated ROC-AUC  : {calibrated_roc_auc:.6f}"
)

print(
    f"Raw PR-AUC         : {raw_pr_auc:.6f}"
)

print(
    f"Calibrated PR-AUC   : {calibrated_pr_auc:.6f}"
)

print("\nOutputs:")
print(
    METRICS_PATH
)

print(
    CURVE_DATA_PATH
)

print(
    PLOT_PATH
)
