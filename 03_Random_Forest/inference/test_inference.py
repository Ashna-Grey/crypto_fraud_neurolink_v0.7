# ============================================================
# test_inference.py
# Test the complete RF inference pipeline
#
# Uses 40 known wallets from the FINAL TEST SET:
#   20 non-fraud
#   20 fraud
#
# Tests:
#   - calibrated μ_RF
#   - tree disagreement σ²_RF
#   - conformal prediction
#   - conformal decision
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from predict import predict_dataframe


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

DATASET_PATH = (
    "C:/Users/ASHNA/Desktop/Ashna/"
    "crypto-fraud-intelligence/github_files/"
    "Crypto-fraud-intelligence-main/"
    "01_Dataset/processed/"
    "ethereum_fraud_25_features.csv"
)

SPLIT_PATH = os.path.join(
    ARTIFACT_DIR,
    "rf_data_split_indices.joblib"
)

FEATURE_SCHEMA_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_schema.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "analysis",
    "rf_40_wallet_test.csv"
)


# ============================================================
# 1. LOAD ARTIFACTS
# ============================================================

split_data = joblib.load(
    SPLIT_PATH
)

test_indices = np.asarray(
    split_data["test_indices"]
)

with open(
    FEATURE_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:

    import json
    schema = json.load(f)

FEATURES = schema.get(
    "feature_order",
    schema["features"]
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

# Only the untouched final test set
test_df = df.loc[
    test_indices
].copy()


# ============================================================
# 3. SELECT 20 NON-FRAUD + 20 FRAUD
# ============================================================

nonfraud = test_df[
    test_df["FLAG"] == 0
].sample(
    n=20,
    random_state=42
)

fraud = test_df[
    test_df["FLAG"] == 1
].sample(
    n=20,
    random_state=42
)

test_sample = pd.concat(
    [
        nonfraud,
        fraud
    ],
    axis=0
)

# Shuffle them
test_sample = test_sample.sample(
    frac=1.0,
    random_state=42
)


# ============================================================
# 4. REMOVE TARGET FROM INPUT
# ============================================================

X_test = test_sample[
    FEATURES
].copy()

actual_labels = test_sample[
    "FLAG"
].copy()


# ============================================================
# 5. RUN INFERENCE
# ============================================================

print("=" * 80)
print("RF INFERENCE TEST")
print("=" * 80)

print(
    f"Testing {len(X_test)} wallets"
)

print(
    f"Non-fraud: {(actual_labels == 0).sum()}"
)

print(
    f"Fraud    : {(actual_labels == 1).sum()}"
)

results = predict_dataframe(
    X_test
)


# ============================================================
# 6. ADD ACTUAL LABEL
# ============================================================

results["actual_FLAG"] = (
    actual_labels.to_numpy()
)


# ============================================================
# 7. CHECK PREDICTION CORRECTNESS
# ============================================================

results["correct"] = (
    results["FLAG"]
    ==
    results["actual_FLAG"]
).astype(int)


# ============================================================
# 8. CONFORMAL CORRECTNESS
# ============================================================

def conformal_covers_true_label(
    row
):

    prediction_set = (
        row["rf_conformal_set"]
    )

    true_label = int(
        row["actual_FLAG"]
    )

    # Example strings:
    # "[0]"
    # "[1]"
    # "[0, 1]"
    # "[]"

    if prediction_set == "[0]":
        prediction_set = [0]

    elif prediction_set == "[1]":
        prediction_set = [1]

    elif prediction_set == "[0, 1]":
        prediction_set = [0, 1]

    elif prediction_set == "[]":
        prediction_set = []

    else:
        raise ValueError(
            f"Unexpected prediction set: "
            f"{prediction_set}"
        )

    return int(
        true_label in prediction_set
    )


results["conformal_covered"] = (
    results.apply(
        conformal_covers_true_label,
        axis=1
    )
)


# ============================================================
# 9. DISPLAY ALL RESULTS
# ============================================================

display_columns = [
    "actual_FLAG",
    "FLAG",
    "fraud_probability",
    "rf_mu_calibrated",
    "rf_sigma_squared",
    "rf_conformal_set",
    "rf_conformal_decision",
    "correct",
    "conformal_covered"
]

print(
    "\n" + "=" * 80
)

print(
    "WALLET-LEVEL RESULTS"
)

print(
    "=" * 80
)

print(
    results[
        display_columns
    ].to_string(
        index=True,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 10. OVERALL METRICS
# ============================================================

accuracy = (
    results["correct"].mean()
)

conformal_coverage = (
    results["conformal_covered"].mean()
)

empty_rate = (
    results["rf_conformal_set"]
    ==
    "[]"
).mean()

fraud_decision_count = (
    results["rf_conformal_decision"]
    ==
    "FRAUD"
).sum()

nonfraud_decision_count = (
    results["rf_conformal_decision"]
    ==
    "NON_FRAUD"
).sum()

abstain_count = (
    results["rf_conformal_decision"]
    ==
    "ABSTAIN"
).sum()

ambiguous_count = (
    results["rf_conformal_decision"]
    ==
    "AMBIGUOUS"
).sum()


print(
    "\n" + "=" * 80
)

print(
    "TEST SUMMARY"
)

print(
    "=" * 80
)

print(
    f"Prediction accuracy       : {accuracy:.4f}"
)

print(
    f"Conformal coverage        : "
    f"{conformal_coverage:.4f}"
)

print(
    f"Empty / abstain rate      : "
    f"{empty_rate:.4f}"
)

print(
    f"FRAUD decisions            : "
    f"{fraud_decision_count}"
)

print(
    f"NON_FRAUD decisions        : "
    f"{nonfraud_decision_count}"
)

print(
    f"ABSTAIN decisions          : "
    f"{abstain_count}"
)

print(
    f"AMBIGUOUS decisions        : "
    f"{ambiguous_count}"
)


# ============================================================
# 11. CHECK BOTH TRUE CLASSES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "PER-CLASS RESULTS"
)

print(
    "=" * 80
)

for label, name in [
    (0, "NON-FRAUD"),
    (1, "FRAUD")
]:

    subset = results[
        results["actual_FLAG"] == label
    ]

    if len(subset) == 0:
        continue

    prediction_accuracy = (
        subset["correct"].mean()
    )

    conformal_class_coverage = (
        subset["conformal_covered"].mean()
    )

    mean_mu = (
        subset["rf_mu_calibrated"].mean()
    )

    mean_sigma_squared = (
        subset["rf_sigma_squared"].mean()
    )

    print(
        f"\n{name}"
    )

    print(
        f"Samples                  : "
        f"{len(subset)}"
    )

    print(
        f"Prediction accuracy     : "
        f"{prediction_accuracy:.4f}"
    )

    print(
        f"Conformal coverage      : "
        f"{conformal_class_coverage:.4f}"
    )

    print(
        f"Mean μ_RF               : "
        f"{mean_mu:.6f}"
    )

    print(
        f"Mean σ²_RF              : "
        f"{mean_sigma_squared:.6f}"
    )


# ============================================================
# 12. UNCERTAINTY COMPARISON
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "UNCERTAINTY BY PREDICTION CORRECTNESS"
)

print(
    "=" * 80
)

correct_subset = results[
    results["correct"] == 1
]

incorrect_subset = results[
    results["correct"] == 0
]

if len(correct_subset) > 0:

    print(
        f"Correct predictions - "
        f"mean σ²: "
        f"{correct_subset['rf_sigma_squared'].mean():.6f}"
    )

if len(incorrect_subset) > 0:

    print(
        f"Incorrect predictions - "
        f"mean σ²: "
        f"{incorrect_subset['rf_sigma_squared'].mean():.6f}"
    )

else:

    print(
        "No incorrect predictions in this sample."
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 14. FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "RF INFERENCE TEST COMPLETE"
)

print(
    "=" * 80
)