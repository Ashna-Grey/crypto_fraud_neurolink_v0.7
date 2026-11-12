# ============================================================
# evaluate_conformal.py
# Ethereum Fraud Detection - Random Forest
# Mondrian / Class-Conditional Split Conformal Evaluation
#
# Purpose:
#   1. Load the trained Random Forest
#   2. Load the training-fitted imputer
#   3. Load the feature schema
#   4. Load the fitted sigmoid probability calibrator
#   5. Load class-specific q_hat values:
#
#        q_0 -> non-fraud
#        q_1 -> fraud
#
#   6. Evaluate conformal prediction on the FINAL
#      UNTOUCHED TEST SET
#
# Metrics:
#   - Overall marginal coverage
#   - Non-fraud coverage
#   - Fraud coverage
#   - Coverage gap
#   - Empty-set rate
#   - Singleton-set rate
#   - Ambiguous [0,1] rate
#   - Mean prediction-set size
#   - Singleton accuracy
#   - Fraud-only set precision
#   - Non-fraud-only set precision
#
# IMPORTANT:
#   - Nothing is fitted here.
#   - q_0 and q_1 are loaded from conformal_config.json.
#   - The final test set is used ONLY for evaluation.
# ============================================================


import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

CONFORMAL_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    CONFORMAL_DIR
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)


# ============================================================
# DATASET
# ============================================================
#
# SAME dataset used by train.py
# ============================================================

DATASET_PATH = (
    "C:/Users/ASHNA/Desktop/Ashna/"
    "crypto-fraud-intelligence/github_files/"
    "Crypto-fraud-intelligence-main/"
    "01_Dataset/processed/"
    "ethereum_fraud_25_features.csv"
)


# ============================================================
# EXISTING ARTIFACTS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features.joblib"
)

IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features_imputer.joblib"
)

SCHEMA_PATH = os.path.join(
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

CONFORMAL_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "conformal_config.json"
)


# ============================================================
# OUTPUTS
# ============================================================

RESULTS_PATH = os.path.join(
    ANALYSIS_DIR,
    "conformal_test_predictions.csv"
)

METRICS_PATH = os.path.join(
    ANALYSIS_DIR,
    "conformal_test_metrics.csv"
)


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Dataset": DATASET_PATH,
    "RF model": MODEL_PATH,
    "Imputer": IMPUTER_PATH,
    "Feature schema": SCHEMA_PATH,
    "Split indices": SPLIT_PATH,
    "Calibrator": CALIBRATOR_PATH,
    "Conformal config": CONFORMAL_CONFIG_PATH
}

print("=" * 80)
print("CHECKING REQUIRED FILES")
print("=" * 80)

for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(
        f"[OK] {name}: {path}"
    )


# ============================================================
# 4. LOAD ARTIFACTS
# ============================================================

print("\n" + "=" * 80)
print("LOADING ARTIFACTS")
print("=" * 80)

rf = joblib.load(
    MODEL_PATH
)

imputer = joblib.load(
    IMPUTER_PATH
)

calibrator = joblib.load(
    CALIBRATOR_PATH
)

with open(
    SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)

with open(
    CONFORMAL_CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    conformal_config = json.load(f)


# ============================================================
# 5. READ FEATURE SCHEMA
# ============================================================

FEATURES = schema.get(
    "feature_order",
    schema["features"]
)

print(
    "RF model:",
    type(rf).__name__
)

print(
    "Number of trees:",
    len(rf.estimators_)
)

print(
    "Feature count:",
    len(FEATURES)
)

print(
    "Calibrator:",
    type(calibrator).__name__
)

print(
    "Conformal method:",
    conformal_config.get(
        "method",
        "unknown"
    )
)


# ============================================================
# 6. READ CONFORMAL PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Expected new structure:
#
# "classes": {
#     "0": {
#         "q_hat": ...
#     },
#     "1": {
#         "q_hat": ...
#     }
# }
# ------------------------------------------------------------

if "classes" not in conformal_config:

    raise ValueError(
        "conformal_config.json does not contain "
        "class-specific 'classes' configuration."
    )

if "0" not in conformal_config["classes"]:

    raise ValueError(
        "Class 0 configuration missing."
    )

if "1" not in conformal_config["classes"]:

    raise ValueError(
        "Class 1 configuration missing."
    )


q_0 = float(
    conformal_config[
        "classes"
    ]["0"]["q_hat"]
)

q_1 = float(
    conformal_config[
        "classes"
    ]["1"]["q_hat"]
)

alpha = float(
    conformal_config.get(
        "alpha",
        0.05
    )
)

target_coverage = float(
    conformal_config.get(
        "target_coverage",
        1.0 - alpha
    )
)

p_threshold_0 = 1.0 - q_0
p_threshold_1 = 1.0 - q_1


print(
    f"Alpha: {alpha:.6f}"
)

print(
    f"Target coverage: {target_coverage:.6f}"
)

print(
    f"q_0 (non-fraud): {q_0:.12f}"
)

print(
    f"q_1 (fraud):     {q_1:.12f}"
)

print(
    f"P(non-fraud) threshold: "
    f"{p_threshold_0:.12f}"
)

print(
    f"P(fraud) threshold:     "
    f"{p_threshold_1:.12f}"
)


# ============================================================
# 7. LOAD SPLIT INDICES
# ============================================================

split_data = joblib.load(
    SPLIT_PATH
)

if "test_indices" not in split_data:

    raise KeyError(
        "test_indices not found in "
        "rf_data_split_indices.joblib"
    )

test_indices = np.asarray(
    split_data["test_indices"]
)

print(
    "\nFinal test samples:",
    len(test_indices)
)


# ============================================================
# 8. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset shape:",
    df.shape
)

if "FLAG" not in df.columns:

    raise ValueError(
        "Target column FLAG not found."
    )


# ============================================================
# 9. VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# 10. BUILD FINAL TEST SET
# ============================================================

X_test = df.loc[
    test_indices,
    FEATURES
].copy()

y_test = df.loc[
    test_indices,
    "FLAG"
].astype(int).copy()

print(
    "\nTest X shape:",
    X_test.shape
)

print(
    "Test y shape:",
    y_test.shape
)

print(
    "\nTest FLAG distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 11. APPLY TRAINING-FITTED IMPUTER
# ============================================================

X_test_imp = imputer.transform(
    X_test
)

print(
    "\nImputed test shape:",
    X_test_imp.shape
)


# ============================================================
# 12. GET CALIBRATED PROBABILITIES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GENERATING CALIBRATED TEST PROBABILITIES"
)

print(
    "=" * 80
)

probabilities = calibrator.predict_proba(
    X_test_imp
)

classes = np.asarray(
    calibrator.classes_
)

if 0 not in classes or 1 not in classes:

    raise ValueError(
        "Expected binary classes {0,1}."
    )

class_0_index = np.where(
    classes == 0
)[0][0]

class_1_index = np.where(
    classes == 1
)[0][0]

p_nonfraud = probabilities[
    :,
    class_0_index
]

p_fraud = probabilities[
    :,
    class_1_index
]


# ============================================================
# 13. GENERATE MONDRIAN PREDICTION SETS
# ============================================================
#
# Class 0 included when:
#
#     1 - P(0|x) <= q_0
#
# equivalent to:
#
#     P(0|x) >= 1 - q_0
#
#
# Class 1 included when:
#
#     1 - P(1|x) <= q_1
#
# equivalent to:
#
#     P(1|x) >= 1 - q_1
#
# ============================================================

prediction_sets = []

for p0, p1 in zip(
    p_nonfraud,
    p_fraud
):

    prediction_set = []

    # --------------------------------------------------------
    # NON-FRAUD
    # --------------------------------------------------------

    if (
        1.0 - p0
    ) <= q_0:

        prediction_set.append(0)

    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

    if (
        1.0 - p1
    ) <= q_1:

        prediction_set.append(1)

    prediction_sets.append(
        prediction_set
    )


# ============================================================
# 14. TRUE-LABEL COVERAGE
# ============================================================

true_labels = y_test.to_numpy()

covered = np.array([
    true_label in prediction_set
    for true_label, prediction_set
    in zip(
        true_labels,
        prediction_sets
    )
])


# ============================================================
# 15. SET SIZES
# ============================================================

set_sizes = np.array([
    len(prediction_set)
    for prediction_set
    in prediction_sets
])

empty_mask = (
    set_sizes == 0
)

singleton_mask = (
    set_sizes == 1
)

ambiguous_mask = (
    set_sizes == 2
)


# ============================================================
# 16. OVERALL COVERAGE
# ============================================================

n_test = len(
    true_labels
)

overall_coverage = float(
    np.mean(covered)
)

overall_coverage_gap = (
    overall_coverage
    -
    target_coverage
)


# ============================================================
# 17. CLASS-SPECIFIC COVERAGE
# ============================================================

class_0_mask = (
    true_labels == 0
)

class_1_mask = (
    true_labels == 1
)


if np.any(class_0_mask):

    class_0_coverage = float(
        np.mean(
            covered[
                class_0_mask
            ]
        )
    )

else:

    class_0_coverage = np.nan


if np.any(class_1_mask):

    class_1_coverage = float(
        np.mean(
            covered[
                class_1_mask
            ]
        )
    )

else:

    class_1_coverage = np.nan


# ============================================================
# 18. PREDICTION-SET RATES
# ============================================================

empty_rate = float(
    np.mean(empty_mask)
)

singleton_rate = float(
    np.mean(singleton_mask)
)

ambiguous_rate = float(
    np.mean(ambiguous_mask)
)

mean_set_size = float(
    np.mean(set_sizes)
)


# ============================================================
# 19. SINGLETON DECISION ACCURACY
# ============================================================

singleton_indices = np.where(
    singleton_mask
)[0]

if len(singleton_indices) > 0:

    singleton_predictions = np.array([
        prediction_sets[i][0]
        for i in singleton_indices
    ])

    singleton_true_labels = (
        true_labels[
            singleton_indices
        ]
    )

    singleton_accuracy = float(
        np.mean(
            singleton_predictions
            ==
            singleton_true_labels
        )
    )

else:

    singleton_accuracy = np.nan


# ============================================================
# 20. FRAUD-ONLY SET PRECISION
# ============================================================

fraud_only_mask = np.array([
    prediction_set == [1]
    for prediction_set
    in prediction_sets
])

nonfraud_only_mask = np.array([
    prediction_set == [0]
    for prediction_set
    in prediction_sets
])


fraud_only_count = int(
    np.sum(
        fraud_only_mask
    )
)

nonfraud_only_count = int(
    np.sum(
        nonfraud_only_mask
    )
)


if fraud_only_count > 0:

    fraud_only_precision = float(
        np.mean(
            true_labels[
                fraud_only_mask
            ]
            ==
            1
        )
    )

else:

    fraud_only_precision = np.nan


if nonfraud_only_count > 0:

    nonfraud_only_precision = float(
        np.mean(
            true_labels[
                nonfraud_only_mask
            ]
            ==
            0
        )
    )

else:

    nonfraud_only_precision = np.nan


# ============================================================
# 21. COVERAGE BY TRUE CLASS
# ============================================================

class_0_total = int(
    np.sum(class_0_mask)
)

class_1_total = int(
    np.sum(class_1_mask)
)

class_0_covered = int(
    np.sum(
        covered[
            class_0_mask
        ]
    )
)

class_1_covered = int(
    np.sum(
        covered[
            class_1_mask
        ]
    )
)


# ============================================================
# 22. DISPLAY FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FINAL TEST-SET MONDRIAN CONFORMAL RESULTS"
)

print(
    "=" * 80
)

print(
    f"Test samples               : {n_test}"
)

print(
    f"Target coverage            : "
    f"{target_coverage:.6f}"
)

print(
    f"Observed overall coverage  : "
    f"{overall_coverage:.6f}"
)

print(
    f"Overall coverage gap       : "
    f"{overall_coverage_gap:+.6f}"
)

print(
    f"Non-fraud coverage         : "
    f"{class_0_coverage:.6f}"
)

print(
    f"Fraud coverage             : "
    f"{class_1_coverage:.6f}"
)

print(
    f"Empty-set rate             : "
    f"{empty_rate:.6f}"
)

print(
    f"Singleton-set rate         : "
    f"{singleton_rate:.6f}"
)

print(
    f"Ambiguous [0,1] rate       : "
    f"{ambiguous_rate:.6f}"
)

print(
    f"Mean prediction-set size   : "
    f"{mean_set_size:.6f}"
)

print(
    f"Singleton accuracy         : "
    f"{singleton_accuracy:.6f}"
)

print(
    f"Fraud-only sets            : "
    f"{fraud_only_count}"
)

print(
    f"Non-fraud-only sets        : "
    f"{nonfraud_only_count}"
)

print(
    f"Fraud-only precision       : "
    f"{fraud_only_precision:.6f}"
)

print(
    f"Non-fraud-only precision   : "
    f"{nonfraud_only_precision:.6f}"
)


# ============================================================
# 23. CREATE RESULT TABLE
# ============================================================

results_df = pd.DataFrame({

    "dataset_index":
        test_indices,

    "actual_FLAG":
        true_labels,

    "probability_nonfraud":
        p_nonfraud,

    "probability_fraud":
        p_fraud,

    "q_hat_nonfraud":
        q_0,

    "q_hat_fraud":
        q_1,

    "threshold_nonfraud":
        p_threshold_0,

    "threshold_fraud":
        p_threshold_1,

    "prediction_set":
        [
            str(
                prediction_set
            )
            for prediction_set
            in prediction_sets
        ],

    "prediction_set_size":
        set_sizes,

    "covered":
        covered.astype(int),

    "is_empty_set":
        empty_mask.astype(int),

    "is_singleton":
        singleton_mask.astype(int),

    "is_ambiguous":
        ambiguous_mask.astype(int)
})


# ============================================================
# 24. ADD INTERPRETABLE DECISION
# ============================================================

def conformal_decision(
    prediction_set
):

    if prediction_set == [0]:

        return "NON_FRAUD"

    elif prediction_set == [1]:

        return "FRAUD"

    elif prediction_set == [0, 1]:

        return "AMBIGUOUS"

    else:

        return "ABSTAIN"


results_df[
    "conformal_decision"
] = [
    conformal_decision(
        prediction_set
    )
    for prediction_set
    in prediction_sets
]


# ============================================================
# 25. SAVE TEST PREDICTIONS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    "\nSaved test prediction sets:"
)

print(
    RESULTS_PATH
)


# ============================================================
# 26. CREATE METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame([

    {
        "metric": "q_hat_nonfraud",
        "value": q_0
    },

    {
        "metric": "q_hat_fraud",
        "value": q_1
    },

    {
        "metric": "probability_threshold_nonfraud",
        "value": p_threshold_0
    },

    {
        "metric": "probability_threshold_fraud",
        "value": p_threshold_1
    },

    {
        "metric": "target_coverage",
        "value": target_coverage
    },

    {
        "metric": "observed_overall_coverage",
        "value": overall_coverage
    },

    {
        "metric": "overall_coverage_gap",
        "value": overall_coverage_gap
    },

    {
        "metric": "nonfraud_class_coverage",
        "value": class_0_coverage
    },

    {
        "metric": "fraud_class_coverage",
        "value": class_1_coverage
    },

    {
        "metric": "empty_set_rate",
        "value": empty_rate
    },

    {
        "metric": "singleton_set_rate",
        "value": singleton_rate
    },

    {
        "metric": "ambiguous_set_rate",
        "value": ambiguous_rate
    },

    {
        "metric": "mean_prediction_set_size",
        "value": mean_set_size
    },

    {
        "metric": "singleton_accuracy",
        "value": singleton_accuracy
    },

    {
        "metric": "fraud_only_set_count",
        "value": fraud_only_count
    },

    {
        "metric": "nonfraud_only_set_count",
        "value": nonfraud_only_count
    },

    {
        "metric": "fraud_only_precision",
        "value": fraud_only_precision
    },

    {
        "metric": "nonfraud_only_precision",
        "value": nonfraud_only_precision
    },

    {
        "metric": "test_nonfraud_count",
        "value": class_0_total
    },

    {
        "metric": "test_fraud_count",
        "value": class_1_total
    },

    {
        "metric": "covered_nonfraud_count",
        "value": class_0_covered
    },

    {
        "metric": "covered_fraud_count",
        "value": class_1_covered
    }
])


# ============================================================
# 27. SAVE METRICS
# ============================================================

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

print(
    "Saved conformal metrics:"
)

print(
    METRICS_PATH
)


# ============================================================
# 28. INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "INTERPRETATION"
)

print(
    "=" * 80
)

if overall_coverage >= target_coverage:

    print(
        "Overall observed coverage meets or exceeds "
        "the target coverage on this test sample."
    )

else:

    print(
        "Overall observed coverage is below "
        "the target coverage on this test sample."
    )


if class_0_coverage >= target_coverage:

    print(
        "Non-fraud class coverage meets the target "
        "on this test sample."
    )

else:

    print(
        "Non-fraud class coverage is below the target "
        "on this test sample."
    )


if class_1_coverage >= target_coverage:

    print(
        "Fraud class coverage meets the target "
        "on this test sample."
    )

else:

    print(
        "WARNING: Fraud class coverage is below "
        "the target on this test sample."
    )


if empty_rate > 0:

    print(
        f"Empty prediction sets: {empty_rate:.2%}"
    )

    print(
        "Application interpretation: ABSTAIN / "
        "INCONCLUSIVE."
    )


if ambiguous_rate > 0:

    print(
        f"Ambiguous [0,1] prediction sets: "
        f"{ambiguous_rate:.2%}"
    )

else:

    print(
        "No ambiguous [0,1] prediction sets "
        "were observed."
    )


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "MONDRIAN CONFORMAL EVALUATION COMPLETE"
)

print(
    "=" * 80
)

print(
    f"q_0                  : {q_0:.12f}"
)

print(
    f"q_1                  : {q_1:.12f}"
)

print(
    f"Target coverage      : "
    f"{target_coverage:.2%}"
)

print(
    f"Observed coverage    : "
    f"{overall_coverage:.2%}"
)

print(
    f"Non-fraud coverage   : "
    f"{class_0_coverage:.2%}"
)

print(
    f"Fraud coverage       : "
    f"{class_1_coverage:.2%}"
)

print(
    f"Empty sets           : "
    f"{empty_rate:.2%}"
)

print(
    f"Singleton sets       : "
    f"{singleton_rate:.2%}"
)

print(
    f"Ambiguous sets       : "
    f"{ambiguous_rate:.2%}"
)

print(
    f"Mean set size        : "
    f"{mean_set_size:.4f}"
)

print(
    "\nOutput files:"
)

print(
    RESULTS_PATH
)

print(
    METRICS_PATH
)