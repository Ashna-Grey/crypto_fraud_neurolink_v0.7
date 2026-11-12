# ============================================================
# fit_conformal.py
# Ethereum Fraud Detection - Random Forest
# Class-Conditional (Mondrian) Split Conformal Prediction
#
# Purpose:
#   1. Load the trained 20-feature Random Forest
#   2. Load the fitted probability calibrator
#   3. Load ONLY the dedicated conformal calibration set
#   4. Generate calibrated probabilities
#   5. Compute class-specific nonconformity scores:
#
#        s_i = 1 - P_calibrated(y_i | x_i)
#
#   6. Compute separate conformal quantiles:
#
#        q_0 -> non-fraud class
#        q_1 -> fraud class
#
#   7. Save q_0 and q_1 for inference
#   8. Save calibration scores for analysis
#
# IMPORTANT:
#   - RF is NOT retrained.
#   - Probability calibrator is NOT retrained.
#   - Final test set is NEVER used here.
#   - q_0 and q_1 are fitted only from the conformal set.
#
# Inference rule:
#
#   Include class c when:
#
#       1 - P_calibrated(c | x) <= q_c
#
# Binary output:
#
#   [0]      -> non-fraud only
#   [1]      -> fraud only
#   [0, 1]   -> ambiguous
#   []       -> abstain / inconclusive
# ============================================================


import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

CONFORMAL_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    CONFORMAL_DIR
)

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
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

# Use the SAME dataset used by train.py.
# Change this only if your repository path differs.

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

CALIBRATION_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "rf_calibration_config.json"
)


# ============================================================
# NEW OUTPUTS
# ============================================================

CONFORMAL_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "conformal_config.json"
)

CONFORMAL_SCORES_PATH = os.path.join(
    ANALYSIS_DIR,
    "conformal_calibration_scores.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "FLAG"

# 95% target marginal coverage
ALPHA = 0.05

TARGET_COVERAGE = 1.0 - ALPHA


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Dataset": DATASET_PATH,
    "RF model": MODEL_PATH,
    "Imputer": IMPUTER_PATH,
    "Feature schema": SCHEMA_PATH,
    "Split indices": SPLIT_PATH,
    "RF calibrator": CALIBRATOR_PATH,
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

    print(
        f"[OK] {name}: {path}"
    )


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
    SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)

FEATURES = schema.get(
    "feature_order",
    schema["features"]
)

print(
    "\nFeature count:",
    len(FEATURES)
)

print(
    "Feature order:"
)

for i, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
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
            f"Missing split key '{key}' "
            f"in {SPLIT_PATH}"
        )

conformal_indices = np.asarray(
    split_data["conformal_indices"]
)

print(
    "\nConformal samples:",
    len(conformal_indices)
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

calibration_method = calibration_config.get(
    "calibration_method",
    "unknown"
)

print(
    "Probability calibration:",
    calibration_method
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
# 12. VERIFY FEATURE SCHEMA
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
# 13. CREATE CONFORMAL DATASET
# ============================================================

X_all = df[
    FEATURES
].copy()

y_all = df[
    TARGET
].astype(int).copy()

X_conf = X_all.loc[
    conformal_indices
].copy()

y_conf = y_all.loc[
    conformal_indices
].copy()

print(
    "\nConformal X shape:",
    X_conf.shape
)

print(
    "Conformal y shape:",
    y_conf.shape
)

print(
    "\nConformal FLAG distribution:"
)

print(
    y_conf.value_counts()
)


# ============================================================
# 14. APPLY TRAINING-FITTED IMPUTER
# ============================================================

X_conf_imp = imputer.transform(
    X_conf
)

print(
    "\nImputed conformal shape:",
    X_conf_imp.shape
)


# ============================================================
# 15. GET CALIBRATED PROBABILITIES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GENERATING CALIBRATED PROBABILITIES"
)

print(
    "=" * 80
)

probabilities = calibrator.predict_proba(
    X_conf_imp
)

classes = np.asarray(
    calibrator.classes_
)

print(
    "Probability matrix shape:",
    probabilities.shape
)

print(
    "Classes:",
    classes
)


# ============================================================
# 16. VERIFY BINARY CLASSES
# ============================================================

if 0 not in classes or 1 not in classes:

    raise ValueError(
        "This implementation requires binary "
        "classes {0, 1}."
    )

class_0_index = np.where(
    classes == 0
)[0][0]

class_1_index = np.where(
    classes == 1
)[0][0]


# ============================================================
# 17. SPLIT BY TRUE CLASS
# ============================================================
#
# Mondrian/class-conditional conformal calibration:
#
#   q_0 is calculated only from true class 0 examples.
#
#   q_1 is calculated only from true class 1 examples.
# ============================================================

y_conf_array = y_conf.to_numpy()

class_0_mask = (
    y_conf_array == 0
)

class_1_mask = (
    y_conf_array == 1
)


n_class_0 = int(
    np.sum(class_0_mask)
)

n_class_1 = int(
    np.sum(class_1_mask)
)

if n_class_0 < 1:

    raise ValueError(
        "No non-fraud samples in conformal set."
    )

if n_class_1 < 1:

    raise ValueError(
        "No fraud samples in conformal set."
    )


# ============================================================
# 18. CALCULATE CLASS-SPECIFIC NONCONFORMITY SCORES
# ============================================================

# ------------------------------------------------------------
# Class 0: non-fraud
#
# score = 1 - P(non-fraud | x)
# ------------------------------------------------------------

class_0_probabilities = (
    probabilities[
        class_0_mask,
        class_0_index
    ]
)

scores_0 = (
    1.0
    -
    class_0_probabilities
)


# ------------------------------------------------------------
# Class 1: fraud
#
# score = 1 - P(fraud | x)
# ------------------------------------------------------------

class_1_probabilities = (
    probabilities[
        class_1_mask,
        class_1_index
    ]
)

scores_1 = (
    1.0
    -
    class_1_probabilities
)


# ============================================================
# 19. FINITE-SAMPLE CLASS-CONDITIONAL QUANTILE
# ============================================================

def conformal_quantile(
    scores,
    alpha
):
    """
    Calculate the finite-sample split-conformal
    quantile.

    For n calibration examples:

        k = ceil((n + 1)(1 - alpha))

    q_hat = k-th smallest score

    The rank is capped at n for the finite-sample
    edge case where k = n + 1.
    """

    scores = np.asarray(
        scores,
        dtype=float
    )

    n = len(scores)

    if n == 0:

        raise ValueError(
            "Cannot calculate conformal quantile "
            "from an empty score set."
        )

    k = int(
        np.ceil(
            (n + 1)
            *
            (1.0 - alpha)
        )
    )

    k = max(
        1,
        min(
            k,
            n
        )
    )

    sorted_scores = np.sort(
        scores
    )

    q_hat = float(
        sorted_scores[k - 1]
    )

    return q_hat, k


# ============================================================
# 20. CALCULATE q_0 AND q_1
# ============================================================

q_0, rank_0 = conformal_quantile(
    scores_0,
    ALPHA
)

q_1, rank_1 = conformal_quantile(
    scores_1,
    ALPHA
)


# ============================================================
# 21. DISPLAY SCORE STATISTICS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CLASS-CONDITIONAL NONCONFORMITY"
)

print(
    "=" * 80
)

print(
    f"Class 0 samples : {n_class_0}"
)

print(
    f"Class 1 samples : {n_class_1}"
)

print(
    "\nClass 0 score statistics:"
)

print(
    f"Mean   : {scores_0.mean():.6f}"
)

print(
    f"Median : {np.median(scores_0):.6f}"
)

print(
    f"Min    : {scores_0.min():.6f}"
)

print(
    f"Max    : {scores_0.max():.6f}"
)

print(
    "\nClass 1 score statistics:"
)

print(
    f"Mean   : {scores_1.mean():.6f}"
)

print(
    f"Median : {np.median(scores_1):.6f}"
)

print(
    f"Min    : {scores_1.min():.6f}"
)

print(
    f"Max    : {scores_1.max():.6f}"
)


# ============================================================
# 22. DISPLAY CLASS-SPECIFIC QUANTILES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CLASS-CONDITIONAL CONFORMAL THRESHOLDS"
)

print(
    "=" * 80
)

print(
    f"Alpha             : {ALPHA:.6f}"
)

print(
    f"Target coverage   : {TARGET_COVERAGE:.6f}"
)

print(
    f"\nq_0 (non-fraud)   : {q_0:.12f}"
)

print(
    f"q_1 (fraud)       : {q_1:.12f}"
)

print(
    f"\nProbability threshold for class 0:"
)

print(
    f"P(non-fraud|x) >= "
    f"{1.0 - q_0:.12f}"
)

print(
    f"\nProbability threshold for class 1:"
)

print(
    f"P(fraud|x) >= "
    f"{1.0 - q_1:.12f}"
)

print(
    f"\nRank for class 0 : {rank_0}"
)

print(
    f"Rank for class 1 : {rank_1}"
)


# ============================================================
# 23. CREATE CALIBRATION-SET PREDICTION SETS
# ============================================================

prediction_sets = []

for probabilities_row in probabilities:

    prediction_set = []

    p0 = probabilities_row[
        class_0_index
    ]

    p1 = probabilities_row[
        class_1_index
    ]

    # Class 0
    if (
        1.0 - p0
    ) <= q_0:

        prediction_set.append(0)

    # Class 1
    if (
        1.0 - p1
    ) <= q_1:

        prediction_set.append(1)

    prediction_sets.append(
        prediction_set
    )


# ============================================================
# 24. DIAGNOSTIC COVERAGE ON CONFORMAL SET
# ============================================================
#
# IMPORTANT:
# This is only diagnostic because q_0 and q_1 were estimated
# from this same conformal set.
# ============================================================

covered = np.array([
    true_label in prediction_set
    for true_label, prediction_set
    in zip(
        y_conf_array,
        prediction_sets
    )
])


overall_coverage = float(
    np.mean(covered)
)


class_0_coverage = float(
    np.mean(
        covered[
            class_0_mask
        ]
    )
)

class_1_coverage = float(
    np.mean(
        covered[
            class_1_mask
        ]
    )
)


# ============================================================
# 25. PREDICTION SET DIAGNOSTICS
# ============================================================

set_sizes = np.array([
    len(
        prediction_set
    )
    for prediction_set
    in prediction_sets
])

empty_count = int(
    np.sum(
        set_sizes == 0
    )
)

singleton_count = int(
    np.sum(
        set_sizes == 1
    )
)

ambiguous_count = int(
    np.sum(
        set_sizes == 2
    )


)

print(
    "\n" + "=" * 80
)

print(
    "CONFORMAL-SET DIAGNOSTICS"
)

print(
    "=" * 80
)

print(
    f"Overall coverage       : "
    f"{overall_coverage:.6f}"
)

print(
    f"Class 0 coverage       : "
    f"{class_0_coverage:.6f}"
)

print(
    f"Class 1 coverage       : "
    f"{class_1_coverage:.6f}"
)

print(
    f"Empty sets             : "
    f"{empty_count}"
)

print(
    f"Singleton sets         : "
    f"{singleton_count}"
)

print(
    f"Ambiguous [0,1] sets   : "
    f"{ambiguous_count}"
)

print(
    f"Mean set size          : "
    f"{set_sizes.mean():.6f}"
)


# ============================================================
# 26. SAVE CALIBRATION SCORES
# ============================================================

true_class_probability = np.zeros(
    len(y_conf),
    dtype=float
)

true_class_probability[
    class_0_mask
] = class_0_probabilities

true_class_probability[
    class_1_mask
] = class_1_probabilities


scores = np.zeros(
    len(y_conf),
    dtype=float
)

scores[
    class_0_mask
] = scores_0

scores[
    class_1_mask
] = scores_1


conformal_scores_df = pd.DataFrame({

    "dataset_index":
        conformal_indices,

    "actual_FLAG":
        y_conf_array,

    "calibrated_probability_nonfraud":
        probabilities[
            :,
            class_0_index
        ],

    "calibrated_probability_fraud":
        probabilities[
            :,
            class_1_index
        ],

    "true_class_probability":
        true_class_probability,

    "nonconformity_score":
        scores,

    "class_specific_q_hat":
        np.where(
            y_conf_array == 0,
            q_0,
            q_1
        ),

    "prediction_set":
        [
            str(
                prediction_set
            )
            for prediction_set
            in prediction_sets
        ]
})


# ============================================================
# 27. SAVE SCORES
# ============================================================

conformal_scores_df.to_csv(
    CONFORMAL_SCORES_PATH,
    index=False
)

print(
    "\nSaved conformal scores:"
)

print(
    CONFORMAL_SCORES_PATH
)


# ============================================================
# 28. SAVE CONFORMAL CONFIGURATION
# ============================================================

conformal_config = {

    "method":
        "mondrian_split_conformal",

    "method_description":
        "Class-conditional split conformal prediction",

    "nonconformity_method":
        "1 - calibrated true-class probability",

    "alpha":
        float(ALPHA),

    "target_coverage":
        float(TARGET_COVERAGE),

    "classes": {

        "0": {
            "label": "non-fraud",
            "sample_count": n_class_0,
            "q_hat": float(q_0),
            "probability_threshold":
                float(1.0 - q_0),
            "finite_sample_rank":
                int(rank_0)
        },

        "1": {
            "label": "fraud",
            "sample_count": n_class_1,
            "q_hat": float(q_1),
            "probability_threshold":
                float(1.0 - q_1),
            "finite_sample_rank":
                int(rank_1)
        }
    },

    "probability_calibration_method":
        calibration_method,

    "model":
        "RandomForestClassifier",

    "feature_count":
        int(len(FEATURES)),

    "features":
        FEATURES
}


# ============================================================
# 29. SAVE CONFIG
# ============================================================

with open(
    CONFORMAL_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        conformal_config,
        f,
        indent=4
    )


print(
    "Saved conformal configuration:"
)

print(
    CONFORMAL_CONFIG_PATH
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "MONDRIAN CONFORMAL CALIBRATION COMPLETE"
)

print(
    "=" * 80
)

print(
    f"Method               : "
    f"Class-conditional split conformal"
)

print(
    f"Alpha                : "
    f"{ALPHA:.4f}"
)

print(
    f"Target coverage      : "
    f"{TARGET_COVERAGE:.2%}"
)

print(
    f"Class 0 q_hat        : "
    f"{q_0:.12f}"
)

print(
    f"Class 1 q_hat        : "
    f"{q_1:.12f}"
)

print(
    f"Class 0 probability  : "
    f"{1.0 - q_0:.12f}"
)

print(
    f"Class 1 probability  : "
    f"{1.0 - q_1:.12f}"
)

print(
    "\nInterpretation:"
)

print(
    "Class 0 is included when "
    "P(non-fraud | x) >= 1 - q_0."
)

print(
    "Class 1 is included when "
    "P(fraud | x) >= 1 - q_1."
)

print(
    "\nArtifacts:"
)

print(
    CONFORMAL_CONFIG_PATH
)

print(
    CONFORMAL_SCORES_PATH
)