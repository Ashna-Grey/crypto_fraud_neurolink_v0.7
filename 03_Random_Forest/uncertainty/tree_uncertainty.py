# ============================================================
# tree_uncertainty.py
# Ethereum Fraud Detection - Random Forest
# Tree-Ensemble Uncertainty Estimation
#
# Purpose:
#   1. Load the already-trained 20-feature Random Forest
#   2. Load the training-only imputer and feature schema
#   3. Load the untouched final-test split
#   4. Obtain fraud probabilities from EVERY individual tree
#   5. Calculate:
#
#        μ_raw(x)
#        = mean probability across RF trees
#
#        σ²_RF(x)
#        = variance across individual tree probabilities
#
#   6. Verify tree-average probability against RF.predict_proba()
#   7. Study whether higher σ²_RF corresponds to higher error
#   8. Save wallet-level uncertainty results
#   9. Save uncertainty evaluation metrics
#  10. Save uncertainty configuration
#
# IMPORTANT:
#   - The Random Forest is NOT retrained.
#   - No calibration is fitted here.
#   - No conformal prediction is fitted here.
#   - The final test set is used only for evaluation.
#
# Research interpretation:
#
#   μ_raw  = ensemble mean RF probability
#   σ²_RF  = tree-disagreement variance
#
#   σ²_RF is treated as an ensemble-disagreement /
#   epistemic-style uncertainty measure.
# ============================================================


import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# Current file:
# 03_Random_Forest/uncertainty/tree_uncertainty.py

UNCERTAINTY_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# 03_Random_Forest/
BASE_DIR = os.path.dirname(
    UNCERTAINTY_DIR
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

# 03_Random_Forest/analysis/
ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET_PATH = "C:/Users/ASHNA/Desktop/Ashna/crypto-fraud-intelligence/github_files/Crypto-fraud-intelligence-main/01_Dataset/processed/ethereum_fraud_25_features.csv"

# ------------------------------------------------------------
# Existing model artifacts
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

# ------------------------------------------------------------
# New outputs
# ------------------------------------------------------------

UNCERTAINTY_RESULTS_PATH = os.path.join(
    ANALYSIS_DIR,
    "rf_tree_uncertainty.csv"
)

UNCERTAINTY_METRICS_PATH = os.path.join(
    ANALYSIS_DIR,
    "uncertainty_metrics.csv"
)

UNCERTAINTY_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "rf_uncertainty_config.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "FLAG"

RANDOM_STATE = 42

# Number of bins used to study uncertainty/error relationship
N_UNCERTAINTY_BINS = 5


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
# 4. FILE CHECKS
# ============================================================

required_files = {
    "Dataset": DATASET_PATH,
    "RF model": MODEL_PATH,
    "Imputer": IMPUTER_PATH,
    "Feature schema": FEATURE_SCHEMA_PATH,
    "Split indices": SPLIT_PATH
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

T = len(rf.estimators_)


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

print("\nFeature order:")

for i, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 8. LOAD DATA SPLIT
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

test_indices = np.asarray(
    split_data["test_indices"]
)

print(
    "\nFinal test samples:",
    len(test_indices)
)


# ============================================================
# 9. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset shape:",
    df.shape
)

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# 10. VERIFY FEATURES
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
# 11. PREPARE FINAL TEST SET
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
    "\nFinal test class distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 12. APPLY TRAINING-ONLY IMPUTER
# ============================================================

X_test_imp = imputer.transform(
    X_test
)

print(
    "\nImputed test shape:",
    X_test_imp.shape
)


# ============================================================
# 13. GET FRAUD CLASS INDEX
# ============================================================

if 1 not in rf.classes_:

    raise ValueError(
        "Fraud class 1 was not found in RF classes."
    )

fraud_class_index = list(
    rf.classes_
).index(1)


# ============================================================
# 14. FUNCTION:
#     INDIVIDUAL TREE FRAUD PROBABILITIES
# ============================================================

def get_tree_fraud_probabilities(
    model,
    X
):
    """
    Return fraud probabilities from every individual
    Random Forest tree.

    Output shape:
        (n_trees, n_samples)

    Each row corresponds to one tree.
    Each column corresponds to one sample.
    """

    tree_probabilities = []

    for tree_number, tree in enumerate(
        model.estimators_,
        start=1
    ):

        # Some individual trees may theoretically contain
        # only one class. Handle this safely.
        tree_classes = list(
            tree.classes_
        )

        if 1 in tree_classes:

            tree_fraud_index = (
                tree_classes.index(1)
            )

            probabilities = (
                tree.predict_proba(X)[
                    :, tree_fraud_index
                ]
            )

        else:

            # If tree has no fraud class,
            # its estimated fraud probability is 0.
            probabilities = np.zeros(
                X.shape[0],
                dtype=float
            )

        tree_probabilities.append(
            probabilities
        )

    return np.asarray(
        tree_probabilities,
        dtype=float
    )


# ============================================================
# 15. FUNCTION:
#     CALCULATE RF μ AND σ²
# ============================================================

def calculate_tree_uncertainty(
    model,
    X
):
    """
    Calculate:

        μ_raw(x)
        = mean individual-tree fraud probability

        σ²_RF(x)
        = sample variance across tree probabilities

    Returns:
        mean_probability
        variance
        standard_deviation
        tree_probabilities
    """

    tree_probabilities = (
        get_tree_fraud_probabilities(
            model,
            X
        )
    )

    # Mean across trees
    mean_probability = np.mean(
        tree_probabilities,
        axis=0
    )

    # Sample variance across trees
    variance = np.var(
        tree_probabilities,
        axis=0,
        ddof=1
    )

    # Standard deviation
    standard_deviation = np.sqrt(
        variance
    )

    return (
        mean_probability,
        variance,
        standard_deviation,
        tree_probabilities
    )


# ============================================================
# 16. CALCULATE TREE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING INDIVIDUAL TREE PROBABILITIES")
print("=" * 80)

(
    mu_raw,
    sigma_squared_rf,
    sigma_rf,
    tree_probabilities
) = calculate_tree_uncertainty(
    rf,
    X_test_imp
)

print(
    "Tree probability matrix shape:",
    tree_probabilities.shape
)

print(
    "Expected:",
    (T, X_test_imp.shape[0])
)


# ============================================================
# 17. VERIFY AGAINST RF.predict_proba()
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING RF PROBABILITY")
print("=" * 80)

rf_probability = rf.predict_proba(
    X_test_imp
)[:, fraud_class_index]

max_probability_difference = np.max(
    np.abs(
        mu_raw
        -
        rf_probability
    )
)

mean_probability_difference = np.mean(
    np.abs(
        mu_raw
        -
        rf_probability
    )
)

print(
    "Maximum absolute difference:",
    f"{max_probability_difference:.12f}"
)

print(
    "Mean absolute difference:",
    f"{mean_probability_difference:.12f}"
)

# The difference should be extremely small.
if max_probability_difference > 1e-10:

    print(
        "\nWARNING:"
        " mean tree probability differs from"
        " RF.predict_proba()."
    )

else:

    print(
        "\n[OK]"
        " Mean individual-tree probability"
        " matches RF.predict_proba()."
    )


# ============================================================
# 18. RF CLASS PREDICTION
# ============================================================

rf_prediction = (
    rf_probability >= 0.50
).astype(int)


# ============================================================
# 19. CALCULATE TEST-SET METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

test_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

test_pr_auc = average_precision_score(
    y_test,
    rf_probability
)


# ============================================================
# 20. UNCERTAINTY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("RF TREE UNCERTAINTY STATISTICS")
print("=" * 80)

print(
    f"Mean μ_raw             : {mu_raw.mean():.6f}"
)

print(
    f"Mean σ²_RF             : {sigma_squared_rf.mean():.6f}"
)

print(
    f"Median σ²_RF           : {np.median(sigma_squared_rf):.6f}"
)

print(
    f"Minimum σ²_RF          : {sigma_squared_rf.min():.6f}"
)

print(
    f"Maximum σ²_RF          : {sigma_squared_rf.max():.6f}"
)

print(
    f"Mean σ_RF              : {sigma_rf.mean():.6f}"
)

print(
    f"Median σ_RF            : {np.median(sigma_rf):.6f}"
)


# ============================================================
# 21. BUILD WALLET-LEVEL RESULTS
# ============================================================

uncertainty_results = pd.DataFrame({

    "dataset_index": test_indices,

    "rf_mu_raw": mu_raw,

    "rf_sigma_squared": sigma_squared_rf,

    "rf_sigma": sigma_rf,

    "rf_prediction": rf_prediction,

    "actual_FLAG": y_test.to_numpy(),

})

uncertainty_results["correct"] = (
    uncertainty_results["rf_prediction"]
    ==
    uncertainty_results["actual_FLAG"]
).astype(int)

uncertainty_results["absolute_error"] = np.abs(
    uncertainty_results["rf_mu_raw"]
    -
    uncertainty_results["actual_FLAG"]
)


# ============================================================
# 22. UNCERTAINTY QUANTILES
# ============================================================

uncertainty_quantiles = (
    uncertainty_results[
        "rf_sigma_squared"
    ]
    .quantile(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00
        ]
    )
)

print(
    "\n" + "=" * 80
)

print(
    "RF σ² QUANTILES"
)

print(
    "=" * 80
)

for quantile, value in (
    uncertainty_quantiles.items()
):

    print(
        f"{quantile:.2f}: {value:.8f}"
    )


# ============================================================
# 23. STUDY UNCERTAINTY VS ERROR
# ============================================================
#
# We divide the test predictions into uncertainty bins.
#
# If σ²_RF is useful as an uncertainty measure,
# higher-uncertainty groups should generally show
# higher error rates.
# ============================================================

try:

    uncertainty_results["uncertainty_bin"] = pd.qcut(
        uncertainty_results[
            "rf_sigma_squared"
        ],
        q=N_UNCERTAINTY_BINS,
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        duplicates="drop"
    )

except ValueError:

    uncertainty_results["uncertainty_bin"] = pd.cut(
        uncertainty_results[
            "rf_sigma_squared"
        ],
        bins=N_UNCERTAINTY_BINS,
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        include_lowest=True
    )


uncertainty_grouped = (
    uncertainty_results
    .groupby(
        "uncertainty_bin",
        observed=False
    )
    .agg(
        samples=(
            "correct",
            "count"
        ),
        accuracy=(
            "correct",
            "mean"
        ),
        mean_sigma_squared=(
            "rf_sigma_squared",
            "mean"
        ),
        mean_sigma=(
            "rf_sigma",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        )
    )
    .reset_index()
)

uncertainty_grouped[
    "error_rate"
] = (
    1.0
    -
    uncertainty_grouped[
        "accuracy"
    ]
)


# ============================================================
# 24. DISPLAY UNCERTAINTY GROUP ANALYSIS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "UNCERTAINTY VS PREDICTION ERROR"
)

print(
    "=" * 80
)

print(
    uncertainty_grouped.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 25. CORRELATION BETWEEN UNCERTAINTY AND ERROR
# ============================================================
#
# We expect a positive relationship:
#
#     σ²_RF ↑  -> error tendency ↑
#
# This is an empirical validation, NOT an assumption.
# ============================================================

uncertainty_error_corr = (
    uncertainty_results[
        "rf_sigma_squared"
    ]
    .corr(
        uncertainty_results[
            "absolute_error"
        ],
        method="spearman"
    )
)

uncertainty_correct_corr = (
    uncertainty_results[
        "rf_sigma_squared"
    ]
    .corr(
        uncertainty_results[
            "correct"
        ],
        method="spearman"
    )
)

print(
    "\nSpearman correlation:"
)

print(
    "σ²_RF vs absolute error:",
    f"{uncertainty_error_corr:.6f}"
)

print(
    "σ²_RF vs correctness   :",
    f"{uncertainty_correct_corr:.6f}"
)


# ============================================================
# 26. CREATE RESEARCH METRICS TABLE
# ============================================================

metrics_rows = []


# ------------------------------------------------------------
# Overall RF performance
# ------------------------------------------------------------

metrics_rows.append({

    "analysis": "overall_test_performance",

    "group": "all",

    "samples": len(
        uncertainty_results
    ),

    "mean_sigma_squared": (
        sigma_squared_rf.mean()
    ),

    "mean_sigma": (
        sigma_rf.mean()
    ),

    "accuracy": test_accuracy,

    "precision": test_precision,

    "recall": test_recall,

    "f1": test_f1,

    "roc_auc": test_roc_auc,

    "pr_auc": test_pr_auc,

    "error_rate": (
        1.0
        -
        test_accuracy
    )
})


# ------------------------------------------------------------
# Uncertainty groups
# ------------------------------------------------------------

for _, row in uncertainty_grouped.iterrows():

    metrics_rows.append({

        "analysis": "uncertainty_bin",

        "group": str(
            row["uncertainty_bin"]
        ),

        "samples": int(
            row["samples"]
        ),

        "mean_sigma_squared": (
            row["mean_sigma_squared"]
        ),

        "mean_sigma": (
            row["mean_sigma"]
        ),

        "accuracy": (
            row["accuracy"]
        ),

        "precision": np.nan,

        "recall": np.nan,

        "f1": np.nan,

        "roc_auc": np.nan,

        "pr_auc": np.nan,

        "error_rate": (
            row["error_rate"]
        )
    })


uncertainty_metrics_df = pd.DataFrame(
    metrics_rows
)


# ============================================================
# 27. ADD CORRELATION ROW
# ============================================================

correlation_row = pd.DataFrame([{

    "analysis": "uncertainty_error_correlation",

    "group": "Spearman",

    "samples": len(
        uncertainty_results
    ),

    "mean_sigma_squared": np.nan,

    "mean_sigma": np.nan,

    "accuracy": np.nan,

    "precision": np.nan,

    "recall": np.nan,

    "f1": np.nan,

    "roc_auc": np.nan,

    "pr_auc": np.nan,

    "error_rate": np.nan,

    "sigma_squared_error_correlation": (
        uncertainty_error_corr
    ),

    "sigma_squared_correctness_correlation": (
        uncertainty_correct_corr
    )
}])


uncertainty_metrics_df = pd.concat(
    [
        uncertainty_metrics_df,
        correlation_row
    ],
    ignore_index=True
)


# ============================================================
# 28. SAVE WALLET-LEVEL UNCERTAINTY
# ============================================================

uncertainty_results.to_csv(
    UNCERTAINTY_RESULTS_PATH,
    index=False
)

print(
    "\nSaved wallet-level uncertainty:"
)

print(
    UNCERTAINTY_RESULTS_PATH
)


# ============================================================
# 29. SAVE UNCERTAINTY METRICS
# ============================================================

uncertainty_metrics_df.to_csv(
    UNCERTAINTY_METRICS_PATH,
    index=False
)

print(
    "Saved uncertainty metrics:"
)

print(
    UNCERTAINTY_METRICS_PATH
)


# ============================================================
# 30. SAVE CONFIGURATION
# ============================================================

uncertainty_config = {

    "model": "RandomForestClassifier",

    "uncertainty_method": (
        "empirical variance of individual-tree "
        "fraud probabilities"
    ),

    "uncertainty_interpretation": (
        "tree ensemble disagreement"
    ),

    "variance_formula": (
        "sigma_squared_RF = "
        "sum((p_t - mean_p)^2) / (T - 1)"
    ),

    "mean_formula": (
        "mu_raw = sum(p_t) / T"
    ),

    "n_estimators": int(T),

    "feature_count": len(FEATURES),

    "features": FEATURES,

    "test_samples": int(
        len(test_indices)
    ),

    "test_accuracy": float(
        test_accuracy
    ),

    "test_precision": float(
        test_precision
    ),

    "test_recall": float(
        test_recall
    ),

    "test_f1": float(
        test_f1
    ),

    "test_roc_auc": float(
        test_roc_auc
    ),

    "test_pr_auc": float(
        test_pr_auc
    ),

    "mean_sigma_squared": float(
        sigma_squared_rf.mean()
    ),

    "median_sigma_squared": float(
        np.median(
            sigma_squared_rf
        )
    ),

    "min_sigma_squared": float(
        sigma_squared_rf.min()
    ),

    "max_sigma_squared": float(
        sigma_squared_rf.max()
    ),

    "spearman_sigma_squared_absolute_error": (
        float(
            uncertainty_error_corr
        )
    ),

    "spearman_sigma_squared_correctness": (
        float(
            uncertainty_correct_corr
        )
    ),

    "random_state": RANDOM_STATE
}


with open(
    UNCERTAINTY_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        uncertainty_config,
        f,
        indent=4
    )


print(
    "Saved uncertainty configuration:"
)

print(
    UNCERTAINTY_CONFIG_PATH
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TREE UNCERTAINTY ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Trees                  : {T}"
)

print(
    f"Features               : {len(FEATURES)}"
)

print(
    f"Test samples            : {len(y_test)}"
)

print(
    f"Mean μ_raw              : {mu_raw.mean():.6f}"
)

print(
    f"Mean σ²_RF              : "
    f"{sigma_squared_rf.mean():.6f}"
)

print(
    f"Median σ²_RF            : "
    f"{np.median(sigma_squared_rf):.6f}"
)

print(
    f"Maximum σ²_RF           : "
    f"{sigma_squared_rf.max():.6f}"
)

print(
    f"Test Accuracy           : "
    f"{test_accuracy:.6f}"
)

print(
    f"Test F1                 : "
    f"{test_f1:.6f}"
)

print(
    f"Test ROC-AUC            : "
    f"{test_roc_auc:.6f}"
)

print(
    f"Test PR-AUC             : "
    f"{test_pr_auc:.6f}"
)

print(
    f"σ² vs absolute error    : "
    f"{uncertainty_error_corr:.6f}"
)

print(
    "\nOutputs:"
)

print(
    UNCERTAINTY_RESULTS_PATH
)

print(
    UNCERTAINTY_METRICS_PATH
)

print(
    UNCERTAINTY_CONFIG_PATH
)