# ============================================================
# predict.py
# Ethereum Fraud Detection - Random Forest Inference
#
# RF pipeline:
#
#   Input
#     ↓
#   Training-fitted imputer
#     ↓
#   Random Forest
#     ↓
#   ┌───────────────────────────────────────────────┐
#   │                                               │
#   │  Raw RF probability                           │
#   │  Tree disagreement variance                  │
#   │  Sigmoid probability calibration             │
#   │  Mondrian conformal prediction               │
#   │                                               │
#   └───────────────────────────────────────────────┘
#
# Final RF outputs:
#
#   μ_RF          = calibrated fraud probability
#   σ²_RF         = tree-disagreement variance
#   C_RF          = conformal prediction set
#
# Conformal classes:
#
#   [0]      → NON_FRAUD
#   [1]      → FRAUD
#   [0, 1]   → AMBIGUOUS
#   []       → ABSTAIN / INCONCLUSIVE
#
# Input:
#   CSV containing the required 20 features
#
# Output:
#   Original input columns +
#   RF prediction, calibrated probability,
#   RF uncertainty and conformal decision
# ============================================================


import os
import json
import argparse

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ARTIFACT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "artifacts"
    )
)


# ============================================================
# ARTIFACT PATHS
# ============================================================

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "rf_ethereum_fraud_20features.joblib"
)

IMPUTER_PATH = os.path.join(
    ARTIFACT_DIR,
    "rf_ethereum_fraud_20features_imputer.joblib"
)

SCHEMA_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_schema.json"
)

CALIBRATOR_PATH = os.path.join(
    ARTIFACT_DIR,
    "rf_calibrator.joblib"
)

CONFORMAL_CONFIG_PATH = os.path.join(
    ARTIFACT_DIR,
    "conformal_config.json"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    required_files = {
        "Random Forest model": MODEL_PATH,
        "Imputer": IMPUTER_PATH,
        "Feature schema": SCHEMA_PATH,
        "RF calibrator": CALIBRATOR_PATH,
        "Conformal configuration": CONFORMAL_CONFIG_PATH
    }

    for name, path in required_files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Training-fitted imputer
    # --------------------------------------------------------

    imputer = joblib.load(
        IMPUTER_PATH
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    features = schema.get(
        "feature_order",
        schema["features"]
    )

    # --------------------------------------------------------
    # Probability calibrator
    # --------------------------------------------------------

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    # --------------------------------------------------------
    # Mondrian conformal configuration
    # --------------------------------------------------------

    with open(
        CONFORMAL_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        conformal_config = json.load(f)

    return (
        model,
        imputer,
        features,
        calibrator,
        conformal_config
    )


# ============================================================
# VALIDATE INPUT FEATURES
# ============================================================

def validate_features(
    df,
    required_features
):

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Input is missing required features:\n"
            +
            "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )


# ============================================================
# GET MONDRIAN CONFORMAL PARAMETERS
# ============================================================

def get_conformal_parameters(
    conformal_config
):

    # --------------------------------------------------------
    # Verify expected class-specific structure
    # --------------------------------------------------------

    if "classes" not in conformal_config:

        raise ValueError(
            "conformal_config.json does not contain "
            "class-specific 'classes'."
        )

    if "0" not in conformal_config["classes"]:

        raise ValueError(
            "Class 0 conformal configuration missing."
        )

    if "1" not in conformal_config["classes"]:

        raise ValueError(
            "Class 1 conformal configuration missing."
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

    # Probability thresholds:
    #
    # class 0:
    #   P(0|x) >= 1 - q_0
    #
    # class 1:
    #   P(1|x) >= 1 - q_1

    threshold_0 = 1.0 - q_0
    threshold_1 = 1.0 - q_1

    return (
        q_0,
        q_1,
        alpha,
        threshold_0,
        threshold_1
    )


# ============================================================
# GET INDIVIDUAL TREE FRAUD PROBABILITIES
# ============================================================

def get_tree_fraud_probabilities(
    model,
    X
):
    """
    Returns fraud probabilities from every individual
    Random Forest tree.

    Output shape:
        (n_trees, n_samples)
    """

    tree_probabilities = []

    for tree in model.estimators_:

        tree_classes = list(
            tree.classes_
        )

        if 1 in tree_classes:

            fraud_index = (
                tree_classes.index(1)
            )

            probabilities = (
                tree.predict_proba(
                    X.to_numpy()
                )[:, fraud_index]
            )

        else:

            # Defensive handling if a tree contains
            # no fraud class.
            probabilities = np.zeros(
                len(X),
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
# CALCULATE RF TREE UNCERTAINTY
# ============================================================

def calculate_rf_uncertainty(
    model,
    X
):
    """
    Calculate:

        μ_raw(x)
        = mean of individual tree probabilities

        σ²_RF(x)
        = sample variance of tree probabilities

        σ_RF(x)
        = sqrt(σ²_RF)

    Returns:
        mu_raw
        sigma_squared_rf
        sigma_rf
        tree_probabilities
    """

    tree_probabilities = (
        get_tree_fraud_probabilities(
            model,
            X
        )
    )

    # --------------------------------------------------------
    # Ensemble mean
    # --------------------------------------------------------

    mu_raw = np.mean(
        tree_probabilities,
        axis=0
    )

    # --------------------------------------------------------
    # Tree-disagreement variance
    # --------------------------------------------------------

    sigma_squared_rf = np.var(
        tree_probabilities,
        axis=0,
        ddof=1
    )

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    sigma_rf = np.sqrt(
        sigma_squared_rf
    )

    return (
        mu_raw,
        sigma_squared_rf,
        sigma_rf,
        tree_probabilities
    )


# ============================================================
# GENERATE MONDRIAN CONFORMAL SET
# ============================================================

def get_conformal_prediction_set(
    probabilities,
    classes,
    q_0,
    q_1
):
    """
    Class-conditional conformal prediction.

    Include class 0 if:
        1 - P(0|x) <= q_0

    Include class 1 if:
        1 - P(1|x) <= q_1
    """

    classes = np.asarray(
        classes
    )

    if 0 not in classes or 1 not in classes:

        raise ValueError(
            "Expected binary classes {0, 1}."
        )

    class_0_index = np.where(
        classes == 0
    )[0][0]

    class_1_index = np.where(
        classes == 1
    )[0][0]

    prediction_sets = []

    for row in probabilities:

        prediction_set = []

        p0 = float(
            row[class_0_index]
        )

        p1 = float(
            row[class_1_index]
        )

        # ----------------------------------------------------
        # Non-fraud
        # ----------------------------------------------------

        if (
            1.0 - p0
        ) <= q_0:

            prediction_set.append(0)

        # ----------------------------------------------------
        # Fraud
        # ----------------------------------------------------

        if (
            1.0 - p1
        ) <= q_1:

            prediction_set.append(1)

        prediction_sets.append(
            prediction_set
        )

    return prediction_sets


# ============================================================
# CONFORMAL DECISION
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

    elif prediction_set == []:

        return "ABSTAIN"

    else:

        return "ABSTAIN"


# ============================================================
# OPERATIONAL RISK LEVEL
# ============================================================

def get_risk_level(
    probability
):
    """
    Existing operational risk mapping.

        < 0.25  → LOW
        < 0.50  → MEDIUM
        < 0.75  → HIGH
        >= 0.75 → CRITICAL

    These are operational thresholds and are separate
    from conformal coverage.
    """

    if probability < 0.25:

        return "LOW"

    elif probability < 0.50:

        return "MEDIUM"

    elif probability < 0.75:

        return "HIGH"

    else:

        return "CRITICAL"


# ============================================================
# COMPLETE RF PREDICTION
# ============================================================

def predict_dataframe(
    df
):

    (
        model,
        imputer,
        features,
        calibrator,
        conformal_config
    ) = load_artifacts()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        df,
        features
    )

    # --------------------------------------------------------
    # Select EXACT training feature order
    # --------------------------------------------------------

    X = df[
        features
    ].copy()

    # --------------------------------------------------------
    # Apply training-fitted imputer
    # --------------------------------------------------------

    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=features,
        index=X.index
    )

    # ========================================================
    # 1. RAW RF PROBABILITY
    # ========================================================

    raw_rf_probabilities = (
        model.predict_proba(
            X_imputed
        )
    )

    model_classes = np.asarray(
        model.classes_
    )

    if 1 not in model_classes:

        raise ValueError(
            "Fraud class 1 not found in RF."
        )

    rf_fraud_index = np.where(
        model_classes == 1
    )[0][0]

    raw_probability = (
        raw_rf_probabilities[
            :,
            rf_fraud_index
        ]
    )

    # ========================================================
    # 2. TREE-DISAGREEMENT UNCERTAINTY
    # ========================================================

    (
        mu_raw,
        sigma_squared_rf,
        sigma_rf,
        tree_probabilities
    ) = calculate_rf_uncertainty(
        model,
        X_imputed
    )

    # --------------------------------------------------------
    # Verify tree mean
    # --------------------------------------------------------

    max_difference = np.max(
        np.abs(
            raw_probability
            -
            mu_raw
        )
    )

    if max_difference > 1e-10:

        raise RuntimeError(
            "Mean individual-tree probability does not "
            "match RandomForest.predict_proba(). "
            f"Maximum difference: {max_difference}"
        )

    # ========================================================
    # 3. SIGMOID-CALIBRATED PROBABILITY
    # ========================================================

    calibrated_probabilities = (
        calibrator.predict_proba(
            X_imputed
        )
    )

    calibrator_classes = np.asarray(
        calibrator.classes_
    )

    if 1 not in calibrator_classes:

        raise ValueError(
            "Fraud class 1 not found in calibrator."
        )

    calibrated_fraud_index = np.where(
        calibrator_classes == 1
    )[0][0]

    calibrated_probability = (
        calibrated_probabilities[
            :,
            calibrated_fraud_index
        ]
    )

    # ========================================================
    # 4. MONDRIAN CONFORMAL PARAMETERS
    # ========================================================

    (
        q_0,
        q_1,
        alpha,
        threshold_0,
        threshold_1
    ) = get_conformal_parameters(
        conformal_config
    )

    # ========================================================
    # 5. CONFORMAL PREDICTION SET
    # ========================================================

    conformal_sets = (
        get_conformal_prediction_set(
            calibrated_probabilities,
            calibrator_classes,
            q_0,
            q_1
        )
    )

    # ========================================================
    # 6. CONFORMAL DECISION
    # ========================================================

    conformal_decisions = [
        conformal_decision(
            prediction_set
        )
        for prediction_set
        in conformal_sets
    ]

    # ========================================================
    # 7. FLAG
    # ========================================================
    #
    # Keep FLAG binary for compatibility with your
    # existing RF interface.
    #
    # IMPORTANT:
    #   The conformal decision is supplied separately.
    #
    #   μ_RF >= 0.50 → FLAG = 1
    #   otherwise     → FLAG = 0
    #
    # This is separate from the conformal decision.
    # ========================================================

    predictions = (
        calibrated_probability >= 0.50
    ).astype(int)

    # ========================================================
    # 8. RISK LEVEL
    # ========================================================

    risk_levels = [
        get_risk_level(
            probability
        )
        for probability
        in calibrated_probability
    ]

    # ========================================================
    # 9. BUILD RESULTS
    # ========================================================

    results = df.copy()

    # --------------------------------------------------------
    # Existing-compatible outputs
    # --------------------------------------------------------

    results["FLAG"] = predictions

    results["fraud_probability"] = (
        calibrated_probability
    )

    results["risk_level"] = (
        risk_levels
    )

    # --------------------------------------------------------
    # Raw RF
    # --------------------------------------------------------

    results["rf_probability_raw"] = (
        raw_probability
    )

    results["rf_mu_raw"] = (
        mu_raw
    )

    # --------------------------------------------------------
    # Final μ_RF
    # --------------------------------------------------------

    results["rf_mu_calibrated"] = (
        calibrated_probability
    )

    # --------------------------------------------------------
    # RF uncertainty
    # --------------------------------------------------------

    results["rf_sigma_squared"] = (
        sigma_squared_rf
    )

    results["rf_sigma"] = (
        sigma_rf
    )

    # --------------------------------------------------------
    # Conformal output
    # --------------------------------------------------------

    results["rf_conformal_set"] = [
        str(
            prediction_set
        )
        for prediction_set
        in conformal_sets
    ]

    results["rf_conformal_set_size"] = [
        len(
            prediction_set
        )
        for prediction_set
        in conformal_sets
    ]

    results["rf_conformal_decision"] = (
        conformal_decisions
    )

    results["rf_conformal_alpha"] = (
        alpha
    )

    results["rf_conformal_coverage_target"] = (
        1.0 - alpha
    )

    # --------------------------------------------------------
    # Class-specific conformal parameters
    # --------------------------------------------------------

    results["rf_conformal_q0"] = (
        q_0
    )

    results["rf_conformal_q1"] = (
        q_1
    )

    results["rf_conformal_threshold_nonfraud"] = (
        threshold_0
    )

    results["rf_conformal_threshold_fraud"] = (
        threshold_1
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    results["rf_n_trees"] = (
        len(
            model.estimators_
        )
    )

    return results


# ============================================================
# SINGLE-WALLET PREDICTION
# ============================================================

def predict_single(
    feature_values
):

    (
        model,
        imputer,
        features,
        calibrator,
        conformal_config
    ) = load_artifacts()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in feature_values
    ]

    if missing_features:

        raise ValueError(
            "Missing features:\n"
            +
            "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Build one-row dataframe
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            [
                feature_values[
                    feature
                ]
                for feature in features
            ]
        ],
        columns=features
    )

    # --------------------------------------------------------
    # Training-fitted imputer
    # --------------------------------------------------------

    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=features
    )

    # ========================================================
    # RAW RF PROBABILITY
    # ========================================================

    raw_probabilities = (
        model.predict_proba(
            X_imputed
        )
    )

    model_classes = np.asarray(
        model.classes_
    )

    rf_fraud_index = np.where(
        model_classes == 1
    )[0][0]

    raw_probability = float(
        raw_probabilities[
            0,
            rf_fraud_index
        ]
    )

    # ========================================================
    # TREE UNCERTAINTY
    # ========================================================

    (
        mu_raw_array,
        sigma_squared_array,
        sigma_array,
        tree_probabilities
    ) = calculate_rf_uncertainty(
        model,
        X_imputed
    )

    mu_raw = float(
        mu_raw_array[0]
    )

    sigma_squared_rf = float(
        sigma_squared_array[0]
    )

    sigma_rf = float(
        sigma_array[0]
    )

    # ========================================================
    # CALIBRATED PROBABILITY
    # ========================================================

    calibrated_probabilities = (
        calibrator.predict_proba(
            X_imputed
        )
    )

    calibrator_classes = np.asarray(
        calibrator.classes_
    )

    calibrated_fraud_index = np.where(
        calibrator_classes == 1
    )[0][0]

    calibrated_probability = float(
        calibrated_probabilities[
            0,
            calibrated_fraud_index
        ]
    )

    # ========================================================
    # CONFORMAL PARAMETERS
    # ========================================================

    (
        q_0,
        q_1,
        alpha,
        threshold_0,
        threshold_1
    ) = get_conformal_parameters(
        conformal_config
    )

    # ========================================================
    # CONFORMAL SET
    # ========================================================

    conformal_sets = (
        get_conformal_prediction_set(
            calibrated_probabilities,
            calibrator_classes,
            q_0,
            q_1
        )
    )

    prediction_set = (
        conformal_sets[0]
    )

    conformal_result = (
        conformal_decision(
            prediction_set
        )
    )

    # ========================================================
    # FLAG
    # ========================================================

    prediction = int(
        calibrated_probability >= 0.50
    )

    # ========================================================
    # RISK
    # ========================================================

    risk = get_risk_level(
        calibrated_probability
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ----------------------------------------------------
        # Existing interface
        # ----------------------------------------------------

        "FLAG":
            prediction,

        "fraud_probability":
            calibrated_probability,

        "risk_level":
            risk,

        # ----------------------------------------------------
        # RF probability
        # ----------------------------------------------------

        "rf_probability_raw":
            raw_probability,

        "rf_mu_raw":
            mu_raw,

        "rf_mu_calibrated":
            calibrated_probability,

        # ----------------------------------------------------
        # RF uncertainty
        # ----------------------------------------------------

        "rf_sigma_squared":
            sigma_squared_rf,

        "rf_sigma":
            sigma_rf,

        # ----------------------------------------------------
        # Conformal
        # ----------------------------------------------------

        "rf_conformal_set":
            prediction_set,

        "rf_conformal_set_size":
            len(
                prediction_set
            ),

        "rf_conformal_decision":
            conformal_result,

        "rf_conformal_alpha":
            alpha,

        "rf_conformal_coverage_target":
            1.0 - alpha,

        "rf_conformal_q0":
            q_0,

        "rf_conformal_q1":
            q_1,

        "rf_conformal_threshold_nonfraud":
            threshold_0,

        "rf_conformal_threshold_fraud":
            threshold_1,

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "rf_n_trees":
            len(
                model.estimators_
            )
    }


# ============================================================
# CSV COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Ethereum fraud detection using the trained "
            "20-feature Random Forest with sigmoid "
            "probability calibration, tree-disagreement "
            "uncertainty and Mondrian conformal prediction."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV"
    )

    parser.add_argument(
        "--output",
        default="rf_predictions.csv",
        help="Path for prediction output CSV"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input path
    # --------------------------------------------------------

    if not os.path.exists(
        args.input
    ):

        raise FileNotFoundError(
            f"Input file not found:\n{args.input}"
        )

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    df = pd.read_csv(
        args.input
    )

    print(
        "=" * 80
    )

    print(
        "ETHEREUM FRAUD DETECTION"
    )

    print(
        "Random Forest + Calibration + "
        "Tree Uncertainty + Mondrian Conformal"
    )

    print(
        "=" * 80
    )

    print(
        f"Input rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    results = predict_dataframe(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        args.output,
        index=False
    )

    print(
        f"\nSaved predictions to: {args.output}"
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print(
        "\nPrediction summary:"
    )

    summary_columns = [
        "FLAG",
        "fraud_probability",
        "risk_level",
        "rf_mu_calibrated",
        "rf_sigma_squared",
        "rf_conformal_set",
        "rf_conformal_decision"
    ]

    print(
        results[
            summary_columns
        ].head(10)
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()