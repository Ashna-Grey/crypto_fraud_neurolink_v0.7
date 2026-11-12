"""
Rule Evaluation

Evaluates the outputs produced by rule_engine.py against
the actual ground-truth FLAG labels.

Expected labels:
    0 -> Non-fraudulent
    1 -> Fraudulent

The script calculates for each rule:

    - Confusion Matrix
    - True Positive (TP)
    - True Negative (TN)
    - False Positive (FP)
    - False Negative (FN)
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Specificity
    - False Positive Rate

It also evaluates the combined rule engine.

IMPORTANT:
    This file evaluates the rules.
    It does NOT modify the rules or their thresholds.
"""

import os

import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/rule_results.csv"

OUTPUT_DIRECTORY = "outputs/evaluation"

RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "rule_evaluation.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "confusion_matrices.txt"
)

# Ground-truth column
LABEL_COLUMN = "FLAG"


# ============================================================
# RULE COLUMNS
# ============================================================

RULE_COLUMNS = [
    "rapid_transaction_activity",
    "fund_splitting",
    "fund_consolidation",
    "new_wallet_high_value",
    "dust_spam_activity",
    "contract_activity",
    "erc20_activity",
    "high_activity",
]


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df: pd.DataFrame) -> None:
    """
    Verify that the evaluation dataset contains the
    ground-truth label and all rule outputs.
    """

    required_columns = [
        LABEL_COLUMN
    ] + RULE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
) -> dict:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        Actual FLAG values.

    y_pred : array-like
        Predicted rule values.

    Returns
    -------
    dict
        Confusion matrix values and classification metrics.
    """

    # Force binary integer arrays
    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )

    # --------------------------------------------------------
    # Confusion Matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Specificity
    #
    # TN / (TN + FP)
    # --------------------------------------------------------

    if (tn + fp) > 0:
        specificity = (
            tn / (tn + fp)
        )
    else:
        specificity = 0.0

    # --------------------------------------------------------
    # False Positive Rate
    #
    # FP / (FP + TN)
    # --------------------------------------------------------

    if (fp + tn) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = 0.0

    # --------------------------------------------------------
    # False Negative Rate
    #
    # FN / (FN + TP)
    # --------------------------------------------------------

    if (fn + tp) > 0:
        false_negative_rate = (
            fn / (fn + tp)
        )
    else:
        false_negative_rate = 0.0

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
    }


# ============================================================
# EVALUATE ONE RULE
# ============================================================

def evaluate_rule(
    df: pd.DataFrame,
    rule_column: str
) -> dict:
    """
    Evaluate a single rule against the FLAG label.
    """

    y_true = df[LABEL_COLUMN]
    y_pred = df[rule_column]

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["rule"] = rule_column

    return metrics


# ============================================================
# EVALUATE ALL RULES
# ============================================================

def evaluate_all_rules(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate every individual rule.

    Returns
    -------
    pandas.DataFrame
        One row per rule.
    """

    results = []

    for rule in RULE_COLUMNS:

        metrics = evaluate_rule(
            df,
            rule
        )

        results.append(metrics)

    return pd.DataFrame(
        results
    )


# ============================================================
# COMBINED RULE ENGINE
# ============================================================

def evaluate_combined_engine(
    df: pd.DataFrame
) -> dict:
    """
    Evaluate the combined rule engine.

    A wallet is classified as suspicious/fraudulent when
    at least one rule is triggered.

    This is intentionally kept separate from the individual
    rule evaluation so that we can later test different
    combination strategies.
    """

    y_true = df[LABEL_COLUMN]

    # --------------------------------------------------------
    # Current combined decision:
    #
    # If ANY rule fires -> predicted fraud = 1
    # Otherwise -> predicted fraud = 0
    # --------------------------------------------------------

    combined_prediction = (
        df[RULE_COLUMNS]
        .sum(axis=1)
        > 0
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        combined_prediction
    )

    metrics["rule"] = "combined_rule_engine"

    return metrics


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    df: pd.DataFrame,
    output_file: str
) -> None:
    """
    Save a readable confusion matrix for every rule.
    """

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CRYPTO FRAUD RULE EVALUATION\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        y_true = df[LABEL_COLUMN]

        # ----------------------------------------------------
        # Individual rules
        # ----------------------------------------------------

        for rule in RULE_COLUMNS:

            y_pred = df[rule]

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            tn, fp, fn, tp = cm.ravel()

            file.write(
                f"{rule}\n"
            )

            file.write(
                "-" * 50
                + "\n"
            )

            file.write(
                "                 Predicted\n"
            )

            file.write(
                "                 0       1\n"
            )

            file.write(
                f"Actual 0      {tn:6d}  {fp:6d}\n"
            )

            file.write(
                f"Actual 1      {fn:6d}  {tp:6d}\n"
            )

            file.write("\n")

        # ----------------------------------------------------
        # Combined engine
        # ----------------------------------------------------

        combined_prediction = (
            df[RULE_COLUMNS]
            .sum(axis=1)
            > 0
        ).astype(int)

        cm = confusion_matrix(
            y_true,
            combined_prediction,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        file.write(
            "combined_rule_engine\n"
        )

        file.write(
            "-" * 50
            + "\n"
        )

        file.write(
            "                 Predicted\n"
        )

        file.write(
            "                 0       1\n"
        )

        file.write(
            f"Actual 0      {tn:6d}  {fp:6d}\n"
        )

        file.write(
            f"Actual 1      {fn:6d}  {tp:6d}\n"
        )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results_df: pd.DataFrame
) -> None:
    """
    Print the evaluation metrics in a readable format.
    """

    print("\n")
    print("=" * 100)
    print("RULE EVALUATION RESULTS")
    print("=" * 100)

    display_columns = [
        "rule",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "false_positive_rate",
    ]

    display_df = results_df[
        display_columns
    ].copy()

    # Convert metrics to percentages
    percentage_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "false_positive_rate",
    ]

    for column in percentage_columns:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    print("\n")
    print(
        "Percentage-based metrics are shown as percentages."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO FRAUD RULE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load rule-engine output
    # --------------------------------------------------------

    print(
        f"\nLoading: {INPUT_FILE}"
    )

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nCould not find '{INPUT_FILE}'.\n"
            "Run rule_engine.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} records."
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_input(
        df
    )

    print(
        "Input validation successful."
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    print(
        f"\nGround-truth column: {LABEL_COLUMN}"
    )

    print(
        "Class distribution:"
    )

    print(
        df[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Evaluate individual rules
    # --------------------------------------------------------

    print(
        "\nEvaluating individual rules..."
    )

    results_df = evaluate_all_rules(
        df
    )

    # --------------------------------------------------------
    # Evaluate combined engine
    # --------------------------------------------------------

    print(
        "Evaluating combined rule engine..."
    )

    combined_result = evaluate_combined_engine(
        df
    )

    combined_df = pd.DataFrame(
        [combined_result]
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    results_df = pd.concat(
        [
            results_df,
            combined_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nMetrics saved to:"
        f"\n{RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Save confusion matrices
    # --------------------------------------------------------

    save_confusion_matrices(
        df,
        CONFUSION_MATRIX_FILE
    )

    print(
        f"Confusion matrices saved to:"
        f"\n{CONFUSION_MATRIX_FILE}"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        results_df
    )

    print(
        "\nEvaluation completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
