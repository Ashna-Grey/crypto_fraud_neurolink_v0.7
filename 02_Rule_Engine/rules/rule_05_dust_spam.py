"""
Rule 05: Dust / Spam Activity

Detects wallets that send many low-value transactions
to many different destination addresses.

Features used:
    - sent_transactions
    - unique_destinations
    - avg_value_sent

Rule condition:
    - At least 15 outgoing transactions
    - At least 15 unique destinations
    - Average sent value is greater than 0
    - Average sent value is at or below the 10th percentile
      of non-zero average sent values in the evaluated dataset

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""

import numpy as np


RULE_NAME = "dust_spam_activity"

# Minimum number of outgoing transactions
MIN_SENT_TRANSACTIONS = 15

# Minimum number of unique destinations
MIN_UNIQUE_DESTINATIONS = 15

# Bottom 10% is considered low-value activity
LOW_VALUE_PERCENTILE = 0.10


def calculate_low_value_threshold(
    avg_value_sent_values
) -> float:
    """
    Calculate the low-value threshold from the evaluated dataset.

    Parameters
    ----------
    avg_value_sent_values : list, numpy.ndarray, or pandas Series
        Average ETH sent by the evaluated wallets.

    Returns
    -------
    float
        10th percentile of non-zero average sent values.
    """

    values = np.asarray(
        avg_value_sent_values,
        dtype=float
    )

    # Keep only finite, positive values
    values = values[
        np.isfinite(values) & (values > 0)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.quantile(
            values,
            LOW_VALUE_PERCENTILE
        )
    )


def dust_spam_activity(
    features: dict,
    low_value_threshold: float
) -> int:
    """
    Evaluate the Dust / Spam Activity rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    low_value_threshold : float
        10th percentile of non-zero average sent values
        calculated from the evaluation dataset.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    sent_transactions = int(
        features.get(
            "sent_transactions",
            0
        )
    )

    unique_destinations = int(
        features.get(
            "unique_destinations",
            0
        )
    )

    avg_value_sent = float(
        features.get(
            "avg_value_sent",
            0.0
        )
    )

    # Minimum outgoing activity
    if sent_transactions < MIN_SENT_TRANSACTIONS:
        return 0

    # Distribution must involve multiple destinations
    if unique_destinations < MIN_UNIQUE_DESTINATIONS:
        return 0

    # Ignore wallets with no outgoing ETH value
    if avg_value_sent <= 0:
        return 0

    # Check whether average sent value is unusually low
    if avg_value_sent <= low_value_threshold:
        return 1

    return 0


def evaluate(
    features: dict,
    low_value_threshold: float
) -> dict:
    """
    Evaluate the rule and return a structured result.

    Parameters
    ----------
    features : dict
        45 extracted wallet features.

    low_value_threshold : float
        Dataset-derived 10th percentile threshold.

    Returns
    -------
    dict
        Rule evaluation result.
    """

    result = dust_spam_activity(
        features,
        low_value_threshold
    )

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
