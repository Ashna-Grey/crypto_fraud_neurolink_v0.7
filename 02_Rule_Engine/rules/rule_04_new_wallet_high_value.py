"""
Rule 04: New Wallet + High Value Activity

Detects relatively new wallets that are handling unusually
large amounts of ETH.

Features used:
    - wallet_lifetime_minutes
    - total_eth_received
    - total_eth_sent
    - total_transactions

Rule condition:
    - Wallet lifetime <= 7 days
    - Total ETH volume is in the top 10% of the evaluated dataset
    - At least 3 total transactions

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""

import numpy as np


RULE_NAME = "new_wallet_high_value"

# 7 days expressed in minutes
NEW_WALLET_THRESHOLD_MINUTES = 7 * 24 * 60

# Top 10% of transaction volume
HIGH_VALUE_PERCENTILE = 0.90

# Minimum number of transactions
MIN_TRANSACTIONS = 3


def calculate_high_value_threshold(
    total_eth_volumes
) -> float:
    """
    Calculate the high-value threshold from the evaluated dataset.

    Parameters
    ----------
    total_eth_volumes : list, numpy.ndarray, or pandas Series
        Total ETH volume for the wallets being evaluated.

    Returns
    -------
    float
        90th percentile ETH volume.
    """

    values = np.asarray(total_eth_volumes, dtype=float)

    # Remove invalid values
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(
        np.quantile(values, HIGH_VALUE_PERCENTILE)
    )


def new_wallet_high_value(
    features: dict,
    high_value_threshold: float
) -> int:
    """
    Evaluate the New Wallet + High Value rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    high_value_threshold : float
        90th percentile of total ETH transaction volume
        calculated from the evaluation dataset.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    wallet_lifetime_minutes = float(
        features.get(
            "wallet_lifetime_minutes",
            0.0
        )
    )

    total_eth_received = float(
        features.get(
            "total_eth_received",
            0.0
        )
    )

    total_eth_sent = float(
        features.get(
            "total_eth_sent",
            0.0
        )
    )

    total_transactions = int(
        features.get(
            "total_transactions",
            0
        )
    )

    # Total ETH moved by the wallet
    total_eth_volume = (
        total_eth_received +
        total_eth_sent
    )

    # New wallet condition
    is_new_wallet = (
        wallet_lifetime_minutes > 0
        and wallet_lifetime_minutes
        <= NEW_WALLET_THRESHOLD_MINUTES
    )

    # High-value condition
    is_high_value = (
        total_eth_volume >= high_value_threshold
    )

    # Minimum activity condition
    has_sufficient_activity = (
        total_transactions >= MIN_TRANSACTIONS
    )

    if (
        is_new_wallet
        and is_high_value
        and has_sufficient_activity
    ):
        return 1

    return 0


def evaluate(
    features: dict,
    high_value_threshold: float
) -> dict:
    """
    Evaluate the rule and return a structured result.

    Parameters
    ----------
    features : dict
        45 extracted wallet features.

    high_value_threshold : float
        Dataset-derived 90th percentile threshold.

    Returns
    -------
    dict
        Rule evaluation result.
    """

    result = new_wallet_high_value(
        features,
        high_value_threshold
    )

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
