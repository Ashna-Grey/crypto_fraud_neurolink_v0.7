"""
Rule 06: Contract Activity Anomaly

Detects wallets showing unusually high ETH activity involving
smart contracts or creating contracts.

Features used:
    - total_eth_sent_contracts
    - avg_value_sent_contract
    - created_contracts

Rule condition:
    - Contract ETH activity is in the top 10% of the
      evaluated dataset OR
    - The wallet has created at least 3 contracts

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""

import numpy as np


RULE_NAME = "contract_activity"


HIGH_CONTRACT_VALUE_PERCENTILE = 0.90
MIN_CREATED_CONTRACTS = 3


def calculate_contract_value_threshold(
    contract_value_values
) -> float:
    """
    Calculate the high contract-value threshold.

    Parameters
    ----------
    contract_value_values : list, numpy.ndarray, or pandas Series
        total_eth_sent_contracts values from the evaluated dataset.

    Returns
    -------
    float
        90th percentile of positive contract ETH activity.
    """

    values = np.asarray(
        contract_value_values,
        dtype=float
    )

    # Keep only valid positive values
    values = values[
        np.isfinite(values) & (values > 0)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.quantile(
            values,
            HIGH_CONTRACT_VALUE_PERCENTILE
        )
    )


def contract_activity(
    features: dict,
    contract_value_threshold: float
) -> int:
    """
    Evaluate the Contract Activity rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    contract_value_threshold : float
        90th percentile of positive total ETH sent to
        contracts in the evaluated dataset.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    total_eth_sent_contracts = float(
        features.get(
            "total_eth_sent_contracts",
            0.0
        )
    )

    created_contracts = int(
        features.get(
            "created_contracts",
            0
        )
    )

    # High-value contract interaction
    high_contract_activity = (
        total_eth_sent_contracts > 0
        and total_eth_sent_contracts
        >= contract_value_threshold
    )

    # Multiple contract creations
    multiple_contract_creation = (
        created_contracts >= MIN_CREATED_CONTRACTS
    )

    # Trigger if either suspicious pattern is present
    if (
        high_contract_activity
        or multiple_contract_creation
    ):
        return 1

    return 0


def evaluate(
    features: dict,
    contract_value_threshold: float
) -> dict:
    """
    Evaluate the rule and return a structured result.

    Parameters
    ----------
    features : dict
        45 extracted wallet features.

    contract_value_threshold : float
        Dataset-derived 90th percentile threshold.

    Returns
    -------
    dict
        Rule evaluation result.
    """

    result = contract_activity(
        features,
        contract_value_threshold
    )

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
