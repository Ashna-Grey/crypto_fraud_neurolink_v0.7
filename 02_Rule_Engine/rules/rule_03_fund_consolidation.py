"""
Rule 03: Fund Consolidation / Fan-In

Detects wallets that receive funds from multiple different
source addresses within a relatively short period.

Features used:
    - received_transactions
    - unique_sources
    - avg_min_between_received

Rule condition:
    - At least 8 incoming transactions
    - At least 5 unique source addresses
    - Average time between incoming transactions is
      greater than 0 and <= 60 minutes

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""


RULE_NAME = "fund_consolidation"


def fund_consolidation(features: dict) -> int:
    """
    Evaluate the Fund Consolidation / Fan-In rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    received_transactions = int(
        features.get("received_transactions", 0)
    )

    unique_sources = int(
        features.get("unique_sources", 0)
    )

    avg_min_between_received = float(
        features.get("avg_min_between_received", 0.0)
    )

    # Minimum incoming transaction requirement
    if received_transactions < 8:
        return 0

    # Funds must originate from multiple different sources
    if unique_sources < 5:
        return 0

    # Incoming transactions must occur within a relatively
    # short average interval
    if not (
        avg_min_between_received > 0
        and avg_min_between_received <= 60
    ):
        return 0

    return 1


def evaluate(features: dict) -> dict:
    """
    Evaluate the rule and return a structured result.

    Parameters
    ----------
    features : dict
        45 extracted wallet features.

    Returns
    -------
    dict
        Rule evaluation result.
    """

    result = fund_consolidation(features)

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
