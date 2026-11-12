"""
Rule 01: Rapid Transaction Activity

Detects wallets that perform transactions at unusually
short average time intervals.

Features used:
    - avg_min_between_sent
    - avg_min_between_received
    - total_transactions

Rule condition:
    - At least 5 total transactions
    - Average incoming OR outgoing transaction interval
      is greater than 0 and <= 10 minutes

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""


RULE_NAME = "rapid_transaction_activity"


def rapid_transaction_activity(features: dict) -> int:
    """
    Evaluate Rapid Transaction Activity rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    avg_min_between_sent = float(
        features.get("avg_min_between_sent", 0.0)
    )

    avg_min_between_received = float(
        features.get("avg_min_between_received", 0.0)
    )

    total_transactions = int(
        features.get("total_transactions", 0)
    )

    # Minimum activity requirement
    if total_transactions < 5:
        return 0

    # Rapid outgoing transactions
    rapid_sent = (
        avg_min_between_sent > 0
        and avg_min_between_sent <= 10
    )

    # Rapid incoming transactions
    rapid_received = (
        avg_min_between_received > 0
        and avg_min_between_received <= 10
    )

    # Trigger if either incoming or outgoing activity is rapid
    if rapid_sent or rapid_received:
        return 1

    return 0


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

    result = rapid_transaction_activity(features)

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
