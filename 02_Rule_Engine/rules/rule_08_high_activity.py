"""
Rule 08: High Activity / Burst Wallet

Detects wallets performing a high number of transactions
relative to their observed wallet lifetime.

Features used:
    - total_transactions
    - wallet_lifetime_minutes

Derived feature:
    - transactions_per_hour

Rule condition:
    - At least 20 total transactions
    AND
    - At least 5 transactions per observed hour

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""


RULE_NAME = "high_activity"


MIN_TOTAL_TRANSACTIONS = 20
MIN_TRANSACTIONS_PER_HOUR = 5.0


def high_activity(features: dict) -> int:
    """
    Evaluate the High Activity / Burst Wallet rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    total_transactions = int(
        features.get(
            "total_transactions",
            0
        )
    )

    wallet_lifetime_minutes = float(
        features.get(
            "wallet_lifetime_minutes",
            0.0
        )
    )

    # Minimum transaction requirement
    if total_transactions < MIN_TOTAL_TRANSACTIONS:
        return 0

    # A valid observed lifetime is required to calculate
    # transaction frequency.
    if wallet_lifetime_minutes <= 0:
        return 0

    # Convert wallet lifetime from minutes to hours
    lifetime_hours = wallet_lifetime_minutes / 60.0

    # Calculate transaction frequency
    transactions_per_hour = (
        total_transactions / lifetime_hours
    )

    # Detect unusually high activity
    if transactions_per_hour >= MIN_TRANSACTIONS_PER_HOUR:
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

    result = high_activity(features)

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
