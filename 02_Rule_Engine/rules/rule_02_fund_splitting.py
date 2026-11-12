"""
Rule 02: Fund Splitting / Fan-Out

Detects wallets that distribute funds across multiple
destination addresses within a relatively short period.

Features used:
    - sent_transactions
    - unique_destinations
    - avg_min_between_sent

Rule condition:
    - At least 8 outgoing transactions
    - At least 5 unique destination addresses
    - Average time between outgoing transactions is
      greater than 0 and <= 60 minutes

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""


RULE_NAME = "fund_splitting"


def fund_splitting(features: dict) -> int:
    """
    Evaluate the Fund Splitting / Fan-Out rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    sent_transactions = int(
        features.get("sent_transactions", 0)
    )

    unique_destinations = int(
        features.get("unique_destinations", 0)
    )

    avg_min_between_sent = float(
        features.get("avg_min_between_sent", 0.0)
    )

    # Minimum outgoing transaction requirement
    if sent_transactions < 8:
        return 0

    # Wallet must distribute funds to multiple destinations
    if unique_destinations < 5:
        return 0

    # Transactions must occur within a relatively short interval
    if not (
        avg_min_between_sent > 0
        and avg_min_between_sent <= 60
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

    result = fund_splitting(features)

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
