"""
Rule 07: ERC-20 Activity Anomaly

Detects wallets showing unusually high ERC-20 token activity,
especially activity involving many destination addresses or
multiple token contracts.

Features used:
    - erc20_transactions
    - erc20_unique_sent_addresses
    - erc20_unique_sent_tokens

Rule condition:
    - At least 20 ERC-20 transactions
    AND
    - At least 10 unique sent addresses
      OR
    - At least 10 unique sent tokens

Returns:
    1 -> Rule triggered / suspicious pattern detected
    0 -> Rule not triggered
"""


RULE_NAME = "erc20_activity"


MIN_ERC20_TRANSACTIONS = 20
MIN_UNIQUE_SENT_ADDRESSES = 10
MIN_UNIQUE_SENT_TOKENS = 10


def erc20_activity(features: dict) -> int:
    """
    Evaluate the ERC-20 Activity rule.

    Parameters
    ----------
    features : dict
        Dictionary containing the 45 extracted wallet features.

    Returns
    -------
    int
        1 if the rule is triggered, otherwise 0.
    """

    erc20_transactions = int(
        features.get(
            "erc20_transactions",
            0
        )
    )

    erc20_unique_sent_addresses = int(
        features.get(
            "erc20_unique_sent_addresses",
            0
        )
    )

    erc20_unique_sent_tokens = int(
        features.get(
            "erc20_unique_sent_tokens",
            0
        )
    )

    # Minimum ERC-20 activity requirement
    if erc20_transactions < MIN_ERC20_TRANSACTIONS:
        return 0

    # Check whether activity is distributed across
    # many addresses or multiple token contracts.
    many_destinations = (
        erc20_unique_sent_addresses
        >= MIN_UNIQUE_SENT_ADDRESSES
    )

    many_tokens = (
        erc20_unique_sent_tokens
        >= MIN_UNIQUE_SENT_TOKENS
    )

    if many_destinations or many_tokens:
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

    result = erc20_activity(features)

    return {
        "rule": RULE_NAME,
        "triggered": result,
    }
