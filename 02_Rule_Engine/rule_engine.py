import pandas as pd

# ============================================================
# RULE IMPORTS
# ============================================================

from rules.rule_01_rapid_activity import evaluate as rule_01
from rules.rule_02_fund_splitting import evaluate as rule_02
from rules.rule_03_fund_consolidation import evaluate as rule_03
from rules.rule_04_new_wallet_high_value import evaluate as rule_04
from rules.rule_05_dust_spam import evaluate as rule_05
from rules.rule_06_contract_activity import evaluate as rule_06
from rules.rule_07_erc20_activity import evaluate as rule_07
from rules.rule_08_high_activity import evaluate as rule_08


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "outputs/features.csv"
OUTPUT_FILE = "outputs/rule_results.csv"


# ============================================================
# ACTIVE RULES
# ============================================================

# Rule 09 is intentionally excluded.
#
# These are the same eight rules used in the current
# rule-based evaluation.

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
# F1 SCORES
# ============================================================

# F1 values obtained from the previous evaluation against FLAG.
#
# These are percentages, but only their relative values matter
# for normalization. Therefore:
#
#       weight_i = F1_i / SUM(F1)
#
# The eight values below sum to 16.69, not 10.01.
#
# The code calculates the sum automatically instead of hard-
# coding it.

F1_SCORES = {
    "rapid_transaction_activity": 2.80,
    "fund_splitting": 1.10,
    "fund_consolidation": 6.43,
    "new_wallet_high_value": 0.24,
    "dust_spam_activity": 0.98,
    "contract_activity": 0.45,
    "erc20_activity": 4.07,
    "high_activity": 0.62,
}


# ============================================================
# VALIDATE F1 SCORE CONFIGURATION
# ============================================================

if set(F1_SCORES.keys()) != set(RULE_COLUMNS):

    missing_f1 = set(RULE_COLUMNS) - set(F1_SCORES.keys())
    extra_f1 = set(F1_SCORES.keys()) - set(RULE_COLUMNS)

    raise ValueError(
        "F1 score configuration does not match active rules.\n"
        f"Missing F1 scores: {missing_f1}\n"
        f"Unexpected F1 scores: {extra_f1}"
    )


TOTAL_F1 = sum(F1_SCORES.values())


if TOTAL_F1 <= 0:
    raise ValueError(
        "Total F1 score must be greater than zero."
    )


# ============================================================
# CALCULATE F1-DERIVED WEIGHTS
# ============================================================

RULE_WEIGHTS = {
    rule: F1_SCORES[rule] / TOTAL_F1
    for rule in RULE_COLUMNS
}


# ============================================================
# VERIFY WEIGHTS
# ============================================================

WEIGHT_SUM = sum(RULE_WEIGHTS.values())


if abs(WEIGHT_SUM - 1.0) > 1e-10:
    raise ValueError(
        f"Rule weights do not sum to 1. "
        f"Current sum = {WEIGHT_SUM}"
    )


# ============================================================
# BASELINE VARIANCE
# ============================================================

# The architecture specifies:
#
# sigma²_rule = sigma_0² / (1 + n_fired)
#
# The architecture does not specify a calibrated numerical
# sigma_0² value. Therefore 1.0 is used as the current
# baseline parameter.

SIGMA_0_SQUARED = 1.0


# ============================================================
# REQUIRED INPUT FEATURES
# ============================================================

# These are the features actually required by the eight rules.
#
# They are taken from the 45-feature output of feature_extractor.py.

REQUIRED_FEATURES = [
    "avg_min_between_sent",
    "avg_min_between_received",
    "wallet_lifetime_minutes",
    "sent_transactions",
    "received_transactions",
    "created_contracts",
    "unique_sources",
    "unique_destinations",
    "avg_value_sent",
    "total_transactions",
    "total_eth_sent",
    "total_eth_received",
    "total_eth_sent_contracts",
    "erc20_transactions",
    "erc20_unique_sent_addresses",
    "erc20_unique_sent_tokens",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


print("=" * 65)
print("RULE-BASED FRAUD DETECTION ENGINE")
print("=" * 65)

print(f"\nInput file : {INPUT_FILE}")
print(f"Input rows : {len(df)}")


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

missing_features = [
    feature
    for feature in REQUIRED_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "\nMissing required feature columns:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )


if "FLAG" not in df.columns:

    raise ValueError(
        "The input dataset must contain the 'FLAG' column."
    )


# ============================================================
# CALCULATE DATASET THRESHOLDS
# ============================================================

# ------------------------------------------------------------
# Rule 4: New Wallet + High Value
# ------------------------------------------------------------

high_value_threshold = (
    df["total_eth_sent"] +
    df["total_eth_received"]
).quantile(0.90)


# ------------------------------------------------------------
# Rule 5: Dust / Spam Activity
# ------------------------------------------------------------

positive_avg_sent = df.loc[
    df["avg_value_sent"] > 0,
    "avg_value_sent"
]

if positive_avg_sent.empty:

    raise ValueError(
        "Cannot calculate Rule 5 threshold: "
        "no positive avg_value_sent values found."
    )

dust_threshold = positive_avg_sent.quantile(0.10)


# ------------------------------------------------------------
# Rule 6: Contract Activity
# ------------------------------------------------------------

positive_contract_values = df.loc[
    df["total_eth_sent_contracts"] > 0,
    "total_eth_sent_contracts"
]

if positive_contract_values.empty:

    raise ValueError(
        "Cannot calculate Rule 6 threshold: "
        "no positive total_eth_sent_contracts values found."
    )

contract_threshold = (
    positive_contract_values.quantile(0.90)
)


print("\nDataset thresholds:")
print(
    f"Rule 4 high-value threshold : "
    f"{high_value_threshold:.6f}"
)

print(
    f"Rule 5 dust threshold       : "
    f"{dust_threshold:.6f}"
)

print(
    f"Rule 6 contract threshold   : "
    f"{contract_threshold:.6f}"
)


# ============================================================
# DISPLAY F1 SCORES AND WEIGHTS
# ============================================================

print("\nF1-based rule weights:")
print("-" * 65)

for rule in RULE_COLUMNS:

    print(
        f"{rule:<35} "
        f"F1 = {F1_SCORES[rule]:>5.2f}%   "
        f"Weight = {RULE_WEIGHTS[rule]:.6f} "
        f"({RULE_WEIGHTS[rule] * 100:.2f}%)"
    )

print("-" * 65)

print(
    f"Total F1 : {TOTAL_F1:.2f}"
)

print(
    f"Weight sum: {WEIGHT_SUM:.6f}"
)


# ============================================================
# RULE RESULT EXTRACTION
# ============================================================

def get_rule_result(result):
    """
    Convert a rule's output into a binary trigger.

    Current rule modules return a dictionary containing
    a 'fired' field.

    Example:

        {
            "fired": True,
            ...
        }

    The rule engine stores only the binary trigger:

        True  -> 1
        False -> 0
    """

    if isinstance(result, dict):

        if "triggered" not in result:

            raise ValueError(
                "Rule returned a dictionary without "
                "the required 'triggered' field."
            )

        return int(bool(result["triggered"]))

    return int(bool(result))


# ============================================================
# EVALUATE ALL EIGHT RULES
# ============================================================

def evaluate_all_rules(row):

    features = row.to_dict()

    results = {}

    # --------------------------------------------------------
    # R1: Rapid Transaction Activity
    # --------------------------------------------------------

    results["rapid_transaction_activity"] = get_rule_result(
        rule_01(features)
    )


    # --------------------------------------------------------
    # R2: Fund Splitting
    # --------------------------------------------------------

    results["fund_splitting"] = get_rule_result(
        rule_02(features)
    )


    # --------------------------------------------------------
    # R3: Fund Consolidation
    # --------------------------------------------------------

    results["fund_consolidation"] = get_rule_result(
        rule_03(features)
    )


    # --------------------------------------------------------
    # R4: New Wallet + High Value
    # --------------------------------------------------------

    results["new_wallet_high_value"] = get_rule_result(
        rule_04(
            features,
            high_value_threshold
        )
    )


    # --------------------------------------------------------
    # R5: Dust / Spam Activity
    # --------------------------------------------------------

    results["dust_spam_activity"] = get_rule_result(
        rule_05(
            features,
            dust_threshold
        )
    )


    # --------------------------------------------------------
    # R6: Contract Activity
    # --------------------------------------------------------

    results["contract_activity"] = get_rule_result(
        rule_06(
            features,
            contract_threshold
        )
    )


    # --------------------------------------------------------
    # R7: ERC-20 Activity
    # --------------------------------------------------------

    results["erc20_activity"] = get_rule_result(
        rule_07(features)
    )


    # --------------------------------------------------------
    # R8: High Activity
    # --------------------------------------------------------

    results["high_activity"] = get_rule_result(
        rule_08(features)
    )


    return results


# ============================================================
# CALCULATE RULE-BASED MU AND SIGMA²
# ============================================================

def calculate_rule_statistics(rule_outputs):
    """
    Calculate the two outputs of the rule-based engine.

    mu_rule:
        Weighted rule risk estimate.

        mu_rule =
            SUM(weight_i * rule_i)
            / SUM(weight_i)

    sigma_squared_rule:
        Rule-based variance.

        sigma²_rule =
            sigma_0² / (1 + n_fired)

    Returns ONLY:

        mu_rule
        sigma_squared_rule
    """

    # --------------------------------------------------------
    # Validate rule outputs
    # --------------------------------------------------------

    missing_rules = [
        rule
        for rule in RULE_COLUMNS
        if rule not in rule_outputs
    ]

    if missing_rules:

        raise ValueError(
            "Missing rule outputs: "
            + ", ".join(missing_rules)
        )


    # --------------------------------------------------------
    # Calculate weighted numerator
    # --------------------------------------------------------

    weighted_sum = 0.0

    for rule in RULE_COLUMNS:

        rule_value = int(
            rule_outputs[rule]
        )

        if rule_value not in (0, 1):

            raise ValueError(
                f"Invalid output for {rule}: "
                f"{rule_value}. "
                f"Expected 0 or 1."
            )

        weighted_sum += (
            RULE_WEIGHTS[rule] *
            rule_value
        )


    # --------------------------------------------------------
    # Calculate mu_rule
    # --------------------------------------------------------

    mu_rule = (
        weighted_sum /
        WEIGHT_SUM
    )


    # --------------------------------------------------------
    # Count fired rules
    # --------------------------------------------------------

    n_fired = sum(
        int(rule_outputs[rule])
        for rule in RULE_COLUMNS
    )


    # --------------------------------------------------------
    # Calculate sigma²_rule
    # --------------------------------------------------------

    sigma_squared_rule = (
        SIGMA_0_SQUARED /
        (1 + n_fired)
    )


    # --------------------------------------------------------
    # Return ONLY mu and sigma²
    # --------------------------------------------------------

    return (
        float(mu_rule),
        float(sigma_squared_rule)
    )


# ============================================================
# RUN THE RULE ENGINE
# ============================================================

rule_results = []


for index, row in df.iterrows():

    results = evaluate_all_rules(row)

    # --------------------------------------------------------
    # Count how many of the 8 rules fired
    # --------------------------------------------------------

    results["rules_triggered"] = sum(
        results[rule]
        for rule in RULE_COLUMNS
    )

    rule_results.append(results)


# ============================================================
# CONVERT RULE RESULTS TO DATAFRAME
# ============================================================

rules_df = pd.DataFrame(
    rule_results
)


# ============================================================
# CALCULATE MU AND SIGMA² FOR EVERY WALLET
# ============================================================

mu_values = []
sigma_squared_values = []


for _, row in rules_df.iterrows():

    rule_outputs = {
        rule: int(row[rule])
        for rule in RULE_COLUMNS
    }


    mu_rule, sigma_squared_rule = (
        calculate_rule_statistics(
            rule_outputs
        )
    )


    mu_values.append(mu_rule)
    sigma_squared_values.append(
        sigma_squared_rule
    )


# ============================================================
# ADD ONLY THE TWO RULE-STAGE STATISTICAL OUTPUTS
# ============================================================

rules_df["mu_rule"] = mu_values

rules_df["sigma_squared_rule"] = (
    sigma_squared_values
)


# ============================================================
# ADD GROUND TRUTH
# ============================================================

rules_df["FLAG"] = (
    df["FLAG"].values
)


# ============================================================
# ADD WALLET IDENTIFIER IF PRESENT
# ============================================================

if "address" in df.columns:

    rules_df.insert(
        0,
        "address",
        df["address"].values
    )

elif "wallet" in df.columns:

    rules_df.insert(
        0,
        "wallet",
        df["wallet"].values
    )


# ============================================================
# SAVE RESULTS
# ============================================================

rules_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

# mu must be between 0 and 1.
if not (
    rules_df["mu_rule"].between(
        0.0,
        1.0
    ).all()
):

    raise ValueError(
        "Invalid mu_rule value detected. "
        "All mu_rule values must be between 0 and 1."
    )


# sigma² must be positive.
if not (
    rules_df["sigma_squared_rule"] > 0
).all():

    raise ValueError(
        "Invalid sigma_squared_rule value detected. "
        "All variance values must be positive."
    )


# Number of fired rules must be between 0 and 8.
if not (
    rules_df["rules_triggered"].between(
        0,
        len(RULE_COLUMNS)
    ).all()
):

    raise ValueError(
        "Invalid rules_triggered value detected."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("RULE ENGINE COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"\nOutput file : {OUTPUT_FILE}"
)

print(
    f"Output rows : {len(rules_df)}"
)


print("\nRule trigger counts:")
print("-" * 65)

for rule in RULE_COLUMNS:

    print(
        f"{rule:<35}"
        f"{int(rules_df[rule].sum())}"
    )


print("\nRule-stage statistics:")
print("-" * 65)

print(
    f"Average rules triggered : "
    f"{rules_df['rules_triggered'].mean():.4f}"
)

print(
    f"Average mu_rule         : "
    f"{rules_df['mu_rule'].mean():.6f}"
)

print(
    f"Average sigma²_rule     : "
    f"{rules_df['sigma_squared_rule'].mean():.6f}"
)


print("\nFirst 10 rule-stage outputs:")
print("-" * 65)

print(
    rules_df[
        RULE_COLUMNS
        + [
            "rules_triggered",
            "mu_rule",
            "sigma_squared_rule",
            "FLAG"
        ]
    ].head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 65)
print("FINAL RULE OUTPUTS:")
print("  μ_rule")
print("  σ²_rule")
print("=" * 65)