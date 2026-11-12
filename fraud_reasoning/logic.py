"""Traceable contextual arbitration. Every threshold is explicit policy, not a fitted fraud law."""
from __future__ import annotations
from typing import Any

# Provisional domain-policy boundaries, not trained classification cutoffs.
POLICY = {
    'high_volume_count': 100,
    'low_rate_per_day': 5.0,
    'high_rate_per_day': 100.0,
    'conflict_review_gap': 0.35,
    'minimum_applicability': 0.40,
    'max_decision_uncertainty': 0.48,
    'rf_strong_fraud_threshold': 0.90,
    'rf_strong_legitimate_threshold': 0.10,
    'minimum_rule_coverage_for_rf_led_decision': 0.50,
    'high_ood_anomaly_percentile': 99.0,
    'fraud_decision_threshold': 0.70,
    'legitimate_decision_threshold': 0.30,
}

# Input aliases are restricted to representations of the SAME quantity.
ALIASES = {
    'total_transactions': ['total_transactions', 'total transactions (including tnx to create contract',
                           'total transactions (including tnx to create contract)'],
    'wallet_lifetime_minutes': ['wallet_lifetime_minutes', 'Time Diff between first and last (Mins)'],
    'avg_min_between_sent': ['avg_min_between_sent', 'Avg min between sent tnx'],
    'avg_min_between_received': ['avg_min_between_received', 'Avg min between received tnx'],
    'sent_transactions': ['sent_transactions', 'Sent tnx'],
    'received_transactions': ['received_transactions', 'Received Tnx'],
    'unique_sources': ['unique_sources', 'Unique Received From Addresses'],
    'unique_destinations': ['unique_destinations', 'Unique Sent To Addresses'],
    'avg_value_sent': ['avg_value_sent', 'avg val sent'],
    'total_eth_sent': ['total_eth_sent', 'total Ether sent'],
    'total_eth_received': ['total_eth_received', 'total ether received'],
    'erc20_transactions': ['erc20_transactions', 'Total ERC20 tnxs'],
    'erc20_unique_sent_tokens': ['erc20_unique_sent_tokens', 'ERC20 uniq sent token name'],
    'erc20_unique_sent_addresses': ['erc20_unique_sent_addresses', 'ERC20 uniq sent addr'],
    'created_contracts': ['created_contracts', 'Number of Created Contracts'],
    'total_eth_sent_contracts': ['total_eth_sent_contracts', 'total ether sent contracts'],
}

def numeric(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        import math
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def facts_from_input(row: dict[str, Any]) -> dict[str, Any]:
    facts = {}
    for key, names in ALIASES.items():
        candidates = [(name, numeric(row[name])) for name in names if name in row and numeric(row[name]) is not None]
        if candidates:
            first = candidates[0][1]
            if any(abs(other-first) > 1e-7 * max(1,abs(first)) for _,other in candidates[1:]):
                raise ValueError(f'Conflicting aliases for {key}: {[name for name,_ in candidates]}')
            facts[key] = first
    # An observed span is NOT a proof of continuous activity or the time since last tx.
    t, span = facts.get('total_transactions'), facts.get('wallet_lifetime_minutes')
    if t is not None and span is not None and span > 0:
        facts['average_transactions_per_day_over_observed_span'] = t * 1440.0 / span
    return facts

def context_arbitration(facts: dict, rf: dict, if_output: dict, rules: dict, policy: dict | None=None) -> dict:
    p = {**POLICY, **(policy or {})}
    trace, supporting, contradicting, dependencies = [], [], [], []
    count = facts.get('total_transactions')
    rate = facts.get('average_transactions_per_day_over_observed_span')
    high_volume = count is not None and count >= p['high_volume_count']
    rf_top = rf.get('reasoning', [])
    count_rf = any('total transactions' in str(v.get('feature','')).lower() or 'transactions' in str(v.get('feature','')).lower() for v in rf_top if v.get('direction') == 'toward_fraud')
    available_rules = rules.get('evaluated', [])
    fired = {r['rule'] for r in available_rules if r.get('triggered')}
    if high_volume:
        supporting.append({'proposition':'high_transaction_volume','sources':['input'], 'features':['total_transactions'], 'value':count, 'does_not_imply_fraud':True})
        trace.append({'rule_id':'VOLUME_CONTEXT','action':'inspect_activity_rate', 'reason':'High aggregate count needs temporal context', 'source_features':['total_transactions']})
    if high_volume and rate is not None and rate <= p['low_rate_per_day']:
        contradicting.append({'proposition':'sustained_high_transaction_rate','reason':'Observed mean daily rate is below provisional policy boundary','mean_rate':round(rate,5)})
        trace.append({'rule_id':'SLOW_VOLUME','action':'discount_volume_only_evidence',
                      'reason':'High lifetime volume is not proof of a fast transaction burst. Slow mean rate does not establish legitimacy.',
                      'source_features':['total_transactions','wallet_lifetime_minutes'], 'effect_on_rf':'reason_specific_not_blanket_override'})
    if high_volume and rate is not None and rate >= p['high_rate_per_day']:
        supporting.append({'proposition':'high_average_transaction_rate','features':['total_transactions','wallet_lifetime_minutes'], 'mean_rate':round(rate,5), 'not_evidence_of_24h_burst':True})
        trace.append({'rule_id':'FAST_VOLUME','action':'record_rate_warning','reason':'Observed mean rate exceeds provisional boundary, not proof of fraudulent burst'})
    if high_volume and rate is None:
        trace.append({'rule_id':'TIME_UNAVAILABLE','action':'mark_temporal_context_unknown', 'reason':'Missing or invalid observed transaction span; cannot claim slow or fast activity'})
    if count_rf and high_volume:
        overlaps = [r['rule'] for r in available_rules if r.get('triggered') and 'total_transactions' in r.get('dependencies',[])]
        if overlaps:
            dependencies.append({'features':['total_transactions'], 'engines':['random_forest','rule_based_engine'], 'rules':overlaps,
                                 'reason':'Shared feature provenance; not proof of statistical independence', 'adjustment':'do_not_count_twice_as_independent_votes'})
    if if_output.get('status') == 'OK':
        trace.append({'rule_id':'IF_ROLE','action':'treat_as_distributional_evidence',
                      'reason':'IF anomaly/normality scores are not fraud probabilities and cannot independently establish legitimacy'})
    # A non-triggered rule is absence of a specific alert, NOT a vote for legitimacy.
    # Rule-support scores and calibrated RF probabilities do not have common semantics.
    # Only flag a *qualitative* tension: fired rules coexisting with RF's strong
    # legitimate prediction, without subtracting uncalibrated numbers.
    if rf.get('status') == 'OK' and fired and float(rf['fraud_probability']) <= p['rf_strong_legitimate_threshold']:
        trace.append({'rule_id':'QUALITATIVE_EVIDENCE_TENSION','action':'investigate_positive_rules',
                      'reason':'Triggered risk rules coexist with a strong RF legitimate prediction; neither source automatically overrides the other',
                      'triggered_rules':sorted(fired)})
    evidence = []
    for item in rf_top[:8]:
        evidence.append({'source_engine':'random_forest','proposition':'rf_local_path_contribution',
                         'feature':item.get('feature'), 'direction':item.get('direction'), 'raw_probability_contribution':item.get('raw_probability_contribution'),
                         'source_features':[item.get('feature')], 'explanation_kind':'exact_tree_path_attribution_not_causal'})
    for item in available_rules:
        evidence.append({'source_engine':'rule_based_engine','proposition':item['rule'], 'triggered':item.get('triggered'),
                         'source_features':item.get('dependencies',[]),'explanation_kind':'deterministic_policy_rule'})
    return {'facts':facts,'supporting_evidence':supporting,'contradictory_evidence':contradicting,
            'dependency_adjustments':dependencies,'arbitration':trace,'normalized_evidence':evidence,
            'policy':p,'reasoning_method':'deterministic_provenance_aware_context_rules_not_LLM'}
