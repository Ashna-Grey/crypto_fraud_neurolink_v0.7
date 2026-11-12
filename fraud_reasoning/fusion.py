"""PROPOSED, UNCALIBRATED evidence arbitration and RF-anchored risk prioritization.

RF p is a calibrated-under-training-conditions fraud estimate, not guaranteed
calibrated after deployment. IF measures reference-distribution anomalousness,
not fraud probability. Rule signals are weighted *positive alerts*, not
negative votes when nothing fires. The following transformations and
credibilities are explicit policy heuristics requiring external validation.

Context discount (proposed): p_ctx = .5+(p-.5)*(1-.5*v) if p>.5 and
SLOW_VOLUME applies, where v is the fraction of positive RF local attribution
from transaction-count features. This weakens a volume-supported fraud claim
without declaring an unobserved counterfactual. Other cases p_ctx=p.

Risk priority (proposed): R = p_ctx + (1-p_ctx)*(.25 * rule_support * rule_coverage
* dependency_discount). This is NOT a calibrated fraud probability and does
not equate missing/untriggered rules to negative evidence.

Review index (proposed): U = clip(.10 + .15*(1-rf_quality) + .10*(1-rule_coverage)
+ .15*(1-if_app) + .20*qualitative_tension + .15*context_adjustment
+ .10*min(1,tree_variance/.25)). The index is NOT epistemic probability.
"""
from __future__ import annotations
from .logic import POLICY


def clip(v):
    return min(1.0, max(0.0, float(v)))


def fuse(rf, if_out, rules, reasoning, *, policy=None):
    conf = {**POLICY, **(policy or {})}
    if rf.get('status') != 'OK':
        return {'fused_risk_score': None, 'uncertainty': 1.0, 'calibrated': False,
                'decision': 'MANUAL_REVIEW', 'reasons': ['RF_UNAVAILABLE']}

    p = clip(rf['fraud_probability'])
    rf_quality = clip(rf.get('input_quality', 0))
    raw_attr = rf.get('attribution', {}).get('reasoning', rf.get('reasoning', []))
    # Full additive tree-path contributions are used for the decision. Top-k
    # explanations are for display only. Fallback supports legacy test data.
    contributions = rf.get('attribution', {}).get('all_feature_contributions')
    if not isinstance(contributions, dict):
        contributions = {str(a.get('feature', '')): float(a.get('raw_probability_contribution') or 0)
                         for a in raw_attr}
    positives = sum(max(0.0, float(value)) for value in contributions.values())
    volume_pos = sum(max(0.0, float(value)) for name, value in contributions.items()
                     if 'total transactions' in name.lower() or name.lower() == 'total_transactions')
    volume_fraction = clip(volume_pos / positives) if positives > 0 else 0.0
    slow_context = any(a.get('rule_id') == 'SLOW_VOLUME' for a in reasoning.get('arbitration', []))
    contextual_discount = 1.0 - 0.5 * volume_fraction if slow_context and p > 0.5 else 1.0
    p_context = .5 + (p - .5) * contextual_discount if slow_context and p > .5 else p

    coverage = clip(rules.get('coverage') or 0)
    rb_support = rules.get('risk_score')
    rb_support = clip(rb_support) if rb_support is not None else None
    triggered = rules.get('triggered_rules', [])
    dependencies = reasoning.get('dependency_adjustments', [])
    dependency_discount = 0.85 if dependencies else 1.0

    # A zero score from *evaluated, non-firing* rules cannot lower RF fraud
    # probability, regardless of the fraction of missing rules.
    additive_rule_signal = (0.25 * rb_support * coverage * dependency_discount
                            if rb_support is not None and triggered else 0.0)
    priority = clip(p_context + (1.0-p_context) * additive_rule_signal)

    if_app = if_out.get('applicability', {}).get('score') if if_out.get('status') == 'OK' else None
    if_app = clip(if_app) if if_app is not None else 0.0
    weak_if = if_app < conf['minimum_applicability']
    qualitative_tension = any(a.get('rule_id') == 'QUALITATIVE_EVIDENCE_TENSION'
                              for a in reasoning.get('arbitration', []))
    tree_variance = max(0.0, float(rf.get('tree_disagreement_variance') or 0))
    review_index = clip(0.10 + 0.15 * (1-rf_quality) + 0.10 * (1-coverage)
                        + 0.15 * (1-if_app) + 0.20 * float(qualitative_tension)
                        + 0.15 * (1-contextual_discount)
                        + 0.10 * min(1.0, tree_variance/.25))

    # Subjective opinions are shown as *accounting of credibility*, NOT as an
    # evidence-combination probability. A non-firing rule makes no opinion.
    rf_eta = clip(.70 * rf_quality * contextual_discount)
    rule_eta = clip(.35 * coverage * dependency_discount) if triggered else 0.0
    opinions = [{'engine': 'random_forest', 'p': p_context, 'eta': rf_eta}]
    if triggered:
        opinions.append({'engine': 'rule_based_engine', 'p': rb_support, 'eta': rule_eta})
    for item in opinions:
        item['belief_fraud'] = item['eta'] * item['p']
        item['belief_legitimate'] = item['eta'] * (1-item['p'])
        item['uncommitted_uncertainty'] = 1-item['eta']

    reasons = []
    advisories = []
    if coverage < 1:
        advisories.append('RULE_EVIDENCE_INCOMPLETE')
    if weak_if:
        reasons.append('IF_TRAINING_APPLICABILITY_LOW_OR_UNAVAILABLE')
    if qualitative_tension:
        reasons.append('QUALITATIVE_EVIDENCE_TENSION')
    if rf.get('conformal_decision') in ('AMBIGUOUS', 'ABSTAIN'):
        reasons.append('RF_CONFORMAL_UNCERTAIN')
    if rf_quality < 1:
        reasons.append('RF_IMPUTED_FEATURES')
    if coverage < conf['minimum_rule_coverage_for_rf_led_decision']:
        reasons.append('INSUFFICIENT_RULE_COVERAGE')
    if review_index > conf['max_decision_uncertainty']:
        reasons.append('EXCESSIVE_UNCERTAINTY')
    anomaly_pct = if_out.get('anomaly_percentile')
    if anomaly_pct is not None and float(anomaly_pct) >= conf['high_ood_anomaly_percentile']:
        reasons.append('EXTREME_IF_REFERENCE_ANOMALY')
    if slow_context and contextual_discount < 0.9 and p >= conf['rf_strong_fraud_threshold']:
        reasons.append('MATERIAL_CONTEXTUAL_RF_CONFLICT')

    # Provisional RF-anchored policy: a score does NOT create a diagnosis.
    # RF conformal label acts as an additional consistency gate. No IF or
    # rule engine is incorrectly treated as independently calibrated fraud.
    decision = 'MANUAL_REVIEW'
    if not reasons:
        conformal = rf.get('conformal_decision')
        if (priority >= conf['fraud_decision_threshold']
                and p >= conf['rf_strong_fraud_threshold']
                and conformal == 'FRAUD'):
            decision = 'FRAUD'
        elif (priority <= conf['legitimate_decision_threshold']
                and p <= conf['rf_strong_legitimate_threshold']
                and conformal in ('NON_FRAUD', 'LEGITIMATE')):
            decision = 'LEGITIMATE'
        else:
            reasons.append('INSUFFICIENT_RF_CONFIDENCE_OR_FUSION_MARGIN')

    return {
        'method': 'proposed_rf_anchored_positive_rule_fusion_with_contextual_arbitration',
        'fused_risk_score': round(priority, 6),
        'uncertainty': round(review_index, 6),
        'calibrated': False,
        'pairwise_conflict_proxy': None,
        'score_semantics': 'uncalibrated_risk_priority_not_fraud_probability',
        'opinion_components': [
            {'engine': o['engine'], 'belief_fraud': round(o['belief_fraud'], 6),
             'belief_legitimate': round(o['belief_legitimate'], 6),
             'uncommitted_uncertainty': round(o['uncommitted_uncertainty'], 6),
             'credibility': round(o['eta'], 6)} for o in opinions],
        'adjustments': {
            'rf_contextual_discount': round(contextual_discount, 6),
            'rf_volume_attribution_fraction_of_positive_top_features': round(volume_fraction, 6),
            'rf_context_adjusted_probability_proxy': round(p_context, 6),
            'rule_dependency_discount': dependency_discount,
            'rule_positive_signal': round(additive_rule_signal, 6),
            'rule_coverage': coverage,
            'if_applicability': if_app,
        },
        'advisories': advisories,
        'reliability_source': 'provisional_explicit_policy_not_Brier_validated',
        'decision': decision,
        'reasons': reasons,
    }
