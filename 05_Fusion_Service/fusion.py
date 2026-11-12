"""Conservative evidence fusion. IF is an applicability signal, never a fraud probability."""
from __future__ import annotations
import math

def clamp(x):
    return max(0.0,min(1.0,float(x)))

def combine(rf_probability, rule_support=None, *, rf_reliability=0.7, rule_reliability=0.35,
            rf_quality=1.0, rule_quality=1.0, applicability=1.0, if_anomaly=None,
            conformal_decision=None, conflict_limit=0.35, min_evidence=0.30):
    """Uncalibrated opinion fusion with uncertainty-preserving conflict discount.

    Reliability defaults are *policy placeholders*, NOT measured Brier skill.
    No posterior probability or statistical coverage is claimed for the fused score.
    """
    if rf_probability is None:
        return dict(decision='ABSTAIN',reason='RF_UNAVAILABLE',fusion_support=None,uncertainty=1.0)
    p=clamp(rf_probability)
    eta_rf=clamp(rf_reliability)*clamp(rf_quality)*clamp(applicability)
    engines=[('rf',p,eta_rf)]
    if rule_support is not None:
        engines.append(('rules',clamp(rule_support),clamp(rule_reliability)*clamp(rule_quality)))
    # Projected probabilities use an explicitly fixed neutral base rate a=0.5.
    projected=[eta*v+(1-eta)*0.5 for _,v,eta in engines]
    conflict=abs(projected[0]-projected[1])*engines[0][2]*engines[1][2] if len(engines)==2 else 0.0
    # Do not assume majority vote; disagreement increases uncertainty rather than silencing a minority.
    shrink=1.0-clamp(conflict)
    weights=[eta*shrink for _,_,eta in engines]
    weight=sum(weights)
    support=(sum(w*v for w,(_,v,_) in zip(weights,engines))/weight) if weight>0 else None
    uncertainty=clamp(1-weight/len(engines)+conflict)
    reasons=[]
    if applicability<0.5: reasons.append('TRAINING_DISTRIBUTION_MISMATCH')
    if rf_quality<0.5: reasons.append('RF_INPUT_QUALITY_LOW')
    if conflict>=conflict_limit: reasons.append('ENGINE_CONFLICT')
    if eta_rf<min_evidence: reasons.append('INSUFFICIENT_RF_EVIDENCE')
    if conformal_decision in ('ABSTAIN','AMBIGUOUS'): reasons.append('RF_CONFORMAL_'+conformal_decision)
    if rule_support is None: reasons.append('RULE_ENGINE_UNAVAILABLE')
    if if_anomaly is None: reasons.append('IF_UNAVAILABLE')
    if reasons:
        decision='ABSTAIN'
    elif support is None:
        decision='ABSTAIN'
    else:
        decision='REVIEW_HIGH_RISK' if support>=0.5 else 'LOWER_PRIORITY'
    return dict(decision=decision,reason=';'.join(reasons) if reasons else 'EVIDENCE_SUFFICIENT',
                fusion_support=round(support,6) if support is not None else None,
                uncertainty=round(uncertainty,6),conflict=round(conflict,6),
                reliability={'rf':round(eta_rf,6),'rules':round(engines[1][2],6) if len(engines)>1 else None},
                if_anomaly_percentile=if_anomaly,
                score_semantics='uncalibrated_priority_score_not_fraud_probability')
