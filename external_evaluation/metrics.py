"""Transparent, dependency-free binary metrics. Abstentions never count as correct predictions."""
import math
from collections import Counter

def finite_probability(x):
    if isinstance(x, bool) or x is None: return None
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) and 0 <= v <= 1 else None

def roc_auc(y, p):
    n1, n0 = sum(y), len(y) - sum(y)
    if not n1 or not n0: return None
    pairs = sorted(zip(p, y))
    i = 0; ranks = 0.0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        ranks += sum(label for _, label in pairs[i:j]) * (i + j + 1) / 2
        i = j
    return (ranks - n1 * (n1 + 1) / 2) / (n1 * n0)

def average_precision(y, p):
    n1 = sum(y)
    if not n1: return None
    pairs = sorted(zip(p, y), reverse=True)
    tp = 0; count = 0; ap = 0.0; i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        tp += sum(label for _, label in pairs[i:j]); count += j - i
        # sklearn-style threshold/group tie handling
        delta = sum(label for _, label in pairs[i:j])
        ap += delta * tp / count
        i = j
    return ap / n1

def wilson_interval(successes, total, z=1.95996398454):
    if total<=0:return None
    phat=successes/total;z2=z*z;den=1+z2/total
    center=(phat+z2/(2*total))/den
    radius=(z/den)*math.sqrt(phat*(1-phat)/total+z2/(4*total*total))
    return [max(0.,center-radius),min(1.,center+radius)]

def calculate(rows, threshold=0.5):
    accepted = []
    for r in rows:
        p = finite_probability(r.get('probability'))
        if p is None or str(r.get('decision', '')).upper() not in ('FRAUD','LEGITIMATE'):
            continue
        label = int(r['label'])
        if label not in (0, 1): continue
        accepted.append((label, p, 1 if p >= threshold else 0))
    if not accepted: return None
    y = [q[0] for q in accepted]; p = [q[1] for q in accepted]; pred = [q[2] for q in accepted]
    tp = sum(a == 1 and b == 1 for a, b in zip(y,pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y,pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y,pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y,pred))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = (2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None
    return {'n':len(y), 'positive_n':sum(y), 'negative_n':len(y)-sum(y),
            'TP':tp,'TN':tn,'FP':fp,'FN':fn,'accuracy':(tp+tn)/len(y),
            'precision':precision,'recall':recall,'f1':f1,
            'accuracy_wilson_95pct_ci':wilson_interval(tp+tn,len(y)),
            'recall_wilson_95pct_ci':wilson_interval(tp,tp+fn),
            'precision_wilson_95pct_ci':wilson_interval(tp,tp+fp),
            'roc_auc':roc_auc(y,p), 'average_precision':average_precision(y,p),
            'brier':sum((lab-score)**2 for lab,score in zip(y,p))/len(y),
            'threshold':threshold}

def summarize(rows, requested, provenance='poison_hunter_retrospective_200'):
    tally = Counter(str(r.get('status','unknown')) for r in rows)
    complete = sum(r.get('complete') is True for r in rows)
    classified = [r for r in rows if r.get('status') == 'classified' and finite_probability(r.get('probability')) is not None]
    provisional = [r for r in classified if r.get('complete') is True]
    mismatch = sum(bool(r.get('decision_mismatch')) for r in rows)
    return {
      'evaluation_type':provenance,'sample_requested':requested,'processed':len(rows),
      'collection_complete':complete,'status_counts':dict(sorted(tally.items())),
      'binary_with_probabilities':len(classified), 'complete_history_binary_predictions':len(provisional),
      'decision_probability_mismatch_count':mismatch,
      'label_counts_processed':dict(sorted(Counter(str(r.get('label')) for r in rows).items())),
      'status_by_ground_truth_label':{str(label):dict(sorted(Counter(str(r.get('status','unknown')) for r in rows if str(r.get('label'))==str(label)).items())) for label in (0,1)},
      'error_samples':[{'address':r.get('address'),'status':r.get('status'),'error':r.get('error')} for r in rows if r.get('error')][:15],
      'provisional_metrics_complete_collections':calculate(provisional),
      'diagnostic_metrics_all_classified':calculate(classified),
      'publication_grade_external_accuracy':None,
      'why_not_publication_grade':[
          'Original D1 address overlap cannot be excluded: original D1 has no wallet identifiers.',
          'Exact D1 feature definitions and preprocessing parity have not been proven; ERC20 values and ETH balance may be missing or incomparable.',
          'Poison-Hunter is a specialized retrospective address-poisoning dataset, not representative of all fraud.',
          'Benign addresses are selected from popular Etherscan wallets; the sampled class balance is artificial.',
          'No event-time cutoff is supplied for attack labels; current-chain information may postdate labeling.',
      ]
    }
