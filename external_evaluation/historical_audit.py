"""Recompute 924 D1 historical metrics from shipped frozen MoE/GAM/RF artifacts.

NO retraining. NO use of Poison-Hunter data. Standard Python only.
"""
import csv
import json
import math
from pathlib import Path
import sys
if __package__:
    from .metrics import calculate
    from .run import atomic_json,csv_write
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.metrics import calculate
    from external_evaluation.run import atomic_json

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
ART=REPO/'fraud_reasoning'/'artifacts'
IF_KEYS=['if_anomaly_percentile','if_applicability','if_bootstrap_variance',
         'log_activity_rate','log_total_transactions','if_training_activity_unsupported']
RULE_NAMES=['rapid_transaction_activity','fund_splitting','fund_consolidation',
            'new_wallet_high_value','dust_spam_activity','contract_activity',
            'erc20_activity','high_activity']
RULE_KEYS=['rule_fired_'+r for r in RULE_NAMES]+['rule_unavailable_'+r for r in RULE_NAMES]+[
            'rule_coverage','rule_trigger_count','log_activity_rate']
GATE_KEYS=['rf_raw_probability','if_anomaly_percentile','if_applicability','log_activity_rate',
           'log_total_transactions','rule_trigger_count','rule_coverage',
           'rf_volume_path_contribution','if_training_activity_unsupported']

def expit(x):
    if x>=0:return 1./(1.+math.exp(-x))
    e=math.exp(x)
    return e/(1.+e)

def vec(row, names):return [float(row.get(k,0) or 0) for k in names]

def dot(x,y):return sum(a*b for a,b in zip(x,y))

def adapter(row, obj, keys):
    x=vec(row,keys)
    xs=[(a-b)/s for a,b,s in zip(x,obj['mean'],obj['scale'])]
    return expit(dot(xs,obj['coef'])+obj['intercept'])

def moe_score(row,artifact):
    scores=[float(row['rf_raw_probability']),
            adapter(row,artifact['if_adapter'],IF_KEYS),
            adapter(row,artifact['rule_adapter'],RULE_KEYS)]
    gate=vec(row,GATE_KEYS);st=artifact['gate_scaler']
    normalized=[1.]+[(a-b)/s for a,b,s in zip(gate,st['mean'],st['scale'])]
    logits=[sum(a*b[j] for a,b in zip(normalized,artifact['gate_weights'])) for j in range(3)]
    mx=max(logits);w=[math.exp(v-mx) for v in logits];sw=sum(w)
    raw=max(1e-7,min(1.-1e-7,sum(a*b for a,b in zip(scores,w))/sw))
    return expit(artifact['calibration_b']+artifact['calibration_a']*math.log(raw/(1.-raw)))

def main():
    meta=list(csv.DictReader((ART/'historical_test_meta_features.csv').open(newline='')))
    stored=list(csv.DictReader((ART/'historical_holdout_predictions.csv').open(newline='')))
    model=json.loads((ART/'contextual_moe_v07.json').read_text())
    assert len(meta)==len(stored)==924,'Unexpected historical source length'
    cases={'RF_raw':[],'RF_calibrated':[],'GAM':[],'MoE':[]}
    preds=[]
    for j,(row,original) in enumerate(zip(meta,stored)):
        label=int(row['FLAG'])
        if label!=int(original['label']):raise ValueError(f'Historical label mismatch on row {j}')
        scores={'RF_raw':float(row['rf_raw_probability']),
                'RF_calibrated':float(original['rf_probability']),
                'GAM':float(original['gam_probability']),
                'MoE':moe_score(row,model)}
        for name,p in scores.items():
            cases[name].append({'label':label,'decision':'FRAUD' if p>=.5 else 'LEGITIMATE','probability':p})
        preds.append({'row':j,'label':label,**{name+'_probability':v for name,v in scores.items()}})
    metrics={name:calculate(r) for name,r in cases.items()}
    claimed=json.loads((ART/'historical_comparison_report.json').read_text())['historical_D1_metrics']
    for name in ['RF_raw','RF_calibrated','GAM','MoE']:
        assert abs(metrics[name]['roc_auc']-claimed[name]['roc_auc'])<1e-5
        assert abs(metrics[name]['brier']-claimed[name]['brier'])<1e-5
    out=HERE/'results';out.mkdir(exist_ok=True)
    report={'evaluation_type':'historical_D1_non_independent',
            'note':'These are original D1 historical partition results, NOT Poison-Hunter validation.',
            'records':924,'metrics':metrics,'report_matches_shipped_reference':True}
    atomic_json(out/'historical_audit.json',report)
    with (out/'historical_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
