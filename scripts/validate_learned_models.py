#!/usr/bin/env python3
"""Recompute historical comparisons, bootstrapped paired uncertainty, subgroups.

This is historical reused D1 data. Do not describe it as external validation.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fraud_reasoning.moe import PortableContextualMoE
from fraud_reasoning.learned import PortableEvidenceGAM
P=ROOT/'fraud_reasoning/artifacts'


def summary(y,p):
    if len(set(y))<2:return {'n':len(y),'fraud':int(sum(y)),'brier':float(brier_score_loss(y,p))}
    return {'n':len(y),'fraud':int(sum(y)),
            'roc_auc':round(float(roc_auc_score(y,p)),6),
            'pr_auc':round(float(average_precision_score(y,p)),6),
            'brier':round(float(brier_score_loss(y,p)),6)}


def main():
    frame=pd.read_csv(P/'historical_test_meta_features.csv')
    y=frame.pop('FLAG').to_numpy(int)
    gam=PortableEvidenceGAM.from_file(P/'evidence_gam_v07.json')
    moe=PortableContextualMoE.from_file(P/'contextual_moe_v07.json')
    rows=frame.to_dict('records')
    scores={'RF_raw':frame.rf_raw_probability.to_numpy(float),
            'GAM':gam.predict_proba(rows)[:,1],
            'MoE':moe.predict_proba(rows)[:,1]}
    earlier=pd.read_csv(P/'historical_holdout_predictions.csv')
    scores['RF_calibrated']=earlier.rf_probability.to_numpy(float)
    report={'historical_D1_metrics':{name:summary(y,p) for name,p in scores.items()},
            'historical_dataset_is_independent_external_test':False}
    print('Historical metrics:',report['historical_D1_metrics'],flush=True)
    # Paired stratified bootstrap among historic test rows. These intervals DO
    # NOT account for source sampling, repeated development, nor shifted data.
    rng=np.random.default_rng(42)
    ind0=np.flatnonzero(y==0);ind1=np.flatnonzero(y==1)
    bootstrap=[]
    for i in range(1000):
        ids=np.r_[rng.choice(ind0,len(ind0),replace=True),rng.choice(ind1,len(ind1),replace=True)]
        yb=y[ids]
        row={}
        for comparison in ['GAM','MoE']:
            row[comparison+'_brier_minus_RF']=brier_score_loss(yb,scores[comparison][ids])-brier_score_loss(yb,scores['RF_calibrated'][ids])
            row[comparison+'_AUC_minus_RF']=roc_auc_score(yb,scores[comparison][ids])-roc_auc_score(yb,scores['RF_calibrated'][ids])
        bootstrap.append(row)
    bs=pd.DataFrame(bootstrap)
    report['paired_bootstrap_95pct_conditional']={col:[round(float(bs[col].quantile(.025)),8),round(float(bs[col].quantile(.975)),8)] for col in bs}
    txn=np.expm1(frame.log_total_transactions.to_numpy(float))
    report['activity_subgroups']={}
    for label,mask in [('0-10',txn<=10),('11-50',(txn>10)&(txn<=50)),('51-200',(txn>50)&(txn<=200)),('>200',txn>200)]:
        report['activity_subgroups'][label]={name:summary(y[mask],p[mask]) for name,p in scores.items()}
    quantiles=pd.qcut(scores['MoE'],5,duplicates='drop')
    report['calibration_reliability_bins']=[]
    for level in quantiles.categories:
        mask=np.asarray(quantiles==level)
        report['calibration_reliability_bins'].append({'predicted_mean':float(np.mean(scores['MoE'][mask])),
             'observed_fraud_fraction':float(np.mean(y[mask])), 'n':int(mask.sum())})
    weights=np.array([[z['learned_contextual_weight'] for z in moe.explain(row)['weights']] for row in rows])
    report['gate_aggregate']={'mean_RF_weight':float(weights[:,0].mean()),'mean_IF_adapter_weight':float(weights[:,1].mean()),'mean_rule_adapter_weight':float(weights[:,2].mean()),
                              'RF_weight_above_90pct_fraction':float((weights[:,0]>.9).mean())}
    (P/'historical_comparison_report.json').write_text(json.dumps(report,indent=2))
    print('Paired conditional intervals:',report['paired_bootstrap_95pct_conditional'],flush=True)
    print('Gate summary',report['gate_aggregate'],flush=True)

if __name__=='__main__':main()
