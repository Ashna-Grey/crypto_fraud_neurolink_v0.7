#!/usr/bin/env python3
"""Reproducible cross-fitted evidence meta-learner using existing D1 split provenance.

Only the original RF *training indices* train the meta model (OOF). Original
calibration indices fit a NEW held-out meta Platt calibrator; original test
indices are used only for local historical evaluation. All duplicates shared
across these roles are excluded. The latter is NOT independent external data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score)

from fraud_reasoning.learned import rf_group_contributions, evidence_row, SplineEvidenceGAM, FEATURE_NAMES, export_portable_gam
from fraud_reasoning.rules import RuleAdapter
from fraud_reasoning.service import FraudIntelligence


def duplicate_clean_splits(data, split):
    cols=[col for col in data if col != 'FLAG']
    used=set();out={}; excluded={}
    for role,key in [('meta_train','train_indices'),('meta_calibration','calibration_indices'),('meta_conformal','conformal_indices'),('meta_historical_test','test_indices')]:
        block=data.iloc[split[key]].copy()
        profile=pd.util.hash_pandas_object(block[cols],index=False).to_numpy()
        allowed=[]
        count=0
        for index,h in zip(block.index,profile):
            if h in used:
                count+=1
            else:
                allowed.append(index);used.add(h)
        out[role]=data.loc[allowed].copy()
        excluded[role]=int(count)
    return out,excluded


def metric(y,p):
    pred=(np.asarray(p)>=.5).astype(int)
    return {'n':int(len(y)),'fraud_count':int(sum(y)),
            'roc_auc':round(float(roc_auc_score(y,p)),6),
            'pr_auc':round(float(average_precision_score(y,p)),6),
            'brier':round(float(brier_score_loss(y,p)),6),
            'log_loss':round(float(log_loss(y,p,labels=[0,1])),6),
            'accuracy':round(float(accuracy_score(y,pred)),6),
            'precision':round(float(precision_score(y,pred,zero_division=0)),6),
            'recall':round(float(recall_score(y,pred,zero_division=0)),6),
            'f1':round(float(f1_score(y,pred,zero_division=0)),6),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def make_model_features(data, rf_scores, groups, if_scores, rules, rf_features):
    records=[]
    for i,(_,row) in enumerate(data.iterrows()):
        row=row.to_dict()
        rr=rules.run(row)
        ifscore=if_scores.iloc[i]
        count = row.get('total transactions (including tnx to create contract')
        missing=float(ifscore['missing_feature_ratio']);outside=float(ifscore['outside_training_minmax_count'])
        app=max(0.,1.-missing-outside/20.)
        try:
            if count is not None and float(count)>200:app=min(app,.2)
        except (TypeError,ValueError):pass
        info={'status':'OK','anomaly_percentile':float(ifscore['anomaly_percentile']),
              'bootstrap_anomaly_variance':float(ifscore['if_sigma_squared']),
              'applicability':{'score':app}}
        contrib={g:float(v[i]) for g,v in groups.items()}
        records.append(evidence_row(row,rf_scores[i],contrib,info,rr,rf_features))
    return pd.DataFrame(records,columns=FEATURE_NAMES)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv'))
    parser.add_argument('--output',default=str(ROOT/'fraud_reasoning/artifacts'))
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--n-estimators',type=int,default=1000)
    parser.add_argument('--max-historical-eval',type=int,default=None)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    print('Loading original model and D1 split metadata...',flush=True)
    service=FraudIntelligence(load_learned=False)
    data=pd.read_csv(args.data)
    if data.columns.duplicated().any() or 'FLAG' not in data:raise ValueError('Invalid source data')
    split=joblib.load(ROOT/'03_Random_Forest/artifacts/rf_data_split_indices.joblib')
    blocks,excluded=duplicate_clean_splits(data,split)
    features=service.rf_features
    train=blocks['meta_train']; cal=blocks['meta_calibration']; conformal=blocks['meta_conformal']; test=blocks['meta_historical_test']
    print('Clean role sizes:',{k:len(v) for k,v in blocks.items()},'excluded duplicates:',excluded,flush=True)
    y_train=train.FLAG.to_numpy(int)
    oof_scores=np.full(len(train),np.nan)
    oof_groups={k:np.full(len(train),np.nan) for k in (
        'rf_volume_path_contribution','rf_temporal_path_contribution',
        'rf_erc20_path_contribution','rf_other_path_contribution')}
    print(f'Cross-fitting {args.folds} RF models (training-only labels)...',flush=True)
    cv=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=48)
    for i,(tr,va) in enumerate(cv.split(train,y_train),1):
        imputer=SimpleImputer(strategy='median')
        xt=imputer.fit_transform(train.iloc[tr][features])
        model=RandomForestClassifier(n_estimators=args.n_estimators,
                                     class_weight='balanced',random_state=42,
                                     n_jobs=-1)
        model.fit(xt,y_train[tr])
        v=train.iloc[va][features]
        oof_scores[va]=model.predict_proba(imputer.transform(v))[:,list(model.classes_).index(1)]
        g=rf_group_contributions(model,imputer,v,features)
        for k in oof_groups:oof_groups[k][va]=g[k]
        print('  fold',i,'/',args.folds,'done',flush=True)
    if not np.isfinite(oof_scores).all():raise AssertionError('Incomplete OOF predictions')

    # Original production RF never trained on its calibration or test indices.
    def reference_features(block):
        blockx=block[features]
        x=service.rf_imputer.transform(blockx)
        r=service.rf_model.predict_proba(x)[:,list(service.rf_model.classes_).index(1)]
        return r,rf_group_contributions(service.rf_model,service.rf_imputer,blockx,features)
    cal_rf,cal_groups=reference_features(cal)
    conf_rf,conf_groups=reference_features(conformal)
    test_rf,test_groups=reference_features(test)

    print('Scoring reference IF on all role partitions (D2-trained, label-free)...',flush=True)
    def if_scores(block):
        cols=service.if_features
        frame=pd.DataFrame([{f:record.get(f) for f in cols} for record in block.to_dict('records')])
        return service.if_predict.score_dataframe(frame,service.if_bundle,input_format='raw-live').reset_index(drop=True)
    if_train=if_scores(train)
    if_cal=if_scores(cal)
    if_conf=if_scores(conformal)
    if_test=if_scores(test)
    rules=service.rule_adapter
    print('Building the evidence matrix...',flush=True)
    x_train=make_model_features(train,oof_scores,oof_groups,if_train,rules,features)
    x_cal=make_model_features(cal,cal_rf,cal_groups,if_cal,rules,features)
    x_conf=make_model_features(conformal,conf_rf,conf_groups,if_conf,rules,features)
    x_test=make_model_features(test,test_rf,test_groups,if_test,rules,features)
    if not x_train.notna().all().all():raise AssertionError('Missing meta matrix values')
    print('Fitting spline GAM on OOF evidence and held-out sigmoid calibration...',flush=True)
    gam=SplineEvidenceGAM(c=.35,n_knots=5)
    gam.fit(x_train,y_train)
    gam.calibrate(x_cal,cal.FLAG.to_numpy(int))
    oof_p=gam.predict_proba(x_train)[:,1]
    test_p=gam.predict_proba(x_test)[:,1]
    print('Historical evaluation:',metric(test.FLAG.to_numpy(int),test_p),flush=True)
    baseline=service.rf_predict.predict_dataframe(test[features])
    rf_calibrated=np.asarray(baseline['rf_mu_calibrated'],float)
    results={'version':'0.7.0','training_mode':'cross_fitted_meta_with_original_RF_source_train_only',
             'base_models':{'rf':'original_RF_1000_trees_deployed_and_1000_trees_OOF',
                            'if':'original_IF_D2_only','rules':'original_eight_rule_activations'},
             'n_estimators_oof':args.n_estimators,'folds':args.folds,
             'duplicate_removed':excluded,
             'counts':{role:len(block) for role,block in blocks.items()},
             'class_balance':{role:float(block.FLAG.mean()) for role,block in blocks.items()},
             'meta_features':FEATURE_NAMES,'calibration':'held_out_sigmoid_on_original_RF_calibration_split_and_separate_conformal_partition',
             'historical_test':{'learned_spline_gam':metric(test.FLAG.to_numpy(int),test_p),
                                'original_rf_calibrated':metric(test.FLAG.to_numpy(int),rf_calibrated),
                                'original_rf_raw':metric(test.FLAG.to_numpy(int),test_rf)},
             'dataset_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
             'limitations':[
                'D1 has been analyzed repeatedly; historical test is NOT independent external evaluation.',
                'The original RF selected its 20 features using original training data. OOF models use those preselected columns.',
                'The original RF retains training-split duplicate multiplicities while OOF RF is trained after exact-profile deduplication.',
                'The original RF calibration and meta calibration reuse the same original D1 calibration split (different score stages).',
                'The meta conformal threshold uses a distinct original conformal partition, whose labels were already used by the legacy RF conformal component but not by the meta predictor.',
                'Calibrated probabilities may not transport to new blockchain populations or changed fraud prevalence.',
                'Rule thresholds and IF training distribution are unchanged. IF is never treated as fraud probability.',
                'Local terms attribute model scores, not causes or evidence of criminal activity.']}
    # Conformal set on calibration predictions. No distribution-free guarantees under shift.
    pc=gam.predict_proba(x_conf)[:,1]
    yy=conformal.FLAG.to_numpy(int)
    nonconf=1.-np.where(yy==1,pc,1.-pc)
    alpha=.10
    n=len(nonconf)
    q_level=min(1.0,np.ceil((n+1)*(1-alpha))/n)
    cutoff=float(np.quantile(nonconf,q_level,method='higher'))
    results['conformal']={'alpha':alpha,'nonconformity_quantile':cutoff,'coverage_assumption':'exchangeability_not_proven_under_dataset_shift'}
    pred_cal=(test_p>=.5).astype(int)
    (output/'learned_evaluation.json').write_text(json.dumps(results,indent=2))
    joblib.dump({'gam':gam,'meta_features':FEATURE_NAMES,'metadata':results},output/'evidence_gam_v07.joblib',compress=3)
    export_portable_gam(gam,results,output/'evidence_gam_v07.json')
    pred_frame=pd.DataFrame({'source_row_index':test.index,'label':test.FLAG.to_numpy(int),
                             'rf_probability':rf_calibrated,'gam_probability':test_p,'gam_decision':pred_cal})
    pred_frame.to_csv(output/'historical_holdout_predictions.csv',index=False)
    pd.DataFrame(x_train).assign(FLAG=y_train).to_csv(output/'oof_meta_features.csv',index=False)
    pd.DataFrame(x_cal).assign(FLAG=cal.FLAG.to_numpy(int)).to_csv(output/'calibration_meta_features.csv',index=False)
    pd.DataFrame(x_conf).assign(FLAG=conformal.FLAG.to_numpy(int)).to_csv(output/'conformal_meta_features.csv',index=False)
    pd.DataFrame(x_test).assign(FLAG=test.FLAG.to_numpy(int)).to_csv(output/'historical_test_meta_features.csv',index=False)
    print('Saved:',output.resolve(),'meta_calibrated',gam.calibration_qualified_,flush=True)


if __name__=='__main__': main()
