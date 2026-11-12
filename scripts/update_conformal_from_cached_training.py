#!/usr/bin/env python3
"""Regenerate disjoint conformal and historical test evidence without refitting OOF RF.

Only safe when GAM training and calibration are already complete and reproducible.
For full retraining use train_learned_arbitration.py instead.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from fraud_reasoning.service import FraudIntelligence
from fraud_reasoning.learned import rf_group_contributions,export_portable_gam
from train_learned_arbitration import duplicate_clean_splits,make_model_features,metric
P=ROOT/'fraud_reasoning/artifacts'
d=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv')
split=joblib.load(ROOT/'03_Random_Forest/artifacts/rf_data_split_indices.joblib')
blocks,excluded=duplicate_clean_splits(d,split)
s=FraudIntelligence(load_learned=False)
conf=blocks['meta_conformal'];test=blocks['meta_historical_test']
features=s.rf_features
raw_frame=conf[features]
rf_scores=s.rf_model.predict_proba(s.rf_imputer.transform(raw_frame))[:,list(s.rf_model.classes_).index(1)]
groups=rf_group_contributions(s.rf_model,s.rf_imputer,raw_frame,features)
ifcols=s.if_features
if_frame=pd.DataFrame([{f:record.get(f) for f in ifcols} for record in conf.to_dict('records')])
print('Scoring new separate conformal group of',len(conf),flush=True)
if_scores=s.if_predict.score_dataframe(if_frame,s.if_bundle,input_format='raw-live').reset_index(drop=True)
xconf=make_model_features(conf,rf_scores,groups,if_scores,s.rule_adapter,features)
xconf.assign(FLAG=conf.FLAG.to_numpy(int)).to_csv(P/'conformal_meta_features.csv',index=False)
prev_predictions=pd.read_csv(P/'historical_holdout_predictions.csv')
prev_test=pd.read_csv(P/'historical_test_meta_features.csv')
assert len(prev_predictions)==len(prev_test)
mask=prev_predictions.source_row_index.isin(test.index).to_numpy()
prev_predictions=prev_predictions.loc[mask].reset_index(drop=True)
prev_test=prev_test.loc[mask].reset_index(drop=True)
assert prev_predictions.source_row_index.tolist()==[int(x) for x in test.index]
learned=joblib.load(P/'evidence_gam_v07.joblib')
gam=learned['gam'];meta=learned['metadata']
cal_p=gam.predict_proba(xconf)[:,1];labels=conf.FLAG.to_numpy(int)
nonconf=1.-np.where(labels==1,cal_p,1.-cal_p)
q=min(1.,np.ceil((len(labels)+1)*.90)/len(labels))
cutoff=float(np.quantile(nonconf,q,method='higher'))
meta['conformal']={'alpha':0.1,'nonconformity_quantile':cutoff,
                   'source':'disjoint_original_conformal_indices',
                   'coverage_assumption':'exchangeability_not_proven_under_dataset_shift'}
meta['counts']={k:len(v) for k,v in blocks.items()}
meta['duplicate_removed']=excluded
meta['class_balance']={k:float(v.FLAG.mean()) for k,v in blocks.items()}
y=prev_test.FLAG.to_numpy(int)
pp=gam.predict_proba(prev_test.drop(columns=['FLAG']))[:,1]
meta['historical_test']['learned_spline_gam']=metric(y,pp)
meta['historical_test']['original_rf_calibrated']=metric(y,prev_predictions.rf_probability.to_numpy(float))
meta['historical_test']['original_rf_raw']=metric(y,prev_test.rf_raw_probability.to_numpy(float))
meta['calibration']='held_out_meta_Platt_original_calibration_indices; separate_original_conformal_indices_for_conformal'
(P/'learned_evaluation.json').write_text(json.dumps(meta,indent=2))
joblib.dump({'gam':gam,'meta_features':learned['meta_features'],'metadata':meta},P/'evidence_gam_v07.joblib',compress=3)
export_portable_gam(gam,meta,P/'evidence_gam_v07.json')
prev_test.to_csv(P/'historical_test_meta_features.csv',index=False)
prev_predictions.to_csv(P/'historical_holdout_predictions.csv',index=False)
print('Conformal q',cutoff,'updated heldout n',len(y),flush=True)
