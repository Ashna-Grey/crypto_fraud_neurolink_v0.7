#!/usr/bin/env python3
"""Optional *real* InterpretML EBM experiment; requires interpret-core.

The deployable default is the already fitted MoE with a fitted GAM comparison.
The EBM path is not marked trained unless this command actually executes.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import joblib
try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError as exc:
    raise SystemExit('InterpretML not available. Install requirements-ebm.txt before this optional experiment; no fitted EBM is claimed.') from exc

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'fraud_reasoning/artifacts'

def read(name):
    x=pd.read_csv(P/name)
    y=x.pop('FLAG').to_numpy(dtype=int)
    return x,y

train,yt=read('oof_meta_features.csv')
cal,yc=read('calibration_meta_features.csv')
test,yh=read('historical_test_meta_features.csv')
feature_names=list(train.columns)
# Explicit interactions reflect the proposed explanation-conditioned tests.
from fraud_reasoning.learned import INTERACTIONS
pairs=[tuple(feature_names.index(n) for n in pair) for pair in INTERACTIONS]
model=ExplainableBoostingClassifier(feature_names=feature_names,
    interactions=pairs,outer_bags=3,learning_rate=.025,max_rounds=3000,
    max_bins=128,max_interaction_bins=32,n_jobs=1,random_state=42)
model.fit(train,yt)
raw_cal=np.clip(model.predict_proba(cal)[:,1],1e-8,1-1e-8)
calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(logit(raw_cal).reshape(-1,1),yc)
raw_test=np.clip(model.predict_proba(test)[:,1],1e-8,1-1e-8)
pred=calibrator.predict_proba(logit(raw_test).reshape(-1,1))[:,1]
metrics={'n_historical':len(yh),'roc_auc':float(roc_auc_score(yh,pred)),
         'pr_auc':float(average_precision_score(yh,pred)),
         'brier':float(brier_score_loss(yh,pred)),
         'not_independent_external_validation':True}
joblib.dump({'ebm':model,'calibrator':calibrator,'feature_names':feature_names,
            'package_version':__import__('interpret').__version__},
            P/'optional_ebm_v07.joblib',compress=3)
(P/'optional_ebm_results.json').write_text(json.dumps(metrics,indent=2))
print(metrics)
