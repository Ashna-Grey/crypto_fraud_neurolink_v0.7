#!/usr/bin/env python3
"""Optional learned contextual MoE experiment on saved leakage-managed meta splits."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax,expit,logit
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from fraud_reasoning.moe import PortableContextualMoE,IF_COLUMNS,RULE_COLUMNS,GATE_COLUMNS

def fit_adapter(x,y,seed=42):
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0,random_state=seed))
    pipe.fit(x,y)
    return pipe

def encode(pipe):
    scaler,model=pipe.steps[0][1],pipe.steps[1][1]
    return {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}

def metrics(y,p):
    return {'roc_auc':float(roc_auc_score(y,p)),
            'pr_auc':float(average_precision_score(y,p)),
            'brier_score':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1]))}

def main():
    out=ROOT/'fraud_reasoning/artifacts'
    def read(name):
        x=pd.read_csv(out/name)
        y=x.pop('FLAG').to_numpy(int)
        return x,y
    train,yt=read('oof_meta_features.csv')
    cal,yc=read('calibration_meta_features.csv')
    conf,yf=read('conformal_meta_features.csv')
    test,yh=read('historical_test_meta_features.csv')
    def inputs(x,cols):return x[cols].to_numpy(float)
    print('Training 4-fold IF and rule fraud-evidence adapters...',flush=True)
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=83)
    train_experts=np.zeros((len(train),3),float)
    train_experts[:,0]=train.rf_raw_probability.to_numpy(float)
    for indices_a,indices_b in cv.split(train,yt):
        for num,cols in [(1,IF_COLUMNS),(2,RULE_COLUMNS)]:
            model=fit_adapter(inputs(train.iloc[indices_a],cols),yt[indices_a])
            train_experts[indices_b,num]=model.predict_proba(inputs(train.iloc[indices_b],cols))[:,1]
    if not np.isfinite(train_experts).all():raise ValueError('Bad OOF adapter predictions')
    if_adapter=fit_adapter(inputs(train,IF_COLUMNS),yt)
    rule_adapter=fit_adapter(inputs(train,RULE_COLUMNS),yt)

    gate_scaler=StandardScaler().fit(inputs(train,GATE_COLUMNS))
    def gate_x(x):
        xx=gate_scaler.transform(inputs(x,GATE_COLUMNS))
        return np.column_stack([np.ones(len(xx)),xx])
    xg=gate_x(train)
    n,dim=xg.shape
    # Minimize proper binary cross-entropy; weights arise through softmax gate.
    def objective(flat):
        w=flat.reshape(dim,3)
        probs=softmax(xg@w,axis=1)
        fused=np.clip(np.sum(probs*train_experts,axis=1),1e-8,1-1e-8)
        loss=-np.mean(yt*np.log(fused)+(1-yt)*np.log1p(-fused)) +.015*np.square(w[1:,:]).sum()
        dp=(fused-yt)/(fused*(1-fused))/n
        dlogits=dp[:,None]*probs*(train_experts-fused[:,None])
        grad=xg.T@dlogits
        grad[1:,:]+=.03*w[1:,:]
        return float(loss),grad.flatten()
    result=minimize(objective,np.zeros(dim*3),method='L-BFGS-B',jac=True,
                    options={'maxiter':450,'ftol':1e-10})
    if not result.success and result.fun>0.5:raise RuntimeError(f'Gate did not converge: {result.message}')
    w=result.x.reshape(dim,3)
    payload={'version':'0.7.0','if_adapter':encode(if_adapter),'rule_adapter':encode(rule_adapter),
             'gate_scaler':{'mean':gate_scaler.mean_.tolist(),'scale':gate_scaler.scale_.tolist()},
             'gate_weights':w.tolist(),'calibration_a':1.,'calibration_b':0.,
             'metadata':{}}
    model=PortableContextualMoE(payload)
    pre_cal=model.predict_proba(cal.to_dict('records'))[:,1]
    sigmoid=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(pre_cal,1e-7,1-1e-7)).reshape(-1,1),yc)
    payload['calibration_a']=float(sigmoid.coef_[0,0]);payload['calibration_b']=float(sigmoid.intercept_[0])
    model=PortableContextualMoE(payload)
    confp=model.predict_proba(conf.to_dict('records'))[:,1]
    nonconf=1.-np.where(yf==1,confp,1-confp)
    n=len(nonconf);q=min(1,np.ceil((n+1)*.90)/n)
    quant=float(np.quantile(nonconf,q,method='higher'))
    testp=model.predict_proba(test.to_dict('records'))[:,1]
    payload['metadata']={
        'train_count':int(len(train)),'calibration_count':int(len(cal)),
        'conformal_count':int(len(conf)),'historical_test_count':int(len(test)),
        'historical_test':metrics(yh,testp),
        'conformal':{'alpha':.1,'nonconformity_quantile':quant},
        'convergence':{'success':bool(result.success),'message':str(result.message),'objective':float(result.fun)},
        'limitations':['Not external independent D1 test','IF and rules are converted to label-trained fraud-evidence adapters, not originally fraud probabilities','Gating weights are predictive weights, not expert truth probabilities']}
    (out/'contextual_moe_v07.json').write_text(json.dumps(payload,indent=2))
    print('MoE historical metrics',payload['metadata']['historical_test'],flush=True)
    print('Saved contextual_moe_v07.json',flush=True)

if __name__=='__main__':main()
