"""Render-ready FastAPI service, with strict schemas and no fabricated outputs."""
from functools import lru_cache
from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .service import FraudIntelligence
from .rules import SPEC
from .logic import POLICY

app=FastAPI(title='Neuro-Symbolic Ethereum Fraud Intelligence',version='0.7.0',
            description='Research prototype: three evidence engines plus held-out-calibrated learned MoE / spline GAM arbitration')

class WalletInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    wallet_id:str|None=None
    features:dict[str,float|int|str|None]=Field(...,min_length=1)
    input_format:Literal['raw-live','preprocessed']='raw-live'
    if_features:dict[str,float|int|str|None]|None=None

class BatchInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    wallets:list[WalletInput]=Field(...,min_length=1,max_length=100)

@lru_cache(maxsize=1)
def service():return FraudIntelligence()

@app.on_event('startup')
def startup():
    service()  # no false healthy state for unloaded model artifacts

@app.get('/')
def root():return {'service':'Neuro-Symbolic Ethereum Fraud Intelligence','version':'0.7.0','docs':'/docs'}

@app.get('/health')
def health():
    try:return {'status':'healthy','models':service().model_status()}
    except Exception as exc:raise HTTPException(status_code=503,detail={'status':'unhealthy','reason':type(exc).__name__}) from exc

@app.get('/models/status')
def model_status():return service().model_status()

@app.get('/schema')
def schema():
    s=service()
    return {'rf_features':s.rf_features,'if_features':s.if_features,
            'rule_dependencies':{name:deps for name,deps,_ in SPEC},
            'rules_missing_frozen_thresholds':[key for _,_,key in SPEC if key and key not in s.rule_adapter.thresholds],
            'if_feature_space_warning':'Default raw-live applies IF-specific transformations; separate IF preprocessed features are required in preprocessed mode.'}

@app.get('/reasoning/rules')
def reasoning_rules():return {'rule_dependencies':{name:deps for name,deps,_ in SPEC},
                              'context_policy':POLICY,'policy_validation':'provisional_not_empirically_calibrated'}

@app.get('/research/methodology')
def methodology():
 s=service()
 return {
 'engines':['original_RF_with_calibration','original_bootstrap_IF','original_eight_rule_modules'],
 'fusion':'Default contextual MoE with learned gating, supervised IF/rule evidence adapters, and held-out sigmoid calibration. GAM alternative and legacy policy available.',
 'if_fraud_probability':False, 'fused_risk_is_calibrated_on_historical_D1':s.fusion_mode!='legacy',
 'reasoning':'Traceable contextual facts; learned MoE gate or GAM terms determines final binary prediction in nonlegacy modes',
 'limitations':['Rule thresholds may be missing','Not independently evaluated','IF training distribution mismatch','No LLM','Not production-validated']}

@app.post('/predict')
def predict(payload:WalletInput):
    try:return service().predict(payload.features,wallet_id=payload.wallet_id,
                                  input_format=payload.input_format,if_features=payload.if_features)
    except (ValueError,TypeError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.post('/predict/batch')
def predict_batch(payload:BatchInput):
    rows=[]
    for i,wallet in enumerate(payload.wallets):
        try:rows.append({'index':i,'status':'OK','result':service().predict(wallet.features,wallet_id=wallet.wallet_id,
                     input_format=wallet.input_format,if_features=wallet.if_features)})
        except (ValueError,TypeError) as exc:rows.append({'index':i,'status':'ERROR','error':str(exc)})
    return {'results':rows,'total':len(rows),'succeeded':sum(r['status']=='OK' for r in rows)}

@app.get('/learned/metrics')
def learned_metrics():
    s=service()
    if s.learned_bundle is None:
        raise HTTPException(status_code=503,detail='Calibrated learned model not loaded')
    return s.learned_bundle['metadata']

@app.get('/learned/evidence-schema')
def learned_evidence_schema():
    from .learned import FEATURE_NAMES, INTERACTIONS
    return {'features': FEATURE_NAMES, 'pairwise_learned_interactions': INTERACTIONS,
            'explanation':'GAM main terms and interactions, not model causal reasoning'}
