from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from .service import FraudService

app=FastAPI(title='Crypto Fraud Intelligence',version='0.2.0',description='Research prototype, not a production fraud decision system')

class WalletInput(BaseModel):
    features: dict[str,Any]=Field(...,description='Raw feature dictionary matching the saved model schemas')
    input_format: str='preprocessed'

@lru_cache(maxsize=1)
def service():return FraudService()

@app.get('/health')
def health():return {'status':'ok','service':'crypto-fraud-intelligence','models':'loaded_on_first_prediction'}

@app.get('/schema')
def schema():
    s=service()
    return {'rf_features':s.rf_features,'if_features':s.if_features,'rule_features':s.rule_required,
            'rule_thresholds_ready':all(k in s.thresholds for k in ('high_value_threshold','dust_threshold','contract_threshold'))}

@app.post('/predict')
def predict(payload:WalletInput):
    try:return service().predict(payload.features,payload.input_format)
    except (ValueError,TypeError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
