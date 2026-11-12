import json
import pathlib
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from fraud_reasoning.logic import facts_from_input, context_arbitration
from fraud_reasoning.fusion import fuse
from fraud_reasoning.api import app

ROOT=pathlib.Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope='session')
def example():
    row=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv',nrows=1).iloc[0].to_dict()
    return {k:(None if pd.isna(v) else float(v)) for k,v in row.items() if k!='FLAG'}

def test_context_slow_volume():
    facts=facts_from_input({'total_transactions':1200,'wallet_lifetime_minutes':300*1440})
    assert facts['average_transactions_per_day_over_observed_span']==4
    rf={'status':'OK','fraud_probability':0.88,'reasoning':[{'feature':'total transactions','direction':'toward_fraud'}]}
    rules={'status':'OK','risk_score':0.2,'evaluated':[]}
    trace=context_arbitration(facts,rf,{'status':'OK'},rules)
    assert any(t['rule_id']=='SLOW_VOLUME' for t in trace['arbitration'])
    assert not any(t['rule_id']=='FAST_VOLUME' for t in trace['arbitration'])

def test_context_fast_volume():
    facts=facts_from_input({'total_transactions':1200,'wallet_lifetime_minutes':1440})
    trace=context_arbitration(facts,{'status':'OK','reasoning':[]},{'status':'OK'}, {'status':'UNAVAILABLE','evaluated':[]})
    assert any(t['rule_id']=='FAST_VOLUME' for t in trace['arbitration'])
    assert not any(t['rule_id']=='SLOW_VOLUME' for t in trace['arbitration'])

def test_missing_time_does_not_claim_slow():
    facts=facts_from_input({'total_transactions':1200})
    trace=context_arbitration(facts,{'status':'OK','reasoning':[]},{'status':'OK'}, {'status':'UNAVAILABLE','evaluated':[]})
    assert any(t['rule_id']=='TIME_UNAVAILABLE' for t in trace['arbitration'])

def test_conflicting_aliases_rejected():
    with pytest.raises(ValueError, match='Conflicting aliases'):
        facts_from_input({'total_transactions':100,'total transactions (including tnx to create contract':200})

def test_dependency_detected():
    facts=facts_from_input({'total_transactions':1200,'wallet_lifetime_minutes':300*1440})
    rf={'status':'OK','fraud_probability':0.9,'reasoning':[{'feature':'total transactions (including tnx to create contract','direction':'toward_fraud'}]}
    rules={'status':'OK','risk_score':0.4,'evaluated':[{'rule':'high_activity','triggered':True,'dependencies':['total_transactions','wallet_lifetime_minutes']}]}
    trace=context_arbitration(facts,rf,{'status':'OK'},rules)
    assert trace['dependency_adjustments']

def test_fusion_never_treats_if_as_fraud_prob():
    rf={'status':'OK','fraud_probability':0.88,'input_quality':1,'conformal_decision':'FRAUD','attribution':{'reasoning':[]}}
    rules={'status':'OK','risk_score':0.1,'coverage':1}
    logic={'dependency_adjustments':[],'arbitration':[]}
    a=fuse(rf,{'status':'OK','applicability':{'score':1},'fraud_probability':0.0},rules,logic)
    b=fuse(rf,{'status':'OK','applicability':{'score':1},'fraud_probability':1.0},rules,logic)
    assert a['fused_risk_score']==b['fused_risk_score']
    assert a['decision']=='MANUAL_REVIEW'

def test_fusion_proportions_bounded():
    rf={'status':'OK','fraud_probability':0.99,'input_quality':1,'conformal_decision':'FRAUD','attribution':{'reasoning':[]}}
    result=fuse(rf,{'status':'UNAVAILABLE'}, {'status':'UNAVAILABLE','risk_score':None,'coverage':0}, {'dependency_adjustments':[],'arbitration':[]})
    assert result['decision']=='MANUAL_REVIEW'
    assert 0<=result['uncertainty']<=1
    for item in result['opinion_components']:
        assert abs(item['belief_fraud']+item['belief_legitimate']+item['uncommitted_uncertainty']-1)<1e-5

def test_health_and_schema(client):
    assert client.get('/health').status_code==200
    schema=client.get('/schema').json()
    assert len(schema['rf_features'])==20
    assert len(schema['if_features'])==20
    assert client.get('/models/status').status_code==200
    assert client.get('/reasoning/rules').status_code==200
    assert client.get('/research/methodology').status_code==200

def test_real_three_engine_prediction(client,example):
    response=client.post('/predict',json={'features':example,'wallet_id':'test1'})
    assert response.status_code==200, response.text
    data=response.json()
    assert 0<=data['random_forest']['fraud_probability']<=1
    assert data['random_forest']['attribution']['reconstruction_error']<1e-5
    assert data['isolation_forest']['status']=='OK'
    assert data['isolation_forest']['fraud_probability'] is None
    assert data['rule_based_engine']['status'] in ('OK','PARTIAL')
    assert data['final_output']['decision'] in ('FRAUD','LEGITIMATE','MANUAL_REVIEW')
    assert data['belief_fusion']['calibrated'] is True

def test_preprocessed_requires_separate_if_features(client,example):
    response=client.post('/predict',json={'features':example,'input_format':'preprocessed'})
    assert response.status_code==422

def test_missing_rf_features(client):
    response=client.post('/predict',json={'features':{'Total ERC20 tnxs':1}})
    assert response.status_code==422

def test_batch_predictions(client,example):
    response=client.post('/predict/batch',json={'wallets':[{'features':example},{'features':{'junk':1}}]})
    assert response.status_code==200,response.text
    content=response.json()
    assert content['succeeded']==1
    assert content['results'][1]['status']=='ERROR'

def test_invalid_nan_rejected(client, example):
    example=dict(example);example['Sent tnx']='NaN'
    assert client.post('/predict',json={'features':example}).status_code==422

def test_context_discount_reduces_rf_credibility():
    rf={'status':'OK','fraud_probability':0.80,'input_quality':1,'conformal_decision':'FRAUD',
        'attribution':{'reasoning':[{'feature':'total transactions (including tnx to create contract',
                                      'raw_probability_contribution':0.20,'direction':'toward_fraud'}]}}
    rules={'status':'OK','risk_score':0.80,'coverage':1}
    based={'dependency_adjustments':[],'arbitration':[]}
    slow={'dependency_adjustments':[],'arbitration':[{'rule_id':'SLOW_VOLUME'}]}
    normal=fuse(rf,{'status':'OK','applicability':{'score':1}},rules,based)
    adjusted=fuse(rf,{'status':'OK','applicability':{'score':1}},rules,slow)
    assert adjusted['adjustments']['rf_contextual_discount']==0.5
    assert adjusted['opinion_components'][0]['credibility'] < normal['opinion_components'][0]['credibility']

def test_rule_adapter_reports_missing_inputs(client,example):
    data=client.post('/predict',json={'features':example}).json()
    assert data['rule_based_engine']['unavailable_rules']
    assert any(x['rule']=='contract_activity' for x in data['rule_based_engine']['unavailable_rules'])
    assert data['final_output']['review_recommended'] is True

def test_api_rejects_ambiguous_aliases(client,example):
    row=dict(example)
    row['total_transactions']=12345
    assert client.post('/predict',json={'features':row}).status_code==422
