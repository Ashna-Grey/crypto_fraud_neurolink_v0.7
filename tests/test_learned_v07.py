"""Actual pretrained portable models, three-engine API, and learned arbitration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fraud_reasoning.api import app,service
from fraud_reasoning.moe import PortableContextualMoE
from fraud_reasoning.learned import PortableEvidenceGAM, rf_group_contributions

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'fraud_reasoning/artifacts'

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:yield c

@pytest.fixture(scope='module')
def wallet():
    x=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv',nrows=1).iloc[0].to_dict()
    return {k:(None if pd.isna(v) else float(v)) for k,v in x.items() if k!='FLAG'}

def test_portable_gam_calibration_matches_sklearn_training_artifact():
    import joblib
    old=joblib.load(ART/'evidence_gam_v07.joblib')['gam']
    new=PortableEvidenceGAM.from_file(ART/'evidence_gam_v07.json')
    frame=pd.read_csv(ART/'historical_test_meta_features.csv').drop(columns=['FLAG']).head(25)
    pp=old.predict_proba(frame)[:,1]
    assert np.allclose(new.predict_proba(frame.to_dict('records'))[:,1],pp,atol=1e-10)
    explanation=new.explain(frame.iloc[0].to_dict())
    assert explanation['reconstruction_error']<1e-12
    assert abs(explanation['calibrated_log_odds']-(explanation['intercept_log_odds']+
              sum(explanation['all_term_contributions'].values())))<1e-10

def test_portable_moe_probabilities_and_gate_weights():
    meta=PortableContextualMoE.from_file(ART/'contextual_moe_v07.json')
    data=pd.read_csv(ART/'historical_test_meta_features.csv').drop(columns=['FLAG']).head(5).to_dict('records')
    probs=meta.predict_proba(data)
    assert probs.shape==(5,2)
    assert np.all((probs>=0)&(probs<=1))
    assert np.allclose(probs.sum(axis=1),1)
    for row in data:
        e=meta.explain(row)
        assert sum(w['learned_contextual_weight'] for w in e['weights'])==pytest.approx(1.,abs=1e-10)
        assert sum(w['raw_weighted_probability_contribution'] for w in e['weights'])==pytest.approx(e['uncalibrated_probability'],abs=1e-10)
        assert 0<=e['fraud_probability']<=1
        assert len(e['gate_reasoning'])==3


def test_gate_changes_with_context_without_hand_tuned_overrides():
    model=PortableContextualMoE.from_file(ART/'contextual_moe_v07.json')
    frame=pd.read_csv(ART/'historical_test_meta_features.csv').drop(columns=['FLAG']).iloc[0].to_dict()
    a=model.explain(frame)
    slow=dict(frame);slow['log_activity_rate']=0.0
    fast=dict(frame);fast['log_activity_rate']=7.0
    b=model.explain(slow);c=model.explain(fast)
    assert not np.allclose([z['unnormalized_gate_logit'] for z in b['gate_reasoning']],
                           [z['unnormalized_gate_logit'] for z in c['gate_reasoning']])
    # Not asserting fraud direction: empirical routing need not be monotone.


def test_rf_group_attribution_reconstructs_raw_probability():
    s=service()
    raw=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv',nrows=4)[s.rf_features]
    by_group=rf_group_contributions(s.rf_model,s.rf_imputer,raw,s.rf_features)
    vec=s.rf_imputer.transform(raw)
    predicted=s.rf_model.predict_proba(vec)[:,list(s.rf_model.classes_).index(1)]
    assert len(by_group)==4
    # Path-group differences alone exclude root baseline; their first differences
    # across records must match RF output differences exactly.
    contributions=np.sum(list(by_group.values()),axis=0)
    assert np.allclose(predicted-predicted[0],contributions-contributions[0],atol=1e-8)


def test_model_artifacts_and_historical_metrics_not_fake(client):
    st=client.get('/models/status').json()
    assert st['version']=='0.7.0'
    assert st['learned_arbitration']['status']=='loaded'
    assert client.get('/learned/metrics').status_code==200
    assert client.get('/learned/evidence-schema').status_code==200
    report=json.loads((ART/'historical_comparison_report.json').read_text())
    assert report['historical_dataset_is_independent_external_test'] is False
    assert report['historical_D1_metrics']['MoE']['n']==924
    assert report['historical_D1_metrics']['RF_calibrated']['roc_auc']>=0


def test_single_and_batch_prediction_expose_three_models_and_learned_explanation(client,wallet):
    response=client.post('/predict',json={'features':wallet,'wallet_id':'v07_test'})
    assert response.status_code==200,response.text
    data=response.json()
    assert data['isolation_forest']['fraud_probability'] is None
    assert data['random_forest']['status']=='OK'
    assert data['rule_based_engine']['status'] in ('PARTIAL','OK')
    assert 0 <= data['learned_arbitration']['fraud_probability'] <= 1
    assert data['final_output']['decision'] in ('FRAUD','LEGITIMATE')
    assert data['final_output']['fraud_probability']==pytest.approx(data['learned_arbitration']['fraud_probability'])
    assert data['learned_arbitration']['active_algorithm'] in ('moe','gam')
    assert data['belief_fusion']['calibrated'] is True
    assert client.post('/predict/batch',json={'wallets':[{'features':wallet},{'features':{'fake':1}}]}).json()['succeeded']==1


def test_missing_core_inputs_rejected(client):
    assert client.post('/predict',json={'features':{'Total ERC20 tnxs':1}}).status_code==422

def test_deployed_inference_matches_saved_holdout_meta_features():
    """Catch a serious bug: model evidence transforms must agree across training and API."""
    from fraud_reasoning.service import FraudIntelligence
    metadata=pd.read_csv(ART/'historical_holdout_predictions.csv')
    reference=pd.read_csv(ART/'historical_test_meta_features.csv')
    original=pd.read_csv(ROOT/'01_Dataset/processed/ethereum_fraud_25_features.csv')
    model=PortableContextualMoE.from_file(ART/'contextual_moe_v07.json')
    service_instance=FraudIntelligence()
    for i in (0, 1, 12):
        row=original.iloc[int(metadata.iloc[i]['source_row_index'])].drop('FLAG').to_dict()
        live=service_instance.predict(row,wallet_id='parity_check')
        evidence=reference.iloc[i].drop('FLAG').to_dict()
        expected=model.predict_proba(evidence)[:,1][0]
        assert live['final_output']['fraud_probability']==pytest.approx(expected,abs=1e-9)
        assert live['learned_arbitration']['active_algorithm']=='moe'
        assert live['isolation_forest']['fraud_probability'] is None
