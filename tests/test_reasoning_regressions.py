"""Regression tests for changes after first live Render report."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from fraud_reasoning.api import app
from fraud_reasoning.fusion import fuse
from fraud_reasoning.logic import context_arbitration, facts_from_input


def example_rf(p=0.95, conformal='FRAUD', transactions_contribution=0.3):
    return {'status': 'OK', 'fraud_probability': p,
            'input_quality': 1, 'conformal_decision': conformal,
            'tree_disagreement_variance': .01,
            'attribution': {'all_feature_contributions': {
                'total transactions (including tnx to create contract': transactions_contribution,
                'avg val received': .1}, 'reasoning': []}}


def if_reference(app=1., percentile=80.):
    return {'status': 'OK', 'applicability': {'score': app}, 'anomaly_percentile': percentile,
            'fraud_probability': None}


def test_missing_optional_rules_do_not_force_abstention_when_rf_strong():
    rules={'status':'PARTIAL','coverage':.75,'risk_score':0.,'triggered_rules':[]}
    reasoning={'arbitration':[], 'dependency_adjustments':[]}
    high=fuse(example_rf(),if_reference(),rules,reasoning)
    low=fuse(example_rf(p=.01,conformal='NON_FRAUD'),if_reference(),rules,reasoning)
    assert high['decision']=='FRAUD'
    assert low['decision']=='LEGITIMATE'
    assert 'RULE_EVIDENCE_INCOMPLETE' in high['advisories']
    assert 'RULE_EVIDENCE_INCOMPLETE' not in high['reasons']


def test_zero_triggered_rules_not_negative_fraud_vote():
    rf=example_rf(p=.94)
    trace={'arbitration':[], 'dependency_adjustments':[]}
    partial=fuse(rf,if_reference(),{'coverage':.75,'risk_score':0,'triggered_rules':[]},trace)
    unavailable=fuse(rf,if_reference(),{'coverage':0,'risk_score':None,'triggered_rules':[]},trace)
    assert partial['fused_risk_score']==pytest.approx(.94)
    assert unavailable['fused_risk_score']==pytest.approx(.94)
    assert partial['pairwise_conflict_proxy'] is None
    assert unavailable['decision']=='MANUAL_REVIEW'


def test_uncalibrated_rule_gap_is_not_diagnostic_conflict():
    rf=example_rf(.99)
    rules={'status':'PARTIAL','risk_score':0.0,'evaluated':[], 'triggered_rules':[], 'coverage':.75}
    trace=context_arbitration(facts_from_input({'total_transactions':12}),rf,if_reference(),rules)
    assert 'ENGINE_DISAGREEMENT' not in [r['rule_id'] for r in trace['arbitration']]


def test_context_discount_uses_all_rf_contributions():
    rf=example_rf(p=.95,transactions_contribution=.30)
    rules={'status':'OK','risk_score':.0,'coverage':1.,'triggered_rules':[]}
    base={'arbitration':[], 'dependency_adjustments':[]}
    slow={'arbitration':[{'rule_id':'SLOW_VOLUME'}], 'dependency_adjustments':[]}
    a=fuse(rf,if_reference(),rules,base)
    b=fuse(rf,if_reference(),rules,slow)
    assert b['adjustments']['rf_contextual_discount'] < 1
    assert b['fused_risk_score'] < a['fused_risk_score']
    assert b['adjustments']['rf_volume_attribution_fraction_of_positive_top_features'] == pytest.approx(.75)
    assert b['decision']=='MANUAL_REVIEW'
    assert 'MATERIAL_CONTEXTUAL_RF_CONFLICT' in b['reasons']


def test_positive_rule_tension_preserved_as_review():
    rf=example_rf(.01,'NON_FRAUD')
    rules={'status':'OK','risk_score':.4,'coverage':1.,'triggered_rules':['high_activity'],
           'evaluated':[{'rule':'high_activity','triggered':True,'dependencies':['total_transactions','wallet_lifetime_minutes']}]}
    logic=context_arbitration(facts_from_input({'total_transactions':500,'wallet_lifetime_minutes':30000}), rf, if_reference(), rules)
    result=fuse(rf, if_reference(), rules, logic)
    assert result['decision']=='MANUAL_REVIEW'
    assert 'QUALITATIVE_EVIDENCE_TENSION' in result['reasons']


def test_ood_and_high_if_anomaly_require_review():
    rf=example_rf()
    rules={'status':'OK','risk_score':0.,'coverage':1.,'triggered_rules':[]}
    trace={'arbitration':[], 'dependency_adjustments':[]}
    assert fuse(rf,if_reference(.2),rules,trace)['decision']=='MANUAL_REVIEW'
    assert fuse(rf,if_reference(1,99.5),rules,trace)['decision']=='MANUAL_REVIEW'


def test_actual_render_regression_cases():
    cases_path=Path(__file__).resolve().parent / 'fixtures' / 'render_cases.json'
    if not cases_path.exists():
        pytest.skip('Optional external Render test-case fixture unavailable')
    with TestClient(app) as client:
        cases=json.loads(cases_path.read_text())
        expected={'d1_label_0':'LEGITIMATE','d1_label_1':'FRAUD'}
        for case in cases:
            resp=client.post('/predict',json=case['payload'])
            if case['case_id'] == 'missing_required_feature':
                assert resp.status_code == 422
            else:
                assert resp.status_code == 200,case['case_id']
                assert resp.json()['final_output']['decision'] in ('FRAUD','LEGITIMATE')
                if case['case_id'] in expected:
                    assert resp.json()['final_output']['decision'] == expected[case['case_id']]
