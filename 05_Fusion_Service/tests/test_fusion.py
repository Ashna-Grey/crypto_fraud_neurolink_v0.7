import importlib
combine=importlib.import_module("05_Fusion_Service.fusion").combine


def test_missing_rf_abstains():
    assert combine(None)['decision']=='ABSTAIN'

def test_if_never_becomes_probability():
    a=combine(0.8,0.7,if_anomaly=0.1)
    b=combine(0.8,0.7,if_anomaly=0.9)
    assert a['fusion_support']==b['fusion_support']

def test_shift_abstains():
    assert combine(0.95,0.9,applicability=0.1)['decision']=='ABSTAIN'

def test_missing_rules_abstains():
    assert combine(0.9,None)['decision']=='ABSTAIN'
