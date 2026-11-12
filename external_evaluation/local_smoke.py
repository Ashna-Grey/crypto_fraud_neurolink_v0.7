"""Run NeuroLink's six shipped API fixtures directly against the frozen v0.7 model.
This is an API/inference contract check, NOT labeled external validation.
Requires original repository dependencies, preferably Python 3.13.
"""
import json
import sys
from pathlib import Path
if __package__:
    from .run import atomic_json,interpret_response
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.run import atomic_json,interpret_response
HERE=Path(__file__).resolve().parent
REPO=HERE.parent
sys.path.insert(0,str(REPO))

def main():
    from fraud_reasoning.service import FraudIntelligence
    s=FraudIntelligence()
    cases=json.loads((REPO/'tests/fixtures/render_cases.json').read_text())
    results=[]
    for case in cases:
        payload=case['payload']
        try:
            res=s.predict(payload['features'],wallet_id=payload.get('wallet_id'),input_format=payload.get('input_format','raw-live'))
            item=interpret_response(res)
            passed=item.get('status')=='classified' and case['case_id']!='missing_required_feature'
        except (ValueError,TypeError) as exc:
            item={'status':'input_rejected','reason':type(exc).__name__}
            passed=case['case_id']=='missing_required_feature'
        results.append({'case_id':case['case_id'],'passed_contract':passed,
                        'status':item.get('status'),'decision':item.get('decision'),
                        'fraud_probability':item.get('probability')})
    report={'type':'local_frozen_v0.7_contract_check_not_external_validation',
            'total':len(results),'passed':sum(r['passed_contract'] for r in results),'cases':results}
    atomic_json(HERE/'results/local_smoke.json',report)
    print(json.dumps(report,indent=2))
    return 0 if report['passed']==len(results) else 1

if __name__=='__main__':raise SystemExit(main())
