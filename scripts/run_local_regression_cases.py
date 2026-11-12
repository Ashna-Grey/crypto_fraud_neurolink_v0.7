"""Run the six fixed regression inputs against the *locally loaded* saved model artifacts.

This is NOT a substitute for running against Render after redeployment.
"""
from pathlib import Path
import json
import sys
from fraud_reasoning.service import FraudIntelligence

ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'tests/fixtures/render_cases.json').read_text())
service=FraudIntelligence()
rows=[]
for case in CASES:
    try:
        result=service.predict(**case['payload'])
        row={'case_id':case['case_id'],'status':'OK',
             'rf_fraud_probability':result['random_forest']['fraud_probability'],
             'if_anomaly_percentile':result['isolation_forest'].get('anomaly_percentile'),
             'if_applicability':result['isolation_forest']['applicability']['score'],
             'rule_score':result['rule_based_engine'].get('risk_score'),
             'rule_coverage':result['rule_based_engine']['coverage'],
             'context_trace':[x['rule_id'] for x in result['reasoning_engine']['arbitration']],
             'risk_priority':result['belief_fusion']['fused_risk_score'],
             'review_index':result['belief_fusion']['uncertainty'],
             'decision':result['final_output']['decision'],
             'reasons':result['final_output']['reasons'],
             'advisories':result['final_output'].get('advisories', [])}
    except (TypeError,ValueError) as e:
        row={'case_id':case['case_id'],'status':'EXPECTED_INPUT_ERROR',
             'error_type':type(e).__name__, 'error':str(e)}
    rows.append(row)
output=ROOT/'examples/local_v06_regression_summary.json'
output.write_text(json.dumps(rows,indent=2))
for r in rows:
    print(f"{r['case_id']}: {r.get('decision',r['status'])} | "+
          f"RF={r.get('rf_fraud_probability','N/A')} | reason={r.get('reasons',r.get('error',''))}")
print(f'Saved {output}')
