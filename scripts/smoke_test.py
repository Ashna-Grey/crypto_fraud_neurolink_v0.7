"""Run the actual saved models and print a short auditable output."""
import json
from pathlib import Path
from fraud_reasoning.service import FraudIntelligence
root=Path(__file__).resolve().parents[1]
request=json.loads((root/'examples/sample_wallet_request.json').read_text())
result=FraudIntelligence().predict(request['features'],wallet_id=request['wallet_id'],input_format=request['input_format'])
print(json.dumps({'RF':result['random_forest']['fraud_probability'],
                  'IF_percentile':result['isolation_forest'].get('anomaly_percentile'),
                  'Rule_status':result['rule_based_engine']['status'],
                  'Rule_score':result['rule_based_engine']['risk_score'],
                  'Decision':result['final_output']['decision'],
                  'Review_reasons':result['final_output']['reasons']},indent=2))
