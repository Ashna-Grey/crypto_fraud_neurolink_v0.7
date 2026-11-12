#!/usr/bin/env python3
"""Run 6 included API cases locally, saving real outputs and statuses."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from fraud_reasoning.api import app

cases=json.loads((ROOT/'tests/fixtures/render_cases.json').read_text())
results=[]
with TestClient(app) as client:
    for case in cases:
        r=client.post('/predict',json=case['payload'])
        body=r.json()
        record={'case_id':case['case_id'],'status_code':r.status_code,'response':body}
        results.append(record)
        if r.status_code==200:
            print(case['case_id'],r.status_code,body['final_output']['decision'],
                  round(body['final_output']['fraud_probability'],6),
                  'review_recommended=',body['final_output']['review_recommended'])
        else:print(case['case_id'],r.status_code,'validation/error')
output=ROOT/'examples/local_v07_smoke_results.json'
output.write_text(json.dumps({'results':results},indent=2))
print('Saved',output)
