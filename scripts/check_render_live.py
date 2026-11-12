#!/usr/bin/env python3
"""Run bundled six API regression cases against your already-deployed Render URL.
Does not fabricate response metrics. Run after updating the service.
"""
import sys,json
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2:
    print('Usage: python scripts/check_render_live.py https://YOUR-SERVICE.onrender.com')
    raise SystemExit(2)
base=sys.argv[1].rstrip('/')
cases=json.loads((ROOT/'tests/fixtures/render_cases.json').read_text())
results={'base_url':base,'endpoints':{},'cases':[]}
with httpx.Client(timeout=90,follow_redirects=True) as client:
    for path in ['/health','/models/status','/learned/metrics']:
        try:
            r=client.get(base+path)
            results['endpoints'][path]={'http_status':r.status_code,'response':r.json()}
            print(path,r.status_code)
        except Exception as exc:
            results['endpoints'][path]={'error':type(exc).__name__+': '+str(exc)}
            print(path,'ERROR:',exc)
    for c in cases:
        try:
            r=client.post(base+'/predict',json=c['payload'])
            record={'case_id':c['case_id'],'http_status':r.status_code,'response':r.json()}
            out=record['response'].get('final_output',{})
            print(c['case_id'],r.status_code,out.get('decision'),out.get('fraud_probability'))
        except Exception as exc:
            record={'case_id':c['case_id'],'error':type(exc).__name__+': '+str(exc)}
            print(c['case_id'],'ERROR:',exc)
        results['cases'].append(record)
output=ROOT/'examples/live_v07_render_results.json'
output.write_text(json.dumps(results,indent=2))
print('Saved',output)
