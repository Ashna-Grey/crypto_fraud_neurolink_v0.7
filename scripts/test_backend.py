#!/usr/bin/env python3
"""Run representative non-mutating requests against local or Render API.

Uses urllib from standard library, no external-client installation.
The fixture's recorded labels are NEVER sent to the prediction endpoint.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'tests/fixtures/render_cases.json'

def request(base, path, payload=None):
    body=None if payload is None else json.dumps(payload).encode('utf-8')
    req=urllib.request.Request(base.rstrip('/')+path,body,
        headers={'Content-Type':'application/json'} if body else {},
        method='POST' if body else 'GET')
    try:
        with urllib.request.urlopen(req,timeout=90) as resp:
            return resp.status,json.load(resp)
    except urllib.error.HTTPError as exc:
        try:data=json.loads(exc.read().decode('utf-8'))
        except Exception:data={'error':str(exc)}
        return exc.code,data

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',default='local_api_test_results.json')
    args=parser.parse_args()
    results={'base_url':args.url,'health':{},'cases':[]}
    for path in ['/health','/models/status','/schema','/learned/metrics']:
        status,response=request(args.url,path)
        results['health'][path]={'http_status':status,'response':response}
        print(f'{path}: {status}')
    for case in json.loads(CASES.read_text()):
        status,response=request(args.url,'/predict',case['payload'])
        final=response.get('final_output',{}) if isinstance(response,dict) else {}
        row={'case_id':case['case_id'],'http_status':status,
             'decision':final.get('decision'),'fraud_probability':final.get('fraud_probability'),
             'review_recommended':final.get('review_recommended'),
             'response':response}
        results['cases'].append(row)
        print(f"{case['case_id']}: HTTP {status}, {final.get('decision')}, p={final.get('fraud_probability')}")
    Path(args.output).write_text(json.dumps(results,indent=2))
    print('Saved',args.output)
    return int(not all(results['health'][p]['http_status']==200 for p in results['health']))

if __name__=='__main__':sys.exit(main())
