"""Append-only Poison-Hunter external evaluation: retriable, resumable, no retraining."""
import argparse
import collections
import csv
import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

if __package__:
    from .network import rpc, request_json, ApiError
    from .feature_builder import RF,IF,make_payload,build_features,transfer_key
    from .metrics import summarize,finite_probability
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from external_evaluation.network import rpc,request_json,ApiError
    from external_evaluation.feature_builder import RF,IF,make_payload,build_features,transfer_key
    from external_evaluation.metrics import summarize,finite_probability

HERE=Path(__file__).resolve().parent
DEFAULT_SERVER='https://crypto-fraud-neurolink.onrender.com'
ADDR=re.compile(r'^0x[a-fA-F0-9]{40}$')

def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def read_manifest(path):
    with open(path,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    ids=set()
    for row in rows:
        address=row.get('address','').lower();label=row.get('label')
        if not ADDR.fullmatch(address) or label not in ('0','1') or address in ids:
            raise ValueError('Manifest includes an invalid, duplicated or unlabeled wallet.')
        ids.add(address);row['address']=address
    return rows

def csv_write(path, rows):
    # Stable schema, so one failed wallet cannot break the CSV writer.
    fields=['address','label','status','decision','probability','complete','incoming_complete',
            'outgoing_complete','rf_observed','if_observed','unique_transfer_events',
            'unique_tx_hashes','erc20_events','native_events','review_recommended',
            'advisories','missing_rf','missing_if','decision_mismatch','error']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:
            rendered={k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}
            w.writerow(rendered)

def fetch_direction(url, wallet, direction, snapshot, max_pages, size, cachedir):
    file=cachedir/f'{wallet}_{direction}.json'
    state={'address':wallet,'direction':direction,'snapshot':snapshot,'page_key':None,
           'pages':0,'complete':False,'transfers':[]}
    if file.exists():
        state=json.loads(file.read_text(encoding='utf-8'))
        if state['address']!=wallet or state['snapshot']!=snapshot or state['direction']!=direction:
            raise ValueError('Cache snapshot/address mismatch: remove this wallet cache before retrying')
    if state['complete']:return state
    # A page cap is per wallet *total*, not per invocation; remains bounded on resume.
    while state['pages'] < max_pages and not state['complete']:
        options={'fromBlock':'0x0','toBlock':snapshot,'category':['external','internal','erc20'],
                 'withMetadata':True,'excludeZeroValue':False,'maxCount':hex(size),'order':'asc'}
        options['toAddress' if direction=='incoming' else 'fromAddress']=wallet
        if state['page_key']:options['pageKey']=state['page_key']
        resp=rpc(url,'alchemy_getAssetTransfers',[options])
        if not isinstance(resp,dict) or not isinstance(resp.get('transfers'),list):
            raise ApiError('Alchemy transfer response malformed')
        state['transfers'].extend(resp['transfers'])
        state['page_key']=resp.get('pageKey')
        state['pages']+=1
        state['complete']=not bool(state['page_key'])
        atomic_json(file,state)
    return state

def wallet_collection(url, address, snapshot, outdir, max_pages, page_size):
    cachedir=outdir/'cache';cachedir.mkdir(parents=True,exist_ok=True)
    inc=fetch_direction(url,address,'incoming',snapshot,max_pages,page_size,cachedir)
    outgoing=fetch_direction(url,address,'outgoing',snapshot,max_pages,page_size,cachedir)
    meta_file=cachedir/f'{address}_balance.json'
    if meta_file.exists():
        bal=json.loads(meta_file.read_text())
        if bal['snapshot']!=snapshot:raise ValueError('Balance snapshot mismatch')
        balance=bal['balance']
    else:
        balance=int(rpc(url,'eth_getBalance',[address,snapshot]),16)/1e18
        atomic_json(meta_file,{'snapshot':snapshot,'address':address,'balance':balance})
    return inc,outgoing,balance

def verify_server(base):
    server={}
    for path in ['/health','/schema','/models/status']:
        try:server[path]=request_json(base.rstrip('/')+path)
        except Exception as exc:server[path]={'error':str(exc)}
    schema=server.get('/schema') or {}
    if schema.get('rf_features') != RF or schema.get('if_features') != IF:
        raise ValueError('DEPLOYMENT SCHEMA MISMATCH: deployed /schema is not the RF/IF schema in the uploaded v0.7 ZIP. Refusing to score misaligned features.')
    if (server.get('/health') or {}).get('status')!='healthy':
        raise ValueError('Deployed /health is not healthy; refuse evaluation until deployment is fixed.')
    active_mode=((server.get('/models/status') or {}).get('learned_arbitration') or {}).get('mode')
    if active_mode=='legacy':
        raise ValueError('Render FUSION_MODE=legacy causes MANUAL_REVIEW and no fraud_probability. '
                         'In Render Dashboard > Environment set FUSION_MODE=moe, then redeploy.')
    if active_mode not in ('moe','gam'):
        raise ValueError('Cannot verify active learned fusion mode. Check /models/status and Render FUSION_MODE=moe.')
    # Smoke-check the response contract using a *shipped model fixture*, not Poison-Hunter labels.
    fixture=Path(__file__).resolve().parents[1]/'tests'/'fixtures'/'render_cases.json'
    if fixture.is_file():
        case=json.loads(fixture.read_text(encoding='utf-8'))[0]
        smoke=request_json(base.rstrip('/')+'/predict',case['payload'])
        parsed=interpret_response(smoke)
        server['/predict-smoke']={'case_id':case['case_id'],**parsed}
        if parsed.get('status')!='classified':
            raise ValueError('DEPLOYED /predict CONTRACT MISMATCH: response from known model fixture is '
                             +str(parsed.get('status'))+'. Uploaded v0.7 MoE returns FRAUD/LEGITIMATE + numerical probability. '
                             'Redeploy the exact uploaded ZIP, or use --engine local.')
    return server

def interpret_response(response):
    if not isinstance(response,dict):return {'status':'invalid_response','error':'Non-object API response'}
    final=response.get('final_output')
    if not isinstance(final,dict):return {'status':'invalid_response','error':'Missing final_output in /predict response'}
    decision=str(final.get('decision') or '').strip().upper()
    prob=finite_probability(final.get('fraud_probability'))
    result={'decision':decision,'probability':prob,'review_recommended':final.get('review_recommended'),
            'advisories':final.get('advisories') or []}
    if decision=='MANUAL_REVIEW':
        result['status']='abstained_manual_review';return result
    if decision not in ('FRAUD','LEGITIMATE'):
        result.update(status='abstained_unknown_decision',error='Unknown final_output.decision')
        return result
    if prob is None:
        result.update(status='abstained_missing_probability',error='Binary decision returned without valid numerical fraud_probability')
        return result
    result['decision_mismatch']=(prob>=.5)!=(decision=='FRAUD')
    if result['decision_mismatch']:
        result['status']='invalid_inconsistent_decision'
        result['error']='Decision and numerical probability disagree at threshold 0.5'
    else:
        result['status']='classified'
    return result

def emit_html(report, rows, output):
    metrics=report['provisional_metrics_complete_collections']
    def val(key):
        if not metrics:return 'N/A'
        item=metrics.get(key)
        return ('N/A' if item is None else (f'{100*item:.2f}%' if key in ('accuracy','precision','recall','f1') else f'{item:.5f}' if isinstance(item,float) else str(item)))
    cards=''.join(f'<section class="stat"><small>{html.escape(label)}</small><strong>{html.escape(val(key))}</strong></section>'
                  for key,label in [('accuracy','Accuracy (provisional)'),('precision','Precision'),('recall','Recall'),('f1','F1'),('roc_auc','ROC-AUC'),('brier','Brier score')])
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ('address','label','status','decision','probability','complete'))+'</tr>' for r in rows)
    warning='Publication-grade external accuracy is NOT established. Provisional metrics include only completed histories with binary scores; feature parity and D1 overlap are unresolved.'
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroLink External Evaluation</title>
<style>body{{background:#0d1010;color:#e6f4e9;font:16px/1.55 system-ui,Arial;margin:0;padding:30px;max-width:1250px}}h1{{color:#8cffae}}.sub{{color:#9dafaa}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:28px 0}}.stat{{padding:20px;background:#15201b;border:1px solid #284332;border-radius:10px}}small{{display:block;color:#a5c7b2}}strong{{font-size:26px;color:#e9ffeb}}.warn{{border-left:4px solid #e9a65b;padding:16px;background:#242017}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #304139;word-break:break-all}}th{{color:#8cffae}}.scroll{{overflow-x:auto}}</style></head><body><h1>NeuroLink · Poison-Hunter evaluation</h1><p class="sub">Snapshot {html.escape(str(report.get('snapshot_block')))} · Processed {report['processed']} / {report['sample_requested']} · Binary scores {report['binary_with_probabilities']} · Completed collections {report['collection_complete']}</p><div class="stats">{cards}</div><p class="warn">{html.escape(warning)}</p><h2>Evaluation outcomes</h2><div class="scroll"><table><thead><tr><th>Address</th><th>True label (1=phishing)</th><th>Status</th><th>Decision</th><th>Fraud probability</th><th>History complete</th></tr></thead><tbody>{table}</tbody></table></div><p class="sub">Scores from a frozen model, not proof of actual criminal activity. No class labels are sent to inference.</p></body></html>'''
    output.write_text(content,encoding='utf-8')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-wallets',type=int,default=5,help='Number of manifest wallets, maximum 200. Default 5.')
    ap.add_argument('--max-pages',type=int,default=100)
    ap.add_argument('--page-size',type=int,default=1000)
    ap.add_argument('--base-url',default=DEFAULT_SERVER)
    ap.add_argument('--manifest',type=Path,default=HERE/'manifest.csv')
    ap.add_argument('--outdir',type=Path,default=HERE/'results')
    ap.add_argument('--probe-only',action='store_true')
    ap.add_argument('--engine',choices=['remote','local'],default='remote',help='local uses the frozen model directly and requires repository Python dependencies')
    ap.add_argument('--allow-partial',action='store_true',help='Send partial histories, marked diagnostic only')
    args=ap.parse_args(argv)
    if not 1<=args.max_wallets<=200:ap.error('--max-wallets must be 1 to 200')
    if not 1<=args.max_pages<=10000:ap.error('--max-pages must be 1 to 10000')
    if not 1<=args.page_size<=1000:ap.error('--page-size must be 1 to 1000')
    args.outdir.mkdir(parents=True,exist_ok=True)
    try:
        local_model=None
        if args.engine=='local':
            project_root=HERE.parent
            if str(project_root) not in sys.path:sys.path.insert(0,str(project_root))
            from fraud_reasoning.service import FraudIntelligence
            local_model=FraudIntelligence()
            if list(local_model.rf_features)!=RF or list(local_model.if_features)!=IF:
                raise ValueError('Local model features disagree with evaluation schema')
            server={'/models/status':local_model.model_status(),'mode':'local_frozen_artifact'}
            print('Local frozen-model artifacts loaded.',flush=True)
        else:
            server=verify_server(args.base_url)
            print('Deployed NeuroLink /schema matched, /health healthy.',flush=True)
        atomic_json(args.outdir/'server_check.json',server)
    except Exception as exc:
        print('ENGINE ERROR:',type(exc).__name__,str(exc),file=sys.stderr)
        return 2
    if args.probe_only:
        print('Probe successful. No Alchemy key or transaction collection needed.')
        return 0
    url=os.environ.get('ALCHEMY_RPC_URL','').strip()
    key=os.environ.get('ALCHEMY_API_KEY','').strip()
    if not url and key:
        # Accept key only, NOT full URL; catch common misconfiguration.
        if key.startswith('https://'):print('Set ALCHEMY_RPC_URL to the full endpoint; ALCHEMY_API_KEY is key only.',file=sys.stderr);return 2
        url='https://eth-mainnet.g.alchemy.com/v2/'+key
    if not url or not url.startswith('https://eth-mainnet.g.alchemy.com/v2/') or len(url.split('/v2/')[-1])<8:
        print('Set ALCHEMY_RPC_URL to the full Alchemy Ethereum Mainnet URL containing your private key.',file=sys.stderr)
        return 2
    snapshot_file=args.outdir/'snapshot_block.txt'
    if snapshot_file.exists():snapshot=snapshot_file.read_text().strip()
    else:
        # Reuse the original five-wallet pilot block to allow one comparable snapshot.
        ref=HERE/'reference_snapshot.txt'
        snapshot=ref.read_text().strip() if ref.exists() else rpc(url,'eth_blockNumber',[])
        snapshot_file.write_text(snapshot,encoding='utf-8')
    if not re.fullmatch(r'0x[0-9a-fA-F]+',snapshot):raise ValueError('Invalid cached snapshot')
    print('Snapshot:',snapshot,flush=True)
    rows=read_manifest(args.manifest)[:args.max_wallets]
    all_rows=[]
    for index,item in enumerate(rows,1):
        addr=item['address'];label=int(item['label'])
        record={'address':addr,'label':label,'status':'pending'}
        try:
            # Rerun prediction even for cached transfers, to detect deployment changes.
            incoming,outgoing,balance=wallet_collection(url,addr,snapshot,args.outdir,args.max_pages,args.page_size)
            complete=incoming['complete'] and outgoing['complete']
            record.update(complete=complete,incoming_complete=incoming['complete'],outgoing_complete=outgoing['complete'])
            feats,quality=build_features(addr,incoming['transfers'],outgoing['transfers'],balance)
            record.update(quality)
            if not complete and not args.allow_partial:
                record['status']='excluded_incomplete_history'
            else:
                payload=make_payload(addr,feats)
                if local_model is not None:
                    response=local_model.predict(payload['features'],wallet_id=addr,input_format='raw-live')
                else:
                    response=request_json(args.base_url.rstrip('/')+'/predict',payload)
                record.update(interpret_response(response))
                if not complete and record['status']=='classified':
                    record['status']='classified_partial_history' # excluded from primary metric
                # Uploaded ZIP only emits FRAUD/LEGITIMATE with fraud_probability in MoE/GAM.
                # MANUAL_REVIEW indicates live deployment mismatch or changed policy.
                if record['status']=='abstained_manual_review':
                    record['error']='Deployed /predict returned MANUAL_REVIEW; uploaded v0.7 MoE/GAM code does not emit this decision.'
        except Exception as exc:
            record['status']='error'
            record['error']=f'{type(exc).__name__}: {str(exc)}'[:220]
        all_rows.append(record)
        csv_write(args.outdir/'predictions.csv',all_rows)
        rep=summarize(all_rows,len(rows));rep['snapshot_block']=snapshot;rep['engine']=args.engine
        rep['server_models_status']=server.get('/models/status',{})
        atomic_json(args.outdir/'report.json',rep)
        emit_html(rep,all_rows,args.outdir/'report.html')
        print(f'{index}/{len(rows)} {addr} {record["status"]} '
              f'p={record.get("probability")}',flush=True)
    print('Reports saved in:',args.outdir.resolve())
    print('Complete-history provisional metrics:',json.dumps(rep['provisional_metrics_complete_collections'],indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
