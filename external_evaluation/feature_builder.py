"""Blockchain transfer to frozen NeuroLink feature names, with explicit unknowns."""
import datetime
import math
import statistics

RF = ['Total ERC20 tnxs','Time Diff between first and last (Mins)','ERC20 uniq rec contract addr',
      'ERC20 uniq rec token name','ERC20 uniq rec addr','Avg min between received tnx',
      'total ether received','total transactions (including tnx to create contract',
      'Unique Received From Addresses','avg val received','total ether balance','Received Tnx',
      'ERC20 max val rec','ERC20 total Ether received','max value received','total Ether sent',
      'Sent tnx','ERC20 min val rec','ERC20 uniq sent token name','min value received']
IF = ['Time Diff between first and last (Mins)','Total ERC20 tnxs',
      'Unique Received From Addresses','ERC20 uniq rec addr','ERC20 uniq rec token name',
      'total ether received','avg val received','Avg min between received tnx',
      'max value received','ERC20 min val rec','min value received','total Ether sent',
      'ERC20 avg val rec','ERC20 max val rec','Avg min between sent tnx','max val sent',
      'avg val sent','Unique Sent To Addresses','ERC20 uniq sent token name','current_eth_balance']
EXTRA = ['ERC20 uniq sent addr','Number of Created Contracts','total ether sent contracts']
TOKEN_VALUE_FIELDS=['ERC20 max val rec','ERC20 min val rec','ERC20 avg val rec','ERC20 total Ether received']

def timestamp(tx):
    try:
        s=(tx.get('metadata') or {}).get('blockTimestamp')
        return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp() if s else None
    except (ValueError,TypeError,OverflowError): return None

def time_gap(items):
    times=sorted(t for tx in items if (t:=timestamp(tx)) is not None)
    return statistics.mean((b-a)/60 for a,b in zip(times,times[1:])) if len(times)>1 else None

def value_stats(items, kind):
    a=[]
    for tx in items:
        v=tx.get('value')
        try:
            if isinstance(v,bool) or v is None or not math.isfinite(float(v)):return None
            a.append(float(v))
        except (ValueError,TypeError):return None
    if not a:return 0.0
    return {'sum':sum,'min':min,'max':max,'avg':statistics.mean}[kind](a)

def transfer_key(tx):
    # Per-transfer ID is preferable to hash-only or hash+endpoints, which collapse
    # distinct transfer logs inside the same on-chain transaction.
    uid=tx.get('uniqueId')
    if uid:return str(uid)
    import json
    return 'fallback:'+json.dumps([tx.get('hash'),tx.get('category'),tx.get('from'),tx.get('to'),
                                     tx.get('logIndex'),tx.get('rawContract')],sort_keys=True)

def build_features(address, incoming, outgoing, balance):
    addr=address.lower();events={}
    for direction, transfers in [('incoming',incoming),('outgoing',outgoing)]:
        for tx in transfers:
            uid=transfer_key(tx)
            e=events.setdefault(uid,{'tx':tx,'incoming':False,'outgoing':False})
            e[direction]=True
    unique=list(events.values())
    native=[e for e in unique if e['tx'].get('category') in ('external','internal')]
    tokens=[e for e in unique if e['tx'].get('category')=='erc20']
    rec=[e['tx'] for e in native if e['incoming'] and (e['tx'].get('from') or '').lower()!=addr]
    sent=[e['tx'] for e in native if e['outgoing'] and (e['tx'].get('to') or '').lower()!=addr]
    erc_rec=[e['tx'] for e in tokens if e['incoming']]
    erc_sent=[e['tx'] for e in tokens if e['outgoing']]
    all_times=[t for e in unique if (t:=timestamp(e['tx'])) is not None]
    span=(max(all_times)-min(all_times))/60 if len(all_times)>1 else (0. if all_times else None)
    hashes={e['tx'].get('hash') for e in unique if e['tx'].get('hash')}
    f={
        'Total ERC20 tnxs':len(tokens),
        'Time Diff between first and last (Mins)':span,
        'ERC20 uniq rec contract addr':len({(t.get('rawContract') or {}).get('address','').lower() for t in erc_rec if (t.get('rawContract') or {}).get('address')}),
        'ERC20 uniq rec token name':len({t['asset'] for t in erc_rec if t.get('asset')}),
        'ERC20 uniq rec addr':len({t['from'].lower() for t in erc_rec if t.get('from')}),
        'Avg min between received tnx':time_gap(rec),
        'total ether received':value_stats(rec,'sum'),
        'total transactions (including tnx to create contract':len(hashes),
        'Unique Received From Addresses':len({t['from'].lower() for t in rec if t.get('from')}),
        'avg val received':value_stats(rec,'avg'),
        'total ether balance':None, # D1's meaning is not verified; DO NOT substitute current balance
        'Received Tnx':len(rec),
        'ERC20 max val rec':None,
        'ERC20 total Ether received':None,
        'max value received':value_stats(rec,'max'),
        'total Ether sent':value_stats(sent,'sum'),
        'Sent tnx':len(sent),
        'ERC20 min val rec':None,
        'ERC20 uniq sent token name':len({t['asset'] for t in erc_sent if t.get('asset')}),
        'min value received':value_stats(rec,'min'),
        'ERC20 avg val rec':None,
        'Avg min between sent tnx':time_gap(sent),
        'max val sent':value_stats(sent,'max'),
        'avg val sent':value_stats(sent,'avg'),
        'Unique Sent To Addresses':len({t['to'].lower() for t in sent if t.get('to')}),
        'current_eth_balance':balance,
        'ERC20 uniq sent addr':len({t['to'].lower() for t in erc_sent if t.get('to')}),
        'Number of Created Contracts':None,
        'total ether sent contracts':None
    }
    # Zero erc20 events makes zero token-quantity aggregates meaningful. Otherwise
    # values for unlike tokens cannot be summed/comparable without D1 parity.
    if not tokens:
        for k in TOKEN_VALUE_FIELDS: f[k]=0.
    quality={'unique_transfer_events':len(unique),'unique_tx_hashes':len(hashes),
             'erc20_events':len(tokens),'native_events':len(native),
             'rf_observed':sum(f.get(k) is not None for k in RF),
             'if_observed':sum(f.get(k) is not None for k in IF),
             'missing_rf': [k for k in RF if f.get(k) is None],
             'missing_if': [k for k in IF if f.get(k) is None],
             'ambiguous_token_value_features':TOKEN_VALUE_FIELDS if tokens else []}
    return f,quality

def make_payload(address,features):
    # No ground-truth label ever enters the model request.
    return {'wallet_id':address,'input_format':'raw-live',
            'features':{k:features.get(k) for k in sorted(set(RF+IF+EXTRA))}}
