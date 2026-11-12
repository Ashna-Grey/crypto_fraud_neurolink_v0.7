"""Resilient HTTPS helpers for Alchemy/Render. Keys never enter output files."""
import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request

TRANSIENT=(http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError,
           ConnectionAbortedError, ConnectionError, BrokenPipeError,TimeoutError,socket.timeout,
           urllib.error.URLError, json.JSONDecodeError)

class ApiError(RuntimeError): pass


def request_json(url, payload=None, retries=6, timeout=110):
    body=None if payload is None else json.dumps(payload,allow_nan=False).encode('utf-8')
    for attempt in range(retries):
        try:
            req=urllib.request.Request(url,body,headers={'Content-Type':'application/json',
                   'User-Agent':'NeuroLink-PoisonHunter-External-Evaluation/2.0'},
                   method='POST' if payload is not None else 'GET')
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                raw=resp.read()
                return json.loads(raw.decode('utf-8'))
        except urllib.error.HTTPError as exc:
            # Never include full URL or request body. It may hold the API key.
            status=exc.code
            if status not in (408,409,425,429,500,502,503,504) or attempt == retries-1:
                raise ApiError(f'HTTP {status}, endpoint rejected request') from None
        except TRANSIENT as exc:
            if attempt==retries-1:raise ApiError(f'{type(exc).__name__}: network request exhausted retries') from None
        if attempt < retries-1: time.sleep(min(18, 1.5*(2**attempt)+random.random()))
    raise ApiError('Request exhausted retries')


def rpc(url,method,params):
    out=request_json(url,{'jsonrpc':'2.0','id':1,'method':method,'params':params})
    if not isinstance(out,dict):raise ApiError('Invalid RPC response shape')
    if 'error' in out:
        err=out['error'];code=err.get('code','unknown') if isinstance(err,dict) else 'unknown'
        raise ApiError(f'{method}: RPC error code {code}')
    if 'result' not in out:raise ApiError(f'{method}: RPC response has no result')
    return out['result']
