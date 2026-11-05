#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 3J: teste de VIABILIDADE das fontes de prova, para TODOS os elegíveis.

Não recalcula scores, preços, receitas, gates ou rankings. Não infere queima
mensal a partir de amostras. Objetivo: determinar quais fontes permitem obter
provas históricas suficientes antes de construir outro coletor caro.
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RPC = {
    'ETH': ['https://ethereum-rpc.publicnode.com', 'https://eth.llamarpc.com'],
    'AVAX': ['https://avalanche-c-chain-rpc.publicnode.com', 'https://api.avax.network/ext/bc/C/rpc'],
}
INJ_LCD = ['https://sentry.lcd.injective.network', 'https://lcd.injective.network']
UA = 'CryptoOpportunity/3J feasibility-read-only'

def fetch(url, payload=None, timeout=12):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'User-Agent': UA, 'Accept':'application/json', 'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2_000_000)
            status = resp.status
        obj = json.loads(raw)
        return {'status':'OK', 'http':status, 'sha256':hashlib.sha256(raw).hexdigest(), 'data':obj}
    except urllib.error.HTTPError as exc:
        return {'status':f'HTTP_{exc.code}', 'http':exc.code, 'sha256':'', 'data':None}
    except Exception as exc:
        return {'status':'ERROR_'+type(exc).__name__, 'http':'', 'sha256':'', 'data':None}

def rpc(url, method, params):
    res=fetch(url, {'jsonrpc':'2.0','id':1,'method':method,'params':params})
    if res['status']=='OK' and (not isinstance(res['data'],dict) or res['data'].get('error') or res['data'].get('result') is None):
        res['status']='RPC_UNAVAILABLE_OR_ERROR'
    return res

def write_csv(path, rows, cols):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def read_rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase3f',required=True)
    ap.add_argument('--phase3g',required=True)
    ap.add_argument('--phase3i',default='')
    ap.add_argument('--output-dir',default='output/fase3j_viabilidade_fontes')
    a=ap.parse_args()
    rows=read_rows(a.phase3f)
    prior={r.get('symbol','').upper():r for r in read_rows(a.phase3g)}
    symbols=list(dict.fromkeys(r.get('symbol','').upper() for r in rows if r.get('symbol')))
    if not symbols:raise SystemExit('ERRO: Fase 3F vazia; nenhuma análise executada.')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    probes=[]; matrix=[]
    def add(symbol,source,operation,url,result,details=''):
        probes.append({'symbol':symbol,'source':source,'operation':operation,'endpoint':url,'status':result['status'],'http':result.get('http',''),'response_sha256':result.get('sha256',''),'details':details})
    for symbol in symbols:
        proof='NOT_PROVEN'; viable='NOT_TESTED'; gap=''
        if symbol in RPC:
            valid=[]
            for endpoint in RPC[symbol]:
                head=rpc(endpoint,'eth_blockNumber',[])
                add(symbol,'EVM_RPC','HEAD',endpoint,head)
                if head['status']!='OK':continue
                try:
                    height=int(head['data']['result'],16)
                except (KeyError,ValueError,TypeError):continue
                # ~30 days: chain-specific conservative height offsets; NOT proof of time alignment.
                old_height=max(1,height-(220000 if symbol=='ETH' else 1300000))
                old=rpc(endpoint,'eth_getBlockByNumber',[hex(old_height),False])
                add(symbol,'EVM_RPC','HISTORICAL_BLOCK',endpoint,old,f'height={old_height}; historical availability only, NOT 30d exact')
                receipts=rpc(endpoint,'eth_getBlockReceipts',[hex(old_height)])
                add(symbol,'EVM_RPC','HISTORICAL_BLOCK_RECEIPTS',endpoint,receipts,f'height={old_height}; support only, no monthly aggregation')
                if old['status']=='OK':valid.append({'rpc':endpoint,'historical_block':True,'historical_receipts':receipts['status']=='OK'})
            if valid:
                viable='HISTORICAL_BLOCK_SOURCE_FOUND'
                gap='Need exact UTC boundary heights, complete contiguous block range, verified issuance and supply accounting; RPC availability alone proves none.'
            else:
                viable='HISTORICAL_BLOCK_SOURCE_NOT_FOUND'
                gap='Find working historical RPC or indexed historical burn and issuance dataset.'
        elif symbol=='INJ':
            successes=[]
            for endpoint in INJ_LCD:
                for operation,path in [('SUPPLY','/cosmos/bank/v1beta1/supply'),('TX_SEARCH','/cosmos/tx/v1beta1/txs?events=message.action%3D%27%2Fcosmos.bank.v1beta1.MsgSend%27&pagination.limit=1'),('BLOCK_LATEST','/cosmos/base/tendermint/v1beta1/blocks/latest')]:
                    res=fetch(endpoint+path)
                    add(symbol,'INJ_LCD',operation,endpoint+path,res)
                    if res['status']=='OK':successes.append(operation)
            viable='PARTIAL_LCD_ACCESS' if successes else 'LCD_UNAVAILABLE'
            gap='Identify official Community BuyBack burn transaction event schema, settled tx hashes, INJ burned and issuance over matching period.'
        elif symbol in ('BEAM','NEXO'):
            viable='IDENTITY_MAPPING_REQUIRED'
            gap='Verify exact project/token and protocol source before calling any onchain endpoint.'
        elif symbol=='XAUT':
            viable='ASSET_SPECIFIC_MODEL_REQUIRED'
            gap='Gold-backed token: verify redemption, reserve attestations, custody and fees; burn-only model is inappropriate.'
        else:
            viable='MECHANISM_AND_CONTRACT_MAPPING_REQUIRED'
            gap='Confirm official token contract, chain and value capture mechanism before historical transaction collection.'
        matrix.append({'symbol':symbol,'original_opportunity_score':next((r.get('opportunity_score_original','') for r in rows if r.get('symbol','').upper()==symbol),''),'phase3g_economic_status':prior.get(symbol,{}).get('economic_status',''),'source_feasibility':viable,'independent_30d_capture_proven':False,'evidence_gap':gap})
        print(f'{symbol}: {viable} | capture_30d=NOT_PROVEN',flush=True)
    write_csv(out/'fase3j_fontes_testadas.csv',probes,['symbol','source','operation','endpoint','status','http','response_sha256','details'])
    write_csv(out/'fase3j_matriz_8_elegiveis.csv',matrix,['symbol','original_opportunity_score','phase3g_economic_status','source_feasibility','independent_30d_capture_proven','evidence_gap'])
    summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'total_eligible':len(symbols),'sources_tested':len(probes),'independent_30d_capture_proven':0,'gates_modified':False,'scope':'Source feasibility only. No duplicate 3F/3G/3H/3I calculations.','warning':'Successful RPC/LCD requests do not prove historical token holder capture or net supply reduction.'}
    (out/'fase3j_resumo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('RESUMO:',json.dumps(summary,ensure_ascii=False))
    print('ARQUIVOS:',out/'fase3j_matriz_8_elegiveis.csv','|',out/'fase3j_fontes_testadas.csv')
if __name__=='__main__':main()
