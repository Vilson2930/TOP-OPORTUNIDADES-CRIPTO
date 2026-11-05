#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 3I: evidência temporal on-chain para TODOS os elegíveis (READ ONLY).

Verifica blocos em ~30 datas UTC para ETH e AVAX, com hash e base-fee burn
EXATO POR BLOCO. NÃO extrapola amostras para totais de 30 dias, nem declara
redução líquida da oferta ou captura pelos holders. Para outros mecanismos,
registra lacunas sem inventar identidades/contratos. Não altera o motor.
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

RPC = {
    'ETH': ['https://ethereum-rpc.publicnode.com', 'https://eth.llamarpc.com'],
    'AVAX': ['https://avalanche-c-chain-rpc.publicnode.com', 'https://api.avax.network/ext/bc/C/rpc'],
}
UA = 'CryptoOpportunity/3I independent-sampled-evidence'

def rpc(url, method, params):
    payload = json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
    request = urllib.request.Request(url, data=payload, headers={'Content-Type':'application/json','User-Agent':UA})
    with urllib.request.urlopen(request, timeout=18) as response:
        raw = response.read(2_000_000)
    obj = json.loads(raw)
    if obj.get('error') or obj.get('result') is None:
        raise RuntimeError(str(obj.get('error', 'null RPC result'))[:180])
    return obj['result'], hashlib.sha256(raw).hexdigest()

def block(url, height):
    obj, digest = rpc(url, 'eth_getBlockByNumber', [hex(height), False])
    if not isinstance(obj, dict):
        raise ValueError('Invalid block')
    if not all(k in obj for k in ('hash','timestamp','baseFeePerGas','gasUsed')):
        raise ValueError('Missing verifiable block fields')
    return obj, digest

def at_or_before(url, timestamp, upper):
    lo, hi = 1, upper
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        obj, _ = block(url, mid)
        ts = int(obj['timestamp'],16)
        if ts <= timestamp:
            found = (mid, obj)
            lo = mid + 1
        else:
            hi = mid - 1
    if found is None:
        raise RuntimeError('No block at or before target timestamp')
    return found

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def verify_chain(symbol, days):
    errors=[]
    for url in RPC[symbol]:
        try:
            head_hex, _ = rpc(url, 'eth_blockNumber', [])
            head=int(head_hex,16)
            latest, _ = block(url, head)
            latest_ts=int(latest['timestamp'],16)
            if abs(time.time()-latest_ts)>86400:
                raise RuntimeError('RPC chain head stale (>24h)')
            # UTC day-end observations: one independent block for each UTC day.
            today=datetime.fromtimestamp(latest_ts, timezone.utc).date()
            samples=[]
            previous_height=None
            for days_ago in range(days,0,-1):
                day=today-timedelta(days=days_ago)
                target=int(datetime(day.year,day.month,day.day,tzinfo=timezone.utc).timestamp())+86399
                h,obj=at_or_before(url,target,head)
                if previous_height is not None and h<=previous_height:
                    raise RuntimeError('Non-increasing historical block heights')
                previous_height=h
                burn=int(obj['baseFeePerGas'],16)*int(obj['gasUsed'],16)
                samples.append({'symbol':symbol,'utc_day':day.isoformat(),'block_height':h,
                    'block_hash':obj['hash'],'block_timestamp_utc':datetime.fromtimestamp(int(obj['timestamp'],16),timezone.utc).isoformat(),
                    'base_fee_wei':int(obj['baseFeePerGas'],16),'gas_used':int(obj['gasUsed'],16),
                    'burn_wei_exact_for_block':str(burn),'burn_native_exact_for_block':f'{burn/1e18:.12f}',
                    'rpc_endpoint':url,'proof_scope':'ONE_BLOCK_PER_UTC_DAY_NOT_DAILY_TOTAL'})
            return samples,errors
        except Exception as exc:
            errors.append({'symbol':symbol,'rpc_endpoint':url,'error':str(exc)[:240]})
    return [],errors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase3f', default='output/fase3f_multiactivos/fase3f_comparacao_todos_elegiveis.csv')
    p.add_argument('--phase3g', default='output/fase3g_captura_multiactivos/fase3g_captura_todos_elegiveis.csv')
    p.add_argument('--phase3h', default='output/fase3h_evidencias_multiactivos/fase3h_matriz_economica_todos_elegiveis.csv')
    p.add_argument('--output-dir', default='output/fase3i_evidencias_30d')
    p.add_argument('--days',type=int,default=30)
    args=p.parse_args()
    if not 1<=args.days<=31: p.error('days must be 1..31')
    paths=[Path(args.phase3f),Path(args.phase3g),Path(args.phase3h)]
    if not all(x.is_file() for x in paths): p.error('Missing prior CSV: restore outputs; DO NOT rerun engine')
    tables=[read_csv(x) for x in paths]
    if any(not t for t in tables): p.error('Empty CSV')
    maps=[]
    for table in tables:
        syms=[r.get('symbol','').strip().upper() for r in table]
        if not all(syms) or len(syms)!=len(set(syms)): p.error('Missing or duplicated symbol')
        maps.append(dict(zip(syms,table)))
    if not (set(maps[0])==set(maps[1])==set(maps[2])): p.error('Different eligible sets; fail closed')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    all_blocks=[];errors=[];matrix=[]
    for idx, (sym,fr) in enumerate(maps[0].items(),1):
        gr=maps[1][sym];hr=maps[2][sym]
        blocks=[];issues=[]
        if sym in RPC:
            blocks,issues=verify_chain(sym,args.days)
            all_blocks.extend(blocks);errors.extend(issues)
        status=('DAILY_BLOCK_SAMPLES_VERIFIED_NOT_30D_TOTAL' if len(blocks)==args.days
                else 'RPC_FAILED_OR_INCOMPLETE' if sym in RPC else 'MECHANISM_SPECIFIC_PROOF_NOT_AVAILABLE')
        row={'symbol':sym,'opportunity_score_original':fr.get('opportunity_score_original',''),
            'revenue_growth_30d':fr.get('revenue_growth_30d',''),
            'price_change_30d':fr.get('price_change_30d',''),
            'revenue_minus_price_pp':fr.get('revenue_minus_price_pp',''),
            'mechanism':hr.get('mechanism_to_verify',''),
            'phase3g_status':gr.get('economic_status',''),
            'onchain_evidence_status':status,'daily_block_samples_verified':len(blocks),
            'burn_total_30d_proven':False,'issuance_30d_proven':False,
            'net_supply_change_proven':False,'holder_capture_proven':False,
            'investment_approval':'NOT_APPROVED_INDEPENDENT_CAPTURE_MISSING',
            'missing_evidence':('Exact 30-day aggregate burns, issuance and supply reconciliation' if sym in RPC
                 else 'Mechanism-specific independently verified settlement/distribution/burn transactions and issuance')}
        matrix.append(row)
        print(f'[{idx}/{len(maps[0])}] {sym}: {status} ({len(blocks)} historical blocks)',flush=True)
    save_csv(out/'fase3i_matriz_8_elegiveis.csv',matrix,list(matrix[0]))
    bfields=['symbol','utc_day','block_height','block_hash','block_timestamp_utc','base_fee_wei','gas_used','burn_wei_exact_for_block','burn_native_exact_for_block','rpc_endpoint','proof_scope']
    save_csv(out/'fase3i_blocos_diarios_verificados.csv',all_blocks,bfields)
    save_csv(out/'fase3i_rpc_errors.csv',errors,['symbol','rpc_endpoint','error'])
    summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'eligible':len(matrix),
             'onchain_daily_blocks_verified':len(all_blocks),'rpc_errors':len(errors),
             'independently_verified_30d_capture':0,'gates_modified':False,
             'critical_limit':'One verified block per day is NOT a 30-day burn total, supply reduction or holder capture.',
             'input_files':[str(x) for x in paths]}
    (out/'fase3i_resumo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('RESUMO:',json.dumps(summary,ensure_ascii=False),flush=True)
    print('ARQUIVOS:',out/'fase3i_matriz_8_elegiveis.csv',out/'fase3i_blocos_diarios_verificados.csv',flush=True)

if __name__=='__main__': main()
