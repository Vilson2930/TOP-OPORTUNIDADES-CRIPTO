#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 3H: evidencias economicas multiactivos, READ ONLY, fail-closed.

Input: Fase 3F + Fase 3G. Does NOT change engine gates or scores.
Samples recent EVM blocks to verify protocol fee burns on ETH and AVAX C-chain.
A sampled block is NOT evidence of a 30-day burn total or token-holder revenue.
Other mechanisms remain unverified unless independent transaction evidence exists.
"""
import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = 'CryptoOpportunityResearch/3H read-only'
RPC = {
    'ETH': ['https://ethereum-rpc.publicnode.com', 'https://eth.llamarpc.com'],
    'AVAX': ['https://avalanche-c-chain-rpc.publicnode.com', 'https://api.avax.network/ext/bc/C/rpc'],
}
MECHANISMS = {
    'ETH': ('EIP1559_BASE_FEE_BURN', 'EVM block baseFeePerGas * gasUsed; supply net requires issuance and all burns'),
    'AVAX': ('C_CHAIN_BASE_FEE_BURN', 'C-chain base fee burn; other chains and issuance not measured'),
    'INJ': ('COMMUNITY_BUYBACK_BURN', 'Requires independently verified buyback settlement/burn tx and issuance'),
    'MNT': ('TOKEN_ECONOMICS_UNVERIFIED', 'Requires verified tokenholder mechanism and chain transactions'),
    'XPL': ('TOKEN_ECONOMICS_UNVERIFIED', 'Requires verified tokenholder mechanism and chain transactions'),
    'XAUT': ('GOLD_BACKED_TOKEN', 'Gold backing/redemption is not protocol revenue distributed to holders'),
    'BEAM': ('IDENTITY_UNRESOLVED', 'Ambiguous token/protocol mapping: no guessed RPC or contract'),
    'NEXO': ('IDENTITY_UNRESOLVED', 'Protocol identity not safely matched to revenue source'),
}

def fetch_json(url, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'User-Agent': USER_AGENT, 'Accept':'application/json', 'Content-Type':'application/json'}, method='POST' if data else 'GET')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(2_000_000)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def rpc_call(url, method, params):
    obj, digest = fetch_json(url, {'jsonrpc':'2.0', 'id':1, 'method':method, 'params':params})
    if 'error' in obj: raise RuntimeError(str(obj['error'])[:240])
    if obj.get('result') is None: raise RuntimeError('RPC returned null result')
    return obj['result'], digest

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

def safe_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError): return None

def sample_evm(symbol, out, sample_count, spacing):
    attempts=[]
    for endpoint in RPC[symbol]:
        try:
            latest, _ = rpc_call(endpoint, 'eth_blockNumber', [])
            height = int(latest,16)
            if height < 1: raise ValueError('invalid chain height')
            blocks=[]
            for idx in range(sample_count):
                n = max(1, height - idx*spacing)
                block, digest = rpc_call(endpoint, 'eth_getBlockByNumber', [hex(n), False])
                if not isinstance(block,dict): raise ValueError('invalid block')
                base=block.get('baseFeePerGas'); used=block.get('gasUsed')
                if base is None or used is None: raise ValueError('missing EIP-1559 fields')
                burn_wei=int(base,16)*int(used,16)
                blocks.append({'symbol':symbol,'chain':'Ethereum mainnet' if symbol=='ETH' else 'Avalanche C-chain',
                               'rpc_url':endpoint,'block_height':n,'block_hash':block.get('hash'),
                               'timestamp_utc':datetime.fromtimestamp(int(block['timestamp'],16),timezone.utc).isoformat(),
                               'base_fee_wei':int(base,16),'gas_used':int(used,16),
                               'base_fee_burn_native':f'{burn_wei / 10**18:.12f}',
                               'rpc_response_sha256':digest,
                               'proof_scope':'SAMPLED_BLOCK_BASE_FEE_BURN_ONLY'})
            return blocks, attempts, 'SAMPLED_ONCHAIN_BLOCK_BURNS_VERIFIED'
        except Exception as e:
            attempts.append({'symbol':symbol,'rpc_url':endpoint,'error':str(e)[:300]})
    return [], attempts, 'RPC_UNAVAILABLE_OR_UNVERIFIED'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase3f',default='output/fase3f_multiactivos/fase3f_comparacao_todos_elegiveis.csv')
    p.add_argument('--phase3g',default='output/fase3g_captura_multiactivos/fase3g_captura_todos_elegiveis.csv')
    p.add_argument('--output-dir',default='output/fase3h_evidencias_multiactivos')
    p.add_argument('--sample-blocks',type=int,default=3)
    p.add_argument('--block-spacing',type=int,default=120)
    args=p.parse_args()
    if not 1<=args.sample_blocks<=10 or not 1<=args.block_spacing<=100000:
        p.error('sample-blocks must be 1..10 and block-spacing 1..100000')
    fpath=Path(args.phase3f); gpath=Path(args.phase3g)
    if not fpath.is_file() or not gpath.is_file():
        p.error('Missing phase 3F/3G CSV: do not rerun engine; restore existing output or Drive backup')
    frows=read_csv(fpath); grows=read_csv(gpath)
    if not frows or not grows: p.error('Empty input CSV')
    if any('symbol' not in r for r in frows+grows): p.error('Missing symbol column')
    fsyms=[r['symbol'].strip().upper() for r in frows]
    gmap={r['symbol'].strip().upper():r for r in grows}
    if len(fsyms)!=len(set(fsyms)) or len(gmap)!=len(grows): p.error('Duplicate symbols: fail closed')
    if set(fsyms)!=set(gmap): p.error('Phase 3F and 3G symbol sets differ: fail closed')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    evidence=[]; queries=[]; summary=[]
    for idx,fr in enumerate(frows,1):
        symbol=fr['symbol'].strip().upper(); gr=gmap[symbol]
        mechanism, note=MECHANISMS.get(symbol,('UNCLASSIFIED','No verified mechanism configured'))
        print(f'[{idx}/{len(frows)}] {symbol}: {mechanism}',flush=True)
        status='NO_INDEPENDENT_ONCHAIN_CAPTURE_PROOF'
        if symbol in RPC:
            blocks, failures, status=sample_evm(symbol,out,args.sample_blocks,args.block_spacing)
            evidence.extend(blocks); queries.extend(failures)
        summary.append({
            'symbol':symbol,
            'opportunity_score_original':fr.get('opportunity_score_original',''),
            'revenue_growth_30d':fr.get('revenue_growth_30d',''),
            'price_change_30d':fr.get('price_change_30d',''),
            'revenue_minus_price_pp':fr.get('revenue_minus_price_pp',''),
            'phase3g_economic_status':gr.get('economic_status',''),
            'mechanism_to_verify':mechanism,
            'mechanism_note':note,
            'chain_evidence_status':status,
            'sampled_block_count':sum(x['symbol']==symbol for x in evidence),
            'historical_30d_burn_verified':False,
            'issuance_same_period_verified':False,
            'net_supply_reduction_proven':False,
            'tokenholder_capture_independently_proven':False,
            'original_holder_value_gate_modified':False,
            'next_missing_evidence':('Verified mechanism-specific settlement/burn/distribution transactions, issuance and supply reconciliation for same UTC period; '
                                     'for ETH/AVAX sampled base-fee burns do not establish 30d total or net supply change.'),
        })
    cols=['symbol','opportunity_score_original','revenue_growth_30d','price_change_30d','revenue_minus_price_pp',
          'phase3g_economic_status','mechanism_to_verify','mechanism_note','chain_evidence_status','sampled_block_count',
          'historical_30d_burn_verified','issuance_same_period_verified','net_supply_reduction_proven',
          'tokenholder_capture_independently_proven','original_holder_value_gate_modified','next_missing_evidence']
    write_csv(out/'fase3h_matriz_economica_todos_elegiveis.csv',summary,cols)
    write_csv(out/'fase3h_blocos_evm_amostrados.csv',evidence,
              ['symbol','chain','rpc_url','block_height','block_hash','timestamp_utc','base_fee_wei','gas_used','base_fee_burn_native','rpc_response_sha256','proof_scope'])
    write_csv(out/'fase3h_erros_rpc.csv',queries,['symbol','rpc_url','error'])
    report={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'READ_ONLY_ALL_3F_ELIGIBLE',
            'total_eligible':len(summary),'independent_holder_capture_proven':0,
            'evm_sampled_block_count':len(evidence),'rpc_errors':len(queries),
            'warning':'Sampled onchain base-fee burns are NOT evidence of 30d burn totals, net token supply reduction, protocol revenue attribution, or tokenholder distributions.',
            'gates_modified':False,'sources':[str(fpath),str(gpath)]}
    (out/'fase3h_resumo.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nRESUMO:',json.dumps(report,ensure_ascii=False),flush=True)
    print('\nMATRIZ:')
    for r in summary:
        print(f"{r['symbol']:6s} | {r['mechanism_to_verify']:27s} | {r['chain_evidence_status']}")
    print('\nARQUIVOS:',out/'fase3h_matriz_economica_todos_elegiveis.csv',out/'fase3h_blocos_evm_amostrados.csv',out/'fase3h_resumo.json')

if __name__=='__main__': main()
