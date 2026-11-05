#!/usr/bin/env python3
"""Fase 3K: quantificacao on-chain verificavel, incremental, fail-closed.

Le a matriz 3F e preserva todos os elegiveis. Para ETH/AVAX calcula
queima EIP-1559 por bloco REAL, com checkpoint, sem extrapolar amostras.
Nao altera gates, scores nem promove ativos. Nao estima emissao.
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

NETWORKS = {
    'ETH': ['https://ethereum-rpc.publicnode.com', 'https://eth.llamarpc.com'],
    'AVAX': ['https://avalanche-c-chain-rpc.publicnode.com', 'https://api.avax.network/ext/bc/C/rpc'],
}

def rpc(url, method, params, timeout=22):
    body = json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
    req = Request(url, data=body, headers={'Content-Type':'application/json','User-Agent':'EconomicAudit/3K'})
    with urlopen(req, timeout=timeout) as r:
        obj=json.load(r)
    if 'error' in obj: raise RuntimeError(str(obj['error'])[:250])
    if obj.get('result') is None: raise RuntimeError('null RPC result')
    return obj['result']

def call(urls, method, params):
    errors=[]
    for url in urls:
        try: return rpc(url,method,params),url
        except Exception as e: errors.append(f'{url}: {type(e).__name__}: {e}')
    raise RuntimeError(' | '.join(errors))

def block(urls, height):
    b,url=call(urls,'eth_getBlockByNumber',[hex(height),False])
    return b,url

def find_height(urls, timestamp, latest):
    lo,hi=1,latest
    while lo < hi:
        mid=(lo+hi)//2
        b,_=block(urls,mid)
        if int(b['timestamp'],16)<timestamp: lo=mid+1
        else: hi=mid
    return lo

def load_rows(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase3f',required=True)
    ap.add_argument('--phase3g')
    ap.add_argument('--output-dir',default='output/fase3k_captura_30d')
    ap.add_argument('--days',type=int,default=30)
    ap.add_argument('--max-blocks-per-network',type=int,default=1500,
                    help='Limite de blocos por execucao/rede. Checkpoint permite continuar sem repetir.')
    ap.add_argument('--pause',type=float,default=0.04)
    args=ap.parse_args()
    if args.days<1 or args.max_blocks_per_network<1: ap.error('days e max-blocks devem ser positivos')
    rows=load_rows(args.phase3f)
    if not rows: raise SystemExit('Fase 3F vazia')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    symbols=[r.get('symbol','').strip().upper() for r in rows]
    if len(set(symbols))!=len(symbols): raise SystemExit('Simbolos duplicados: pare e revise identidades')
    results=[]
    for row in rows:
        symbol=row.get('symbol','').strip().upper()
        item={'symbol':symbol,'opportunity_score_original':row.get('opportunity_score_original',''),
              'phase3f_revenue_minus_price_pp':row.get('revenue_minus_price_pp',''),
              'mechanism':'EIP1559_BASE_FEE_BURN' if symbol=='ETH' else 'C_CHAIN_BASE_FEE_BURN' if symbol=='AVAX' else 'NOT_VERIFIED_HERE',
              'verified_blocks':0,'total_blocks_window':'','verified_burn_native':'',
              'burn_window_coverage_pct':'','burn_30d_status':'NOT_MEASURED',
              'issuance_30d_status':'NOT_VERIFIED','net_supply_change_status':'NOT_PROVEN',
              'holder_capture_status':'NOT_PROVEN','reason':''}
        if symbol not in NETWORKS:
            item['reason']='Requer verificacao de mecanismo especifico, contrato e eventos; sem inferencia por agregador.'
            results.append(item);print(f'{symbol}: mecanismo especifico pendente',flush=True);continue
        urls=NETWORKS[symbol]
        statepath=out/f'{symbol.lower()}_checkpoint.json'
        try:
            latest_obj,_=call(urls,'eth_getBlockByNumber',['latest',False]);latest=int(latest_obj['number'],16)
            # Congelar a janela na primeira execucao para que checkpoints sejam comparaveis.
            if statepath.exists():
                state=json.loads(statepath.read_text())
                if state.get('days')!=args.days: raise RuntimeError('Checkpoint tem janela diferente. Use outro output-dir.')
                start,end=int(state['start_height']),int(state['end_height'])
            else:
                end=latest
                start_ts=int(latest_obj['timestamp'],16)-args.days*86400
                start=find_height(urls,start_ts,end)
                state={'symbol':symbol,'days':args.days,'start_height':start,'end_height':end,
                       'last_height':start-1,'burn_wei':0,'verified_blocks':0,
                       'errors':[],'created_utc':datetime.now(timezone.utc).isoformat()}
                statepath.write_text(json.dumps(state,indent=2))
            total=end-start+1
            remaining=min(args.max_blocks_per_network,end-int(state['last_height']))
            print(f'{symbol}: janela congelada {start}..{end} ({total} blocos); processar ate {remaining} nesta execucao',flush=True)
            for height in range(int(state['last_height'])+1, min(end,int(state['last_height'])+args.max_blocks_per_network)+1):
                try:
                    b,_=block(urls,height)
                    base=b.get('baseFeePerGas');used=b.get('gasUsed')
                    if base is None or used is None: raise RuntimeError('baseFeePerGas/gasUsed ausente')
                    burn=int(base,16)*int(used,16)
                    state['burn_wei']+=burn;state['last_height']=height;state['verified_blocks']+=1
                except Exception as e:
                    state['errors'].append({'height':height,'error':str(e)[:300]})
                    print(f'{symbol}: erro no bloco {height}: {e}',flush=True)
                    break # nao pular blocos: fail closed
                if state['verified_blocks']%100==0:
                    statepath.write_text(json.dumps(state,indent=2))
                    print(f'{symbol}: {state["verified_blocks"]}/{total} blocos verificados',flush=True)
                if args.pause:time.sleep(args.pause)
            statepath.write_text(json.dumps(state,indent=2))
            item['verified_blocks']=state['verified_blocks'];item['total_blocks_window']=total
            item['verified_burn_native']=f'{state["burn_wei"]/10**18:.12f}'
            item['burn_window_coverage_pct']=round(100*state['verified_blocks']/total,5)
            complete=state['last_height']==end and not state['errors']
            item['burn_30d_status']='FULL_WINDOW_ONCHAIN_VERIFIED' if complete else 'PARTIAL_EXACT_BLOCK_SUM_NOT_30D_TOTAL'
            item['reason']='Queima base-fee de blocos verificados; emissao e oferta liquida nao verificadas.'
            if state['errors']:item['reason']+=' Ultimo erro: '+state['errors'][-1]['error'][:120]
        except Exception as e:
            item['burn_30d_status']='RPC_OR_CHECKPOINT_ERROR'
            item['reason']=str(e)[:350]
        results.append(item)
        print(f'{symbol}: {item["burn_30d_status"]}; cobertura={item["burn_window_coverage_pct"]}%; captura holders=NAO COMPROVADA',flush=True)
    fields=list(results[0].keys())
    csvpath=out/'fase3k_matriz_8_elegiveis.csv';write_csv(csvpath,results,fields)
    summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'eligible':len(results),
             'independent_holder_capture_proven':0,'gates_modified':False,
             'critical_limit':'Queima base-fee de ETH/AVAX nao equivale a captura por holders; emissao e oferta liquida NAO comprovadas. Valores parciais NAO sao totais de 30d.',
             'checkpoint':'Rerode mesma celula para continuar ETH/AVAX sem recomputar blocos anteriores.',
             'rows':results}
    (out/'fase3k_resumo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('RESULTADOS:',csvpath,flush=True)
    print('AVISO: captura independente comprovada = 0; nenhum gate alterado.',flush=True)

if __name__=='__main__':main()
