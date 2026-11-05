#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASE 3M: verificador de evidencias transacionais, READ ONLY e fail-closed.

Nao reexecuta motor ou fases anteriores. Recebe evidencia EXTERNA em CSV,
verifica transacoes EVM em ETH e AVAX, e nunca promove gates.

CSV opcional --evidence com colunas:
symbol,chain,tx_hash,expected_token_contract,expected_burn_amount_raw,source_url

- ETH/AVAX: verifica recibo on-chain, status e logs ERC-20 Transfer para 0x0
  ou 0x...dead, quando contrato e quantidade forem fornecidos.
- Sem contrato: valida apenas a existencia da transacao, NAO a captura.
- INJ e demais: nao afirma prova por falta de adaptador independente.
- Mesmo uma queima validada nao prova 30 dias, reducao liquida nem retorno ao holder.
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

RPC = {
 'ETH': ['https://ethereum-rpc.publicnode.com','https://eth.llamarpc.com'],
 'AVAX': ['https://avalanche-c-chain-rpc.publicnode.com','https://api.avax.network/ext/bc/C/rpc'],
}
TRANSFER_TOPIC='0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
ZERO='0x'+'0'*40
DEAD='0x'+'0'*36+'dead'
HEX64=re.compile(r'^0x[0-9a-fA-F]{64}$')
ADDR=re.compile(r'^0x[0-9a-fA-F]{40}$')


def read_rows(path):
 with open(path,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def rpc(endpoint,method,params):
 body=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()
 req=urllib.request.Request(endpoint,data=body,headers={'Content-Type':'application/json','User-Agent':'CryptoOpportunity3M/1.0'})
 with urllib.request.urlopen(req,timeout=12) as r: data=json.load(r)
 if 'error' in data: raise RuntimeError(str(data['error'])[:200])
 return data.get('result')


def receipt(chain,tx_hash):
 errors=[]
 for endpoint in RPC[chain]:
  try:
   rec=rpc(endpoint,'eth_getTransactionReceipt',[tx_hash])
   if not rec: raise ValueError('transaction receipt not found')
   if rec.get('transactionHash','').lower()!=tx_hash.lower(): raise ValueError('hash mismatch')
   return rec,endpoint,''
  except Exception as exc: errors.append(type(exc).__name__+': '+str(exc)[:130])
 return None,'',' | '.join(errors)


def burn_logs(rec,contract):
 total=0; matches=[]
 for log in rec.get('logs',[]):
  if log.get('address','').lower()!=contract.lower(): continue
  topics=log.get('topics',[])
  if len(topics)<3 or topics[0].lower()!=TRANSFER_TOPIC: continue
  to='0x'+topics[2][-40:].lower()
  if to not in (ZERO,DEAD): continue
  try: amount=int(log['data'],16)
  except (ValueError,KeyError,TypeError): continue
  total+=amount
  matches.append({'log_index':log.get('logIndex'),'recipient':to,'raw_amount':str(amount)})
 return total,matches


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--phase3l',default='output/fase3l_consolidacao/fase3l_matriz_decisao_todos_elegiveis.csv')
 p.add_argument('--evidence',default='output/fase3m_evidencias_externas.csv')
 p.add_argument('--output-dir',default='output/fase3m_verificacao_independente')
 args=p.parse_args()
 base=Path(args.phase3l)
 if not base.is_file(): raise SystemExit('FALTA FASE 3L: '+str(base))
 eligible=read_rows(base)
 symbols=[r['symbol'].strip().upper() for r in eligible]
 if len(symbols)!=len(set(symbols)): raise SystemExit('Simbolos duplicados na Fase 3L')
 ev_path=Path(args.evidence)
 evidence=read_rows(ev_path) if ev_path.is_file() else []
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 results=[]
 for n,e in enumerate(evidence,1):
  symbol=e.get('symbol','').strip().upper();chain=e.get('chain','').strip().upper()
  tx=e.get('tx_hash','').strip();contract=e.get('expected_token_contract','').strip()
  raw=e.get('expected_burn_amount_raw','').strip()
  r={'row':n,'symbol':symbol,'chain':chain,'tx_hash':tx,'status':'','verified_burn_raw':'','endpoint':'','reason':'','source_url':e.get('source_url','')}
  if symbol not in symbols: r['status']='REJECTED_NOT_ELIGIBLE'
  elif chain!=symbol or chain not in RPC: r['status']='UNSUPPORTED_CHAIN_OR_IDENTITY'
  elif not HEX64.fullmatch(tx): r['status']='INVALID_TX_HASH'
  elif not contract or not ADDR.fullmatch(contract): r['status']='MISSING_OR_INVALID_BURN_CONTRACT'
  elif not raw.isdigit(): r['status']='MISSING_OR_INVALID_EXPECTED_AMOUNT'
  else:
   rec,endpoint,error=receipt(chain,tx)
   r['endpoint']=endpoint
   if rec is None: r['status']='RPC_NOT_VERIFIED';r['reason']=error
   elif int(rec.get('status','0x0'),16)!=1: r['status']='TRANSACTION_FAILED'
   else:
    actual,logs=burn_logs(rec,contract)
    r['verified_burn_raw']=str(actual)
    if actual==0: r['status']='NO_BURN_LOG_FOUND'
    elif actual!=int(raw): r['status']='BURN_AMOUNT_MISMATCH'
    else: r['status']='TX_BURN_LOG_VERIFIED';r['reason']='One transaction only; not net supply or holder capture'
  results.append(r)
  print(f'[{n}/{len(evidence)}] {symbol} {r["status"]}',flush=True)
  time.sleep(0.1)
 columns=['row','symbol','chain','tx_hash','status','verified_burn_raw','endpoint','reason','source_url']
 with (out/'fase3m_transacoes_verificadas.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(results)
 counts={s:sum(r['status']=='TX_BURN_LOG_VERIFIED' and r['symbol']==s for r in results) for s in symbols}
 matrix=[{'symbol':s,'transactions_burn_verified':counts[s],'holder_capture_30d_proven':False,'original_gates_modified':False,'status':'TRANSACTION_BURN_EVIDENCE_PARTIAL' if counts[s] else 'NO_INDEPENDENT_TRANSACTION_PROOF'} for s in symbols]
 with (out/'fase3m_matriz_8_elegiveis.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
 summary={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'eligible':len(symbols),'submitted_transactions':len(evidence),'verified_burn_transactions':sum(counts.values()),'holder_capture_30d_proven':0,'gates_modified':False,'evidence_file_exists':ev_path.is_file(),'warning':'No transactions supplied means NO new proof. ERC20 burn event is not proof of net supply reduction, protocol revenue attribution or holder capture. Native ETH/AVAX fee burns are not ERC20 Transfer logs.'}
 (out/'fase3m_resumo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 print('FASE 3M:',json.dumps(summary,ensure_ascii=False),flush=True)
 print('OUTPUT:',out.resolve(),flush=True)
 if not evidence: print('SEM NOVA EVIDENCIA: nao repetir a fase; obter hashes/contratos oficiais antes de nova execucao.',flush=True)

if __name__=='__main__':main()
