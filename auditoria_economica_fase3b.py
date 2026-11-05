#!/usr/bin/env python3
"""Fase 3B: auditoria read-only de evidencias economicas do Crypto Opportunity Engine.
Nao recalcula scores, nao promove gates, nao confirma captura economica.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DEFAULT_SYMBOLS = ('INJ', 'AVAX', 'XPL', 'ETH')
FIELDS = [
 'coin_id','symbol','name','price','market_cap','fdv','return_30d','return_90d','return_180d','return_365d','price_change_30d',
 'annualized_revenue','annualized_fees','revenue_growth','fees_growth','revenue_growth_method','revenue_annualization_basis',
 'revenue_data_source','revenue_data_protocol_slug','revenue_match_status','revenue_source_status',
 'holder_revenue','holder_revenue_annualized','holder_revenue_yield','holder_capture_ratio','fee_distribution_ratio',
 'holder_value_protocol_revenue','holder_value_protocol_fees','holder_value_protocol_slug','holder_value_data_source',
 'holder_value_holders_match_status','holder_value_revenue_match_status','holder_value_fees_match_status',
 'holder_value_cross_source_consistent','holder_value_fee_ratio_verified','holder_value_capture_ratio_verified',
 'holder_value_evidence_count','holder_value_evidence_sufficient','holder_value_evidence_status','holder_value_quality_gate',
 'buyback_active','burn_active','token_required_for_usage','gas_token','economic_staking','real_yield',
 'fundamental_change_90d','fundamental_price_divergence','fundamental_price_divergence_score',
 'revenue_price_divergence','revenue_price_divergence_score','opportunity_score','fundamental_gate','revenue_gate',
 'holder_value_gate','dilution_gate','quality_gate','hard_block','quality_failure_reasons','signal',
 'last_updated','market_source','history_source'
]

def missing(v):
    if v is None: return True
    try:
        if bool(pd.isna(v)): return True
    except (TypeError, ValueError): pass
    return isinstance(v,str) and not v.strip()

def number(v):
    if missing(v): return None
    try:
        x=float(v)
        return x if pd.notna(x) and abs(x)!=float('inf') else None
    except (ValueError,TypeError): return None

def truth(v):
    return str(v).strip().lower() in ('true','1','yes') if not missing(v) else False

def as_json(v):
    if missing(v): return None
    if isinstance(v,(bool,int,float,str)):
        if isinstance(v,float) and not pd.notna(v): return None
        return v
    return str(v)

def analyze(row):
    d={k:as_json(row.get(k)) for k in FIELDS}
    flags=[]
    rev=number(d['annualized_revenue'])
    holder=number(d['holder_revenue'])
    ratio=number(d['holder_capture_ratio'])
    if rev is not None and holder is not None and abs(rev-holder)<=max(0.01,abs(rev)*1e-9):
        flags.append('HOLDER_REVENUE_EQUALS_PROTOCOL_REVENUE_REQUIRES_INDEPENDENT_ATTRIBUTION')
    if ratio is not None and ratio>=0.999:
        flags.append('REPORTED_CAPTURE_RATIO_100_PERCENT_NOT_PROOF_OF_TOKEN_DISTRIBUTION')
    if holder is None: flags.append('HOLDER_REVENUE_MISSING')
    if missing(d['revenue_growth']): flags.append('REVENUE_GROWTH_MISSING')
    if missing(d['revenue_price_divergence_score']): flags.append('REVENUE_PRICE_DIVERGENCE_SCORE_MISSING')
    if missing(d['fundamental_price_divergence_score']): flags.append('FUNDAMENTAL_PRICE_DIVERGENCE_SCORE_MISSING')
    if not truth(d['holder_value_evidence_sufficient']): flags.append('HOLDER_VALUE_EVIDENCE_NOT_SUFFICIENT')
    if not truth(d['holder_value_gate']): flags.append('HOLDER_VALUE_GATE_FAILED')
    if truth(d['hard_block']): flags.append('HARD_BLOCK_PRESENT')
    if truth(d['holder_value_capture_ratio_verified']):
        flags.append('INTERNAL_CAPTURE_RATIO_VERIFIED_FLAG_NOT_INDEPENDENT_ECONOMIC_PROOF')
    methods=[k for k in ('buyback_active','burn_active','economic_staking','real_yield','fee_distribution_ratio') if not missing(d[k])]
    if not methods: flags.append('NO_EXPLICIT_ECONOMIC_MECHANISM_FIELDS_POPULATED')
    # Returns are stored as fractions in the engine; do not equate TVL growth with revenue growth.
    price90=number(d['return_90d'])
    fund90=number(d['fundamental_change_90d'])
    d['observed_90d_fundamental_minus_price']=round(fund90-price90,8) if price90 is not None and fund90 is not None else None
    d['observed_90d_comparison_note']='DESCRIPTIVE_ONLY_NOT_ORIGINAL_ENGINE_SCORE' if d['observed_90d_fundamental_minus_price'] is not None else 'INSUFFICIENT_COMPARABLE_90D_DATA'
    d['audit_flags']='|'.join(flags)
    d['audit_flag_count']=len(flags)
    d['economic_capture_independently_proven']=False
    d['audit_conclusion']='INVESTIGATE_ONLY_NO_GATE_PROMOTION'
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='output/crypto_diagnostic_all.csv')
    ap.add_argument('--output-dir',default='output/fase3b_auditoria')
    ap.add_argument('--symbols',default=','.join(DEFAULT_SYMBOLS))
    args=ap.parse_args()
    source=Path(args.input)
    if not source.is_file(): raise FileNotFoundError(source)
    df=pd.read_csv(source,low_memory=False)
    if 'symbol' not in df.columns: raise ValueError('Missing symbol column')
    symbols=[x.strip().upper() for x in args.symbols.split(',') if x.strip()]
    subset=df[df['symbol'].astype(str).str.upper().isin(symbols)].copy()
    if 'opportunity_score' in subset.columns:
        subset=subset.sort_values('opportunity_score',ascending=False,na_position='last')
    found=set(subset['symbol'].astype(str).str.upper())
    absent=[s for s in symbols if s not in found]
    if absent: raise ValueError('Missing requested symbols: '+', '.join(absent))
    rows=[analyze(row) for _,row in subset.iterrows()]
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    audit=pd.DataFrame(rows)
    csv_path=out/'fase3b_economic_audit.csv'
    json_path=out/'fase3b_economic_audit.json'
    audit.to_csv(csv_path,index=False,encoding='utf-8-sig')
    payload={'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'source_file':str(source),
      'source_rows':len(df),'target_symbols':symbols,'rows':rows,
      'scope':'READ_ONLY_EVIDENCE_DIAGNOSTIC','gates_modified':False,'scores_modified':False,
      'warning':'Aggregator matching and internal verified flags do not prove tokenholder cash flows.'}
    json_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('=== FASE 3B - AUDITORIA ECONOMICA ===')
    print('Registros no universo:',len(df),'| registros selecionados:',len(audit))
    cols=['symbol','opportunity_score','annualized_revenue','revenue_growth','holder_revenue',
      'holder_capture_ratio','holder_value_evidence_count','holder_value_evidence_status',
      'revenue_price_divergence_score','observed_90d_fundamental_minus_price',
      'holder_value_gate','hard_block','audit_flag_count']
    print(audit[cols].to_string(index=False))
    print('\n=== ALERTAS POR ATIVO ===')
    for row in rows:
        print('\n'+str(row['symbol'])+':')
        for flag in row['audit_flags'].split('|'): print(' -',flag)
    print('\nCSV:',csv_path.resolve())
    print('JSON:',json_path.resolve())
    print('Nenhum score ou gate alterado. Nenhuma captura economica confirmada automaticamente.')

if __name__=='__main__': main()
