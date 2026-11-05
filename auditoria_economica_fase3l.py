#!/usr/bin/env python3
"""Fase 3L: decisao comparativa READ ONLY, sem rede e sem recalcular gates.
Consolida 3F..3K; nunca transforma serie de agregador ou amostra de bloco
em prova de captura, queima mensal ou reducao liquida da oferta.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {
    '3f': 'output/fase3f_multiactivos/fase3f_comparacao_todos_elegiveis.csv',
    '3g': 'output/fase3g_captura_multiactivos/fase3g_captura_todos_elegiveis.csv',
    '3h': 'output/fase3h_evidencias_multiactivos/fase3h_matriz_economica_todos_elegiveis.csv',
    '3i': 'output/fase3i_evidencias_30d/fase3i_matriz_8_elegiveis.csv',
    '3j': 'output/fase3j_viabilidade_fontes/fase3j_matriz_8_elegiveis.csv',
    '3k': 'output/fase3k_captura_30d/fase3k_matriz_8_elegiveis.csv',
}
MECHANISMS = {
    'ETH': 'EIP-1559: queima de taxa base; emissao de validadores a conciliar',
    'AVAX': 'Queima de taxas C-Chain; emissao de staking a conciliar',
    'INJ': 'Community BuyBack: queimas liquidadas e emissao a verificar',
    'XAUT': 'Token lastreado em ouro: lastro/resgate e custos, nao queima de taxas',
    'MNT': 'Mecanismo de captura e contrato a verificar',
    'XPL': 'Mecanismo de captura e contrato a verificar',
    'BEAM': 'Identidade do projeto/contrato e mecanismo a verificar',
    'NEXO': 'Identidade do projeto/contrato e mecanismo a verificar',
}

def load(path, required=False):
    p = Path(path)
    if not p.is_file():
        if required:
            raise FileNotFoundError(f'Arquivo obrigatorio ausente: {p}. Restaure o backup, nao execute o motor novamente.')
        return None
    with p.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'symbol' not in rows[0]:
        raise ValueError(f'CSV vazio ou sem coluna symbol: {p}')
    index = {}
    for r in rows:
        s = (r.get('symbol') or '').strip().upper()
        if not s or s in index:
            raise ValueError(f'Simbolo vazio/duplicado em {p}: {s!r}')
        index[s] = r
    return index

def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def true(v):
    return str(v).strip().lower() in ('true','1','sim','yes')

def first(row, *names):
    for name in names:
        if row.get(name) not in (None, ''):
            return row[name]
    return ''

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        ap.add_argument('--phase'+key, default=default)
    ap.add_argument('--output-dir', default='output/fase3l_consolidacao')
    a = ap.parse_args()
    sources = {key: getattr(a, 'phase'+key) for key in DEFAULTS}
    tables = {key: load(path, required=key in ('3f','3g')) for key,path in sources.items()}
    base = tables['3f']
    if set(base) != set(tables['3g']):
        raise ValueError('3F e 3G nao possuem o mesmo universo. Abortado sem misturar execucoes.')
    for phase, tab in tables.items():
        if tab is not None and set(tab) != set(base):
            raise ValueError(f'{phase} possui universo diferente de 3F. Abortado para nao misturar execucoes.')
    rows = []
    for symbol, f in base.items():
        g = tables['3g'][symbol]
        h = (tables['3h'] or {}).get(symbol, {})
        i = (tables['3i'] or {}).get(symbol, {})
        j = (tables['3j'] or {}).get(symbol, {})
        k = (tables['3k'] or {}).get(symbol, {})
        rg = num(f.get('revenue_growth_30d'))
        pg = num(f.get('price_change_30d'))
        diff = num(f.get('revenue_minus_price_pp'))
        status = f.get('comparison_status','')
        comparable = status == 'ALIGNED_60D_DESCRIPTIVE' and all(x is not None for x in (rg,pg,diff))
        if comparable and rg > 0 and diff > 0:
            diagnostic = 'RECEITA_CRESCENTE_PRECO_ATRASADO_DESCRITIVO'
            priority = 1
        elif comparable and rg <= 0:
            diagnostic = 'RECEITA_EM_QUEDA_NAO_E_MELHORIA_ECONOMICA'
            priority = 3
        elif comparable:
            diagnostic = 'SEM_DIVERGENCIA_FAVORAVEL_COM_RECEITA_CRESCENTE'
            priority = 3
        else:
            diagnostic = 'DADOS_INSUFICIENTES_PARA_COMPARACAO'
            priority = 2
        # Fail closed: no upstream phase independently established 30d holder capture.
        # Even a positive upstream boolean cannot bypass validation of transaction-level evidence.
        capture = 'NAO_COMPROVADA_INDEPENDENTEMENTE'
        if symbol in ('BEAM','NEXO'):
            gap = 'Validar identidade inequivoca do ativo/protocolo e series historicas; depois mecanismo de captura.'
        elif symbol == 'XAUT':
            gap = 'Validar lastro, direito de resgate, custos e risco de contraparte; modelo nao comparavel a receita de protocolo.'
        elif symbol in ('ETH','AVAX'):
            gap = 'Fonte agregada verificavel: queima TOTAL 30d, emissao 30d e reconciliacao da oferta no mesmo UTC; amostras nao bastam.'
        elif symbol == 'INJ':
            gap = 'Hashes de Community BuyBack liquidados, INJ queimado e emissao no mesmo UTC; reconciliar oferta.'
        else:
            gap = 'Validar contrato/mecanismo e evento de captura mensuravel, com diluicao no mesmo periodo.'
        rows.append({
            'symbol':symbol,
            'opportunity_score_original':first(f,'opportunity_score_original'),
            'fundamental_gate_original':first(f,'fundamental_gate_original'),
            'revenue_gate_original':first(f,'revenue_gate_original'),
            'dilution_gate_original':first(f,'dilution_gate_original'),
            'holder_value_gate_original':first(f,'holder_value_gate_original'),
            'revenue_growth_30d':rg,
            'price_change_30d':pg,
            'revenue_minus_price_pp':diff,
            'comparison_status':status,
            'phase3g_economic_status':g.get('economic_status',''),
            'phase3h_chain_evidence_status':h.get('chain_evidence_status','NAO_DISPONIVEL'),
            'phase3i_status':first(i,'historical_status','chain_evidence_status','status'),
            'phase3j_source_feasibility':j.get('source_feasibility','NAO_DISPONIVEL'),
            'phase3k_status':first(k,'chain_evidence_status','status','burn_status'),
            'mechanism_to_verify':MECHANISMS.get(symbol,'Mecanismo nao validado'),
            'independent_holder_capture_30d':capture,
            'net_supply_change_30d_verified':False,
            'investment_approval':'NAO_APROVADO_EVIDENCIA_INSUFICIENTE',
            'economic_diagnostic':diagnostic,
            'investigation_priority':priority,
            'critical_missing_evidence':gap,
            'original_gates_modified':False,
        })
    rows.sort(key=lambda r:(r['investigation_priority'], -(r['revenue_minus_price_pp'] if r['revenue_minus_price_pp'] is not None else -9999),r['symbol']))
    out = Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    csvpath = out/'fase3l_matriz_decisao_todos_elegiveis.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={
        'audit_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'READ_ONLY_CONSOLIDATION_NO_NETWORK_NO_GATE_CHANGE',
        'eligible_count':len(rows),
        'aligned_comparisons':sum(r['comparison_status']=='ALIGNED_60D_DESCRIPTIVE' for r in rows),
        'positive_revenue_positive_divergence':sum(r['investigation_priority']==1 for r in rows),
        'independent_30d_holder_capture_proven':0,
        'investment_approvals':0,
        'inputs':{k:{'path':v,'available':tables[k] is not None} for k,v in sources.items()},
        'caveats':['Consolidacao nao verifica novas transacoes ou dados de mercado.',
                   'Receita de protocolo nao equivale a captura por holders.',
                   'Queimas amostradas nao equivalem a queima total ou reducao liquida.',
                   'Diferenca de percentuais nao e valuation nem recomendacao de compra.',
                   'Nenhum gate, score ou classificacao do motor foi alterado.'],
    }
    (out/'fase3l_resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FASE 3L | CONSOLIDACAO DOS ELEGIVEIS | SEM REDE',flush=True)
    for r in rows:
        d=r['revenue_minus_price_pp']
        print(f"{r['symbol']:6s} | score={r['opportunity_score_original']} | divergencia={f'{d:+.2f}pp' if d is not None else 'N/D'} | {r['economic_diagnostic']} | CAPTURA NAO COMPROVADA",flush=True)
    print('RESUMO:',json.dumps(summary,ensure_ascii=False),flush=True)
    print('ARQUIVOS:',csvpath,out/'fase3l_resumo.json',flush=True)

if __name__=='__main__':
    main()
