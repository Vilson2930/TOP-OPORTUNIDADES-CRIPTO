#!/usr/bin/env python3
"""Fase 3D: INJ - auditoria read-only de mecanismo, dados on-chain e divergencia.

Nenhuma consulta de leitura comprova por si so um burn historico. Nao altera
scores, gates, rankings, sinais ou arquivos de entrada. Falha de forma fechada.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LCD = 'https://sentry.lcd.injective.network'
CHAIN_ENDPOINTS = {
    'auction_params': '/injective/auction/v1beta1/params',
    'auction_current': '/injective/auction/v1beta1/current_auction_basket',
    'auction_round': '/injective/auction/v1beta1/auction_round',
    'mint_inflation': '/injective/mint/v1beta1/inflation',
    'mint_params': '/injective/mint/v1beta1/params',
    'bank_supply_inj': '/cosmos/bank/v1beta1/supply/inj',
    'bank_total_supply': '/cosmos/bank/v1beta1/supply',
    'node_info': '/cosmos/base/tendermint/v1beta1/node_info',
}
DOCUMENTARY_SOURCE = 'https://www.sec.gov/Archives/edgar/data/2073616/000199937126024117/inj-s1a_110226.htm'


def scalar(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x.item() if hasattr(x, 'item') else x


def number(x):
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)


def inj_row(df):
    if 'symbol' not in df.columns:
        raise ValueError('Coluna symbol ausente')
    subset = df[df['symbol'].astype(str).str.upper().eq('INJ')]
    if len(subset) != 1:
        raise ValueError(f'Esperado exatamente 1 INJ, encontrados {len(subset)}')
    return subset.iloc[0]


def get_with_retries(session, url, timeout, retries):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=False)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(10, 2 ** attempt))
                continue
            if r.status_code != 200:
                return None, None, f'HTTP_{r.status_code}'
            if len(r.content) > 3_000_000:
                return None, None, 'RESPONSE_TOO_LARGE'
            obj = r.json()
            if not isinstance(obj, dict):
                return None, None, 'NON_OBJECT_JSON'
            return obj, r.content, 'OK'
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                return None, None, f'{type(e).__name__}: {str(e)[:160]}'
            time.sleep(min(10, 2 ** attempt))
    return None, None, 'UNKNOWN_ERROR'


def find_nested(data, *keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]
    return None


def run(args):
    df = read_csv(Path(args.input))
    inj = inj_row(df)
    out = Path(args.output_dir)
    raw_dir = out / 'raw_chain'
    raw_dir.mkdir(parents=True, exist_ok=True)
    audit_time = datetime.now(timezone.utc).isoformat()
    result = {
        'symbol': 'INJ', 'audit_at_utc': audit_time,
        'audit_mode': 'READ_ONLY_NO_GATE_PROMOTION',
        'documentary_mechanism': 'COMMUNITY_BUYBACK_REPLACED_BURN_AUCTION_OCT_2025',
        'documentary_source_url': DOCUMENTARY_SOURCE,
        'documentary_source_type': 'SEC_FILED_DISCLOSURE_NOT_TRANSACTION_PROOF',
        'mechanism_documented': True,
        'historical_burn_transactions_independently_verified': False,
        'net_token_supply_reduction_proven': False,
        'capture_independently_proven': False,
        'revenue_price_divergence_confirmed': False,
        'holder_value_gate': scalar(inj.get('holder_value_gate')),
        'opportunity_score_original': number(inj.get('opportunity_score')),
        'annualized_revenue_original': number(inj.get('annualized_revenue')),
        'holder_revenue_original': number(inj.get('holder_revenue')),
        'holder_capture_ratio_original': number(inj.get('holder_capture_ratio')),
        'revenue_growth_original': number(inj.get('revenue_growth')),
        'return_30d_original': number(inj.get('return_30d')),
        'return_90d_original': number(inj.get('return_90d')),
        'fundamental_price_divergence_score_original': number(inj.get('fundamental_price_divergence_score')),
        'revenue_price_divergence_score_original': number(inj.get('revenue_price_divergence_score')),
    }
    if result['holder_capture_ratio_original'] == 1.0:
        result['capture_attribution_warning'] = 'RATIO_1_NOT_INDEPENDENT_DISTRIBUTION_PROOF'
    else:
        result['capture_attribution_warning'] = 'REQUIRES_INDEPENDENT_TRANSACTION_EVIDENCE'

    evidence = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'CryptoOpportunityEngine-ReadOnlyAudit/3D (research)', 'Accept': 'application/json'})
    for name, endpoint in CHAIN_ENDPOINTS.items():
        url = LCD + endpoint
        if args.offline:
            obj, body, status = None, None, 'OFFLINE_NOT_QUERIED'
        else:
            obj, body, status = get_with_retries(session, url, args.timeout, args.retries)
        item = {'query': name, 'url': url, 'status': status, 'sha256': None,
                'chain_response_available': obj is not None, 'verified_historical_burn': False}
        if obj is not None:
            file_path = raw_dir / (name + '.json')
            file_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
            item['sha256'] = sha256(body)
            item['raw_file'] = str(file_path)
            if name == 'bank_supply_inj':
                supply = find_nested(obj, 'amount')
                if isinstance(supply, dict):
                    item['supply_denom'] = supply.get('denom')
                    item['supply_amount_raw'] = supply.get('amount')
            if name == 'mint_inflation':
                item['reported_inflation_raw'] = find_nested(obj, 'inflation')
            if name == 'auction_round':
                item['round_data_present'] = bool(obj)
        evidence.append(item)
        print(f'{name}: {status}', flush=True)
        if not args.offline:
            time.sleep(args.pause)

    # Historical economic observation from 3C is separate from chain evidence.
    phase3c_path = Path(args.phase3c)
    if phase3c_path.is_file():
        phase3c = inj_row(read_csv(phase3c_path))
        for key in ('collection_status', 'source_identity_status',
                    'revenue_30d_vs_previous30d', 'holdersrevenue_30d_vs_previous30d',
                    'holdersrevenue_last30_sum'):
            result['phase3c_' + key] = scalar(phase3c.get(key))
        result['phase3c_source'] = str(phase3c_path)
    else:
        result['phase3c_source'] = 'NOT_AVAILABLE'

    # A return_30d from snapshot and 3C's moving 30-day window are NOT assumed
    # to share identical start/end dates. No spurious divergence is computed.
    result['historical_comparison_status'] = 'NOT_COMPARABLE_WITHOUT_ALIGNED_PRICE_AND_REVENUE_DATES'
    result['chain_queries_ok'] = sum(x['status'] == 'OK' for x in evidence)
    result['chain_queries_total'] = len(evidence)
    result['evidence_gap'] = (
        'Need on-chain settled Community BuyBack transactions with event/tx hashes, '
        'INJ amounts burned, and issuance over identical observation periods; '
        'need price and revenue daily series with identical UTC windows.'
    )
    result['conclusion'] = 'DOCUMENTED_MECHANISM_BUT_CAPTURE_AND_NET_BURN_NOT_YET_VERIFIED'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fase3d_inj_evidence.json').write_text(
        json.dumps({'summary': result, 'chain_queries': evidence}, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')
    pd.DataFrame([result]).to_csv(out / 'fase3d_inj_summary.csv', index=False)
    pd.DataFrame(evidence).to_csv(out / 'fase3d_chain_queries.csv', index=False)
    print('\n=== FASE 3D - INJ ===')
    print(f"Mecanismo documental: {result['documentary_mechanism']}")
    print(f"Consultas de blockchain OK: {result['chain_queries_ok']}/{result['chain_queries_total']}")
    print('Queimas historicas comprovadas nesta etapa: NAO')
    print('Reducao liquida da oferta comprovada: NAO')
    print('Captura economica independente comprovada: NAO')
    print('Divergencia preco/receita com janelas alinhadas: NAO')
    print('Scores/gates alterados: NAO')
    print('CSV:', out / 'fase3d_inj_summary.csv')
    print('JSON:', out / 'fase3d_inj_evidence.json')


def main():
    p = argparse.ArgumentParser(description='Fase 3D INJ: auditoria documental e consultas read-only on-chain')
    p.add_argument('--input', default='output/crypto_diagnostic_all.csv')
    p.add_argument('--phase3c', default='output/fase3c_auditoria/fase3c_economic_evidence.csv')
    p.add_argument('--output-dir', default='output/fase3d_inj')
    p.add_argument('--timeout', type=float, default=12)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--pause', type=float, default=0.3)
    p.add_argument('--offline', action='store_true', help='Valida fluxo sem acessar a rede')
    args = p.parse_args()
    if args.timeout <= 0 or not (0 <= args.retries <= 5) or args.pause < 0:
        p.error('timeout > 0; retries 0..5; pause >= 0')
    run(args)


if __name__ == '__main__':
    main()
