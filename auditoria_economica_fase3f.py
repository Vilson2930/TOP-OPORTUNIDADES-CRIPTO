#!/usr/bin/env python3
"""Fase 3F: triagem economica MULTIATIVOS, somente leitura, fail-closed.

Seleciona TODOS os ativos que passaram fundamental_gate, revenue_gate e
(dilution_gate, quando presente). Nao altera gates/scores/classificacoes.
Compara receita diaria DefiLlama e preco diario DefiLlama em 60 datas UTC
consecutivas. NAO infere captura de valor a partir de revenue/holdersRevenue.
Requer Python 3.10+, pandas, requests.
"""
import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

BASE_REVENUE = 'https://api.llama.fi/summary/fees/'
BASE_PRICE = 'https://coins.llama.fi/chart/coingecko:'
SAFE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')

def clean(x):
    if x is None:
        return None
    try:
        if bool(pd.isna(x)):
            return None
    except (ValueError, TypeError):
        pass
    return str(x).strip()

def number(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def is_true(x):
    return str(x).strip().lower() in ('true', '1', 'yes', 'sim')

def get(session, url, params, timeout, retries):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout, allow_redirects=False)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(8, 2 ** attempt))
                continue
            if r.status_code != 200:
                return None, f'HTTP_{r.status_code}', None, r.url
            if len(r.content) > 12_000_000:
                return None, 'RESPONSE_TOO_LARGE', None, r.url
            obj = r.json()
            if not isinstance(obj, dict):
                return None, 'INVALID_JSON_ROOT', None, r.url
            return obj, 'OK', r.content, r.url
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                return None, type(exc).__name__, None, url
            time.sleep(min(8, 2 ** attempt))
    return None, 'UNKNOWN', None, url

def chart(payload, kind, coin_id=None):
    if not isinstance(payload, dict):
        return {}
    if kind == 'revenue':
        raw = payload.get('totalDataChart', [])
        pairs = [(p[0], p[1]) for p in raw if isinstance(p, (list, tuple)) and len(p) >= 2]
    else:
        coins = payload.get('coins') or {}
        coin = coins.get('coingecko:' + coin_id, {}) if isinstance(coins, dict) else {}
        raw = coin.get('prices', []) if isinstance(coin, dict) else []
        pairs = [(p.get('timestamp'), p.get('price')) for p in raw if isinstance(p, dict)]
    output = {}
    for ts, val in pairs:
        t, v = number(ts), number(val)
        if t is None or v is None or t <= 0 or v < 0 or (kind == 'price' and v <= 0):
            continue
        day = datetime.fromtimestamp(t, timezone.utc).date()
        # Daily points: use the last available observation of each UTC day.
        if day not in output or t >= output[day][0]:
            output[day] = (t, v)
    return {day: value for day, (_, value) in output.items()}

def compare(rev, price):
    common = set(rev) & set(price)
    if not common:
        return {'comparison_status': 'NO_COMMON_DATES'}
    # Last complete day shared by both series, never today's partial data.
    end = min(max(common), datetime.now(timezone.utc).date() - timedelta(days=1))
    days = [end - timedelta(days=i) for i in range(59, -1, -1)]
    if not all(d in rev and d in price for d in days):
        return {'comparison_status': 'INCOMPLETE_60_CONSECUTIVE_DAYS', 'latest_common_day': str(end),
                'complete_days': sum(d in rev and d in price for d in days)}
    previous, recent = days[:30], days[30:]
    r0, r1 = sum(rev[d] for d in previous), sum(rev[d] for d in recent)
    p0, p1 = price[previous[-1]], price[recent[-1]]
    if r0 <= 0 or p0 <= 0:
        return {'comparison_status': 'INVALID_BASELINE'}
    rg, pg = r1 / r0 - 1, p1 / p0 - 1
    return {'comparison_status': 'ALIGNED_60D_DESCRIPTIVE', 'start_utc': str(days[0]),
            'previous_end_utc': str(previous[-1]), 'recent_end_utc': str(recent[-1]),
            'revenue_previous30_usd': r0, 'revenue_last30_usd': r1,
            'revenue_growth_30d': rg, 'price_previous_close_usd': p0,
            'price_recent_close_usd': p1, 'price_change_30d': pg,
            'revenue_minus_price_pp': 100 * (rg - pg)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='output/crypto_diagnostic_all.csv')
    ap.add_argument('--output-dir', default='output/fase3f_multiactivos')
    ap.add_argument('--timeout', type=int, default=15)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--pause', type=float, default=0.2)
    ap.add_argument('--offline', action='store_true', help='Somente arquivos previamente coletados')
    args = ap.parse_args()
    source = Path(args.input)
    if not source.is_file():
        ap.error(f'Arquivo nao encontrado: {source}')
    df = pd.read_csv(source, low_memory=False)
    required = ['symbol', 'coin_id', 'fundamental_gate', 'revenue_gate',
                'revenue_data_protocol_slug', 'holder_value_protocol_slug']
    missing = [c for c in required if c not in df.columns]
    if missing:
        ap.error('Colunas obrigatorias ausentes: ' + ', '.join(missing))
    gates = ['fundamental_gate', 'revenue_gate']
    if 'dilution_gate' in df.columns:
        gates.append('dilution_gate')
    mask = pd.Series(True, index=df.index)
    for g in gates:
        mask &= df[g].map(is_true)
    selected = df.loc[mask].copy()
    out = Path(args.output_dir)
    cache = out / 'raw'
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'CryptoOpportunityEngine-ReadOnly/3F', 'Accept': 'application/json'})
    records = []
    print(f'FASE 3F MULTIATIVOS | universo={len(df)} | elegiveis={len(selected)} | gates={gates}', flush=True)
    for idx, (_, item) in enumerate(selected.iterrows(), 1):
        sym = clean(item.get('symbol')) or ''
        coin = clean(item.get('coin_id')) or ''
        slug_rev = clean(item.get('revenue_data_protocol_slug')) or ''
        slug_holder = clean(item.get('holder_value_protocol_slug')) or ''
        record = {'symbol': sym, 'coin_id': coin, 'protocol_slug_revenue': slug_rev,
                  'protocol_slug_holder': slug_holder, 'fundamental_gate_original': clean(item.get('fundamental_gate')),
                  'revenue_gate_original': clean(item.get('revenue_gate')),
                  'dilution_gate_original': clean(item.get('dilution_gate')),
                  'holder_value_gate_original': clean(item.get('holder_value_gate')),
                  'opportunity_score_original': number(item.get('opportunity_score')),
                  'hard_block_original': clean(item.get('hard_block')),
                  'quality_failure_reasons_original': clean(item.get('quality_failure_reasons')),
                  'capture_independently_proven_by_phase3f': False,
                  'capture_proof_status': 'NOT_EVALUATED_ONCHAIN',
                  'gate_modified': False, 'data_identity_verified_independently': False}
        if not SAFE.fullmatch(coin) or not SAFE.fullmatch(slug_rev):
            record['comparison_status'] = 'MISSING_OR_UNSAFE_COIN_ID_OR_REVENUE_SLUG'
            records.append(record)
            continue
        if slug_holder and slug_holder != slug_rev:
            record['identity_warning'] = 'REVENUE_HOLDER_SLUG_MISMATCH'
        else:
            record['identity_warning'] = 'AGGREGATOR_IDENTITY_NOT_INDEPENDENTLY_VERIFIED'
        key = hashlib.sha256((coin + '|' + slug_rev).encode()).hexdigest()[:16]
        rev_file, price_file = cache / (key + '_revenue.json'), cache / (key + '_price.json')
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - 90 * 86400
        jobs = [('revenue', rev_file, BASE_REVENUE + quote(slug_rev), {'dataType': 'dailyRevenue'}),
                ('price', price_file, BASE_PRICE + quote(coin),
                 {'start': start_ts, 'span': 90, 'period': '1d'})]
        payloads = {}
        for kind, path, url, params in jobs:
            payload = None
            status = 'CACHE_MISSING'
            final_url = url
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    status = 'CACHE_OK'
                except (ValueError, OSError):
                    status = 'CACHE_INVALID'
            if payload is None and not args.offline:
                payload, status, raw, final_url = get(session, url, params, args.timeout, args.retries)
                if raw is not None:
                    path.write_bytes(raw)
                time.sleep(max(0, args.pause))
            payloads[kind] = payload
            record[kind + '_api_status'] = status
            record[kind + '_source_url'] = final_url
            record[kind + '_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if payloads.get('revenue') is None or payloads.get('price') is None:
            record['comparison_status'] = 'SOURCE_UNAVAILABLE'
        else:
            record.update(compare(chart(payloads.get('revenue'), 'revenue'),
                                  chart(payloads.get('price'), 'price', coin)))
        records.append(record)
        print(f'[{idx}/{len(selected)}] {sym} ({coin}): {record["comparison_status"]}', flush=True)
    report = pd.DataFrame(records)
    if report.empty:
        report = pd.DataFrame(columns=['symbol', 'coin_id', 'comparison_status'])
    # Ordering is diagnostic only, NOT investment ranking or gate promotion.
    if 'revenue_minus_price_pp' in report.columns:
        report = report.sort_values('revenue_minus_price_pp', ascending=False, na_position='last', kind='stable')
    csv_path = out / 'fase3f_comparacao_todos_elegiveis.csv'
    report.to_csv(csv_path, index=False, encoding='utf-8-sig')
    summary = {'audit_at_utc': datetime.now(timezone.utc).isoformat(),
               'source': str(source), 'universe_count': int(len(df)),
               'selected_count': int(len(selected)), 'selection_gates': gates,
               'compared_count': int((report['comparison_status'] == 'ALIGNED_60D_DESCRIPTIVE').sum()),
               'capture_independently_proven_count': 0,
               'methodology': '30d revenue vs preceding 30d; token price between corresponding period-end closes; UTC contiguous days',
               'limitations': ['Protocol revenue does not prove token holder capture.',
                               'Revenue-minus-price percentage points is descriptive, not a valuation model.',
                               'Phase 3F does not independently verify onchain burns, distributions or token ownership.',
                               'CoinGecko/DefiLlama mapping is aggregator-based; protocol-token identity not independently proven.',
                               'No original gates, scores or ranking modified.']}
    (out / 'fase3f_resumo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nRESUMO:', json.dumps(summary, ensure_ascii=False), flush=True)
    print('\nCOMPARACAO (diagnostica, NAO ranking de compra):', flush=True)
    columns = [c for c in ['symbol', 'opportunity_score_original', 'revenue_growth_30d',
              'price_change_30d', 'revenue_minus_price_pp', 'comparison_status',
              'capture_proof_status'] if c in report.columns]
    print(report[columns].to_string(index=False), flush=True)
    print('\nARQUIVOS:', csv_path, '|', out / 'fase3f_resumo.json', flush=True)

if __name__ == '__main__':
    main()
