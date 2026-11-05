#!/usr/bin/env python3
"""Fase 3G — auditoria incremental de captura economica para TODOS os elegiveis.

Nao executa motor, nao altera gates, nao presume que receita de protocolo seja
rendimento de holders. Usa saida da Fase 3F e consulta series publicas distintas
(dailyFees, dailyRevenue, dailyHoldersRevenue) da DefiLlama, sem duplicar 3F.
Um valor de dailyHoldersRevenue NAO constitui prova on-chain de captura.

Requisitos: Python 3.10+, pandas, requests.
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

SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
TYPES = ('dailyFees', 'dailyRevenue', 'dailyHoldersRevenue')

def clean(v):
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v).strip()

def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def fetch(session, url, params, timeout, retries):
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, allow_redirects=False)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(8, 2 ** attempt))
                continue
            if resp.status_code != 200:
                return None, f'HTTP_{resp.status_code}', None
            if len(resp.content) > 12_000_000:
                return None, 'TOO_LARGE', None
            payload = resp.json()
            if not isinstance(payload, dict):
                return None, 'INVALID_ROOT', None
            return payload, 'OK', resp.content
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                return None, type(exc).__name__, None
            time.sleep(min(8, 2 ** attempt))
    return None, 'UNKNOWN', None

def daily(payload):
    result = {}
    for point in payload.get('totalDataChart', []) if isinstance(payload, dict) else []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        t, val = num(point[0]), num(point[1])
        if t is None or val is None or t <= 0 or val < 0:
            continue
        day = datetime.fromtimestamp(t, timezone.utc).date()
        if day not in result or t >= result[day][0]:
            result[day] = (t, val)
    return {day: val for day, (_, val) in result.items()}

def evaluate(series):
    revenue, holder, fees = (series.get(t, {}) for t in ('dailyRevenue', 'dailyHoldersRevenue', 'dailyFees'))
    out = {'capture_independently_proven': False, 'capture_proof_status': 'NOT_VERIFIED_ONCHAIN',
           'holder_series_is_transaction_proof': False}
    if not revenue:
        out['economic_status'] = 'REVENUE_SERIES_MISSING'
        return out
    end = min(max(revenue), datetime.now(timezone.utc).date() - timedelta(days=1))
    dates = [end - timedelta(days=i) for i in range(29, -1, -1)]
    out['period_start_utc'], out['period_end_utc'] = str(dates[0]), str(dates[-1])
    for typ, points, prefix in [('dailyRevenue', revenue, 'revenue'),
                                ('dailyHoldersRevenue', holder, 'holders'),
                                ('dailyFees', fees, 'fees')]:
        available = all(d in points for d in dates)
        out[prefix + '_complete_30d'] = available
        out[prefix + '_sum_30d_usd'] = round(sum(points[d] for d in dates), 8) if available else None
    if not out['revenue_complete_30d']:
        out['economic_status'] = 'REVENUE_30D_INCOMPLETE'
        return out
    if not out['holders_complete_30d']:
        out['economic_status'] = 'HOLDER_SERIES_MISSING_OR_INCOMPLETE'
        return out
    rev = out['revenue_sum_30d_usd']
    hol = out['holders_sum_30d_usd']
    out['reported_holder_to_revenue_ratio'] = round(hol / rev, 8) if rev > 0 else None
    identical = all(abs(holder[d] - revenue[d]) <= max(1e-7, abs(revenue[d]) * 1e-9) for d in dates)
    out['holder_revenue_identical_to_revenue_daily'] = identical
    if identical:
        out['economic_status'] = 'AGGREGATOR_SERIES_IDENTICAL_ATTRIBUTION_UNVERIFIED'
    elif hol > 0:
        out['economic_status'] = 'REPORTED_HOLDER_REVENUE_NEEDS_TOKEN_AND_ONCHAIN_PROOF'
    else:
        out['economic_status'] = 'NO_POSITIVE_REPORTED_HOLDER_REVENUE'
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase3f', default='output/fase3f_multiactivos/fase3f_comparacao_todos_elegiveis.csv')
    ap.add_argument('--output-dir', default='output/fase3g_captura_multiactivos')
    ap.add_argument('--timeout', type=int, default=18)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--pause', type=float, default=0.2)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    source = Path(args.phase3f)
    if not source.is_file():
        ap.error(f'Fase 3F ausente: {source}. Nao repetir o motor; executar somente 3F se necessario.')
    df = pd.read_csv(source, low_memory=False)
    required = {'symbol', 'coin_id', 'protocol_slug_revenue', 'protocol_slug_holder',
                'opportunity_score_original', 'comparison_status'}
    if not required.issubset(df.columns):
        ap.error('Colunas ausentes: ' + ', '.join(sorted(required - set(df.columns))))
    out = Path(args.output_dir)
    cache = out / 'raw'
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'CryptoOpportunityEngine-ReadOnly/3G', 'Accept': 'application/json'})
    rows = []
    print(f'FASE 3G | TODOS ELEGIVEIS DA 3F: {len(df)} | READ ONLY', flush=True)
    for idx, (_, item) in enumerate(df.iterrows(), 1):
        symbol = clean(item.get('symbol'))
        revslug = clean(item.get('protocol_slug_revenue'))
        holslug = clean(item.get('protocol_slug_holder'))
        record = {'symbol': symbol, 'coin_id': clean(item.get('coin_id')),
                  'opportunity_score_original': num(item.get('opportunity_score_original')),
                  'phase3f_comparison_status': clean(item.get('comparison_status')),
                  'phase3f_revenue_minus_price_pp': num(item.get('revenue_minus_price_pp')),
                  'protocol_slug_revenue': revslug, 'protocol_slug_holder': holslug,
                  'capture_independently_proven': False,
                  'capture_proof_status': 'NOT_VERIFIED_ONCHAIN',
                  'original_gates_modified': False}
        if not SAFE.fullmatch(revslug) or not SAFE.fullmatch(holslug):
            record['economic_status'] = 'MISSING_OR_UNSAFE_PROTOCOL_SLUG'
            rows.append(record)
            print(f'[{idx}/{len(df)}] {symbol}: {record["economic_status"]}', flush=True)
            continue
        if revslug != holslug:
            record['economic_status'] = 'REVENUE_HOLDER_PROTOCOL_IDENTITY_MISMATCH'
            rows.append(record)
            print(f'[{idx}/{len(df)}] {symbol}: {record["economic_status"]}', flush=True)
            continue
        series = {}
        for typ in TYPES:
            key = hashlib.sha256(f'{revslug}|{typ}'.encode()).hexdigest()[:20]
            file = cache / f'{key}.json'
            payload, status = None, 'CACHE_MISSING'
            if file.is_file():
                try:
                    payload = json.loads(file.read_text(encoding='utf-8'))
                    status = 'CACHE_OK'
                except (ValueError, OSError):
                    status = 'CACHE_INVALID'
            url = 'https://api.llama.fi/summary/fees/' + quote(revslug)
            if payload is None and not args.offline:
                payload, status, raw = fetch(session, url, {'dataType': typ}, args.timeout, args.retries)
                if raw is not None:
                    file.write_bytes(raw)
                time.sleep(max(0, args.pause))
            record[typ + '_status'] = status
            record[typ + '_url'] = url + '?dataType=' + typ
            record[typ + '_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else ''
            series[typ] = daily(payload)
        record.update(evaluate(series))
        record['identity_status'] = 'AGGREGATOR_MAPPING_NOT_INDEPENDENTLY_VERIFIED'
        rows.append(record)
        print(f'[{idx}/{len(df)}] {symbol}: {record["economic_status"]}', flush=True)
    report = pd.DataFrame(rows)
    if report.empty:
        report = pd.DataFrame(columns=['symbol', 'economic_status', 'capture_independently_proven'])
    csv = out / 'fase3g_captura_todos_elegiveis.csv'
    report.to_csv(csv, index=False, encoding='utf-8-sig')
    summary = {'audit_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_phase3f': str(source), 'total_elegiveis': len(df),
               'analisados': len(report), 'captura_independente_comprovada': 0,
               'status_counts': report['economic_status'].value_counts(dropna=False).to_dict(),
               'scope': 'Incremental: holdersRevenue/fees/revenue for all phase3F candidates; no repeat of price comparison.',
               'limitation': 'DefiLlama holdersRevenue is aggregator reporting, not independent onchain proof of token holder capture. Mechanism-specific onchain proof remains pending.',
               'gates_originals_modified': False}
    (out / 'fase3g_resumo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    cols = [x for x in ('symbol', 'opportunity_score_original', 'phase3f_revenue_minus_price_pp',
                       'revenue_sum_30d_usd', 'holders_sum_30d_usd',
                       'holder_revenue_identical_to_revenue_daily', 'economic_status') if x in report.columns]
    print('\nRESULTADOS (NAO SAO APROVACOES):\n' + report[cols].to_string(index=False), flush=True)
    print('\nRESUMO: ' + json.dumps(summary, ensure_ascii=False), flush=True)
    print('\nARQUIVOS:', csv, '|', out / 'fase3g_resumo.json', flush=True)

if __name__ == '__main__':
    main()
