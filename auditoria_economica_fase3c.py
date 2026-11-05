#!/usr/bin/env python3
"""Fase 3C: coleta read-only de series economicas publicas DefiLlama.

Nao modifica motor, scores, gates, ou classificacoes. Dados agregados de
holders revenue NAO sao prova independente de fluxo ao token especifico.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

SYMBOLS = ('INJ', 'AVAX', 'XPL', 'ETH')
DATA_TYPES = ('dailyFees', 'dailyRevenue', 'dailyHoldersRevenue')
BASE = 'https://api.llama.fi/summary/fees/'


def num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def clean(value):
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, 'item') else value


def get_json(session, url, datatype, timeout, retries, pause):
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params={'dataType': datatype}, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(12, 2 ** attempt + pause))
                continue
            resp.raise_for_status()
            obj = resp.json()
            if not isinstance(obj, dict):
                raise ValueError('JSON root is not an object')
            return obj, None, resp.url
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                return None, f'{type(exc).__name__}: {str(exc)[:200]}', url
            time.sleep(min(12, 2 ** attempt + pause))
    return None, 'UNEXPECTED', url


def daily_chart(payload):
    """Accept only explicitly dated numeric [timestamp, value] chart points."""
    if not isinstance(payload, dict):
        return {}
    result = {}
    for item in payload.get('totalDataChart') or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts, value = num(item[0]), num(item[1])
        if ts is None or value is None or ts <= 0 or value < 0:
            continue
        day = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d')
        result[day] = value
    return result


def window_compare(chart, end_day):
    """30d current vs preceding 30d, ending at latest available day."""
    dates = sorted(d for d in chart if d <= end_day)
    if len(dates) < 60:
        return None, None, None, 'LESS_THAN_60_DAILY_OBSERVATIONS'
    from datetime import date, timedelta
    end = date.fromisoformat(end_day)
    current_days = [(end - timedelta(days=i)).isoformat() for i in range(30)]
    prior_days = [(end - timedelta(days=i)).isoformat() for i in range(30, 60)]
    if not all(d in chart for d in current_days + prior_days):
        return None, None, None, 'NON_CONTIGUOUS_60_DAYS'
    current = sum(chart[d] for d in current_days)
    prior = sum(chart[d] for d in prior_days)
    growth = current / prior - 1 if prior > 0 else None
    return current, prior, growth, 'COMPARABLE_30D_WINDOWS' if prior > 0 else 'ZERO_PRIOR_BASE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/crypto_diagnostic_all.csv')
    parser.add_argument('--output-dir', default='output/fase3c_auditoria')
    parser.add_argument('--symbols', default=','.join(SYMBOLS))
    parser.add_argument('--timeout', type=int, default=20)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--pause', type=float, default=0.6)
    parser.add_argument('--offline', action='store_true', help='Use somente cache ja coletado')
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    df = pd.read_csv(source, low_memory=False)
    needed = {'symbol', 'coin_id', 'holder_value_protocol_slug', 'revenue_data_protocol_slug'}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    subset = df[df.symbol.astype(str).str.upper().isin(symbols)].copy()
    if subset.empty:
        raise ValueError('No selected assets in input')
    absent = sorted(set(symbols) - set(subset.symbol.astype(str).str.upper()))
    if absent:
        raise ValueError('Symbols missing: ' + ', '.join(absent))
    if subset.symbol.astype(str).str.upper().duplicated().any():
        raise ValueError('Ambiguous duplicate symbol: select by coin_id before audit')

    output = Path(args.output_dir)
    cache = output / 'raw_defillama'
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'CryptoOpportunityEngine-ReadOnlyResearch/3C', 'Accept': 'application/json'})
    records = []
    for _, item in subset.iterrows():
        sym = str(item['symbol']).upper()
        slug_holder = clean(item.get('holder_value_protocol_slug'))
        slug_revenue = clean(item.get('revenue_data_protocol_slug'))
        slug = str(slug_holder or slug_revenue or '').strip()
        record = {
            'symbol': sym, 'coin_id': clean(item.get('coin_id')),
            'defillama_slug': slug or None,
            'slug_consistency': 'MATCH' if slug_holder and slug_revenue and str(slug_holder) == str(slug_revenue) else 'NOT_ESTABLISHED',
            'original_opportunity_score': clean(item.get('opportunity_score')),
            'original_holder_value_gate': clean(item.get('holder_value_gate')),
            'original_hard_block': clean(item.get('hard_block')),
            'engine_annualized_revenue': clean(item.get('annualized_revenue')),
            'engine_holder_revenue': clean(item.get('holder_revenue')),
            'engine_holder_capture_ratio': clean(item.get('holder_capture_ratio')),
            'engine_revenue_growth': clean(item.get('revenue_growth')),
            'engine_return_90d': clean(item.get('return_90d')),
            'capture_independently_proven': False,
            'gate_modified': False,
            'source_identity_status': 'NOT_VERIFIED',
            'methodology_url': None,
            'methodology_text': None,
            'evidence_limit': 'AGGREGATOR_DATA_NOT_ONCHAIN_PROOF_OF_TOKEN_CAPTURE',
        }
        if not slug or '/' in slug or '\\' in slug:
            record['collection_status'] = 'NO_SAFE_PROTOCOL_SLUG'
            records.append(record)
            continue
        url = BASE + quote(slug, safe='')
        source_names = []
        success = 0
        for kind in DATA_TYPES:
            cache_file = cache / f'{sym}_{kind}.json'
            payload, err, final_url = None, None, url + '?dataType=' + kind
            if args.offline:
                if cache_file.exists():
                    try:
                        payload = json.loads(cache_file.read_text(encoding='utf-8'))
                    except (ValueError, OSError) as exc:
                        err = 'CACHE_INVALID: ' + str(exc)[:150]
                else:
                    err = 'OFFLINE_CACHE_MISSING'
            else:
                payload, err, final_url = get_json(session, url, kind, args.timeout, args.retries, args.pause)
                if payload is not None:
                    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                time.sleep(args.pause)
            prefix = kind.replace('daily', '').lower()
            record[prefix + '_source_url'] = final_url
            record[prefix + '_fetch_status'] = 'OK' if payload is not None else (err or 'FAILED')
            record[prefix + '_raw_cache'] = str(cache_file) if payload is not None else None
            record[prefix + '_raw_sha256'] = hashlib.sha256(cache_file.read_bytes()).hexdigest() if payload is not None and cache_file.exists() else None
            if payload is None:
                continue
            success += 1
            source_names.append(str(payload.get('name') or ''))
            if kind == 'dailyHoldersRevenue':
                record['methodology_url'] = payload.get('methodologyURL')
                method = payload.get('methodology')
                record['methodology_text'] = json.dumps(method, ensure_ascii=False)[:2000] if method is not None else None
            record[prefix + '_protocol_name'] = payload.get('name')
            record[prefix + '_gecko_id'] = payload.get('gecko_id')
            record[prefix + '_latest_fetch_ok'] = payload.get('latestFetchIsOk')
            record[prefix + '_reported_30d'] = num(payload.get('total30d'))
            chart = daily_chart(payload)
            record[prefix + '_daily_points'] = len(chart)
            record[prefix + '_last_day'] = max(chart) if chart else None
            if chart:
                a, b, growth, state = window_compare(chart, max(chart))
                record[prefix + '_last30_sum'] = a
                record[prefix + '_previous30_sum'] = b
                record[prefix + '_30d_vs_previous30d'] = growth
                record[prefix + '_comparison_status'] = state
            else:
                record[prefix + '_comparison_status'] = 'NO_DATED_HISTORY'
        record['collection_status'] = f'{success}/3_TYPES_FETCHED'
        record['source_names'] = '|'.join(source_names)
        gecko_ids = [record.get(k + '_gecko_id') for k in ('fees', 'revenue', 'holdersrevenue')]
        valid_ids = [str(x) for x in gecko_ids if x]
        record['source_identity_status'] = ('GECKO_ID_MATCH_AGGREGATOR_ONLY' if valid_ids and all(x == str(record['coin_id']) for x in valid_ids)
                                            else 'GECKO_ID_CONFLICT_OR_MISSING')
        # Deliberately no automatic confirmation even when aggregator gecko_id matches.
        records.append(record)
        print(f'{sym}: {record["collection_status"]} | {record["source_identity_status"]}', flush=True)

    result = pd.DataFrame(records)
    csv_path = output / 'fase3c_economic_evidence.csv'
    json_path = output / 'fase3c_economic_evidence.json'
    result.to_csv(csv_path, index=False, encoding='utf-8-sig')
    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_input': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'read_only': True, 'engine_gates_modified': False, 'engine_scores_modified': False,
        'disclaimer': 'DefiLlama classifications are aggregator assertions, not independently verified token distributions or burns.',
        'records': records,
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('\n=== FASE 3C: EVIDENCIAS ECONOMICAS (SEM PROMOCAO DE GATES) ===')
    cols = ['symbol', 'collection_status', 'source_identity_status', 'revenue_30d_vs_previous30d',
            'holdersrevenue_30d_vs_previous30d', 'holdersrevenue_last30_sum', 'capture_independently_proven']
    print(result[cols].to_string(index=False))
    print('\nCSV:', csv_path.resolve(), '\nJSON:', json_path.resolve())
    print('Dados novos sao apenas evidencias externas rastreaveis; nenhuma aprovacao automatica.')


if __name__ == '__main__':
    main()
