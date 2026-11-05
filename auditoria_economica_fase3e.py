#!/usr/bin/env python3
"""Fase 3E - INJ: evidencia on-chain e divergencia em janelas alinhadas.

Somente leitura. Nao altera scores, gates, classificacoes, entradas ou codigo do motor.
Evidencia de burn e atribuicao ao Community BuyBack sao coisas distintas.
Requer: requests, pandas. Python 3.10+.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

LCDS = [
    'https://sentry.lcd.injective.network',
    'https://lcd.injective.network',
    'https://injective-api.polkachu.com',
]
PRICE_URL = 'https://coins.llama.fi/chart/coingecko:injective-protocol'
REVENUE_URL = 'https://api.llama.fi/summary/fees/injective?dataType=dailyRevenue'
# Event queries are discovery only: event naming and completeness vary by node.
EVENTS = ["message.module='auction'", "message.action='/injective.auction.v1beta1.MsgBid'", "burn.amount EXISTS"]
MAX_RESPONSE = 8_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def request(session, url, timeout, retries=1, headers=None):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout, headers=headers or {}, allow_redirects=False)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1 + attempt)
                continue
            if r.status_code != 200:
                return None, None, f'HTTP_{r.status_code}'
            if len(r.content) > MAX_RESPONSE:
                return None, None, 'TOO_LARGE'
            return r.json(), r.content, 'OK'
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                return None, None, f'{type(exc).__name__}:{str(exc)[:100]}'
            time.sleep(1 + attempt)
    return None, None, 'UNKNOWN'


def save_raw(folder, label, raw):
    if raw is None:
        return None
    path = folder / (label + '.json')
    path.write_bytes(raw)
    return str(path)


def get_chain(session, folder, name, path, timeout, rows, headers=None):
    for i, base in enumerate(LCDS):
        url = base + path
        obj, raw, status = request(session, url, timeout, headers=headers)
        raw_path = save_raw(folder, f'{name}_node{i+1}', raw)
        rows.append({'query': name, 'node': base, 'url': url, 'status': status,
                     'sha256': digest(raw) if raw else None, 'raw_file': raw_path,
                     'block_header_requested': (headers or {}).get('x-cosmos-block-height')})
        if status == 'OK' and isinstance(obj, dict):
            return obj, base
    return None, None


def extract_supply(obj):
    if not isinstance(obj, dict):
        return None
    val = obj.get('amount')
    if isinstance(val, dict) and val.get('denom') == 'inj':
        return number(val.get('amount'))
    for row in obj.get('supply', []) if isinstance(obj.get('supply'), list) else []:
        if isinstance(row, dict) and row.get('denom') == 'inj':
            return number(row.get('amount'))
    return None


def latest_height(obj):
    try:
        return int(obj['block']['header']['height'])
    except (KeyError, TypeError, ValueError):
        return None


def chart_points(obj):
    if not isinstance(obj, dict):
        return []
    coins = obj.get('coins', {})
    coin = coins.get('coingecko:injective-protocol', {}) if isinstance(coins, dict) else {}
    values = coin.get('prices', []) if isinstance(coin, dict) else []
    points = []
    for item in values:
        if not isinstance(item, dict):
            continue
        t = number(item.get('timestamp'))
        p = number(item.get('price'))
        if t is not None and p is not None and p > 0:
            points.append((datetime.fromtimestamp(t, timezone.utc).date(), p))
    return sorted(points)


def revenue_points(obj):
    if not isinstance(obj, dict):
        return []
    points = []
    for item in obj.get('totalDataChart', []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            t, v = number(item[0]), number(item[1])
            if t is not None and v is not None and v >= 0:
                points.append((datetime.fromtimestamp(t, timezone.utc).date(), v))
    return sorted(points)


def windows(price, revenue):
    # Same complete UTC dates for both data series; no interpolation of missing days.
    pp = dict(price)
    rr = dict(revenue)
    common = sorted(set(pp) & set(rr))
    if not common:
        return {'status': 'NO_COMMON_DATES'}
    end = min(common[-1], datetime.now(timezone.utc).date() - timedelta(days=1))
    days = [end - timedelta(days=i) for i in range(59, -1, -1)]
    if not all(d in pp and d in rr for d in days):
        return {'status': 'INCOMPLETE_ALIGNED_60D', 'latest_common_utc': str(end),
                'days_present': sum(d in pp and d in rr for d in days)}
    old, recent = days[:30], days[30:]
    old_rev, new_rev = sum(rr[d] for d in old), sum(rr[d] for d in recent)
    # Compare closing price at the end of the two non-overlapping windows.
    p0, p1 = pp[old[-1]], pp[recent[-1]]
    if old_rev <= 0 or p0 <= 0:
        return {'status': 'ZERO_BASELINE', 'old_revenue': old_rev, 'old_price': p0}
    rev_change = new_rev / old_rev - 1
    price_change = p1 / p0 - 1
    return {'status': 'ALIGNED_60D_OBSERVATION', 'start_utc': str(days[0]),
            'mid_utc': str(old[-1]), 'end_utc': str(recent[-1]),
            'revenue_previous30_usd': old_rev, 'revenue_last30_usd': new_rev,
            'revenue_change': rev_change, 'price_previous_close_usd': p0,
            'price_recent_close_usd': p1, 'price_change': price_change,
            'revenue_minus_price_pct_points': 100 * (rev_change - price_change),
            'warning': 'Descriptive divergence only; not evidence of undervaluation or holder capture.'}


def tx_candidates(obj):
    if not isinstance(obj, dict):
        return []
    txs = obj.get('tx_responses', [])
    if not isinstance(txs, list):
        return []
    results = []
    for tx in txs[:25]:
        if not isinstance(tx, dict):
            continue
        events = []
        for log in tx.get('logs', []) or []:
            if isinstance(log, dict):
                events.extend(log.get('events', []) or [])
        events.extend(tx.get('events', []) or [])
        results.append({'txhash': tx.get('txhash'), 'height': tx.get('height'),
                        'code': tx.get('code'), 'timestamp': tx.get('timestamp'),
                        'event_types': sorted({str(e.get('type')) for e in events if isinstance(e, dict)}),
                        'buyback_attribution_verified': False,
                        'burn_amount_verified': False,
                        'warning': 'Candidate only. Requires decoded tx, amount, module identity and settled burn proof.'})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='output/crypto_diagnostic_all.csv')
    ap.add_argument('--phase3c', default='output/fase3c_auditoria/fase3c_economic_evidence.csv')
    ap.add_argument('--output-dir', default='output/fase3e_inj')
    ap.add_argument('--timeout', type=int, default=12)
    ap.add_argument('--height-gap', type=int, default=100000,
                    help='Historical block gap, NOT a fixed number of days')
    args = ap.parse_args()
    source = Path(args.input)
    if not source.is_file():
        ap.error(f'Missing input: {source}')
    df = pd.read_csv(source, low_memory=False)
    if 'symbol' not in df.columns or df['symbol'].astype(str).str.upper().eq('INJ').sum() != 1:
        ap.error('Expected exactly one INJ row in original diagnostic')
    phase3c = Path(args.phase3c)
    if not phase3c.is_file():
        ap.error(f'Missing Phase 3C input: {phase3c}')
    out = Path(args.output_dir)
    raw_dir = out / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'CryptoOpportunityResearch-ReadOnly/3E'})
    query_rows = []
    chain, chain_node = get_chain(session, raw_dir, 'latest_block', '/cosmos/base/tendermint/v1beta1/blocks/latest', args.timeout, query_rows)
    height = latest_height(chain)
    current, current_node = get_chain(session, raw_dir, 'supply_current', '/cosmos/bank/v1beta1/supply/inj', args.timeout, query_rows)
    if extract_supply(current) is None:
        current, current_node = get_chain(session, raw_dir, 'supply_current_fallback', '/cosmos/bank/v1beta1/supply', args.timeout, query_rows)
    current_supply = extract_supply(current)
    historical_supply = None
    historical_height = None
    if height and height > args.height_gap:
        historical_height = height - args.height_gap
        old, _ = get_chain(session, raw_dir, 'supply_historical', '/cosmos/bank/v1beta1/supply/inj', args.timeout,
                           query_rows, headers={'x-cosmos-block-height': str(historical_height)})
        # Header may be ignored by a node: never accept this as independently proven history.
        historical_supply = extract_supply(old)
    candidates = []
    for index, event in enumerate(EVENTS):
        path = '/cosmos/tx/v1beta1/txs?' + urlencode({'events': event, 'pagination.limit': '20', 'order_by': 'ORDER_BY_DESC'})
        obj, _ = get_chain(session, raw_dir, f'tx_event_{index+1}', path, args.timeout, query_rows)
        candidates.extend(tx_candidates(obj))
    unique = {c['txhash']: c for c in candidates if c.get('txhash')}
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - 75 * 86400
    price_query = PRICE_URL + '?' + urlencode({'start': start, 'span': 75, 'period': '1d'})
    price_obj, price_raw, price_status = request(session, price_query, args.timeout)
    save_raw(raw_dir, 'price_defillama', price_raw)
    rev_obj, rev_raw, rev_status = request(session, REVENUE_URL, args.timeout)
    save_raw(raw_dir, 'revenue_defillama', rev_raw)
    comparison = windows(chart_points(price_obj), revenue_points(rev_obj))
    # No proof of supply change from two snapshots unless chain height headers are
    # independently confirmed, and no proof of attribution from event search alone.
    report = {
        'symbol': 'INJ', 'audit_at_utc': now(), 'mode': 'READ_ONLY_FAIL_CLOSED',
        'original_input': str(source), 'phase3c_input': str(phase3c),
        'latest_chain_height': height, 'latest_chain_node': chain_node,
        'current_supply_base_units': current_supply, 'current_supply_node': current_node,
        'historical_requested_height': historical_height,
        'historical_supply_unverified_base_units': historical_supply,
        'historical_supply_header_honored_verified': False,
        'net_supply_reduction_proven': False,
        'community_buyback_burn_verified': False,
        'buyback_tx_candidates_count': len(unique),
        'price_api_status': price_status, 'revenue_api_status': rev_status,
        'comparison': comparison,
        'capture_independently_proven': False, 'holder_value_gate_promoted': False,
        'cautions': [
            'Search hits are NOT confirmed buyback burns.',
            'Supply observations at different block heights are NOT confirmed until archive headers are validated.',
            'Supply delta alone cannot attribute changes to burn versus issuance.',
            'Protocol revenue does not equal income paid to INJ holders.',
            'Aligned revenue-price divergence is descriptive and does not prove undervaluation.',
        ],
    }
    (out / 'fase3e_inj_evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(query_rows).to_csv(out / 'fase3e_chain_queries.csv', index=False)
    pd.DataFrame(list(unique.values()), columns=['txhash','height','code','timestamp','event_types',
        'buyback_attribution_verified','burn_amount_verified','warning']).to_csv(out / 'fase3e_tx_candidates.csv', index=False)
    flat = {k: v for k, v in report.items() if not isinstance(v, (list, dict))}
    flat.update({'aligned_' + k: v for k, v in comparison.items()})
    pd.DataFrame([flat]).to_csv(out / 'fase3e_inj_summary.csv', index=False)
    print('=== FASE 3E | INJ | READ ONLY ===')
    print('Chain height:', height, '| Current supply (base units):', current_supply)
    print('Historical supply: NOT INDEPENDENTLY VERIFIED')
    print('Candidate tx hashes:', len(unique), '| Community BuyBack burns verified: 0')
    print('Price API:', price_status, '| Revenue API:', rev_status)
    print('Aligned comparison:', json.dumps(comparison, ensure_ascii=False))
    print('Net supply reduction proven: False | Holder capture proven: False')
    print('Outputs:', out.resolve())


if __name__ == '__main__':
    main()
