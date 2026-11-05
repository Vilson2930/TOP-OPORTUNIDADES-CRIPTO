#!/usr/bin/env python3
"""Phase 2C.1: read-only identity evidence audit, fail-closed.

This script NEVER promotes a token identity to verified from self-reported
ERC20 metadata or a single aggregator catalogue. It compares exact platform+
contract identities with CoinGecko's bulk catalogue and optionally a manually
reviewed independent official-source attestation CSV. No investment gates change.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

CATALOGUE_URL = 'https://api.coingecko.com/api/v3/coins/list?include_platform=true'
HEX_ADDRESS = re.compile(r'^0x[0-9a-fA-F]{40}$')
EVIDENCE_COLUMNS = ['coin_id','platform_id','contract_address','official_source_url','official_source_contract','official_source_platform','reviewer','reviewed_at_utc','attestation_sha256']
OUTPUT_COLUMNS = ['coin_id','symbol','name','platform_id','contract_address','phase2c_status','phase2b_status','onchain_name','onchain_symbol','name_match','symbol_match','catalogue_status','catalogue_exact_match','catalogue_sha256','catalogue_retrieved_at_utc','metadata_assessment','independent_source_status','official_source_url','identity_evidence_status','identity_verified','economic_mechanism_verified','audit_note']


def norm(v):
    if pd.isna(v):
        return ''
    return str(v).strip()


def address_norm(v):
    v = norm(v)
    return v.lower() if HEX_ADDRESS.fullmatch(v) else v


def load_catalogue(cache_path, timeout, max_age_hours, offline):
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if offline or age <= max_age_hours * 3600:
            raw = cache_path.read_bytes()
            return json.loads(raw), hashlib.sha256(raw).hexdigest(), 'CACHE', datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat()
    if offline:
        return None, '', 'OFFLINE_NO_CACHE', ''
    try:
        req = Request(CATALOGUE_URL, headers={'User-Agent': 'CryptoOpportunityEngine-IdentityAudit/1.0', 'Accept': 'application/json'})
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
        data = json.loads(raw)
        if not isinstance(data, list) or not data:
            raise ValueError('Invalid or empty catalogue')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        return data, hashlib.sha256(raw).hexdigest(), 'LIVE', datetime.now(timezone.utc).isoformat()
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f'[phase2c1] Catalogue unavailable: {type(exc).__name__}: {exc}', file=sys.stderr)
        return None, '', 'UNAVAILABLE', ''


def load_attestations(path):
    """Optional exact, human-reviewed official-source evidence, never auto-approved.

    Attestations are only QUEUED_FOR_REVIEW: a CSV assertion does not prove
    authenticity of a website, chain or token.
    """
    if not path:
        return {}
    df = pd.read_csv(path, dtype=str).fillna('')
    missing = set(EVIDENCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Attestation CSV missing columns: {sorted(missing)}')
    indexed = {}
    for _, row in df.iterrows():
        key = (norm(row['coin_id']), norm(row['platform_id']), address_norm(row['contract_address']))
        indexed.setdefault(key, []).append(row.to_dict())
    return indexed


def classify_metadata(row):
    nm = norm(row.get('name_match')).lower() == 'true'
    sm = norm(row.get('symbol_match')).lower() == 'true'
    if nm and sm:
        return 'NAME_AND_SYMBOL_MATCH_UNTRUSTED'
    if sm:
        return 'SYMBOL_MATCH_NAME_VARIANT_UNTRUSTED'
    if nm:
        return 'NAME_MATCH_SYMBOL_VARIANT_UNTRUSTED'
    if norm(row.get('onchain_name')) or norm(row.get('onchain_symbol')):
        return 'METADATA_DIFFERENT_REVIEW_REQUIRED'
    return 'METADATA_NOT_AVAILABLE'


def run(args):
    source = Path(args.input)
    target = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    df = pd.read_csv(source, dtype=str).fillna('')
    required = {'coin_id','platform_id','contract_address','identity_status','phase2b_status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Input missing columns: {sorted(missing)}')
    catalogue, digest, catalogue_status, retrieved = load_catalogue(Path(args.cache), args.timeout, args.cache_hours, args.offline)
    by_id = {}
    if catalogue is not None:
        for item in catalogue:
            if isinstance(item, dict) and item.get('id'):
                by_id[item['id']] = item
    attestations = load_attestations(args.attestations)
    rows = []
    for _, row in df.iterrows():
        coin = norm(row['coin_id'])
        platform = norm(row['platform_id'])
        address = address_norm(row['contract_address'])
        status = norm(row['identity_status'])
        bytecode = norm(row['phase2b_status'])
        item = by_id.get(coin)
        exact = False
        cat_status = 'CATALOGUE_UNAVAILABLE' if catalogue is None else 'COIN_ID_NOT_IN_CATALOGUE'
        if item is not None:
            platforms = item.get('platforms') or {}
            expected = address_norm(platforms.get(platform, ''))
            if expected and address and expected == address:
                exact = True
                cat_status = 'CATALOGUE_EXACT_PLATFORM_ADDRESS'
            elif not platform or not address:
                cat_status = 'NO_COMPARABLE_PLATFORM_ADDRESS'
            else:
                cat_status = 'CATALOGUE_MISMATCH_OR_MISSING_PLATFORM'
        key = (coin, platform, address)
        att = attestations.get(key, [])
        official_url = ''
        independent = 'NOT_PROVIDED'
        if att:
            urls = sorted({norm(a.get('official_source_url')) for a in att if norm(a.get('official_source_url'))})
            official_url = ';'.join(urls)
            independent = 'ATTESTATION_REQUIRES_MANUAL_AUTHENTICITY_REVIEW'
        if bytecode != 'EVM_BYTECODE_PRESENT':
            outcome = 'NOT_ELIGIBLE_FOR_EVM_IDENTITY_REVIEW'
        elif not exact:
            outcome = 'UNVERIFIED_CATALOGUE_NOT_CORROBORATED'
        elif att:
            outcome = 'REVIEW_QUEUE_OFFICIAL_ATTESTATION_UNVALIDATED'
        else:
            outcome = 'CATALOGUE_CORROBORATED_NOT_INDEPENDENTLY_VERIFIED'
        result = {
            'coin_id': coin,
            'symbol': norm(row.get('symbol')),
            'name': norm(row.get('name')),
            'platform_id': platform,
            'contract_address': norm(row['contract_address']),
            'phase2c_status': status,
            'phase2b_status': bytecode,
            'onchain_name': norm(row.get('onchain_name')),
            'onchain_symbol': norm(row.get('onchain_symbol')),
            'name_match': norm(row.get('name_match')),
            'symbol_match': norm(row.get('symbol_match')),
            'catalogue_status': cat_status,
            'catalogue_exact_match': exact,
            'catalogue_sha256': digest,
            'catalogue_retrieved_at_utc': retrieved,
            'metadata_assessment': classify_metadata(row),
            'independent_source_status': independent,
            'official_source_url': official_url,
            'identity_evidence_status': outcome,
            'identity_verified': False,
            'economic_mechanism_verified': False,
            'audit_note': 'CoinGecko is an aggregator, not independent official proof; on-chain token metadata is self-reported; manual official attestation is unverified until independently authenticated.'
        }
        rows.append(result)
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f'[phase2c1] Input records: {len(df)}')
    print(f'[phase2c1] Catalogue status: {catalogue_status}')
    print('[phase2c1] Evidence status counts:')
    print(out['identity_evidence_status'].value_counts(dropna=False).to_string())
    print(f'[phase2c1] Output: {target}')
    print('[phase2c1] No identities or economic mechanisms were automatically verified; no gates or scores modified.')


def main():
    parser = argparse.ArgumentParser(description='Read-only token identity evidence audit, fail-closed')
    parser.add_argument('--input', required=True, help='Phase 2C CSV')
    parser.add_argument('--output', required=True, help='Phase 2C.1 evidence CSV')
    parser.add_argument('--cache', default='output/coingecko_platform_catalogue.json')
    parser.add_argument('--cache-hours', type=float, default=12)
    parser.add_argument('--timeout', type=float, default=25)
    parser.add_argument('--offline', action='store_true', help='Use existing catalogue cache only')
    parser.add_argument('--attestations', help='Optional curated official-source attestations CSV; never auto-verifies')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
