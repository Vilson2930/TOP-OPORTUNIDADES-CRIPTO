"""CRYPTO OPPORTUNITY ENGINE — Phase 2A: dynamic on-chain contract discovery.

Read-only research. Fetches CoinGecko's aggregated coin/platform catalogue in ONE
request, matches ONLY exact coin_id, and emits contract-address candidates.
No chain activity, burn, buyback, staking, token ownership, economic benefit,
verification flags or scores are inferred. The output NEVER feeds scoring.

Usage:
 python holder_value_onchain_discovery.py \
   --input output/crypto_diagnostic_all.csv \
   --output output/holder_value_onchain_candidates.csv

For reproducible offline tests:
 python holder_value_onchain_discovery.py --input diagnostic.csv \
   --catalogue-json coins_list_snapshot.json

Dependencies: pandas, requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CATALOGUE_URL = "https://api.coingecko.com/api/v3/coins/list"
TIMEOUT = (8, 35)
EVM_ADDRESS = re.compile(r"^0x[a-fA-F0-9]{40}$")
OUTPUT_COLUMNS = [
    "coin_id", "symbol", "name", "platform_id", "contract_address",
    "contract_format", "discovery_status", "identity_match", "source_url",
    "catalogue_sha256", "retrieved_at_utc", "economic_mechanism_verified",
    "audit_note",
]


def fetch_catalogue(session: requests.Session) -> list[dict]:
    for attempt in range(3):
        try:
            response = session.get(
                CATALOGUE_URL,
                params={"include_platform": "true"},
                timeout=TIMEOUT,
                headers={"Accept": "application/json"},
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("Unexpected catalogue response: expected list")
            return data
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Catalogue unavailable")


def build_candidates(dataset: pd.DataFrame, catalogue: list[dict], *, timestamp: str | None = None) -> pd.DataFrame:
    required = {"coin_id", "symbol", "name"}
    if not required.issubset(dataset.columns):
        raise ValueError("Diagnostic missing columns: " + ", ".join(sorted(required - set(dataset.columns))))
    if not isinstance(catalogue, list):
        raise ValueError("Catalogue must be a list")
    snapshot = json.dumps(catalogue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    by_id: dict[str, list[dict]] = {}
    for item in catalogue:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            by_id.setdefault(item["id"].strip(), []).append(item)
    records = []
    for _, asset in dataset.iterrows():
        coin_id = str(asset["coin_id"]).strip()
        base = {
            "coin_id": coin_id, "symbol": asset["symbol"], "name": asset["name"],
            "catalogue_sha256": digest, "retrieved_at_utc": timestamp,
            "economic_mechanism_verified": False, "source_url": CATALOGUE_URL,
        }
        matches = by_id.get(coin_id, [])
        if len(matches) != 1:
            records.append({**base, "discovery_status": "NO_EXACT_ID" if not matches else "AMBIGUOUS_ID",
                            "identity_match": False, "audit_note": "No unique exact CoinGecko ID; no ticker fallback"})
            continue
        entry = matches[0]
        # Reject conflicting symbols: catalogue IDs are authoritative for lookup,
        # but a symbol mismatch is a reason for manual identity reconciliation.
        if str(entry.get("symbol", "")).strip().lower() != str(asset["symbol"]).strip().lower():
            records.append({**base, "discovery_status": "SYMBOL_CONFLICT", "identity_match": False,
                            "audit_note": "CoinGecko ID matched but symbol differs; review identity"})
            continue
        platforms = entry.get("platforms")
        if not isinstance(platforms, dict) or not platforms:
            records.append({**base, "discovery_status": "NO_PLATFORM_CONTRACT", "identity_match": True,
                            "audit_note": "No contract listed; native assets may have no token contract"})
            continue
        emitted = 0
        for platform, address in sorted(platforms.items()):
            if not isinstance(platform, str) or not isinstance(address, str):
                continue
            platform, address = platform.strip(), address.strip()
            if not platform or not address:
                continue
            # Non-EVM chains have distinct address encodings. Never claim validation
            # beyond syntactic EVM address shape.
            fmt = "EVM_SHAPE_ONLY" if EVM_ADDRESS.fullmatch(address) else "UNVALIDATED_NON_EVM_OR_OTHER"
            records.append({**base, "platform_id": platform, "contract_address": address,
                            "contract_format": fmt, "identity_match": True,
                            "discovery_status": "CONTRACT_CANDIDATE_UNVERIFIED",
                            "audit_note": "Catalogue-listed address only; not on-chain checked; mechanism unverified"})
            emitted += 1
        if not emitted:
            records.append({**base, "discovery_status": "NO_PLATFORM_CONTRACT", "identity_match": True,
                            "audit_note": "No usable platform/address entries"})
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic contract candidate discovery, read-only")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output/holder_value_onchain_candidates.csv")
    parser.add_argument("--catalogue-json", help="Offline CoinGecko catalogue snapshot")
    args = parser.parse_args()
    dataset = pd.read_csv(args.input, dtype=str, keep_default_na=False, low_memory=False)
    if args.catalogue_json:
        catalogue = json.loads(Path(args.catalogue_json).read_text(encoding="utf-8"))
    else:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "CryptoOpportunityEngine-OnchainResearch/1.0"})
            catalogue = fetch_catalogue(session)
    result = build_candidates(dataset, catalogue)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[onchain_discovery] Assets: {len(dataset)}")
    print(f"[onchain_discovery] Contract candidates: {(result['discovery_status'] == 'CONTRACT_CANDIDATE_UNVERIFIED').sum()}")
    print(f"[onchain_discovery] Output: {target}")
    print("[onchain_discovery] No on-chain events verified; no scores/gates changed")


if __name__ == "__main__":
    main()
