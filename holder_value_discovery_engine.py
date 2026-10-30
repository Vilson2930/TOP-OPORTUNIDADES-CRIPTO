"""CRYPTO OPPORTUNITY ENGINE — dynamic Holder Value source discovery (phase 1).

Discovers candidate protocol identities and official URLs from the aggregated
DefiLlama protocol catalogue. Does NOT verify any economic mechanism, assign
scores, or modify the engine. Ambiguous identities fail closed.

python holder_value_discovery_engine.py --input output/crypto_diagnostic_all.csv \
    --output output/holder_value_discovery.csv

Dependency: pandas, requests (already used by the engine).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

CATALOGUE_URL = "https://api.llama.fi/protocols"
TIMEOUT = (8, 25)
OUTPUT_COLUMNS = [
    "coin_id", "symbol", "name", "discovery_status", "protocol_name",
    "protocol_slug", "protocol_id", "protocol_symbol", "protocol_category",
    "protocol_chains", "candidate_url", "candidate_url_kind", "catalogue_source",
    "catalogue_retrieved_at_utc", "economic_mechanism_verified", "audit_note",
]


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = unicodedata.normalize("NFKD", str(value).lower().strip())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in raw if not unicodedata.combining(c)))


def safe_public_https(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlparse(value.strip())
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.port not in (None, 443)):
            return ""
        host = parsed.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
            return ""
        return value.strip()
    except ValueError:
        return ""


def fetch_catalogue(session: requests.Session) -> list[dict]:
    response = session.get(CATALOGUE_URL, timeout=TIMEOUT, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected DefiLlama catalogue structure")
    return [p for p in payload if isinstance(p, dict)]


def discover(dataset: pd.DataFrame, catalogue: list[dict], timestamp: str | None = None) -> pd.DataFrame:
    """Return research candidates for every input asset, without making claims of verification."""
    if dataset is None:
        raise ValueError("Input diagnostic is required")
    if not {"coin_id", "name", "symbol"}.issubset(dataset.columns):
        raise ValueError("Diagnostic must include coin_id, name and symbol")
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    index: dict[str, set[int]] = {}
    for i, p in enumerate(catalogue):
        for field in ("name", "slug"):
            key = normalize(p.get(field))
            if key:
                index.setdefault(key, set()).add(i)
    records: list[dict] = []
    for _, asset in dataset.iterrows():
        base = {
            "coin_id": asset.get("coin_id", ""), "symbol": asset.get("symbol", ""),
            "name": asset.get("name", ""), "catalogue_source": CATALOGUE_URL,
            "catalogue_retrieved_at_utc": timestamp,
            "economic_mechanism_verified": False,
        }
        identities = list(dict.fromkeys(filter(None, (normalize(asset.get("name")), normalize(asset.get("coin_id"))))))
        matches = [index[k] for k in identities if k in index]
        if not matches:
            records.append({**base, "discovery_status": "NO_UNIQUE_MATCH", "audit_note": "No exact protocol name/slug identity; ticker not used"})
            continue
        union = set().union(*matches)
        if len(union) != 1 or any(len(m) != 1 for m in matches):
            records.append({**base, "discovery_status": "AMBIGUOUS_IDENTITY", "audit_note": "Conflicting or multiple exact protocol matches"})
            continue
        p = catalogue[next(iter(union))]
        chains = p.get("chains")
        if not isinstance(chains, list):
            chains = []
        details = {
            **base, "protocol_name": p.get("name", ""), "protocol_slug": p.get("slug", ""),
            "protocol_id": p.get("id", ""), "protocol_symbol": p.get("symbol", ""),
            "protocol_category": p.get("category", ""),
            "protocol_chains": ";".join(str(c) for c in chains),
        }
        # The catalogue's `url` is the protocol homepage, NOT necessarily docs,
        # nor proof of token-level cash flows.
        homepage = safe_public_https(p.get("url"))
        if homepage:
            records.append({**details, "discovery_status": "CANDIDATE_FOUND_UNVERIFIED",
                            "candidate_url": homepage, "candidate_url_kind": "PROTOCOL_HOMEPAGE",
                            "audit_note": "Catalogue identity and homepage candidate only; economic mechanism unverified"})
        else:
            records.append({**details, "discovery_status": "IDENTITY_ONLY_NO_URL",
                            "audit_note": "Catalogue identity found, but no acceptable HTTPS homepage"})
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic, fail-closed Holder Value source discovery")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output/holder_value_discovery.csv")
    parser.add_argument("--catalogue-json", help="Optional offline catalogue snapshot for reproducible testing")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Diagnostic not found: {source}")
    dataset = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    if args.catalogue_json:
        catalogue = json.loads(Path(args.catalogue_json).read_text(encoding="utf-8"))
    else:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "CryptoOpportunityEngine-Discovery/1.0"})
            catalogue = fetch_catalogue(session)
    if not isinstance(catalogue, list):
        raise ValueError("Catalogue must be a list")
    output = discover(dataset, catalogue)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[holder_value_discovery] Assets: {len(output)}")
    print(f"[holder_value_discovery] Candidate URLs: {(output['discovery_status'] == 'CANDIDATE_FOUND_UNVERIFIED').sum()}")
    print(f"[holder_value_discovery] Output: {target}")
    print("[holder_value_discovery] No mechanism verified, no gate or score changed")


if __name__ == "__main__":
    main()
