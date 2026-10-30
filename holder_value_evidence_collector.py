"""CRYPTO OPPORTUNITY ENGINE — Holder Value evidence collector.

Independent, fail-closed documentary collector. Fetches ONLY explicitly supplied
public HTTPS evidence URLs, stores a content digest and a small text excerpt,
and emits a research report. It NEVER sets verified flags, assigns scores,
changes ranking gates, or infers buybacks/burns/staking from keywords.

Usage:
 python holder_value_evidence_collector.py --registry holder_value_sources.csv \
   --diagnostic output/crypto_diagnostic_all.csv --output output/holder_value_source_audit.csv

Registry CSV headers:
 coin_id,symbol,mechanism,source_url,source_type,period,reference,flow_id

mechanism: buyback | burn | token_usage | economic_staking | real_yield
source_type: official_docs | governance | explorer | financial_report

A source fetch proves only that an accessible document was retrieved; economic
verification requires human review of the exact mechanism and token identity.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

FIELDS = ("coin_id", "symbol", "mechanism", "source_url", "source_type", "period", "reference", "flow_id")
MECHANISMS = {"buyback", "burn", "token_usage", "economic_staking", "real_yield"}
SOURCE_TYPES = {"official_docs", "governance", "explorer", "financial_report"}
MAX_BYTES = 1_000_000
TIMEOUT = (5, 15)
OUTPUT_COLUMNS = list(FIELDS) + [
    "fetch_status", "http_status", "retrieved_at_utc", "content_sha256",
    "content_type", "content_bytes", "excerpt", "audit_note", "economically_verified",
]


def _safe_host(url: str) -> tuple[bool, str]:
    """Reject private/loopback destinations, credentials and non-HTTPS URLs."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False, "HTTPS public URL required"
        if parsed.port not in (None, 443):
            return False, "Nonstandard port rejected"
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
            return False, "Local hostname rejected"
        # DNS is checked before connecting. In adversarial environments, DNS
        # rebinding remains possible: run only with trusted registries.
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        ips = {info[4][0] for info in infos}
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            return False, "Nonpublic DNS address rejected"
        return True, ""
    except (ValueError, OSError) as exc:
        return False, f"URL/DNS validation failed: {type(exc).__name__}"


def _fetch(session: requests.Session, url: str) -> dict:
    safe, reason = _safe_host(url)
    if not safe:
        return {"fetch_status": "REJECTED_URL", "audit_note": reason}
    try:
        # Redirects disabled so an approved public URL cannot redirect to a
        # private host. Streamed reads enforce an upper bound.
        with session.get(url, timeout=TIMEOUT, allow_redirects=False, stream=True,
                         headers={"Accept": "text/html,text/plain,application/json,application/pdf"}) as response:
            status = response.status_code
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            if status != 200:
                return {"fetch_status": "HTTP_NOT_OK", "http_status": status,
                        "audit_note": "Document not retrieved; redirects not followed"}
            if content_type not in {"text/html", "text/plain", "application/json", "application/pdf"}:
                return {"fetch_status": "UNSUPPORTED_CONTENT", "http_status": status,
                        "content_type": content_type, "audit_note": "Unsupported MIME type"}
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    return {"fetch_status": "TOO_LARGE", "http_status": status,
                            "content_type": content_type, "audit_note": "Document exceeds 1 MB limit"}
                chunks.append(chunk)
            data = b"".join(chunks)
            digest = hashlib.sha256(data).hexdigest()
            if content_type == "application/pdf":
                excerpt = "[PDF: hash recorded; contents not interpreted]"
            else:
                excerpt = re.sub(r"\s+", " ", data.decode("utf-8", errors="replace"))[:400]
            return {"fetch_status": "RETRIEVED_UNVERIFIED", "http_status": status,
                    "content_type": content_type, "content_bytes": len(data),
                    "content_sha256": digest, "excerpt": excerpt,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "audit_note": "Document retrieved; mechanism and token attribution NOT verified"}
    except requests.RequestException as exc:
        return {"fetch_status": "NETWORK_ERROR", "audit_note": type(exc).__name__}


def collect(registry: pd.DataFrame, diagnostic: pd.DataFrame | None = None) -> pd.DataFrame:
    """No score/verification fields from the engine are written or changed."""
    missing = set(FIELDS) - set(registry.columns)
    if missing:
        raise ValueError("Registry missing columns: " + ", ".join(sorted(missing)))
    known_ids = set()
    known_symbols = set()
    if diagnostic is not None:
        if "coin_id" in diagnostic:
            known_ids = set(diagnostic["coin_id"].dropna().astype(str).str.strip())
        if "symbol" in diagnostic:
            known_symbols = set(diagnostic["symbol"].dropna().astype(str).str.strip().str.upper())
    records = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "CryptoOpportunityEngine-EvidenceResearch/1.0"})
        for _, item in registry.iterrows():
            row = {key: str(item[key]).strip() if pd.notna(item[key]) else "" for key in FIELDS}
            row["economically_verified"] = False
            mechanism = row["mechanism"].lower()
            source_type = row["source_type"].lower()
            if mechanism not in MECHANISMS or source_type not in SOURCE_TYPES:
                row.update(fetch_status="INVALID_REGISTRY", audit_note="Unknown mechanism/source_type")
            elif not row["source_url"] or not row["reference"] or not row["period"]:
                row.update(fetch_status="INCOMPLETE_REGISTRY", audit_note="URL, reference and period required")
            elif not row["coin_id"]:
                row.update(fetch_status="MISSING_COIN_ID", audit_note="Exact coin_id required")
            elif diagnostic is not None and (row["coin_id"] not in known_ids if known_ids else
                                             row["symbol"].upper() not in known_symbols):
                row.update(fetch_status="ASSET_NOT_IN_DIAGNOSTIC", audit_note="Identity not in diagnostic")
            else:
                row.update(_fetch(session, row["source_url"]))
            records.append(row)
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only documentary Holder Value collector")
    parser.add_argument("--registry", required=True, help="Curated CSV of documentary sources")
    parser.add_argument("--diagnostic", help="Optional diagnostic CSV for coin_id identity check")
    parser.add_argument("--output", default="output/holder_value_source_audit.csv")
    args = parser.parse_args()
    registry_path = Path(args.registry)
    if not registry_path.is_file():
        parser.error(f"Registry not found: {registry_path}")
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    diagnostic = pd.read_csv(args.diagnostic, dtype=str, low_memory=False) if args.diagnostic else None
    result = collect(registry, diagnostic)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[holder_value_source_audit] Registry rows: {len(result)}")
    print(f"[holder_value_source_audit] Retrieved, not verified: {(result['fetch_status'] == 'RETRIEVED_UNVERIFIED').sum() if len(result) else 0}")
    print(f"[holder_value_source_audit] Output: {target}")
    print("[holder_value_source_audit] No gate, score, or verified flag modified")


if __name__ == "__main__":
    main()
