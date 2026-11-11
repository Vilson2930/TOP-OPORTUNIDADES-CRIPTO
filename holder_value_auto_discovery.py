"""Read-only, bounded documentary candidate discovery for Holder Value.

Inputs: output/holder_value_discovery.csv and optional
        output/crypto_diagnostic_all.csv.
Outputs: output/holder_value_auto_sources.csv,
         output/holder_value_auto_summary.json.

Never asserts economic verification or changes scoring / Quality Gates.
Candidate documents are links, not evidence of token-holder value capture.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import pandas as pd
import requests

MAX_CANDIDATES = 30
MAX_BYTES = 250_000
TIMEOUT = (5, 12)
MAX_LINKS_PER_SITE = 5
KEYWORDS = {
    "buyback": ("buyback", "buy-back", "repurchase"),
    "burn": ("burn", "burning", "tokenomics"),
    "distribution": ("distribution", "revenue share", "fee sharing", "fees"),
    "staking": ("staking", "real yield", "yield", "rewards"),
    "governance": ("governance", "docs", "documentation", "whitepaper"),
}
FIELDS = [
    "coin_id", "symbol", "name", "protocol_name", "protocol_slug",
    "discovery_status", "candidate_url", "candidate_url_kind",
    "source_catalogue", "priority_score", "priority_basis", "fetch_status",
    "http_status", "content_type", "content_sha256", "retrieved_at_utc",
    "excerpt", "document_candidates_json", "document_candidate_count",
    "evidence_scope", "economic_mechanism_verified", "audit_note",
]


def _valid_url(url: str) -> bool:
    """Reject malformed URLs, IP literals, private/local names and non-HTTPS."""
    try:
        p = urlparse(str(url).strip())
        if p.scheme != "https" or not p.hostname or p.username or p.password:
            return False
        if p.port not in (None, 443):
            return False
        host = p.hostname.lower().rstrip(".")
        if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
            return False
        if not re.fullmatch(r"[a-z0-9.-]+", host) or "." not in host:
            return False
        if any(not label or label.startswith("-") or label.endswith("-") for label in host.split(".")):
            return False
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            return True
    except (ValueError, AttributeError):
        return False


def _public_dns(url: str) -> bool:
    """Best-effort DNS check; requests can re-resolve, so not a strict SSRF sandbox.

    Run only against trusted catalogue-supplied domains in a restricted runner.
    """
    host = urlparse(url).hostname
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses:
            return False
        return all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (OSError, ValueError):
        return False


def _safe_number(value) -> float | None:
    try:
        result = float(value)
        if pd.isna(result) or result in (float("inf"), float("-inf")):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _priority_table(diagnostic: pd.DataFrame | None) -> dict[str, tuple[float, str]]:
    """Prioritize for research only; no score or gate modification."""
    if diagnostic is None or diagnostic.empty or "coin_id" not in diagnostic.columns:
        return {}
    options = ("opportunity_score_observed", "opportunity_score", "holder_value_score")
    table = {}
    for _, row in diagnostic.iterrows():
        coin_id = str(row.get("coin_id", "")).strip()
        if not coin_id or coin_id in table:
            continue
        for field in options:
            value = _safe_number(row.get(field))
            if value is not None:
                table[coin_id] = (value, field)
                break
    return table


def _extract_links(page: str, base_url: str) -> list[dict]:
    """Discover same-host HTTPS documentary links; do not fetch or verify them."""
    base_host = urlparse(base_url).hostname
    found = []
    seen = set()
    # HTML attribute parser is intentionally conservative; not a full browser.
    for match in re.finditer(r'<a\b[^>]*\bhref\s*=\s*(["\'])(.*?)\1[^>]*>(.*?)</a\s*>',
                             page, flags=re.I | re.S):
        raw_href = html.unescape(match.group(2)).strip()
        if not raw_href or raw_href.startswith(("#", "mailto:", "javascript:", "data:")):
            continue
        target = urldefrag(urljoin(base_url, raw_href))[0]
        if not _valid_url(target) or urlparse(target).hostname != base_host:
            continue
        label = re.sub(r"<[^>]*>", " ", match.group(3))
        label = re.sub(r"\s+", " ", html.unescape(label)).strip()[:120]
        haystack = (target + " " + label).lower()
        tags = [key for key, terms in KEYWORDS.items() if any(term in haystack for term in terms)]
        if not tags or target in seen:
            continue
        seen.add(target)
        found.append({"url": target, "label": label, "topics": tags,
                      "verification_status": "LINK_ONLY_NOT_RETRIEVED"})
        if len(found) >= MAX_LINKS_PER_SITE:
            break
    return found


def _fetch(url: str, session: requests.Session, *, check_dns: bool = True) -> dict:
    if not _valid_url(url) or (check_dns and not _public_dns(url)):
        return {"fetch_status": "UNSAFE_OR_UNRESOLVED_URL",
                "audit_note": "URL failed HTTPS/public DNS validation"}
    try:
        with session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False,
                         headers={"Accept": "text/html,text/plain,application/json"}) as response:
            status = response.status_code
            mime = response.headers.get("Content-Type", "").split(";")[0].lower().strip()
            if status != 200:
                return {"fetch_status": "HTTP_NOT_OK", "http_status": status,
                        "content_type": mime, "audit_note": "Non-200 or redirect; not verified"}
            if mime not in {"text/html", "text/plain", "application/json"}:
                return {"fetch_status": "UNSUPPORTED_CONTENT", "http_status": status,
                        "content_type": mime, "audit_note": "Unsupported content type"}
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=32768):
                size += len(chunk)
                if size > MAX_BYTES:
                    return {"fetch_status": "TOO_LARGE", "http_status": status,
                            "content_type": mime, "audit_note": "Exceeded bounded download"}
                chunks.append(chunk)
            content = b"".join(chunks)
            decoded = content.decode("utf-8", "replace")
            links = _extract_links(decoded, url) if mime == "text/html" else []
            return {
                "fetch_status": "HOMEPAGE_RETRIEVED_UNVERIFIED",
                "http_status": status, "content_type": mime,
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "excerpt": re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", decoded))[:400],
                "document_candidates_json": json.dumps(links, ensure_ascii=False),
                "document_candidate_count": len(links),
                "audit_note": "Homepage and links only; no mechanism, attribution, period or on-chain proof",
            }
    except requests.RequestException as exc:
        return {"fetch_status": "NETWORK_ERROR", "audit_note": type(exc).__name__}


def collect(discovery: pd.DataFrame, max_candidates: int = MAX_CANDIDATES,
            session: requests.Session | None = None,
            diagnostic: pd.DataFrame | None = None,
            *, check_dns: bool = True) -> pd.DataFrame:
    """Select unique, eligible catalogue candidates; sort by observed score."""
    required = {"coin_id", "symbol", "name", "discovery_status", "candidate_url"}
    missing = required - set(discovery.columns)
    if missing:
        raise ValueError("Discovery CSV missing: " + ", ".join(sorted(missing)))
    if not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be 1..{MAX_CANDIDATES}")
    selected = discovery.loc[
        discovery["discovery_status"].fillna("").astype(str).eq("CANDIDATE_FOUND_UNVERIFIED")
    ].copy()
    selected["coin_id"] = selected["coin_id"].astype(str).str.strip()
    selected = selected[selected["coin_id"].ne("")]
    selected = selected.drop_duplicates(subset=["coin_id"], keep=False)
    priorities = _priority_table(diagnostic)
    selected["_priority"] = selected["coin_id"].map(
        lambda x: priorities.get(x, (float("-inf"), ""))[0]
    )
    selected = selected.sort_values(["_priority", "coin_id"],
                                    ascending=[False, True], kind="stable").head(max_candidates)
    own_session = session is None
    if own_session:
        session = requests.Session()
    rows = []
    try:
        for _, item in selected.iterrows():
            row = {field: "" for field in FIELDS}
            for field in ("coin_id", "symbol", "name", "protocol_name", "protocol_slug",
                          "discovery_status", "candidate_url", "candidate_url_kind"):
                if field in selected.columns and pd.notna(item[field]):
                    row[field] = str(item[field]).strip()
            priority = priorities.get(row["coin_id"])
            row.update({
                "source_catalogue": "https://api.llama.fi/protocols",
                "priority_score": priority[0] if priority else "",
                "priority_basis": priority[1] if priority else "NO_DIAGNOSTIC_SCORE",
                "evidence_scope": "PROTOCOL_HOMEPAGE_AND_LINK_CANDIDATES_ONLY",
                "economic_mechanism_verified": False,
                "document_candidates_json": "[]", "document_candidate_count": 0,
            })
            if not _valid_url(row["candidate_url"]):
                row.update(fetch_status="INVALID_CANDIDATE",
                           audit_note="Missing or invalid HTTPS catalogue URL")
            else:
                row.update(_fetch(row["candidate_url"], session, check_dns=check_dns))
            rows.append(row)
    finally:
        if own_session:
            session.close()
    return pd.DataFrame(rows, columns=FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Holder Value source discovery")
    parser.add_argument("--discovery", default="output/holder_value_discovery.csv")
    parser.add_argument("--diagnostic", default="output/crypto_diagnostic_all.csv")
    parser.add_argument("--output", default="output/holder_value_auto_sources.csv")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    source = Path(args.discovery)
    if not source.is_file():
        parser.error(f"Discovery report not found: {source}")
    discovery = pd.read_csv(source, dtype=str, keep_default_na=False)
    diagnostic_path = Path(args.diagnostic)
    diagnostic = (pd.read_csv(diagnostic_path, dtype=str, keep_default_na=False)
                  if diagnostic_path.is_file() else None)
    results = collect(discovery, max_candidates=args.max_candidates, diagnostic=diagnostic)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "diagnostic_used": diagnostic is not None,
        "selection_method": "descending diagnostic score then coin_id" if diagnostic is not None else "coin_id",
        "candidates_checked": len(results),
        "homepages_retrieved": int((results["fetch_status"] == "HOMEPAGE_RETRIEVED_UNVERIFIED").sum()),
        "document_links_discovered": int(pd.to_numeric(results["document_candidate_count"], errors="coerce").fillna(0).sum()),
        "economic_mechanisms_verified": 0,
        "approval_flags_modified": False,
        "warning": "Document links are research leads only; economic mechanisms and on-chain events remain unverified",
    }
    output.with_name("holder_value_auto_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
