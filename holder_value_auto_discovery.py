"""Bounded, read-only Holder Value documentary research (v3).

Inputs: output/holder_value_discovery.csv; optional output/crypto_diagnostic_all.csv.
Outputs: output/holder_value_auto_sources.csv, output/holder_value_auto_documents.csv,
         output/holder_value_auto_claims.csv, output/holder_value_auto_summary.json.

Document mentions and HTTP retrieval NEVER verify economic capture, on-chain
transactions, token attribution, payment periods, or independent evidence.
No investment dataset, scoring field, or Quality Gate is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import ipaddress
import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import pandas as pd
import requests

MAX_CANDIDATES = 30
MAX_BYTES = 250_000
MAX_LINKS_PER_SITE = 5
MAX_DOCUMENTS_TOTAL = 30
TIMEOUT = (5, 12)
TOPICS = {
    "buyback": ("buyback", "buy-back", "repurchase", "token buy back"),
    "burn": ("burn", "burning", "fee burn", "tokenomics"),
    "distribution": ("distribution", "revenue share", "fee sharing", "fee distribution"),
    "staking": ("staking", "real yield", "staking rewards", "staker"),
    "documentation": ("docs", "documentation", "whitepaper", "economics", "tokenomics"),
}
ECONOMIC_TOPICS = ("buyback", "burn", "distribution", "staking")
FIELDS = [
    "coin_id", "symbol", "name", "protocol_name", "protocol_slug",
    "discovery_status", "candidate_url", "candidate_url_kind", "source_catalogue",
    "priority_score", "priority_basis", "fetch_status", "http_status", "content_type",
    "content_sha256", "retrieved_at_utc", "excerpt", "document_candidates_json",
    "document_candidate_count", "evidence_scope", "economic_mechanism_verified", "audit_note",
]
DOCUMENT_FIELDS = [
    "coin_id", "symbol", "protocol_slug", "source_homepage", "document_url",
    "link_label", "link_topics", "fetch_status", "http_status", "content_type",
    "content_sha256", "retrieved_at_utc", "document_excerpt", "mentioned_topics",
    "token_mentioned", "mechanism_documented", "period_verified", "onchain_verified",
    "independent_source_verified", "economic_mechanism_verified", "evidence_scope", "audit_note",
]


CLAIM_FIELDS = [
    "coin_id", "symbol", "protocol_slug", "document_url", "content_sha256",
    "retrieved_at_utc", "mechanism_type", "matched_expression",
    "context_excerpt", "token_mentioned_in_context", "period_mentioned_in_context",
    "potential_double_counting", "source_type", "evidence_status",
    "economic_mechanism_verified", "independent_verification_required",
    "audit_note",
]

# Discovery patterns are intentionally only leads. In particular, a generic
# mention of staking, burn or distribution is not a token-holder payment.
CLAIM_PATTERNS = {
    "BUYBACK": (r"\bbuy[ -]?backs?\b", r"\brepurchas(?:e|es|ed|ing)\b"),
    "BURN": (r"\btoken[s]? burn(?:ed|ing|s)?\b", r"\bburn(?:ed|ing|s)? (?:of )?tokens?\b",
             r"\bfee burn\b"),
    "FEE_DISTRIBUTION": (r"\bfee(?:s)? distribut(?:ion|ed|ions)\b",
                         r"\brevenue shar(?:e|ing)\b", r"\bfee shar(?:e|ing)\b"),
    "STAKING_YIELD": (r"\bstaking rewards?\b", r"\breal yield\b",
                      r"\bstakers? receiv(?:e|es|ing)\b"),
}
PERIOD_PATTERN = re.compile(
    r"\b(?:20\d{2}|(?:last|past)\s+\d+\s+(?:days?|months?|years?)|"
    r"\d+\s*[- ]?day|monthly|quarterly|annually|per year)\b", re.I
)


def extract_claims(documents: pd.DataFrame) -> pd.DataFrame:
    """Extract bounded textual leads from retrieved excerpts, NEVER verification.

    Full-page content is not retained by the original collector. This routine
    only searches the first 1,200 visible characters of each fetched document;
    an empty result means no matching lead in that excerpt, not no mechanism.
    """
    rows = []
    for _, doc in documents.iterrows():
        if str(doc.get("fetch_status", "")) != "RETRIEVED_UNVERIFIED":
            continue
        excerpt = str(doc.get("document_excerpt", ""))
        symbol = str(doc.get("symbol", "")).strip()
        if not excerpt:
            continue
        for mechanism, patterns in CLAIM_PATTERNS.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.finditer(pattern, excerpt, re.I))
            # At most two leads of each type per document, avoiding repeated text.
            seen = set()
            for match in sorted(matches, key=lambda m: m.start()):
                context = excerpt[max(0, match.start()-160):min(len(excerpt), match.end()+160)]
                key = (mechanism, context.casefold())
                if key in seen:
                    continue
                seen.add(key)
                token_mentioned = bool(symbol and re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(symbol) + r"(?![A-Za-z0-9])",
                    context, re.I))
                rows.append({
                    "coin_id": doc.get("coin_id", ""),
                    "symbol": symbol,
                    "protocol_slug": doc.get("protocol_slug", ""),
                    "document_url": doc.get("document_url", ""),
                    "content_sha256": doc.get("content_sha256", ""),
                    "retrieved_at_utc": doc.get("retrieved_at_utc", ""),
                    "mechanism_type": mechanism,
                    "matched_expression": match.group(0),
                    "context_excerpt": context,
                    "token_mentioned_in_context": token_mentioned,
                    "period_mentioned_in_context": bool(PERIOD_PATTERN.search(context)),
                    "potential_double_counting": mechanism in {"BUYBACK", "BURN"},
                    "source_type": "PROTOCOL_DOCUMENT_CANDIDATE",
                    "evidence_status": "TEXTUAL_LEAD_UNVERIFIED",
                    "economic_mechanism_verified": False,
                    "independent_verification_required": True,
                    "audit_note": "Text mention only; verify token identity, cash flow, period and independent on-chain or financial records",
                })
                if len(seen) >= 2:
                    break
    return pd.DataFrame(rows, columns=CLAIM_FIELDS)


def _valid_url(url: str) -> bool:
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
    try:
        host = urlparse(url).hostname
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(x[4][0]).is_global for x in addresses)
    except (OSError, ValueError):
        return False


def _safe_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _priority_table(diagnostic: pd.DataFrame | None) -> dict:
    """Research ordering: Holder Value first, then revenue and opportunity as tie breakers.

    These are existing diagnostic scores, not a newly invented economic score.
    Missing values are not treated as zero or proof of economic capture.
    """
    if diagnostic is None or diagnostic.empty or "coin_id" not in diagnostic.columns:
        return {}
    result = {}
    for _, row in diagnostic.iterrows():
        coin_id = str(row.get("coin_id", "")).strip()
        if not coin_id or coin_id in result:
            continue
        columns = ("holder_value_score", "revenue_score", "opportunity_score_observed", "opportunity_score")
        values = tuple(_safe_number(row.get(key)) for key in columns)
        result[coin_id] = values
    return result


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and self.href is None:
            self.href = dict(attrs).get("href", "")
            self.label = []

    def handle_data(self, data):
        if self.href is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href is not None:
            self.links.append((self.href, " ".join(self.label)))
            self.href = None
            self.label = []


def _topics(text: str) -> list[str]:
    value = text.lower()
    return [topic for topic, terms in TOPICS.items() if any(term in value for term in terms)]


def _extract_links(page: str, base_url: str) -> list[dict]:
    """Same-host links only. A link is a research lead, never proof."""
    parser = _LinkParser()
    parser.feed(page)
    base_host = urlparse(base_url).hostname
    found, seen = [], set()
    for href, label in parser.links:
        href = html.unescape(str(href)).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "data:")):
            continue
        target = urldefrag(urljoin(base_url, href))[0]
        if not _valid_url(target) or urlparse(target).hostname != base_host or target in seen:
            continue
        label = re.sub(r"\s+", " ", label).strip()[:120]
        topics = _topics(target + " " + label)
        if not topics:
            continue
        seen.add(target)
        found.append({"url": target, "label": label, "topics": topics,
                      "verification_status": "LINK_ONLY_NOT_RETRIEVED"})
    found.sort(key=lambda x: (-sum(t in ECONOMIC_TOPICS for t in x["topics"]), x["url"]))
    return found[:MAX_LINKS_PER_SITE]


def _fetch(url: str, session: requests.Session, *, check_dns: bool = True) -> dict:
    """Small HTTPS fetch, no redirects, never interprets content as verified proof.

    DNS check is best effort, not a network sandbox: run in a restricted CI runner.
    """
    if not _valid_url(url) or (check_dns and not _public_dns(url)):
        return {"fetch_status": "UNSAFE_OR_UNRESOLVED_URL", "audit_note": "URL/public DNS rejected"}
    try:
        with session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False,
                         headers={"Accept": "text/html,text/plain,application/json"}) as response:
            status = response.status_code
            mime = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if status != 200:
                return {"fetch_status": "HTTP_NOT_OK", "http_status": status,
                        "content_type": mime, "audit_note": "HTTP failure or redirect"}
            if mime not in {"text/html", "text/plain", "application/json"}:
                return {"fetch_status": "UNSUPPORTED_CONTENT", "http_status": status,
                        "content_type": mime, "audit_note": "Unsupported MIME"}
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=32768):
                size += len(chunk)
                if size > MAX_BYTES:
                    return {"fetch_status": "TOO_LARGE", "http_status": status,
                            "content_type": mime, "audit_note": "Download exceeds byte limit"}
                chunks.append(chunk)
            content = b"".join(chunks)
            decoded = content.decode("utf-8", "replace")
            links = _extract_links(decoded, url) if mime == "text/html" else []
            plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", decoded)).strip()
            return {"fetch_status": "RETRIEVED_UNVERIFIED", "http_status": status,
                    "content_type": mime, "content_sha256": hashlib.sha256(content).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "excerpt": plain[:1200], "links": links,
                    "mentioned_topics": _topics(plain),
                    "audit_note": "Document fetched, but no token-holder economic proof established"}
    except requests.RequestException as exc:
        return {"fetch_status": "NETWORK_ERROR", "audit_note": type(exc).__name__}


def collect(discovery: pd.DataFrame, max_candidates: int = MAX_CANDIDATES,
            session: requests.Session | None = None, diagnostic: pd.DataFrame | None = None,
            *, check_dns: bool = True, max_documents: int = MAX_DOCUMENTS_TOTAL,
            return_documents: bool = False):
    """Backward compatible: returns source DataFrame unless return_documents=True."""
    required = {"coin_id", "symbol", "name", "discovery_status", "candidate_url"}
    missing = required - set(discovery.columns)
    if missing:
        raise ValueError("Discovery CSV missing: " + ", ".join(sorted(missing)))
    if not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be 1..{MAX_CANDIDATES}")
    if not 0 <= max_documents <= MAX_DOCUMENTS_TOTAL:
        raise ValueError(f"max_documents must be 0..{MAX_DOCUMENTS_TOTAL}")
    selected = discovery.loc[
        discovery["discovery_status"].fillna("").astype(str).eq("CANDIDATE_FOUND_UNVERIFIED")
    ].copy()
    selected["coin_id"] = selected["coin_id"].astype(str).str.strip()
    selected = selected[selected["coin_id"].ne("")]
    selected = selected.drop_duplicates(subset=["coin_id"], keep=False)
    priorities = _priority_table(diagnostic)
    def order_key(coin_id):
        values = priorities.get(coin_id, (None, None, None, None))
        return tuple(-x if x is not None else float("inf") for x in values)
    selected["_sort"] = selected["coin_id"].map(order_key)
    selected = selected.sort_values(["_sort", "coin_id"], kind="stable").head(max_candidates)
    owns_session = session is None
    if owns_session:
        session = requests.Session()
    rows, documents, fetched_urls = [], [], set()
    try:
        for _, item in selected.iterrows():
            row = {field: "" for field in FIELDS}
            for field in ("coin_id", "symbol", "name", "protocol_name", "protocol_slug",
                          "discovery_status", "candidate_url", "candidate_url_kind"):
                if field in selected.columns and pd.notna(item[field]):
                    row[field] = str(item[field]).strip()
            scores = priorities.get(row["coin_id"], (None, None, None, None))
            basis = next((key for key, value in zip(
                ("holder_value_score", "revenue_score", "opportunity_score_observed", "opportunity_score"),
                scores) if value is not None), "NO_DIAGNOSTIC_SCORE")
            row.update({"source_catalogue": "https://api.llama.fi/protocols",
                        "priority_score": next((v for v in scores if v is not None), ""),
                        "priority_basis": basis,
                        "evidence_scope": "HOMEPAGE_AND_DOCUMENTARY_LEADS_ONLY",
                        "economic_mechanism_verified": False,
                        "document_candidates_json": "[]", "document_candidate_count": 0})
            if not _valid_url(row["candidate_url"]):
                row.update(fetch_status="INVALID_CANDIDATE", audit_note="Invalid HTTPS catalogue URL")
            else:
                result = _fetch(row["candidate_url"], session, check_dns=check_dns)
                links = result.pop("links", [])
                result.pop("mentioned_topics", None)
                if result.get("fetch_status") == "RETRIEVED_UNVERIFIED":
                    result["fetch_status"] = "HOMEPAGE_RETRIEVED_UNVERIFIED"
                row.update(result)
                row["document_candidates_json"] = json.dumps(links, ensure_ascii=False)
                row["document_candidate_count"] = len(links)
                for link in links:
                    if len(documents) >= max_documents:
                        break
                    doc_url = link["url"]
                    if doc_url in fetched_urls:
                        continue
                    fetched_urls.add(doc_url)
                    doc = {field: "" for field in DOCUMENT_FIELDS}
                    doc.update({"coin_id": row["coin_id"], "symbol": row["symbol"],
                                "protocol_slug": row["protocol_slug"],
                                "source_homepage": row["candidate_url"],
                                "document_url": doc_url, "link_label": link["label"],
                                "link_topics": json.dumps(link["topics"], ensure_ascii=False),
                                "mechanism_documented": False, "period_verified": False,
                                "onchain_verified": False, "independent_source_verified": False,
                                "economic_mechanism_verified": False,
                                "evidence_scope": "DOCUMENT_CONTENT_RESEARCH_ONLY"})
                    fetched = _fetch(doc_url, session, check_dns=check_dns)
                    doc.update({"fetch_status": fetched.get("fetch_status", ""),
                                "http_status": fetched.get("http_status", ""),
                                "content_type": fetched.get("content_type", ""),
                                "content_sha256": fetched.get("content_sha256", ""),
                                "retrieved_at_utc": fetched.get("retrieved_at_utc", ""),
                                "document_excerpt": fetched.get("excerpt", ""),
                                "mentioned_topics": json.dumps(fetched.get("mentioned_topics", []), ensure_ascii=False),
                                "token_mentioned": bool(row["symbol"] and re.search(
                                    r"(?<![a-zA-Z0-9])" + re.escape(row["symbol"]) + r"(?![a-zA-Z0-9])",
                                    fetched.get("excerpt", ""), flags=re.I)),
                                "audit_note": fetched.get("audit_note", "")})
                    documents.append(doc)
            rows.append(row)
    finally:
        if owns_session:
            session.close()
    sources = pd.DataFrame(rows, columns=FIELDS)
    docs = pd.DataFrame(documents, columns=DOCUMENT_FIELDS)
    return (sources, docs) if return_documents else sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Holder Value documentary research")
    parser.add_argument("--discovery", default="output/holder_value_discovery.csv")
    parser.add_argument("--diagnostic", default="output/crypto_diagnostic_all.csv")
    parser.add_argument("--output", default="output/holder_value_auto_sources.csv")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--max-documents", type=int, default=MAX_DOCUMENTS_TOTAL)
    args = parser.parse_args()
    source = Path(args.discovery)
    if not source.is_file():
        parser.error(f"Discovery report not found: {source}")
    discovery = pd.read_csv(source, dtype=str, keep_default_na=False)
    diagnostic_path = Path(args.diagnostic)
    diagnostic = (pd.read_csv(diagnostic_path, dtype=str, keep_default_na=False)
                  if diagnostic_path.is_file() else None)
    results, docs = collect(discovery, args.max_candidates, diagnostic=diagnostic,
                            max_documents=args.max_documents, return_documents=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, encoding="utf-8-sig")
    doc_path = output.with_name("holder_value_auto_documents.csv")
    docs.to_csv(doc_path, index=False, encoding="utf-8-sig")
    claims = extract_claims(docs)
    claim_path = output.with_name("holder_value_auto_claims.csv")
    claims.to_csv(claim_path, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "diagnostic_used": diagnostic is not None,
        "selection_method": "holder_value, revenue, opportunity (descending); coin_id tie-break",
        "candidates_checked": len(results),
        "homepages_retrieved": int((results["fetch_status"] == "HOMEPAGE_RETRIEVED_UNVERIFIED").sum()),
        "document_links_discovered": int(pd.to_numeric(results["document_candidate_count"], errors="coerce").fillna(0).sum()),
        "documents_attempted": len(docs),
        "documents_retrieved": int((docs["fetch_status"] == "RETRIEVED_UNVERIFIED").sum()),
        "textual_leads_unverified": len(claims),
        "documents_with_textual_leads": int(claims["document_url"].nunique()),
        "claims_report": str(claim_path),
        "claim_extraction_scope": "FIRST_1200_CHARACTERS_OF_RETRIEVED_DOCUMENTS",
        "economic_mechanisms_verified": 0, "approval_flags_modified": False,
        "warning": "Textual leads are NOT proof of token value capture, payment, period or independent on-chain verification",
    }
    output.with_name("holder_value_auto_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
