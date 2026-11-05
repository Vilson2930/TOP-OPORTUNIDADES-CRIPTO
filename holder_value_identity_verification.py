"""CRYPTO OPPORTUNITY ENGINE — Fase 2C: auditoria conservadora de identidade EVM.

Somente leitura. Consulta ERC-20 name(), symbol(), decimals() em contratos com
bytecode confirmado na Fase 2B. Metadados on-chain são autodeclarados: uma
correspondência NÃO comprova identidade oficial. Nunca emite IDENTITY_VERIFIED
sem evidência externa independente (não implementada nesta fase).

Uso:
 python holder_value_identity_verification.py \
   --input output/holder_value_onchain_verification.csv \
   --output output/holder_value_identity_verification.csv \
   --rpc-map output/holder_value_rpc_map.json

Requer: requests, pandas. Não altera outros arquivos, gates, scores ou evidências.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

# Chain IDs esperados: mapeamento fechado, sem inferir rede a partir do endereço.
CHAIN_IDS = {
    "ethereum": 1,
    "binance-smart-chain": 56,
    "base": 8453,
    "arbitrum-one": 42161,
    "polygon-pos": 137,
    "avalanche": 43114,
    "optimistic-ethereum": 10,
}
SELECTORS = {"name": "0x06fdde03", "symbol": "0x95d89b41", "decimals": "0x313ce567"}
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
HEX_RE = re.compile(r"^0x(?:[0-9a-fA-F]{2})*$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def decode_abi_string(hex_result):
    """Decode dynamic ABI string or legacy bytes32; reject malformed payloads."""
    if not isinstance(hex_result, str) or not HEX_RE.fullmatch(hex_result):
        raise ValueError("invalid_hex_response")
    data = bytes.fromhex(hex_result[2:])
    if len(data) == 32:
        raw = data.rstrip(b"\x00")
    elif len(data) >= 64:
        offset = int.from_bytes(data[:32], "big")
        if offset % 32 or offset + 32 > len(data):
            raise ValueError("invalid_abi_offset")
        length = int.from_bytes(data[offset:offset + 32], "big")
        if length > 1024 or offset + 32 + length > len(data):
            raise ValueError("invalid_abi_length")
        raw = data[offset + 32:offset + 32 + length]
    else:
        raise ValueError("empty_or_short_abi")
    value = raw.decode("utf-8", errors="strict").strip("\x00 \t\r\n")
    if not value or any(ord(ch) < 32 for ch in value):
        raise ValueError("invalid_abi_text")
    return value[:256]


def decode_decimals(hex_result):
    if not isinstance(hex_result, str) or not HEX_RE.fullmatch(hex_result):
        raise ValueError("invalid_hex_response")
    raw = bytes.fromhex(hex_result[2:])
    if len(raw) != 32:
        raise ValueError("invalid_decimals_abi")
    number = int.from_bytes(raw, "big")
    if number > 255:
        raise ValueError("out_of_range_decimals")
    return number


def rpc(session, endpoint, method, params, timeout, retries):
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            response = session.post(
                endpoint,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=timeout,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"http_{response.status_code}")
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
                raise RuntimeError("invalid_jsonrpc_response")
            if "error" in body:
                raise RuntimeError("rpc_error:" + str(body["error"])[:160])
            if "result" not in body:
                raise RuntimeError("missing_rpc_result")
            return body["result"]
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc)[:180]
            if attempt < retries:
                time.sleep(min(2 ** attempt, 3))
    raise RuntimeError(last_error)


def load_rpc_map(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RPC map must be an object")
    result = {}
    for platform, value in raw.items():
        if platform not in CHAIN_IDS:
            continue  # fail closed on unrecognized platforms
        if not isinstance(value, str):
            raise ValueError(f"RPC URL must be a string: {platform}")
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"Only HTTPS RPC endpoints are supported: {platform}")
        result[platform] = value
    return result


def classify(expected_name, expected_symbol, actual_name, actual_symbol):
    en, es = normalize_text(expected_name), normalize_text(expected_symbol)
    an, ass = normalize_text(actual_name), normalize_text(actual_symbol)
    name_match = bool(en and an and en == an)
    symbol_match = bool(es and ass and es == ass)
    if name_match and symbol_match:
        return "IDENTITY_PARTIAL", name_match, symbol_match, "Name and symbol match; self-reported contract metadata is not independent proof"
    if (en and an and en != an) or (es and ass and es != ass):
        return "IDENTITY_CONFLICT", name_match, symbol_match, "Catalog and contract metadata differ; bridged/wrapped tokens can legitimately differ"
    return "IDENTITY_UNVERIFIED", name_match, symbol_match, "Insufficient ERC-20 metadata for comparison"


def run(args):
    source = Path(args.input)
    destination = Path(args.output)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must differ")
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {"coin_id", "symbol", "name", "platform_id", "contract_address", "verification_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    rpc_map = load_rpc_map(args.rpc_map)
    session = requests.Session()
    session.headers.update({"User-Agent": "CryptoOpportunityEngine-IdentityAudit/1.0"})
    chain_checks = {}
    for platform, endpoint in rpc_map.items():
        try:
            actual = rpc(session, endpoint, "eth_chainId", [], args.timeout, args.retries)
            actual_id = int(actual, 16)
            chain_checks[platform] = (actual_id == CHAIN_IDS[platform], actual_id, "")
        except Exception as exc:
            chain_checks[platform] = (False, None, str(exc)[:180])

    cache = {}
    rows = []
    for i, record in enumerate(df.to_dict("records"), 1):
        platform = record["platform_id"]
        address = record["contract_address"]
        original_status = record["verification_status"]
        row = {
            "coin_id": record["coin_id"], "symbol": record["symbol"], "name": record["name"],
            "platform_id": platform, "contract_address": address,
            "phase2b_status": original_status, "identity_status": "IDENTITY_UNVERIFIED",
            "onchain_name": "", "onchain_symbol": "", "onchain_decimals": "",
            "name_match": False, "symbol_match": False,
            "rpc_chain_id": "", "rpc_endpoint_host": "", "checked_at_utc": utc_now(),
            "identity_verified": False, "economic_mechanism_verified": False,
            "audit_note": "", "rpc_error": "",
        }
        if original_status != "EVM_BYTECODE_PRESENT":
            row["audit_note"] = "Phase 2B did not confirm EVM bytecode; not queried"
        elif not ADDRESS_RE.fullmatch(address):
            row["audit_note"] = "Invalid EVM address despite Phase 2B status"
        elif platform not in CHAIN_IDS or platform not in rpc_map:
            row["audit_note"] = "No supported RPC configured for platform"
        elif not chain_checks[platform][0]:
            row["audit_note"] = "RPC chain ID mismatch or unavailable; no contract calls made"
            row["rpc_error"] = chain_checks[platform][2] or "chain_id_mismatch"
        else:
            endpoint = rpc_map[platform]
            row["rpc_chain_id"] = CHAIN_IDS[platform]
            row["rpc_endpoint_host"] = urlparse(endpoint).hostname
            key = (platform, address.lower())
            if key not in cache:
                metadata = {}
                errors = []
                for field, selector in SELECTORS.items():
                    try:
                        result = rpc(session, endpoint, "eth_call", [{"to": address, "data": selector}, "latest"], args.timeout, args.retries)
                        metadata[field] = decode_decimals(result) if field == "decimals" else decode_abi_string(result)
                    except Exception as exc:
                        errors.append(f"{field}:{str(exc)[:100]}")
                cache[key] = (metadata, "; ".join(errors))
            metadata, errors = cache[key]
            row["onchain_name"] = metadata.get("name", "")
            row["onchain_symbol"] = metadata.get("symbol", "")
            row["onchain_decimals"] = metadata.get("decimals", "")
            row["rpc_error"] = errors
            status, nm, sm, note = classify(record["name"], record["symbol"], row["onchain_name"], row["onchain_symbol"])
            row["identity_status"] = status
            row["name_match"] = nm
            row["symbol_match"] = sm
            row["audit_note"] = note
        rows.append(row)
        if i % 50 == 0:
            print(f"[phase2c] Processed {i}/{len(df)} records", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(destination, index=False, encoding="utf-8")
    print(f"[phase2c] Input records: {len(df)}")
    print(f"[phase2c] RPC networks configured: {len(rpc_map)}")
    print("[phase2c] Identity status counts:")
    print(result_df["identity_status"].value_counts(dropna=False).to_string())
    print("[phase2c] RPC chain checks:")
    for platform, (ok, actual, error) in chain_checks.items():
        print(f"  {platform}: {'OK' if ok else 'FAIL'} (chain_id={actual}, error={error})")
    print(f"[phase2c] Output: {destination}")
    print("[phase2c] No identity independently verified; no economic evidence, scores or gates modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rpc-map", required=True)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    if args.timeout <= 0 or not (0 <= args.retries <= 4):
        parser.error("timeout must be positive; retries must be 0..4")
    run(args)


if __name__ == "__main__":
    main()
