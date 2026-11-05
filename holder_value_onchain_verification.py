"""CRYPTO OPPORTUNITY ENGINE — Fase 2B, verificação conservadora de contratos EVM.

Somente leitura. Confirma presença de bytecode no endereço candidato via JSON-RPC
(eth_getCode), NÃO valida identidade oficial do token, nem buyback/burn/staking.
Não altera scores, flags de evidência ou gates.

Uso:
 python holder_value_onchain_verification.py --input output/holder_value_onchain_candidates.csv --output output/holder_value_onchain_verification.csv

Dependências: pandas, requests. Configuração opcional: --rpc-map caminho.json,
com chaves platform_id do CoinGecko e valores URL HTTPS de RPC EVM.
Por segurança, o padrão não consulta RPCs externos; informe --rpc-map explicitamente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

REQUIRED = {"coin_id", "platform_id", "contract_address"}
ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
COLUMNS = [
    "coin_id", "symbol", "name", "platform_id", "contract_address",
    "verification_status", "rpc_chain_id", "bytecode_sha256", "bytecode_bytes",
    "rpc_endpoint_host", "checked_at_utc", "identity_verified",
    "economic_mechanism_verified", "audit_note",
]


def rpc_call(session: requests.Session, url: str, method: str, params: list, timeout: int) -> object:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or "error" in body or "result" not in body:
        raise ValueError(f"RPC response invalid for {method}")
    return body["result"]


def load_rpc_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("RPC map must be a JSON object")
    result = {}
    for platform, url in content.items():
        if not isinstance(platform, str) or not isinstance(url, str):
            raise ValueError("RPC map keys and values must be strings")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"RPC URL must be HTTPS with no embedded credentials: {platform}")
        result[platform.strip().lower()] = url
    return result


def verify_contracts(data: pd.DataFrame, rpc_map: dict[str, str], timeout: int = 12, delay: float = 0.15) -> pd.DataFrame:
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if data.empty:
        return pd.DataFrame(columns=COLUMNS)

    session = requests.Session()
    session.headers.update({"User-Agent": "CryptoOpportunityEngine-ReadOnly-EVMAudit/1.0"})
    cache: dict[tuple[str, str], dict] = {}
    chain_cache: dict[str, tuple[str, str | None]] = {}
    records = []

    for row in data.to_dict("records"):
        platform = str(row.get("platform_id") or "").strip().lower()
        address = str(row.get("contract_address") or "").strip()
        if address.lower() in {"nan", "none"}:
            address = ""
        stamp = datetime.now(timezone.utc).isoformat()
        record = {
            "coin_id": row.get("coin_id"), "symbol": row.get("symbol"),
            "name": row.get("name"), "platform_id": row.get("platform_id"),
            "contract_address": row.get("contract_address"),
            "verification_status": "NOT_CHECKED", "rpc_chain_id": None,
            "bytecode_sha256": None, "bytecode_bytes": None,
            "rpc_endpoint_host": None, "checked_at_utc": stamp,
            "identity_verified": False, "economic_mechanism_verified": False,
            "audit_note": "No on-chain check performed",
        }
        if not address:
            record.update(verification_status="NO_CONTRACT_ADDRESS", audit_note="Native asset or no catalogue contract; not a failure")
        elif not ADDRESS.fullmatch(address):
            record.update(verification_status="NON_EVM_ADDRESS_OR_INVALID_SHAPE", audit_note="Not a 20-byte hex EVM address; chain-specific adapter required")
        elif platform not in rpc_map:
            record.update(verification_status="RPC_NOT_CONFIGURED", audit_note="No explicit RPC mapping for platform; no network request made")
        else:
            url = rpc_map[platform]
            record["rpc_endpoint_host"] = urlparse(url).hostname
            key = (platform, address.lower())
            if key not in cache:
                try:
                    if platform not in chain_cache:
                        chain_id = rpc_call(session, url, "eth_chainId", [], timeout)
                        if not isinstance(chain_id, str) or not chain_id.startswith("0x"):
                            raise ValueError("Invalid eth_chainId")
                        chain_cache[platform] = (str(int(chain_id, 16)), None)
                    chain_id = chain_cache[platform][0]
                    code = rpc_call(session, url, "eth_getCode", [address, "latest"], timeout)
                    if not isinstance(code, str) or not code.startswith("0x") or len(code) % 2 != 0:
                        raise ValueError("Invalid eth_getCode")
                    raw = bytes.fromhex(code[2:])
                    cache[key] = {
                        "verification_status": "EVM_BYTECODE_PRESENT" if raw else "NO_BYTECODE_AT_ADDRESS",
                        "rpc_chain_id": chain_id,
                        "bytecode_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
                        "bytecode_bytes": len(raw),
                        "audit_note": "Bytecode exists; token identity and economic mechanisms NOT verified" if raw else "No deployed bytecode at latest block; may be an EOA or incorrect chain",
                    }
                except (requests.RequestException, ValueError, TypeError, OverflowError) as exc:
                    cache[key] = {"verification_status": "RPC_ERROR", "audit_note": f"RPC query failed: {type(exc).__name__}: {str(exc)[:160]}"}
                if delay > 0:
                    time.sleep(delay)
            record.update(cache[key])
        records.append(record)
    return pd.DataFrame(records, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2B: validação somente leitura de bytecode EVM")
    parser.add_argument("--input", required=True, help="CSV de contratos candidatos da Fase 2A")
    parser.add_argument("--output", default="output/holder_value_onchain_verification.csv")
    parser.add_argument("--rpc-map", help="JSON platform_id -> HTTPS RPC URL; sem este argumento não há consultas externas")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()
    if args.timeout < 1 or args.delay < 0:
        parser.error("timeout must be >=1 and delay must be >=0")
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Input CSV not found: {source}")
    try:
        rpc_map = load_rpc_map(args.rpc_map)
        data = pd.read_csv(source, dtype={"coin_id": "string", "platform_id": "string", "contract_address": "string"}, low_memory=False)
        result = verify_contracts(data, rpc_map, timeout=args.timeout, delay=args.delay)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[phase2b] Input records: {len(data)}")
    print(f"[phase2b] RPC networks configured: {len(rpc_map)}")
    print("[phase2b] Verification status counts:")
    print(result["verification_status"].value_counts(dropna=False).to_string())
    print(f"[phase2b] Output: {output}")
    print("[phase2b] No economic evidence, score, or gate was modified.")


if __name__ == "__main__":
    main()
