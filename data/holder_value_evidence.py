"""CRYPTO OPPORTUNITY ENGINE — independent EVM block-burn evidence (read only).

Scope: verify EIP-1559 base-fee burn for *individual EVM blocks* using two
independent RPC endpoints, without extrapolating a sampled block to 30 days.
This module DOES NOT verify buybacks, holder distributions, real yield, or
30-day burn totals, and NEVER writes holder_value_engine approval flags.

Run:
 python -m data.holder_value_evidence --input fase3h_blocos_evm_amostrados.csv \
   --output-dir output/evidence_eth
 python -m data.holder_value_evidence --input fase3i_blocos_diarios_verificados.csv \
   --output-dir output/evidence_avax

RPC URLs can be overridden with environment variables:
 HOLDER_EVIDENCE_ETH_RPC_1, HOLDER_EVIDENCE_ETH_RPC_2,
 HOLDER_EVIDENCE_AVAX_RPC_1, HOLDER_EVIDENCE_AVAX_RPC_2.

Dependency: requests. No keys, wallets or private data are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

RPC_DEFAULTS = {
    "ETH": (
        "https://ethereum-rpc.publicnode.com",
        "https://eth.llamarpc.com",
    ),
    "AVAX": (
        "https://api.avax.network/ext/bc/C/rpc",
        "https://avalanche-c-chain-rpc.publicnode.com",
    ),
}
CHAIN_IDS = {"ETH": 1, "AVAX": 43114}
MAX_ROWS = 500  # Prevent accidental multi-million-block scans.
MAX_RETRIES = 2
TIMEOUT_SECONDS = 15
HASH_PATTERN = re.compile(r"^0x[0-9a-fA-F]{64}$")


class EvidenceError(Exception):
    """RPC or evidence validation failure; never treated as verification."""


def _hex_int(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]+", value):
        raise EvidenceError("Invalid RPC hex integer")
    return int(value, 16)


def _rpc(session: requests.Session, url: str, method: str, params: list) -> Any:
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("error") or "result" not in payload:
                raise EvidenceError("RPC response missing result or returned error")
            return payload["result"]
        except (requests.RequestException, ValueError, EvidenceError) as exc:
            if attempt + 1 == MAX_RETRIES:
                raise EvidenceError(f"{method}: {type(exc).__name__}: {exc}") from exc
            time.sleep(0.4 * (attempt + 1))
    raise EvidenceError("RPC attempts exhausted")


def _endpoints(symbol: str) -> tuple[str, str]:
    if symbol not in RPC_DEFAULTS:
        raise EvidenceError(f"Unsupported chain: {symbol}")
    defaults = RPC_DEFAULTS[symbol]
    urls = tuple(os.getenv(f"HOLDER_EVIDENCE_{symbol}_RPC_{i}", defaults[i - 1]).strip()
                 for i in (1, 2))
    if not all(url.startswith("https://") for url in urls) or urls[0] == urls[1]:
        raise EvidenceError("Two distinct HTTPS RPC URLs are required")
    return urls


def _validate_block(block: Any, expected_height: int, expected_chain: int) -> dict:
    if not isinstance(block, dict):
        raise EvidenceError("Missing block")
    height = _hex_int(block.get("number"))
    if height != expected_height:
        raise EvidenceError("Block height mismatch")
    block_hash = block.get("hash")
    if not isinstance(block_hash, str) or not HASH_PATTERN.fullmatch(block_hash):
        raise EvidenceError("Invalid block hash")
    base_fee = _hex_int(block.get("baseFeePerGas"))
    gas_used = _hex_int(block.get("gasUsed"))
    timestamp = _hex_int(block.get("timestamp"))
    if base_fee < 0 or gas_used < 0 or timestamp <= 0:
        raise EvidenceError("Invalid burn inputs")
    return {
        "chain_id": expected_chain,
        "block_height": height,
        "block_hash": block_hash.lower(),
        "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        "base_fee_wei": base_fee,
        "gas_used": gas_used,
        "burn_wei_exact_for_block": base_fee * gas_used,
    }


def verify_sample(row: dict, session: requests.Session | None = None) -> dict:
    """Verify one supplied block against two RPCs; no approval fields emitted."""
    symbol = str(row.get("symbol", "")).strip().upper()
    result = {
        "symbol": symbol,
        "block_height": row.get("block_height", ""),
        "status": "NOT_VERIFIED",
        "proof_scope": "ONE_EVM_BLOCK_BASE_FEE_BURN_ONLY",
        "source_file_proof_scope": row.get("proof_scope", ""),
        "economic_capture_verified": False,
        "holder_value_gate_approved": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if symbol not in CHAIN_IDS:
            raise EvidenceError("Unsupported symbol; no chain inference")
        height = int(str(row.get("block_height", "")).strip())
        if height < 0:
            raise EvidenceError("Negative block height")
        supplied_hash = str(row.get("block_hash", "")).strip().lower()
        if not HASH_PATTERN.fullmatch(supplied_hash):
            raise EvidenceError("Supplied block hash is missing/invalid")
        urls = _endpoints(symbol)
        s = session or requests.Session()
        observations = []
        for url in urls:
            chain_id = _hex_int(_rpc(s, url, "eth_chainId", []))
            if chain_id != CHAIN_IDS[symbol]:
                raise EvidenceError("RPC chain ID mismatch")
            block = _rpc(s, url, "eth_getBlockByNumber", [hex(height), False])
            observation = _validate_block(block, height, chain_id)
            observation["rpc_url"] = url
            observations.append(observation)
        comparable = ("chain_id", "block_height", "block_hash", "timestamp_utc",
                      "base_fee_wei", "gas_used", "burn_wei_exact_for_block")
        if any(observations[0][k] != observations[1][k] for k in comparable):
            raise EvidenceError("Independent RPC observations conflict")
        verified = observations[0]
        if verified["block_hash"] != supplied_hash:
            raise EvidenceError("Supplied block hash differs from RPC consensus")
        # Source snapshots are corroborative; decimal values are not compared
        # because CSV display may have rounded them.
        if row.get("base_fee_wei") not in (None, ""):
            if int(str(row["base_fee_wei"])) != verified["base_fee_wei"]:
                raise EvidenceError("Source base fee differs from RPC consensus")
        if row.get("gas_used") not in (None, ""):
            if int(str(row["gas_used"])) != verified["gas_used"]:
                raise EvidenceError("Source gas used differs from RPC consensus")
        result.update(verified)
        result.update({
            "status": "VERIFIED_BLOCK_BURN_ONLY",
            "rpc_count": 2,
            "rpc_endpoints": urls,
            "burn_native_exact_for_block": str(
                Decimal(verified["burn_wei_exact_for_block"]) / Decimal(10**18)),
            "evidence_sha256": hashlib.sha256(json.dumps(
                {k: verified[k] for k in comparable}, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
        })
    except (EvidenceError, ValueError, TypeError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result


def verify_csv(input_path: str | Path, output_dir: str | Path,
               session: requests.Session | None = None) -> dict:
    """Bounded, reproducible audit; never mutates the investment dataset."""
    source = Path(input_path)
    destination = Path(output_dir)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"symbol", "block_height", "block_hash"}.issubset(reader.fieldnames):
            raise EvidenceError("CSV needs symbol, block_height, block_hash")
        rows = []
        for row in reader:
            if len(rows) >= MAX_ROWS:
                raise EvidenceError(f"More than {MAX_ROWS} blocks: refusing unbounded scan")
            rows.append(row)
    s = session or requests.Session()
    results = [verify_sample(row, session=s) for row in rows]
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "source_csv": source.name,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(results),
        "verified_block_count": sum(r["status"] == "VERIFIED_BLOCK_BURN_ONLY" for r in results),
        "unverified_block_count": sum(r["status"] != "VERIFIED_BLOCK_BURN_ONLY" for r in results),
        "thirty_day_total_verified": False,
        "holder_economic_capture_verified": False,
        "approval_flags_modified": False,
        "records": results,
    }
    (destination / "holder_value_evidence_blocks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ("symbol", "block_height", "status", "block_hash", "timestamp_utc",
              "base_fee_wei", "gas_used", "burn_wei_exact_for_block",
              "burn_native_exact_for_block", "evidence_sha256", "reason", "proof_scope")
    with (destination / "holder_value_evidence_blocks.csv").open(
        "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing fase3h/fase3i CSV")
    parser.add_argument("--output-dir", default="output/holder_value_evidence")
    args = parser.parse_args()
    report = verify_csv(args.input, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
