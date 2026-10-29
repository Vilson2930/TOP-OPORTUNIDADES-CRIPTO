"""CRYPTO OPPORTUNITY ENGINE — auditoria de evidências Holder Value.

Camada independente, somente leitura. Não atribui scores, não cria flags de
verificação e não altera os gates. Gera uma fila reproduzível de investigação
para os ativos presentes no diagnóstico consolidado do motor.

Uso:
    python holder_value_evidence_audit.py --input crypto_diagnostic_all.csv
    python holder_value_evidence_audit.py --input crypto_diagnostic_all.csv --output holder_value_evidence_audit.csv

Dependência: pandas.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


MECHANISMS = ("buyback", "burn")
ECONOMIC_FIELDS = (
    "real_yield_score", "fee_distribution_score", "buyback_score",
    "burn_score", "token_usage_score", "economic_staking_score",
)
EVIDENCE_FIELDS = ("source", "period", "reference")
OUTPUT_DEFAULT = "holder_value_evidence_audit.csv"


def _value(row: pd.Series, field: str) -> Any:
    return row[field] if field in row.index else None


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, bool):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null", "0", "false", "true"}


def _verified(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def _mechanism_status(row: pd.Series, mechanism: str) -> tuple[str, str]:
    prefix = f"{mechanism}_evidence_"
    flag = _verified(_value(row, prefix + "verified"))
    missing = [field for field in EVIDENCE_FIELDS if not _present(_value(row, prefix + field))]
    if flag and not missing:
        return "DECLARED_VERIFIED", "Exige auditoria externa da fonte e da operação"
    if flag:
        return "INCOMPLETE_VERIFICATION", "Faltam: " + ", ".join(missing)
    if len(missing) < len(EVIDENCE_FIELDS):
        return "DOCUMENTATION_PENDING", "Documentação parcial; verificação não declarada"
    return "NO_DOCUMENTED_EVIDENCE", "Sem fonte, período e referência operacional"


def _numeric_present(row: pd.Series, field: str) -> bool:
    value = pd.to_numeric(_value(row, field), errors="coerce")
    return bool(pd.notna(value))


def build_audit_queue(dataset: pd.DataFrame) -> pd.DataFrame:
    """Produz diagnósticos; nunca altera ou promove o dataset de entrada."""
    if dataset is None or dataset.empty:
        return pd.DataFrame(columns=[
            "symbol", "name", "holder_value_score", "holder_value_quality_gate",
            "holder_value_evidence_count", "holder_value_confidence",
            "buyback_audit_status", "buyback_audit_note", "burn_audit_status",
            "burn_audit_note", "missing_economic_components", "audit_priority",
        ])

    records = []
    for _, row in dataset.iterrows():
        buyback_status, buyback_note = _mechanism_status(row, "buyback")
        burn_status, burn_note = _mechanism_status(row, "burn")
        missing = [field for field in ECONOMIC_FIELDS if not _numeric_present(row, field)]
        score = pd.to_numeric(_value(row, "holder_value_score"), errors="coerce")
        evidence_count = pd.to_numeric(_value(row, "holder_value_evidence_count"), errors="coerce")
        confidence = pd.to_numeric(_value(row, "holder_value_confidence"), errors="coerce")
        gate = _verified(_value(row, "holder_value_quality_gate"))
        # Ordenação operacional, não sinal de compra ou recomendação.
        if not gate and pd.notna(score) and score >= 40:
            priority = "REVIEW_EVIDENCE"
        elif not gate:
            priority = "DATA_INSUFFICIENT_OR_BELOW_GATE"
        else:
            priority = "GATE_PASSED_REVIEW_PROVENANCE"
        records.append({
            "symbol": _value(row, "symbol"),
            "name": _value(row, "name"),
            "coin_id": _value(row, "coin_id"),
            "holder_value_score": score,
            "holder_value_quality_gate": gate,
            "holder_value_evidence_count": evidence_count,
            "holder_value_confidence": confidence,
            "holder_value_evidence_status": _value(row, "holder_value_evidence_status"),
            "holder_value_data_source": _value(row, "holder_value_data_source"),
            "holder_value_protocol_slug": _value(row, "holder_value_protocol_slug"),
            "holder_value_holders_match_status": _value(row, "holder_value_holders_match_status"),
            "holder_value_fee_ratio_verified": _verified(_value(row, "holder_value_fee_ratio_verified")),
            "buyback_audit_status": buyback_status,
            "buyback_audit_note": buyback_note,
            "burn_audit_status": burn_status,
            "burn_audit_note": burn_note,
            "missing_economic_components": ";".join(missing),
            "audit_priority": priority,
            "audit_disclaimer": "Diagnóstico documental; nenhuma evidência validada por este script",
        })

    result = pd.DataFrame.from_records(records)
    order = {
        "REVIEW_EVIDENCE": 0,
        "DATA_INSUFFICIENT_OR_BELOW_GATE": 1,
        "GATE_PASSED_REVIEW_PROVENANCE": 2,
    }
    result["_priority"] = result["audit_priority"].map(order)
    result = result.sort_values(
        ["_priority", "holder_value_score"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    ).drop(columns="_priority")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria somente leitura de evidências Holder Value")
    parser.add_argument("--input", required=True, help="CSV completo de diagnóstico do motor")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="CSV da fila de auditoria")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Arquivo de entrada não encontrado: {source}")
    dataset = pd.read_csv(source, low_memory=False)
    audit = build_audit_queue(dataset)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[holder_value_audit] Ativos: {len(audit)}")
    print(f"[holder_value_audit] Revisão de evidências: {(audit['audit_priority'] == 'REVIEW_EVIDENCE').sum() if not audit.empty else 0}")
    print(f"[holder_value_audit] Relatório: {target}")
    print("[holder_value_audit] Nenhum gate, score ou flag de verificação foi modificado.")


if __name__ == "__main__":
    main()
