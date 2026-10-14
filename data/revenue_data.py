"""CRYPTO OPPORTUNITY ENGINE — Revenue Data Layer.

Coleta agregada gratuita de DefiLlama Dimensions, com correspondência conservadora.
Não modifica scoring, pesos, Quality Gates ou outros módulos.
Valores não disponíveis permanecem NaN; taxas não são tratadas como receita.
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

REQUEST_TIMEOUT = 25
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2.0
DEFILLAMA_FEES_OVERVIEW_URL = "https://api.llama.fi/overview/fees"
DATA_TYPES = ("dailyRevenue", "dailyFees")
USER_AGENT = "CRYPTO-OPPORTUNITY-ENGINE/1.1 (+revenue-data-layer)"


def _safe_float(value) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _first_available(row: pd.Series, fields: Iterable[str]) -> float:
    for field in fields:
        if field in row.index:
            value = _safe_float(row.get(field))
            if pd.notna(value):
                return value
    return np.nan


def _normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def _normalize_symbol(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().strip())


def _annualize_period_value(total_1y, total_30d, total_7d, total_24h) -> float:
    for value, multiplier in ((total_1y, 1.0), (total_30d, 365 / 30),
                              (total_7d, 365 / 7), (total_24h, 365)):
        v = _safe_float(value)
        if pd.notna(v) and v >= 0:
            return float(v * multiplier)
    return np.nan


def _annualization_basis(record: dict) -> str:
    for key, label in (("total1y", "TRAILING_1Y"), ("total30d", "RUN_RATE_30D"),
                       ("total7d", "RUN_RATE_7D"), ("total24h", "RUN_RATE_24H")):
        v = _safe_float(record.get(key))
        if pd.notna(v) and v >= 0:
            return label
    return "UNAVAILABLE"


def _calculate_short_term_growth(total_30d, total_7d) -> float:
    """Últimos 7 dias versus 23 dias anteriores, em médias diárias disjuntas.

    Não equivale a crescimento 90d; o campo legado 'revenue_growth' permanece
    sendo um proxy de aceleração recente para preservar a interface do engine.
    """
    t30, t7 = _safe_float(total_30d), _safe_float(total_7d)
    if pd.isna(t30) or pd.isna(t7) or t30 <= 0 or t7 < 0 or t7 > t30:
        return np.nan
    previous_23 = t30 - t7
    if previous_23 <= 0:
        return np.nan
    return float((t7 / 7) / (previous_23 / 23) - 1)


def _request_overview(session: requests.Session, data_type: str) -> Optional[dict]:
    params = {"excludeTotalDataChart": "true",
              "excludeTotalDataChartBreakdown": "true", "dataType": data_type}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(DEFILLAMA_FEES_OVERVIEW_URL, params=params,
                                   timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("protocols"), list):
                    return payload
                print(f"[revenue_data] {data_type}: formato inesperado.")
                return None
            if response.status_code in (401, 403, 404, 429):
                print(f"[revenue_data] {data_type}: HTTP {response.status_code}; sem dados.")
                return None
            if response.status_code >= 500 and attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue
            print(f"[revenue_data] {data_type}: HTTP {response.status_code}.")
            return None
        except (requests.RequestException, ValueError) as exc:
            print(f"[revenue_data] {data_type}: tentativa {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
    return None


def _extract_protocols(payload) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    return [p for p in payload.get("protocols", []) if isinstance(p, dict)]


def _protocol_record(protocol: dict, data_type: str) -> Dict:
    record = {
        "data_type": data_type,
        "protocol_name": protocol.get("name"),
        "protocol_display_name": protocol.get("displayName"),
        "protocol_slug": protocol.get("slug"),
        "protocol_symbol": protocol.get("symbol"),
        "total_24h": _safe_float(protocol.get("total24h")),
        "total_7d": _safe_float(protocol.get("total7d")),
        "total_30d": _safe_float(protocol.get("total30d")),
        "total_1y": _safe_float(protocol.get("total1y")),
        "annualization_basis": _annualization_basis(protocol),
    }
    record["name_key"] = _normalize_text(record["protocol_name"])
    record["display_name_key"] = _normalize_text(record["protocol_display_name"])
    record["slug_key"] = _normalize_text(record["protocol_slug"])
    record["symbol_key"] = _normalize_symbol(record["protocol_symbol"])
    record["annualized"] = _annualize_period_value(record["total_1y"],
        record["total_30d"], record["total_7d"], record["total_24h"])
    record["short_term_growth"] = _calculate_short_term_growth(
        record["total_30d"], record["total_7d"])
    return record


def _build_dimension_table(session: requests.Session, data_type: str) -> pd.DataFrame:
    payload = _request_overview(session, data_type)
    records = [_protocol_record(p, data_type) for p in _extract_protocols(payload)]
    table = pd.DataFrame(records)
    return table[table["annualized"].notna()].copy() if not table.empty else table


def _candidate_matches(row: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Somente match inequívoco. Símbolos isolados podem ser reutilizados."""
    if table.empty:
        return table
    name = _normalize_text(row.get("name"))
    coin_id = _normalize_text(row.get("coin_id"))
    symbol = _normalize_symbol(row.get("symbol"))
    keys = ("name_key", "display_name_key", "slug_key")
    # Nome completo tem prioridade; ID CoinGecko e slug DefiLlama não são
    # necessariamente iguais e nunca são tratados como equivalência garantida.
    for identity in (name, coin_id):
        if not identity:
            continue
        mask = pd.Series(False, index=table.index)
        for key in keys:
            mask |= table[key].eq(identity)
        candidates = table.loc[mask]
        if len(candidates) == 1:
            return candidates
        if len(candidates) > 1:
            # Uma coincidência textual múltipla não pode ser resolvida por receita.
            return table.iloc[0:0]
    if symbol:
        candidates = table.loc[table["symbol_key"].eq(symbol)]
        if len(candidates) == 1:
            # Sem correspondência nominal, um ticker não prova identidade.
            # Preservar NaN é preferível a atribuir receita de outro projeto.
            return table.iloc[0:0]
    return table.iloc[0:0]


def _best_match(row: pd.Series, table: pd.DataFrame) -> Optional[pd.Series]:
    matches = _candidate_matches(row, table)
    return matches.iloc[0] if len(matches) == 1 else None


def _existing_revenue(row: pd.Series) -> float:
    return _first_available(row, ("annualized_revenue", "revenue_annualized",
        "protocol_revenue_annualized", "revenue_365d"))


def _existing_fees(row: pd.Series) -> float:
    return _first_available(row, ("annualized_fees", "fees_annualized",
        "protocol_fees_annualized", "fees_365d"))


def _evaluate_asset(row: pd.Series, revenue_table: pd.DataFrame,
                    fees_table: pd.DataFrame) -> Dict:
    rev = _best_match(row, revenue_table)
    fees = _best_match(row, fees_table)
    revenue_value = _safe_float(rev.get("annualized")) if rev is not None else _existing_revenue(row)
    fees_value = _safe_float(fees.get("annualized")) if fees is not None else _existing_fees(row)
    revenue_growth = _safe_float(rev.get("short_term_growth")) if rev is not None else np.nan
    fees_growth = _safe_float(fees.get("short_term_growth")) if fees is not None else np.nan
    available = sum(pd.notna(v) for v in (revenue_value, fees_value, revenue_growth, fees_growth))
    return {
        "annualized_revenue": revenue_value,
        "annualized_fees": fees_value,
        "revenue_growth": revenue_growth,
        "fees_growth": fees_growth,
        "revenue_data_protocol_slug": rev.get("protocol_slug") if rev is not None else None,
        "revenue_data_protocol_name": rev.get("protocol_name") if rev is not None else None,
        "fees_data_protocol_slug": fees.get("protocol_slug") if fees is not None else None,
        "fees_data_protocol_name": fees.get("protocol_name") if fees is not None else None,
        "revenue_data_completeness_source": round(available / 4, 4),
        "revenue_data_source": "DEFILLAMA_DIMENSIONS" if rev is not None or fees is not None else (
            "EXISTING_DATA" if available else "DATA_INSUFFICIENT"),
        "revenue_annualization_basis": rev.get("annualization_basis") if rev is not None else None,
        "fees_annualization_basis": fees.get("annualization_basis") if fees is not None else None,
        "revenue_growth_method": "LAST_7D_VS_PREVIOUS_23D" if pd.notna(revenue_growth) else None,
        "fees_growth_method": "LAST_7D_VS_PREVIOUS_23D" if pd.notna(fees_growth) else None,
    }


def enrich_with_revenue_data(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset is None:
        return pd.DataFrame()
    if dataset.empty:
        return dataset.copy()
    result = dataset.copy()
    print(f"[revenue_data] Processando {len(result)} ativos; coleta agregada.")
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        revenue_table = _build_dimension_table(session, "dailyRevenue")
        fees_table = _build_dimension_table(session, "dailyFees")
    print(f"[revenue_data] Protocolos: Revenue={len(revenue_table)}, Fees={len(fees_table)}.")
    evaluations = [_evaluate_asset(row, revenue_table, fees_table)
                   for _, row in result.iterrows()]
    evaluation_df = pd.DataFrame(evaluations, index=result.index)
    for column in evaluation_df.columns:
        result[column] = evaluation_df[column]
    for col in ("annualized_revenue", "annualized_fees", "revenue_growth", "fees_growth"):
        print(f"[revenue_data] {col}: {int(result[col].notna().sum())}/{len(result)} ativos.")
    print("[revenue_data] Match ambíguo rejeitado; ausências preservadas.")
    return result


if __name__ == "__main__":
    from data.market_data import build_market_dataset
    market = build_market_dataset(include_history=False)
    enriched = enrich_with_revenue_data(market)
    columns = ["symbol", "name", "annualized_revenue", "annualized_fees",
               "revenue_growth", "fees_growth", "revenue_data_source"]
    print(enriched[[c for c in columns if c in enriched.columns]].head(50).to_string(index=False))
