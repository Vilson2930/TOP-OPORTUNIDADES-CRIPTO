"""
CRYPTO OPPORTUNITY ENGINE
Holder Value Data Layer

OBJETIVO:
Fornecer ao holder_value_engine dados observáveis de captura de valor
pelo token/holder sem transformar ausência de informação em nota zero.

PRINCÍPIOS:
- protocolo bom != token bom;
- receita do protocolo != receita do holder;
- yield de emissão não é tratado como real yield;
- ausência de dado permanece NaN;
- nenhuma métrica qualitativa (buyback, burn, utility, staking econômico)
  é inventada;
- evitar chamadas HTTP individuais por ativo.

FONTE PRIMÁRIA:
DefiLlama Dimensions / Fees overview, usando o dataset agregado de
Holders Revenue quando disponível.

A camada faz poucas chamadas agregadas e depois associa os protocolos
aos ativos por symbol/name/coin_id. Não faz uma requisição por ativo.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÃO LOCAL
# ============================================================

REQUEST_TIMEOUT = 20
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2.0

DEFILLAMA_FEES_OVERVIEW_URL = "https://api.llama.fi/overview/fees"

# Somente datasets agregados. Nunca uma URL por ativo.
DATA_TYPES = (
    "dailyHoldersRevenue",
    "dailyRevenue",
    "dailyFees",
)

USER_AGENT = (
    "CRYPTO-OPPORTUNITY-ENGINE/1.0 "
    "(+holder-value-data-layer)"
)


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value) -> float:
    try:
        if value is None:
            return np.nan

        result = float(value)

        if not np.isfinite(result):
            return np.nan

        return result

    except (TypeError, ValueError, OverflowError):
        return np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan

    result = numerator / denominator

    if not np.isfinite(result):
        return np.nan

    return float(result)


def _first_available(row: pd.Series, fields: Iterable[str]) -> float:
    for field in fields:
        if field not in row.index:
            continue

        value = _safe_float(row.get(field))

        if not pd.isna(value):
            return value

    return np.nan


def _normalize_text(value) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"[^a-z0-9]+", "", text)

    return text


def _normalize_symbol(value) -> str:
    if value is None:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).strip().upper(),
    )


def _annualize_period_value(
    total_1y: float,
    total_30d: float,
    total_7d: float,
    total_24h: float,
) -> float:
    """
    Prioridade para janela mais longa.
    Valores são anualizados apenas quando necessário.
    """

    if not pd.isna(total_1y) and total_1y >= 0:
        return float(total_1y)

    if not pd.isna(total_30d) and total_30d >= 0:
        return float(total_30d * (365.0 / 30.0))

    if not pd.isna(total_7d) and total_7d >= 0:
        return float(total_7d * (365.0 / 7.0))

    if not pd.isna(total_24h) and total_24h >= 0:
        return float(total_24h * 365.0)

    return np.nan


def _extract_protocols(payload) -> list[dict]:
    if isinstance(payload, dict):
        protocols = payload.get("protocols")

        if isinstance(protocols, list):
            return [
                item
                for item in protocols
                if isinstance(item, dict)
            ]

    return []


def _request_overview(
    session: requests.Session,
    data_type: str,
) -> Optional[dict]:

    params = {
        "excludeTotalDataChart": "true",
        "excludeTotalDataChartBreakdown": "true",
        "dataType": data_type,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                DEFILLAMA_FEES_OVERVIEW_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                payload = response.json()

                if isinstance(payload, dict):
                    return payload

                print(
                    "[holder_value_data] "
                    f"{data_type}: resposta JSON inesperada."
                )

                return None

            if response.status_code in (403, 404):
                print(
                    "[holder_value_data] "
                    f"{data_type}: HTTP {response.status_code}. "
                    "Dataset indisponível."
                )
                return None

            if response.status_code == 429:
                print(
                    "[holder_value_data] "
                    f"{data_type}: HTTP 429. "
                    "Fail-fast para evitar execução longa."
                )
                return None

            if response.status_code >= 500:
                print(
                    "[holder_value_data] "
                    f"{data_type}: HTTP {response.status_code} "
                    f"(tentativa {attempt}/{MAX_RETRIES})."
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue

                return None

            print(
                "[holder_value_data] "
                f"{data_type}: HTTP {response.status_code}."
            )
            return None

        except requests.RequestException as error:
            print(
                "[holder_value_data] "
                f"{data_type}: erro de rede "
                f"(tentativa {attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            return None

        except (ValueError, TypeError) as error:
            print(
                "[holder_value_data] "
                f"{data_type}: resposta inválida: {error}"
            )
            return None

    return None


# ============================================================
# NORMALIZAÇÃO DEFILLAMA
# ============================================================

def _protocol_record(
    protocol: dict,
    data_type: str,
) -> Dict:

    name = protocol.get("name")
    slug = protocol.get("slug")
    display_name = protocol.get("displayName")
    symbol = protocol.get("symbol")

    total_24h = _safe_float(protocol.get("total24h"))
    total_7d = _safe_float(protocol.get("total7d"))
    total_30d = _safe_float(protocol.get("total30d"))
    total_1y = _safe_float(protocol.get("total1y"))

    annualized = _annualize_period_value(
        total_1y=total_1y,
        total_30d=total_30d,
        total_7d=total_7d,
        total_24h=total_24h,
    )

    return {
        "data_type": data_type,
        "protocol_name": name,
        "protocol_display_name": display_name,
        "protocol_slug": slug,
        "protocol_symbol": symbol,
        "name_key": _normalize_text(name),
        "display_name_key": _normalize_text(display_name),
        "slug_key": _normalize_text(slug),
        "symbol_key": _normalize_symbol(symbol),
        "total_24h": total_24h,
        "total_7d": total_7d,
        "total_30d": total_30d,
        "total_1y": total_1y,
        "annualized": annualized,
    }


def _build_dimension_table(
    session: requests.Session,
    data_type: str,
) -> pd.DataFrame:

    payload = _request_overview(
        session=session,
        data_type=data_type,
    )

    if payload is None:
        return pd.DataFrame()

    protocols = _extract_protocols(payload)

    records = [
        _protocol_record(protocol, data_type)
        for protocol in protocols
    ]

    table = pd.DataFrame(records)

    if table.empty:
        return table

    table = table[
        table["annualized"].notna()
    ].copy()

    return table


# ============================================================
# MATCHING
# ============================================================

def _candidate_matches(
    row: pd.Series,
    table: pd.DataFrame,
) -> pd.DataFrame:

    if table.empty:
        return table

    symbol_key = _normalize_symbol(
        row.get("symbol")
    )

    name_key = _normalize_text(
        row.get("name")
    )

    coin_id_key = _normalize_text(
        row.get("coin_id")
    )

    masks = []

    # Coin/protocol name and slug are safer than symbol alone.
    if coin_id_key:
        masks.append(
            (table["slug_key"] == coin_id_key)
            | (table["name_key"] == coin_id_key)
            | (table["display_name_key"] == coin_id_key)
        )

    if name_key:
        masks.append(
            (table["name_key"] == name_key)
            | (table["display_name_key"] == name_key)
            | (table["slug_key"] == name_key)
        )

    if masks:
        combined = masks[0].copy()

        for mask in masks[1:]:
            combined = combined | mask

        matched = table[combined].copy()

        if not matched.empty:
            return matched

    # Symbol é fallback, porque símbolos podem ser ambíguos.
    if symbol_key:
        symbol_matches = table[
            table["symbol_key"] == symbol_key
        ].copy()

        if len(symbol_matches) == 1:
            return symbol_matches

    return pd.DataFrame(
        columns=table.columns
    )


def _best_match(
    row: pd.Series,
    table: pd.DataFrame,
) -> Optional[pd.Series]:

    matches = _candidate_matches(
        row=row,
        table=table,
    )

    if matches.empty:
        return None

    # Se houver mais de um protocolo relacionado ao mesmo token,
    # usar o maior valor econômico observável evita escolher um adapter
    # secundário quando existe o protocolo principal.
    matches = matches.sort_values(
        "annualized",
        ascending=False,
        na_position="last",
    )

    return matches.iloc[0]


# ============================================================
# HOLDER VALUE METRICS
# ============================================================

def _extract_existing_protocol_fees(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "annualized_fees",
            "fees_annualized",
            "protocol_fees_annualized",
            "fees_365d",
            "protocol_fees",
        ],
    )


def _extract_existing_protocol_revenue(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "annualized_revenue",
            "revenue_annualized",
            "protocol_revenue_annualized",
            "revenue_365d",
            "protocol_revenue",
        ],
    )


def _calculate_fee_distribution_ratio(
    holder_revenue: float,
    protocol_fees: float,
) -> float:

    ratio = _safe_divide(
        holder_revenue,
        protocol_fees,
    )

    if pd.isna(ratio):
        return np.nan

    # Holders revenue não deve superar fees economicamente.
    # Pequenas diferenças de janela/fonte podem ocorrer; limitamos a 1.
    return float(
        np.clip(
            ratio,
            0.0,
            1.0,
        )
    )


def _calculate_holder_revenue_yield(
    holder_revenue: float,
    market_cap: float,
) -> float:

    return _safe_divide(
        holder_revenue,
        market_cap,
    )


def _evaluate_asset(
    row: pd.Series,
    holders_table: pd.DataFrame,
    revenue_table: pd.DataFrame,
    fees_table: pd.DataFrame,
) -> Dict:

    holders_match = _best_match(
        row=row,
        table=holders_table,
    )

    revenue_match = _best_match(
        row=row,
        table=revenue_table,
    )

    fees_match = _best_match(
        row=row,
        table=fees_table,
    )

    holder_revenue = (
        _safe_float(
            holders_match.get("annualized")
        )
        if holders_match is not None
        else np.nan
    )

    protocol_revenue = (
        _safe_float(
            revenue_match.get("annualized")
        )
        if revenue_match is not None
        else _extract_existing_protocol_revenue(row)
    )

    protocol_fees = (
        _safe_float(
            fees_match.get("annualized")
        )
        if fees_match is not None
        else _extract_existing_protocol_fees(row)
    )

    market_cap = _safe_float(
        row.get("market_cap")
    )

    fee_distribution_ratio = (
        _calculate_fee_distribution_ratio(
            holder_revenue=holder_revenue,
            protocol_fees=protocol_fees,
        )
    )

    holder_revenue_yield = (
        _calculate_holder_revenue_yield(
            holder_revenue=holder_revenue,
            market_cap=market_cap,
        )
    )

    # Não chamamos isso de "real_yield" automaticamente.
    # Holder revenue / market cap é uma métrica observável de rendimento
    # econômico, mas real yield exige distinguir origem orgânica de emissão.
    # O holder_value_engine pode usar holder_revenue e fee_distribution_ratio.

    source_slug = None
    source_name = None

    for match in (
        holders_match,
        revenue_match,
        fees_match,
    ):
        if match is not None:
            source_slug = match.get("protocol_slug")
            source_name = match.get("protocol_name")
            break

    available_metrics = sum(
        1
        for value in (
            holder_revenue,
            fee_distribution_ratio,
            holder_revenue_yield,
        )
        if not pd.isna(value)
    )

    completeness = round(
        available_metrics / 3.0,
        4,
    )

    if completeness >= 1.0:
        quality = "HIGH"
    elif completeness >= 0.6667:
        quality = "GOOD"
    elif completeness > 0:
        quality = "PARTIAL"
    else:
        quality = "DATA_INSUFFICIENT"

    return {
        "holder_revenue": holder_revenue,
        "holder_revenue_annualized": holder_revenue,
        "holder_revenue_yield": holder_revenue_yield,
        "fee_distribution_ratio": fee_distribution_ratio,
        "holder_value_protocol_revenue": protocol_revenue,
        "holder_value_protocol_fees": protocol_fees,
        "holder_value_protocol_slug": source_slug,
        "holder_value_protocol_name": source_name,
        "holder_value_data_completeness_source": completeness,
        "holder_value_data_quality_source": quality,
        "holder_value_data_source": (
            "DEFILLAMA_DIMENSIONS"
            if available_metrics > 0
            else "DATA_INSUFFICIENT"
        ),
        # Campos qualitativos permanecem NaN até existir fonte verificável.
        "buyback_active": np.nan,
        "burn_active": np.nan,
        "token_required_for_usage": np.nan,
        "gas_token": np.nan,
        "economic_staking": np.nan,
        "staking_yield": np.nan,
        "real_yield": np.nan,
    }


# ============================================================
# DATAFRAME ENRICHMENT
# ============================================================

def enrich_with_holder_value_data(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset is None:
        return pd.DataFrame()

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    print(
        "[holder_value_data] "
        f"Processando {len(result)} ativos."
    )

    print(
        "[holder_value_data] "
        "Modo agregado ativo; sem HTTP individual por ativo."
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    print(
        "[holder_value_data] "
        "Carregando Holders Revenue agregado."
    )

    holders_table = _build_dimension_table(
        session=session,
        data_type="dailyHoldersRevenue",
    )

    print(
        "[holder_value_data] "
        f"Holders Revenue: {len(holders_table)} protocolos utilizáveis."
    )

    print(
        "[holder_value_data] "
        "Carregando Revenue agregado."
    )

    revenue_table = _build_dimension_table(
        session=session,
        data_type="dailyRevenue",
    )

    print(
        "[holder_value_data] "
        f"Revenue: {len(revenue_table)} protocolos utilizáveis."
    )

    print(
        "[holder_value_data] "
        "Carregando Fees agregado."
    )

    fees_table = _build_dimension_table(
        session=session,
        data_type="dailyFees",
    )

    print(
        "[holder_value_data] "
        f"Fees: {len(fees_table)} protocolos utilizáveis."
    )

    evaluations = []
    total = len(result)

    for position, (_, row) in enumerate(
        result.iterrows(),
        start=1,
    ):
        if (
            position == 1
            or position == total
            or position % 25 == 0
        ):
            print(
                "[holder_value_data] "
                f"{row.get('symbol', 'UNKNOWN')} "
                f"({position}/{total})"
            )

        evaluations.append(
            _evaluate_asset(
                row=row,
                holders_table=holders_table,
                revenue_table=revenue_table,
                fees_table=fees_table,
            )
        )

    evaluation_df = pd.DataFrame(
        evaluations,
        index=result.index,
    )

    # Evita sufixos _x/_y e garante que esta camada seja a fonte canônica
    # somente para os campos que ela realmente produz.
    for column in evaluation_df.columns:
        if column in result.columns:
            result = result.drop(
                columns=[column],
                errors="ignore",
            )

        result[column] = evaluation_df[column]

    matched = int(
        result["holder_revenue"]
        .notna()
        .sum()
    )

    fee_distribution_available = int(
        result["fee_distribution_ratio"]
        .notna()
        .sum()
    )

    print(
        "[holder_value_data] "
        f"Concluído: {len(result)} ativos."
    )

    print(
        "[holder_value_data] "
        f"Ativos com Holders Revenue: {matched}."
    )

    print(
        "[holder_value_data] "
        "Ativos com Fee Distribution Ratio: "
        f"{fee_distribution_available}."
    )

    print(
        "[holder_value_data] "
        "Campos qualitativos sem fonte verificável permanecem NaN."
    )

    return result


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":
    from data.market_data import build_market_dataset

    market = build_market_dataset(
        include_history=False
    )

    market = market.head(100)

    enriched = enrich_with_holder_value_data(
        market
    )

    columns = [
        "symbol",
        "name",
        "market_cap",
        "holder_revenue",
        "holder_revenue_yield",
        "fee_distribution_ratio",
        "holder_value_protocol_name",
        "holder_value_data_completeness_source",
        "holder_value_data_quality_source",
        "holder_value_data_source",
    ]

    available = [
        column
        for column in columns
        if column in enriched.columns
    ]

    print(
        enriched[available]
        .sort_values(
            "holder_revenue",
            ascending=False,
            na_position="last",
        )
        .head(50)
        .to_string(index=False)
    )
