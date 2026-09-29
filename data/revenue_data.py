"""
CRYPTO OPPORTUNITY ENGINE
Revenue Data Layer

OBJETIVO:
Fornecer dados observáveis de Revenue e Fees ao revenue_engine
sem alterar a lógica de scoring, pesos ou Quality Gates.

PRINCÍPIOS:
- usar dados agregados da DefiLlama Dimensions;
- evitar chamadas HTTP individuais por ativo;
- ausência de dado permanece NaN;
- não inventar crescimento quando a fonte não fornece base comparável;
- separar coleta de dados da lógica de decisão.

FONTE PRIMÁRIA:
DefiLlama Dimensions / Fees overview.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests


REQUEST_TIMEOUT = 20
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2.0

DEFILLAMA_FEES_OVERVIEW_URL = "https://api.llama.fi/overview/fees"

DATA_TYPES = (
    "dailyRevenue",
    "dailyFees",
)

USER_AGENT = (
    "CRYPTO-OPPORTUNITY-ENGINE/1.0 "
    "(+revenue-data-layer)"
)


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


def _first_available(
    row: pd.Series,
    fields: Iterable[str],
) -> float:
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
    if not pd.isna(total_1y) and total_1y >= 0:
        return float(total_1y)

    if not pd.isna(total_30d) and total_30d >= 0:
        return float(total_30d * (365.0 / 30.0))

    if not pd.isna(total_7d) and total_7d >= 0:
        return float(total_7d * (365.0 / 7.0))

    if not pd.isna(total_24h) and total_24h >= 0:
        return float(total_24h * 365.0)

    return np.nan


def _calculate_short_term_growth(
    total_30d: float,
    total_7d: float,
) -> float:
    """
    Compara a média diária dos últimos 7 dias com a média diária
    dos últimos 30 dias.

    É uma medida observável de aceleração recente, não um histórico
    sintético de 90 dias.
    """
    if (
        pd.isna(total_30d)
        or pd.isna(total_7d)
        or total_30d <= 0
    ):
        return np.nan

    daily_30d = total_30d / 30.0
    daily_7d = total_7d / 7.0

    if daily_30d <= 0:
        return np.nan

    return float(
        daily_7d / daily_30d - 1.0
    )


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
                    "[revenue_data] "
                    f"{data_type}: resposta JSON inesperada."
                )
                return None

            if response.status_code in (403, 404):
                print(
                    "[revenue_data] "
                    f"{data_type}: HTTP {response.status_code}. "
                    "Dataset indisponível."
                )
                return None

            if response.status_code == 429:
                print(
                    "[revenue_data] "
                    f"{data_type}: HTTP 429. "
                    "Fail-fast para evitar execução longa."
                )
                return None

            if response.status_code >= 500:
                print(
                    "[revenue_data] "
                    f"{data_type}: HTTP {response.status_code} "
                    f"(tentativa {attempt}/{MAX_RETRIES})."
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue

                return None

            print(
                "[revenue_data] "
                f"{data_type}: HTTP {response.status_code}."
            )
            return None

        except requests.RequestException as error:
            print(
                "[revenue_data] "
                f"{data_type}: erro de rede "
                f"(tentativa {attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            return None

        except (ValueError, TypeError) as error:
            print(
                "[revenue_data] "
                f"{data_type}: resposta inválida: {error}"
            )
            return None

    return None


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

    short_term_growth = _calculate_short_term_growth(
        total_30d=total_30d,
        total_7d=total_7d,
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
        "short_term_growth": short_term_growth,
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

    matches = matches.sort_values(
        "annualized",
        ascending=False,
        na_position="last",
    )

    return matches.iloc[0]


def _existing_revenue(row: pd.Series) -> float:
    return _first_available(
        row,
        [
            "annualized_revenue",
            "revenue_annualized",
            "protocol_revenue_annualized",
            "revenue_365d",
            "protocol_revenue",
            "holder_value_protocol_revenue",
        ],
    )


def _existing_fees(row: pd.Series) -> float:
    return _first_available(
        row,
        [
            "annualized_fees",
            "fees_annualized",
            "protocol_fees_annualized",
            "fees_365d",
            "protocol_fees",
            "holder_value_protocol_fees",
        ],
    )


def _evaluate_asset(
    row: pd.Series,
    revenue_table: pd.DataFrame,
    fees_table: pd.DataFrame,
) -> Dict:

    revenue_match = _best_match(
        row=row,
        table=revenue_table,
    )

    fees_match = _best_match(
        row=row,
        table=fees_table,
    )

    protocol_revenue = (
        _safe_float(revenue_match.get("annualized"))
        if revenue_match is not None
        else _existing_revenue(row)
    )

    protocol_fees = (
        _safe_float(fees_match.get("annualized"))
        if fees_match is not None
        else _existing_fees(row)
    )

    revenue_growth = (
        _safe_float(
            revenue_match.get("short_term_growth")
        )
        if revenue_match is not None
        else np.nan
    )

    fees_growth = (
        _safe_float(
            fees_match.get("short_term_growth")
        )
        if fees_match is not None
        else np.nan
    )

    revenue_slug = (
        revenue_match.get("protocol_slug")
        if revenue_match is not None
        else None
    )

    revenue_name = (
        revenue_match.get("protocol_name")
        if revenue_match is not None
        else None
    )

    fees_slug = (
        fees_match.get("protocol_slug")
        if fees_match is not None
        else None
    )

    fees_name = (
        fees_match.get("protocol_name")
        if fees_match is not None
        else None
    )

    available = sum(
        1
        for value in (
            protocol_revenue,
            protocol_fees,
            revenue_growth,
            fees_growth,
        )
        if not pd.isna(value)
    )

    completeness = round(
        available / 4.0,
        4,
    )

    return {
        "annualized_revenue":
            protocol_revenue,

        "annualized_fees":
            protocol_fees,

        # O revenue_engine aceita estes aliases.
        # A fonte agregada não fornece histórico 90d ponto-a-ponto.
        # Portanto usamos o campo genérico de growth para a aceleração
        # recente observável 7d versus média diária de 30d.
        "revenue_growth":
            revenue_growth,

        "fees_growth":
            fees_growth,

        "revenue_data_protocol_slug":
            revenue_slug,

        "revenue_data_protocol_name":
            revenue_name,

        "fees_data_protocol_slug":
            fees_slug,

        "fees_data_protocol_name":
            fees_name,

        "revenue_data_completeness_source":
            completeness,

        "revenue_data_source":
            (
                "DEFILLAMA_DIMENSIONS"
                if available > 0
                else "DATA_INSUFFICIENT"
            ),
    }


def enrich_with_revenue_data(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset is None:
        return pd.DataFrame()

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    print(
        "[revenue_data] "
        f"Processando {len(result)} ativos."
    )

    print(
        "[revenue_data] "
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
        "[revenue_data] "
        "Carregando Revenue agregado."
    )

    revenue_table = _build_dimension_table(
        session=session,
        data_type="dailyRevenue",
    )

    print(
        "[revenue_data] "
        f"Revenue: {len(revenue_table)} protocolos utilizáveis."
    )

    print(
        "[revenue_data] "
        "Carregando Fees agregado."
    )

    fees_table = _build_dimension_table(
        session=session,
        data_type="dailyFees",
    )

    print(
        "[revenue_data] "
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
                "[revenue_data] "
                f"{row.get('symbol', 'UNKNOWN')} "
                f"({position}/{total})"
            )

        evaluations.append(
            _evaluate_asset(
                row=row,
                revenue_table=revenue_table,
                fees_table=fees_table,
            )
        )

    evaluation_df = pd.DataFrame(
        evaluations,
        index=result.index,
    )

    # A nova camada é canônica apenas para os campos que produz.
    # Não altera engines, pesos ou thresholds.
    for column in evaluation_df.columns:
        if column in result.columns:
            result = result.drop(
                columns=[column],
                errors="ignore",
            )

        result[column] = evaluation_df[column]

    revenue_available = int(
        result["annualized_revenue"]
        .notna()
        .sum()
    )

    fees_available = int(
        result["annualized_fees"]
        .notna()
        .sum()
    )

    revenue_growth_available = int(
        result["revenue_growth"]
        .notna()
        .sum()
    )

    fees_growth_available = int(
        result["fees_growth"]
        .notna()
        .sum()
    )

    print(
        "[revenue_data] "
        f"Concluído: {len(result)} ativos."
    )

    print(
        "[revenue_data] "
        f"Ativos com Revenue: {revenue_available}."
    )

    print(
        "[revenue_data] "
        f"Ativos com Fees: {fees_available}."
    )

    print(
        "[revenue_data] "
        f"Ativos com Revenue Growth observável: "
        f"{revenue_growth_available}."
    )

    print(
        "[revenue_data] "
        f"Ativos com Fees Growth observável: "
        f"{fees_growth_available}."
    )

    return result


if __name__ == "__main__":

    from data.market_data import (
        build_market_dataset,
    )

    market = build_market_dataset(
        include_history=False
    )

    enriched = enrich_with_revenue_data(
        market
    )

    columns = [
        "symbol",
        "name",
        "annualized_revenue",
        "annualized_fees",
        "revenue_growth",
        "fees_growth",
        "revenue_data_completeness_source",
        "revenue_data_source",
    ]

    available_columns = [
        column
        for column in columns
        if column in enriched.columns
    ]

    print(
        enriched[
            available_columns
        ]
        .sort_values(
            "annualized_revenue",
            ascending=False,
            na_position="last",
        )
        .head(50)
        .to_string(index=False)
    )
