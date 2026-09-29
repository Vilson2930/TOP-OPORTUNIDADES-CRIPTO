"""
CRYPTO OPPORTUNITY ENGINE
On-Chain Data Layer

FOCO:
Encontrar OPORTUNIDADES.

O módulo procura confirmação on-chain de melhora fundamental:
- crescimento de atividade;
- crescimento de transações;
- crescimento de usuários/endereços;
- entrada de capital;
- crescimento de stablecoins;
- crescimento de utilização;
- divergência positiva entre atividade on-chain e preço.

IMPORTANTE:
On-chain confirma oportunidade.
Não substitui fundamentos, receita, captura de valor,
tokenomics, diluição ou valuation.

Fonte primária gratuita:
DefiLlama Public API.

O módulo foi construído com fallback.
Ausência de determinada métrica não gera dado artificial.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config


LLAMA_BASE_URL = "https://api.llama.fi"
STABLECOINS_BASE_URL = "https://stablecoins.llama.fi"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


# ============================================================
# HTTP
# ============================================================

def _request(
    base_url: str,
    endpoint: str,
    params: Optional[Dict] = None,
) -> Optional[object]:

    url = f"{base_url}{endpoint}"

    headers = {
        "accept": "application/json",
        "user-agent": f"{config.ENGINE_NAME}/{config.ENGINE_VERSION}",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:

                wait = RETRY_BACKOFF_SECONDS * attempt

                print(
                    f"[onchain_data] Rate limit. "
                    f"Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:

                wait = RETRY_BACKOFF_SECONDS * attempt

                print(
                    f"[onchain_data] HTTP "
                    f"{response.status_code}. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            return None

        except requests.RequestException:

            if attempt == MAX_RETRIES:
                return None

            time.sleep(
                RETRY_BACKOFF_SECONDS * attempt
            )

    return None


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value) -> float:

    try:

        if value is None:
            return np.nan

        return float(value)

    except (TypeError, ValueError):
        return np.nan


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def _normalize(value: str) -> str:

    return str(
        value or ""
    ).strip().lower()


# ============================================================
# CHAINS
# ============================================================

def fetch_chains() -> pd.DataFrame:

    data = _request(
        LLAMA_BASE_URL,
        "/v2/chains",
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    rows: List[Dict] = []

    for item in data:

        name = str(
            item.get("name", "")
        ).strip()

        token_symbol = str(
            item.get("tokenSymbol", "")
        ).upper().strip()

        if not name:
            continue

        rows.append(
            {
                "chain_name": name,
                "chain_token_symbol":
                    token_symbol,
                "chain_tvl":
                    _safe_float(
                        item.get("tvl")
                    ),
                "chain_gecko_id":
                    item.get(
                        "gecko_id"
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MATCH TOKEN ↔ CHAIN
# ============================================================

def match_chain(
    symbol: str,
    name: str,
    chains: pd.DataFrame,
) -> Optional[pd.Series]:

    if chains.empty:
        return None

    symbol = str(
        symbol or ""
    ).upper().strip()

    name_norm = _normalize(name)

    symbol_matches = chains[
        chains[
            "chain_token_symbol"
        ] == symbol
    ]

    if len(symbol_matches) == 1:
        return symbol_matches.iloc[0]

    if len(symbol_matches) > 1:

        exact = symbol_matches[
            symbol_matches[
                "chain_name"
            ].str.lower() == name_norm
        ]

        if len(exact) == 1:
            return exact.iloc[0]

    exact_name = chains[
        chains[
            "chain_name"
        ].str.lower() == name_norm
    ]

    if len(exact_name) == 1:
        return exact_name.iloc[0]

    return None


# ============================================================
# HISTÓRICO TVL DA CHAIN
# ============================================================

def fetch_chain_tvl_history(
    chain_name: str,
) -> pd.DataFrame:

    data = _request(
        LLAMA_BASE_URL,
        f"/v2/historicalChainTvl/{chain_name}",
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []

    for item in data:

        timestamp = item.get(
            "date"
        )

        tvl = _safe_float(
            item.get("tvl")
        )

        if timestamp is None:
            continue

        try:

            date = pd.to_datetime(
                int(timestamp),
                unit="s",
                utc=True,
            )

        except Exception:
            continue

        rows.append(
            {
                "date": date,
                "chain_tvl": tvl,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .dropna(
            subset=["chain_tvl"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# STABLECOINS POR CHAIN
# ============================================================

def fetch_stablecoin_chains() -> pd.DataFrame:

    data = _request(
        STABLECOINS_BASE_URL,
        "/stablecoinchains",
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []

    for item in data:

        chain = str(
            item.get("name", "")
        ).strip()

        if not chain:
            continue

        total = item.get(
            "totalCirculatingUSD",
            {},
        )

        if isinstance(total, dict):

            current = _safe_float(
                total.get("peggedUSD")
            )

        else:

            current = _safe_float(
                total
            )

        rows.append(
            {
                "chain_name": chain,
                "stablecoin_supply":
                    current,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# HISTÓRICO STABLECOIN
# ============================================================

def fetch_stablecoin_history(
    chain_name: str,
) -> pd.DataFrame:

    data = _request(
        STABLECOINS_BASE_URL,
        f"/stablecoincharts/{chain_name}",
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []

    for item in data:

        timestamp = item.get(
            "date"
        )

        total = item.get(
            "totalCirculatingUSD",
            {},
        )

        if isinstance(total, dict):

            value = _safe_float(
                total.get("peggedUSD")
            )

        else:

            value = _safe_float(
                total
            )

        if timestamp is None:
            continue

        try:

            date = pd.to_datetime(
                int(timestamp),
                unit="s",
                utc=True,
            )

        except Exception:
            continue

        rows.append(
            {
                "date": date,
                "stablecoin_supply":
                    value,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .dropna(
            subset=["stablecoin_supply"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# MÉTRICAS HISTÓRICAS
# ============================================================

def _value_days_ago(
    history: pd.DataFrame,
    column: str,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    latest_date = history[
        "date"
    ].iloc[-1]

    target = (
        latest_date
        - pd.Timedelta(days=days)
    )

    previous = history[
        history["date"] <= target
    ]

    if previous.empty:
        return np.nan

    return _safe_float(
        previous[
            column
        ].iloc[-1]
    )


def calculate_growth(
    history: pd.DataFrame,
    column: str,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    current = _safe_float(
        history[
            column
        ].iloc[-1]
    )

    previous = _value_days_ago(
        history,
        column,
        days,
    )

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous <= 0
    ):
        return np.nan

    return (
        current / previous
        - 1
    )


def calculate_acceleration(
    history: pd.DataFrame,
    column: str,
) -> float:

    if history.empty:
        return np.nan

    current = _safe_float(
        history[
            column
        ].iloc[-1]
    )

    value_30d = _value_days_ago(
        history,
        column,
        30,
    )

    value_60d = _value_days_ago(
        history,
        column,
        60,
    )

    if any(
        pd.isna(value)
        for value in [
            current,
            value_30d,
            value_60d,
        ]
    ):
        return np.nan

    if (
        value_30d <= 0
        or value_60d <= 0
    ):
        return np.nan

    recent_growth = (
        current / value_30d
        - 1
    )

    previous_growth = (
        value_30d / value_60d
        - 1
    )

    return (
        recent_growth
        - previous_growth
    )


# ============================================================
# CHAIN ACTIVITY PROXY
# ============================================================

def calculate_chain_metrics(
    history: pd.DataFrame,
) -> Dict[str, float]:

    if history.empty:

        return {
            "chain_tvl_current":
                np.nan,
            "chain_tvl_growth_30d":
                np.nan,
            "chain_tvl_growth_90d":
                np.nan,
            "chain_tvl_growth_180d":
                np.nan,
            "chain_tvl_acceleration":
                np.nan,
        }

    return {
        "chain_tvl_current":
            _safe_float(
                history[
                    "chain_tvl"
                ].iloc[-1]
            ),

        "chain_tvl_growth_30d":
            calculate_growth(
                history,
                "chain_tvl",
                30,
            ),

        "chain_tvl_growth_90d":
            calculate_growth(
                history,
                "chain_tvl",
                90,
            ),

        "chain_tvl_growth_180d":
            calculate_growth(
                history,
                "chain_tvl",
                180,
            ),

        "chain_tvl_acceleration":
            calculate_acceleration(
                history,
                "chain_tvl",
            ),
    }


# ============================================================
# STABLECOIN INFLOW
# ============================================================

def calculate_stablecoin_metrics(
    history: pd.DataFrame,
) -> Dict[str, float]:

    if history.empty:

        return {
            "stablecoin_supply":
                np.nan,
            "stablecoin_growth_30d":
                np.nan,
            "stablecoin_growth_90d":
                np.nan,
            "stablecoin_growth_180d":
                np.nan,
            "stablecoin_acceleration":
                np.nan,
        }

    return {
        "stablecoin_supply":
            _safe_float(
                history[
                    "stablecoin_supply"
                ].iloc[-1]
            ),

        "stablecoin_growth_30d":
            calculate_growth(
                history,
                "stablecoin_supply",
                30,
            ),

        "stablecoin_growth_90d":
            calculate_growth(
                history,
                "stablecoin_supply",
                90,
            ),

        "stablecoin_growth_180d":
            calculate_growth(
                history,
                "stablecoin_supply",
                180,
            ),

        "stablecoin_acceleration":
            calculate_acceleration(
                history,
                "stablecoin_supply",
            ),
    }


# ============================================================
# DIVERGÊNCIA ON-CHAIN × PREÇO
# ============================================================

def calculate_onchain_price_divergence(
    onchain_growth: float,
    price_return: float,
) -> float:

    if (
        pd.isna(onchain_growth)
        or pd.isna(price_return)
    ):
        return np.nan

    return (
        onchain_growth
        - price_return
    )


# ============================================================
# NORMALIZAÇÃO SCORE
# ============================================================

def _growth_score(
    value: float,
    lower: float = -0.30,
    upper: float = 0.70,
) -> float:

    if pd.isna(value):
        return np.nan

    if value <= lower:
        return 0.0

    if value >= upper:
        return 100.0

    return float(
        np.clip(
            (
                (value - lower)
                / (upper - lower)
            )
            * 100,
            0,
            100,
        )
    )


def _acceleration_score(
    value: float,
) -> float:

    return _growth_score(
        value,
        lower=-0.20,
        upper=0.40,
    )


# ============================================================
# ONCHAIN OPPORTUNITY SCORE
# ============================================================

def calculate_onchain_opportunity_score(
    chain_metrics: Dict[str, float],
    stablecoin_metrics: Dict[str, float],
) -> float:

    components = [
        (
            _growth_score(
                chain_metrics.get(
                    "chain_tvl_growth_30d"
                )
            ),
            0.15,
        ),
        (
            _growth_score(
                chain_metrics.get(
                    "chain_tvl_growth_90d"
                )
            ),
            0.25,
        ),
        (
            _growth_score(
                chain_metrics.get(
                    "chain_tvl_growth_180d"
                )
            ),
            0.10,
        ),
        (
            _acceleration_score(
                chain_metrics.get(
                    "chain_tvl_acceleration"
                )
            ),
            0.15,
        ),
        (
            _growth_score(
                stablecoin_metrics.get(
                    "stablecoin_growth_30d"
                )
            ),
            0.10,
        ),
        (
            _growth_score(
                stablecoin_metrics.get(
                    "stablecoin_growth_90d"
                )
            ),
            0.15,
        ),
        (
            _acceleration_score(
                stablecoin_metrics.get(
                    "stablecoin_acceleration"
                )
            ),
            0.10,
        ),
    ]

    weighted_sum = 0.0
    available_weight = 0.0

    for score, weight in components:

        if pd.isna(score):
            continue

        weighted_sum += (
            score * weight
        )

        available_weight += weight

    if available_weight == 0:
        return np.nan

    return round(
        weighted_sum
        / available_weight,
        2,
    )


# ============================================================
# ENRIQUECIMENTO
# ============================================================

def enrich_with_onchain_data(
    market_df: pd.DataFrame,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:

    if market_df.empty:
        return market_df.copy()

    chains = fetch_chains()

    records = []

    total = len(market_df)

    for position, (_, row) in enumerate(
        market_df.iterrows(),
        start=1,
    ):

        coin_id = row.get(
            "coin_id"
        )

        symbol = row.get(
            "symbol"
        )

        name = row.get(
            "name"
        )

        print(
            f"[onchain_data] "
            f"{symbol} "
            f"({position}/{total})"
        )

        record = {
            "coin_id":
                coin_id,

            "onchain_chain_match":
                False,

            "onchain_chain":
                None,

            "chain_tvl_current":
                np.nan,

            "chain_tvl_growth_30d":
                np.nan,

            "chain_tvl_growth_90d":
                np.nan,

            "chain_tvl_growth_180d":
                np.nan,

            "chain_tvl_acceleration":
                np.nan,

            "stablecoin_supply":
                np.nan,

            "stablecoin_growth_30d":
                np.nan,

            "stablecoin_growth_90d":
                np.nan,

            "stablecoin_growth_180d":
                np.nan,

            "stablecoin_acceleration":
                np.nan,

            "onchain_price_divergence_30d":
                np.nan,

            "onchain_price_divergence_90d":
                np.nan,

            "onchain_opportunity_score":
                np.nan,
        }

        match = match_chain(
            symbol=symbol,
            name=name,
            chains=chains,
        )

        if match is None:

            records.append(
                record
            )

            continue

        chain_name = match[
            "chain_name"
        ]

        chain_history = (
            fetch_chain_tvl_history(
                chain_name
            )
        )

        stablecoin_history = (
            fetch_stablecoin_history(
                chain_name
            )
        )

        chain_metrics = (
            calculate_chain_metrics(
                chain_history
            )
        )

        stablecoin_metrics = (
            calculate_stablecoin_metrics(
                stablecoin_history
            )
        )

        score = (
            calculate_onchain_opportunity_score(
                chain_metrics,
                stablecoin_metrics,
            )
        )

        price_30d = _safe_float(
            row.get(
                "return_30d"
            )
        )

        price_90d = _safe_float(
            row.get(
                "return_90d"
            )
        )

        fundamental_30d = (
            chain_metrics.get(
                "chain_tvl_growth_30d"
            )
        )

        fundamental_90d = (
            chain_metrics.get(
                "chain_tvl_growth_90d"
            )
        )

        divergence_30d = (
            calculate_onchain_price_divergence(
                fundamental_30d,
                price_30d,
            )
        )

        divergence_90d = (
            calculate_onchain_price_divergence(
                fundamental_90d,
                price_90d,
            )
        )

        record.update(
            {
                "onchain_chain_match":
                    True,

                "onchain_chain":
                    chain_name,

                **chain_metrics,

                **stablecoin_metrics,

                "onchain_price_divergence_30d":
                    divergence_30d,

                "onchain_price_divergence_90d":
                    divergence_90d,

                "onchain_opportunity_score":
                    score,
            }
        )

        records.append(
            record
        )

        time.sleep(
            sleep_seconds
        )

    onchain_df = pd.DataFrame(
        records
    )

    return market_df.merge(
        onchain_df,
        on="coin_id",
        how="left",
    )


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    from data.market_data import (
        build_market_dataset,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=50,
    )

    result = enrich_with_onchain_data(
        market
    )

    columns = [
        "symbol",
        "name",
        "onchain_chain",
        "chain_tvl_growth_30d",
        "chain_tvl_growth_90d",
        "chain_tvl_acceleration",
        "stablecoin_growth_30d",
        "stablecoin_growth_90d",
        "onchain_price_divergence_90d",
        "onchain_opportunity_score",
    ]

    available = [
        column
        for column in columns
        if column in result.columns
    ]

    print(
        result[
            available
        ]
        .sort_values(
            "onchain_opportunity_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
