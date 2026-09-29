"""
CRYPTO OPPORTUNITY ENGINE
Market Data Layer

Responsável por:
- Construir o universo inicial.
- Excluir BTC.
- Excluir stablecoins.
- Excluir wrapped assets.
- Coletar preço, market cap, FDV, volume e supply.
- Calcular retornos e drawdown.
- Preparar dados para o Opportunity Engine.

Fonte principal:
CoinGecko Public API.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


# ============================================================
# STABLECOINS
# ============================================================

STABLECOIN_SYMBOLS = {
    "USDT",
    "USDC",
    "DAI",
    "FDUSD",
    "USDE",
    "USDS",
    "TUSD",
    "USDP",
    "PYUSD",
    "FRAX",
    "LUSD",
    "GUSD",
    "USD0",
    "USD1",
    "USDD",
    "CRVUSD",
    "SUSD",
    "EURC",
    "EURT",
    "EURS",
    "RLUSD",
}


# ============================================================
# WRAPPED ASSETS
# ============================================================

WRAPPED_SYMBOLS = {
    "WBTC",
    "WETH",
    "WBNB",
    "WMATIC",
    "WAVAX",
    "WSOL",
    "WFTM",
    "WCELO",
    "WGLMR",
    "WONE",
    "CBBTC",
    "TBTC",
    "STETH",
    "WSTETH",
    "RETH",
}


# ============================================================
# HTTP
# ============================================================

def _request(
    endpoint: str,
    params: Optional[Dict] = None,
) -> Optional[object]:

    url = f"{COINGECKO_BASE_URL}{endpoint}"

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
                    f"[market_data] Rate limit CoinGecko. "
                    f"Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:

                wait = RETRY_BACKOFF_SECONDS * attempt

                print(
                    f"[market_data] CoinGecko HTTP "
                    f"{response.status_code}. Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            print(
                f"[market_data] CoinGecko HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )

            return None

        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:

                print(
                    f"[market_data] Falha definitiva "
                    f"CoinGecko: {exc}"
                )

                return None

            wait = RETRY_BACKOFF_SECONDS * attempt

            time.sleep(wait)

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


# ============================================================
# FILTROS
# ============================================================

def is_excluded_symbol(symbol: str) -> bool:

    symbol = str(symbol).upper().strip()

    return symbol in config.EXCLUDED_SYMBOLS


def is_stablecoin(
    symbol: str,
    name: str = "",
) -> bool:

    symbol = str(symbol).upper().strip()
    name = str(name).lower().strip()

    if symbol in STABLECOIN_SYMBOLS:
        return True

    stable_terms = [
        "stablecoin",
        "stable coin",
        "usd stable",
        "dollar stable",
    ]

    return any(term in name for term in stable_terms)


def is_wrapped_asset(
    symbol: str,
    name: str = "",
) -> bool:

    symbol = str(symbol).upper().strip()
    name = str(name).lower().strip()

    if symbol in WRAPPED_SYMBOLS:
        return True

    wrapped_terms = [
        "wrapped bitcoin",
        "wrapped ether",
        "wrapped ethereum",
        "wrapped bnb",
        "wrapped sol",
        "wrapped avalanche",
    ]

    return any(term in name for term in wrapped_terms)


# ============================================================
# UNIVERSO COINGECKO
# ============================================================

def fetch_market_page(
    page: int,
    per_page: int = 250,
) -> List[Dict]:

    params = {
        "vs_currency": config.BASE_CURRENCY.lower(),
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h,7d,30d",
    }

    data = _request(
        "/coins/markets",
        params=params,
    )

    if not isinstance(data, list):
        return []

    return data


def fetch_market_universe() -> pd.DataFrame:

    target = config.UNIVERSE_MAX_ASSETS

    per_page = 250

    pages = int(np.ceil(target / per_page))

    records: List[Dict] = []

    for page in range(1, pages + 1):

        data = fetch_market_page(
            page=page,
            per_page=per_page,
        )

        if not data:
            break

        records.extend(data)

        if len(records) >= target:
            break

        time.sleep(1.2)

    records = records[:target]

    rows = []

    for coin in records:

        symbol = str(
            coin.get("symbol", "")
        ).upper().strip()

        name = str(
            coin.get("name", "")
        ).strip()

        coin_id = str(
            coin.get("id", "")
        ).strip()

        if not symbol or not coin_id:
            continue

        if is_excluded_symbol(symbol):
            continue

        if (
            config.EXCLUDE_STABLECOINS
            and is_stablecoin(symbol, name)
        ):
            continue

        if (
            config.EXCLUDE_WRAPPED_ASSETS
            and is_wrapped_asset(symbol, name)
        ):
            continue

        market_cap = _safe_float(
            coin.get("market_cap")
        )

        daily_volume = _safe_float(
            coin.get("total_volume")
        )

        if (
            not pd.isna(market_cap)
            and market_cap < config.MIN_MARKET_CAP_USD
        ):
            continue

        if (
            not pd.isna(daily_volume)
            and daily_volume < config.MIN_DAILY_VOLUME_USD
        ):
            continue

        circulating_supply = _safe_float(
            coin.get("circulating_supply")
        )

        total_supply = _safe_float(
            coin.get("total_supply")
        )

        max_supply = _safe_float(
            coin.get("max_supply")
        )

        fdv = _safe_float(
            coin.get("fully_diluted_valuation")
        )

        market_cap_fdv_ratio = _safe_divide(
            market_cap,
            fdv,
        )

        circulating_total_ratio = _safe_divide(
            circulating_supply,
            total_supply,
        )

        rows.append(
            {
                "coin_id": coin_id,
                "symbol": symbol,
                "name": name,

                "price": _safe_float(
                    coin.get("current_price")
                ),

                "market_cap": market_cap,

                "market_cap_rank": coin.get(
                    "market_cap_rank"
                ),

                "fdv": fdv,

                "daily_volume": daily_volume,

                "circulating_supply": circulating_supply,

                "total_supply": total_supply,

                "max_supply": max_supply,

                "market_cap_fdv_ratio":
                    market_cap_fdv_ratio,

                "circulating_total_ratio":
                    circulating_total_ratio,

                "ath": _safe_float(
                    coin.get("ath")
                ),

                "ath_change_percentage":
                    _safe_float(
                        coin.get(
                            "ath_change_percentage"
                        )
                    ),

                "atl": _safe_float(
                    coin.get("atl")
                ),

                "price_change_24h":
                    _safe_float(
                        coin.get(
                            "price_change_percentage_24h"
                        )
                    ),

                "price_change_7d":
                    _safe_float(
                        coin.get(
                            "price_change_percentage_7d_in_currency"
                        )
                    ),

                "price_change_30d":
                    _safe_float(
                        coin.get(
                            "price_change_percentage_30d_in_currency"
                        )
                    ),

                "last_updated":
                    coin.get("last_updated"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["coin_id"]
    )

    df = df.sort_values(
        by="market_cap",
        ascending=False,
        na_position="last",
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# HISTÓRICO DE PREÇO
# ============================================================

def fetch_price_history(
    coin_id: str,
    days: int = 365,
) -> pd.DataFrame:

    params = {
        "vs_currency": config.BASE_CURRENCY.lower(),
        "days": days,
        "interval": "daily",
    }

    data = _request(
        f"/coins/{coin_id}/market_chart",
        params=params,
    )

    if not isinstance(data, dict):
        return pd.DataFrame()

    prices = data.get("prices", [])

    volumes = data.get("total_volumes", [])

    market_caps = data.get("market_caps", [])

    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(
        prices,
        columns=[
            "timestamp",
            "price",
        ],
    )

    price_df["date"] = pd.to_datetime(
        price_df["timestamp"],
        unit="ms",
        utc=True,
    )

    price_df = price_df[
        [
            "date",
            "price",
        ]
    ]

    if volumes:

        volume_df = pd.DataFrame(
            volumes,
            columns=[
                "timestamp",
                "volume",
            ],
        )

        volume_df["date"] = pd.to_datetime(
            volume_df["timestamp"],
            unit="ms",
            utc=True,
        )

        price_df = price_df.merge(
            volume_df[
                [
                    "date",
                    "volume",
                ]
            ],
            on="date",
            how="left",
        )

    if market_caps:

        cap_df = pd.DataFrame(
            market_caps,
            columns=[
                "timestamp",
                "historical_market_cap",
            ],
        )

        cap_df["date"] = pd.to_datetime(
            cap_df["timestamp"],
            unit="ms",
            utc=True,
        )

        price_df = price_df.merge(
            cap_df[
                [
                    "date",
                    "historical_market_cap",
                ]
            ],
            on="date",
            how="left",
        )

    price_df = price_df.sort_values(
        "date"
    ).reset_index(drop=True)

    return price_df


# ============================================================
# MÉTRICAS DE PREÇO
# ============================================================

def calculate_return(
    history: pd.DataFrame,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    if len(history) < 2:
        return np.nan

    current_price = history[
        "price"
    ].iloc[-1]

    target_date = (
        history["date"].iloc[-1]
        - pd.Timedelta(days=days)
    )

    historical = history[
        history["date"] <= target_date
    ]

    if historical.empty:
        return np.nan

    old_price = historical[
        "price"
    ].iloc[-1]

    if old_price <= 0:
        return np.nan

    return (
        current_price / old_price
        - 1
    )


def calculate_drawdown(
    history: pd.DataFrame,
) -> float:

    if history.empty:
        return np.nan

    prices = history[
        "price"
    ].dropna()

    if prices.empty:
        return np.nan

    peak = prices.max()

    current = prices.iloc[-1]

    if peak <= 0:
        return np.nan

    return (
        current / peak
        - 1
    )


def calculate_price_metrics(
    history: pd.DataFrame,
) -> Dict[str, float]:

    if history.empty:

        return {
            "return_30d": np.nan,
            "return_90d": np.nan,
            "return_180d": np.nan,
            "return_365d": np.nan,
            "drawdown_365d": np.nan,
            "volume_growth_30d": np.nan,
        }

    metrics = {
        "return_30d":
            calculate_return(
                history,
                30,
            ),

        "return_90d":
            calculate_return(
                history,
                90,
            ),

        "return_180d":
            calculate_return(
                history,
                180,
            ),

        "return_365d":
            calculate_return(
                history,
                365,
            ),

        "drawdown_365d":
            calculate_drawdown(
                history,
            ),
    }

    if (
        "volume" in history.columns
        and len(history) >= 60
    ):

        recent = history[
            "volume"
        ].tail(30).mean()

        previous = history[
            "volume"
        ].iloc[-60:-30].mean()

        if (
            not pd.isna(previous)
            and previous > 0
        ):
            metrics[
                "volume_growth_30d"
            ] = (
                recent / previous
                - 1
            )

        else:
            metrics[
                "volume_growth_30d"
            ] = np.nan

    else:

        metrics[
            "volume_growth_30d"
        ] = np.nan

    return metrics


# ============================================================
# DIVERGÊNCIA PREÇO
# ============================================================

def calculate_price_underreaction(
    fundamental_change: float,
    price_change: float,
) -> float:

    if (
        pd.isna(fundamental_change)
        or pd.isna(price_change)
    ):
        return np.nan

    return (
        fundamental_change
        - price_change
    )


# ============================================================
# ENRIQUECIMENTO DO UNIVERSO
# ============================================================

def enrich_with_price_history(
    universe: pd.DataFrame,
    top_n: Optional[int] = None,
    days: int = 365,
) -> pd.DataFrame:

    if universe.empty:
        return universe.copy()

    df = universe.copy()

    if top_n is not None:
        working = df.head(top_n).copy()
    else:
        working = df.copy()

    metrics_records = []

    total = len(working)

    for index, row in working.iterrows():

        coin_id = row["coin_id"]

        symbol = row["symbol"]

        print(
            f"[market_data] Histórico "
            f"{symbol} "
            f"({len(metrics_records) + 1}/{total})"
        )

        history = fetch_price_history(
            coin_id=coin_id,
            days=days,
        )

        metrics = calculate_price_metrics(
            history
        )

        metrics["coin_id"] = coin_id

        metrics_records.append(
            metrics
        )

        time.sleep(1.2)

    metrics_df = pd.DataFrame(
        metrics_records
    )

    result = working.merge(
        metrics_df,
        on="coin_id",
        how="left",
    )

    return result


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_market_data_completeness(
    row: pd.Series,
) -> float:

    required_fields = [
        "price",
        "market_cap",
        "daily_volume",
        "circulating_supply",
        "fdv",
    ]

    available = 0

    for field in required_fields:

        value = row.get(field)

        if (
            value is not None
            and not pd.isna(value)
        ):
            available += 1

    return (
        available
        / len(required_fields)
    )


def apply_data_quality(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    result = df.copy()

    result[
        "market_data_completeness"
    ] = result.apply(
        calculate_market_data_completeness,
        axis=1,
    )

    result[
        "market_data_quality_pass"
    ] = (
        result[
            "market_data_completeness"
        ]
        >= config.MIN_DATA_COMPLETENESS
    )

    return result


# ============================================================
# PIPELINE
# ============================================================

def build_market_dataset(
    include_history: bool = False,
    history_top_n: Optional[int] = None,
) -> pd.DataFrame:

    print(
        "[market_data] Construindo universo..."
    )

    universe = fetch_market_universe()

    if universe.empty:

        print(
            "[market_data] Universo vazio."
        )

        return universe

    universe = apply_data_quality(
        universe
    )

    universe = universe[
        universe[
            "market_data_quality_pass"
        ]
    ].copy()

    if include_history:

        universe = enrich_with_price_history(
            universe=universe,
            top_n=history_top_n,
            days=365,
        )

    universe = universe.reset_index(
        drop=True
    )

    print(
        f"[market_data] Universo elegível: "
        f"{len(universe)} ativos."
    )

    return universe


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    dataset = build_market_dataset(
        include_history=False
    )

    if dataset.empty:

        print(
            "Nenhum ativo encontrado."
        )

    else:

        columns = [
            "symbol",
            "name",
            "market_cap",
            "fdv",
            "daily_volume",
            "circulating_total_ratio",
            "market_cap_fdv_ratio",
        ]

        available_columns = [
            column
            for column in columns
            if column in dataset.columns
        ]

        print(
            dataset[
                available_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )
