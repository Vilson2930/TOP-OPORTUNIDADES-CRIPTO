"""
CRYPTO OPPORTUNITY ENGINE
Market Data Layer

Fonte principal:
- CoinGecko

Fallback automático:
- CoinPaprika

Responsável por:
- Construir o universo inicial.
- Excluir BTC.
- Excluir stablecoins.
- Excluir wrapped assets.
- Coletar preço, market cap, FDV, volume e supply.
- Calcular retornos e drawdown.
- Manter estrutura compatível com os demais engines.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config


# ============================================================
# APIs
# ============================================================

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINPAPRIKA_BASE_URL = "https://api.coinpaprika.com/v1"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


# ============================================================
# COLUNAS CANÔNICAS
# ============================================================

MARKET_COLUMNS = [
    "coin_id",
    "symbol",
    "name",
    "price",
    "market_cap",
    "market_cap_rank",
    "fdv",
    "daily_volume",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "market_cap_fdv_ratio",
    "circulating_total_ratio",
    "ath",
    "ath_change_percentage",
    "atl",
    "price_change_24h",
    "price_change_7d",
    "price_change_30d",
    "last_updated",
    "market_source",
]

HISTORY_METRIC_COLUMNS = [
    "return_30d",
    "return_90d",
    "return_180d",
    "return_365d",
    "drawdown_365d",
    "volume_growth_30d",
]


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
# EMPTY DATAFRAME
# ============================================================

def _empty_market_dataframe(
    include_history: bool = False,
) -> pd.DataFrame:

    columns = list(MARKET_COLUMNS)

    columns.extend(
        [
            "market_data_completeness",
            "market_data_quality_pass",
        ]
    )

    if include_history:
        columns.extend(HISTORY_METRIC_COLUMNS)

    columns = list(dict.fromkeys(columns))

    return pd.DataFrame(columns=columns)


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


def _headers() -> Dict[str, str]:

    return {
        "accept": "application/json",
        "user-agent": (
            "Mozilla/5.0 "
            "(X11; Linux x86_64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
    }


# ============================================================
# HTTP
# ============================================================

def _http_get(
    url: str,
    params: Optional[Dict] = None,
    provider: str = "API",
) -> Optional[object]:

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                params=params,
                headers=_headers(),
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                try:
                    return response.json()

                except ValueError:
                    print(
                        f"[market_data] "
                        f"{provider}: JSON inválido."
                    )

            elif response.status_code == 429:

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                    * 3
                )

                print(
                    f"[market_data] "
                    f"{provider}: HTTP 429. "
                    f"Tentativa {attempt}/{MAX_RETRIES}. "
                    f"Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            elif response.status_code in {401, 403}:

                print(
                    f"[market_data] "
                    f"{provider}: "
                    f"HTTP {response.status_code}. "
                    "Acesso recusado."
                )

                return None

            elif 500 <= response.status_code < 600:

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    f"[market_data] "
                    f"{provider}: "
                    f"HTTP {response.status_code}. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            else:

                print(
                    f"[market_data] "
                    f"{provider}: "
                    f"HTTP {response.status_code}. "
                    f"{response.text[:200]}"
                )

                return None

        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:

                print(
                    f"[market_data] "
                    f"{provider}: "
                    f"falha definitiva: {exc}"
                )

                return None

            wait = (
                RETRY_BACKOFF_SECONDS
                * attempt
            )

            print(
                f"[market_data] "
                f"{provider}: erro de conexão. "
                f"Retry em {wait}s."
            )

            time.sleep(wait)

    return None


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

    return any(
        term in name
        for term in stable_terms
    )


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

    return any(
        term in name
        for term in wrapped_terms
    )


def _asset_allowed(
    symbol: str,
    name: str,
    market_cap: float,
    daily_volume: float,
) -> bool:

    if not symbol:
        return False

    if is_excluded_symbol(symbol):
        return False

    if (
        config.EXCLUDE_STABLECOINS
        and is_stablecoin(
            symbol,
            name,
        )
    ):
        return False

    if (
        config.EXCLUDE_WRAPPED_ASSETS
        and is_wrapped_asset(
            symbol,
            name,
        )
    ):
        return False

    if (
        pd.isna(market_cap)
        or market_cap < config.MIN_MARKET_CAP_USD
    ):
        return False

    if (
        pd.isna(daily_volume)
        or daily_volume < config.MIN_DAILY_VOLUME_USD
    ):
        return False

    return True


# ============================================================
# COINGECKO
# ============================================================

def fetch_coingecko_market_page(
    page: int,
    per_page: int = 250,
) -> List[Dict]:

    url = (
        f"{COINGECKO_BASE_URL}"
        "/coins/markets"
    )

    params = {
        "vs_currency": config.BASE_CURRENCY.lower(),
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h,7d,30d",
    }

    data = _http_get(
        url=url,
        params=params,
        provider="CoinGecko",
    )

    if not isinstance(data, list):
        return []

    return data


def fetch_coingecko_universe() -> pd.DataFrame:

    target = config.UNIVERSE_MAX_ASSETS
    per_page = 250

    pages = int(
        np.ceil(
            target / per_page
        )
    )

    records: List[Dict] = []

    for page in range(1, pages + 1):

        print(
            "[market_data] "
            f"CoinGecko página "
            f"{page}/{pages}..."
        )

        data = fetch_coingecko_market_page(
            page=page,
            per_page=per_page,
        )

        if not data:
            break

        records.extend(data)

        if len(records) >= target:
            break

        time.sleep(1.2)

    if not records:
        return _empty_market_dataframe()

    rows = []

    for coin in records[:target]:

        symbol = str(
            coin.get(
                "symbol",
                "",
            )
        ).upper().strip()

        name = str(
            coin.get(
                "name",
                "",
            )
        ).strip()

        coin_id = str(
            coin.get(
                "id",
                "",
            )
        ).strip()

        if not coin_id:
            continue

        market_cap = _safe_float(
            coin.get("market_cap")
        )

        daily_volume = _safe_float(
            coin.get("total_volume")
        )

        if not _asset_allowed(
            symbol=symbol,
            name=name,
            market_cap=market_cap,
            daily_volume=daily_volume,
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
            coin.get(
                "fully_diluted_valuation"
            )
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
                "market_cap_fdv_ratio": _safe_divide(
                    market_cap,
                    fdv,
                ),
                "circulating_total_ratio": _safe_divide(
                    circulating_supply,
                    total_supply,
                ),
                "ath": _safe_float(
                    coin.get("ath")
                ),
                "ath_change_percentage": _safe_float(
                    coin.get(
                        "ath_change_percentage"
                    )
                ),
                "atl": _safe_float(
                    coin.get("atl")
                ),
                "price_change_24h": _safe_float(
                    coin.get(
                        "price_change_percentage_24h"
                    )
                ),
                "price_change_7d": _safe_float(
                    coin.get(
                        "price_change_percentage_7d_in_currency"
                    )
                ),
                "price_change_30d": _safe_float(
                    coin.get(
                        "price_change_percentage_30d_in_currency"
                    )
                ),
                "last_updated": coin.get(
                    "last_updated"
                ),
                "market_source": "COINGECKO",
            }
        )

    if not rows:
        return _empty_market_dataframe()

    return pd.DataFrame(
        rows,
        columns=MARKET_COLUMNS,
    )


# ============================================================
# COINPAPRIKA
# ============================================================

def fetch_coinpaprika_tickers() -> List[Dict]:

    url = (
        f"{COINPAPRIKA_BASE_URL}"
        "/tickers"
    )

    data = _http_get(
        url=url,
        params={
            "quotes":
                config.BASE_CURRENCY.upper(),
        },
        provider="CoinPaprika",
    )

    if not isinstance(data, list):
        return []

    return data


def fetch_coinpaprika_universe() -> pd.DataFrame:

    print(
        "[market_data] "
        "Ativando fallback CoinPaprika..."
    )

    records = fetch_coinpaprika_tickers()

    if not records:

        print(
            "[market_data] "
            "CoinPaprika não retornou ativos."
        )

        return _empty_market_dataframe()

    quote_currency = (
        config.BASE_CURRENCY.upper()
    )

    rows = []

    for coin in records:

        quotes = coin.get(
            "quotes",
            {},
        )

        if not isinstance(quotes, dict):
            continue

        quote = quotes.get(
            quote_currency,
            {},
        )

        if not isinstance(quote, dict):
            continue

        symbol = str(
            coin.get(
                "symbol",
                "",
            )
        ).upper().strip()

        name = str(
            coin.get(
                "name",
                "",
            )
        ).strip()

        coin_id = str(
            coin.get(
                "id",
                "",
            )
        ).strip()

        if not coin_id or not symbol:
            continue

        market_cap = _safe_float(
            quote.get("market_cap")
        )

        daily_volume = _safe_float(
            quote.get("volume_24h")
        )

        if not _asset_allowed(
            symbol=symbol,
            name=name,
            market_cap=market_cap,
            daily_volume=daily_volume,
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

        price = _safe_float(
            quote.get("price")
        )

        fdv_supply = max_supply

        if (
            pd.isna(fdv_supply)
            or fdv_supply <= 0
        ):
            fdv_supply = total_supply

        if (
            not pd.isna(price)
            and not pd.isna(fdv_supply)
            and fdv_supply > 0
        ):
            fdv = price * fdv_supply
        else:
            fdv = np.nan

        rows.append(
            {
                "coin_id": coin_id,
                "symbol": symbol,
                "name": name,
                "price": price,
                "market_cap": market_cap,
                "market_cap_rank": coin.get(
                    "rank"
                ),
                "fdv": fdv,
                "daily_volume": daily_volume,
                "circulating_supply": circulating_supply,
                "total_supply": total_supply,
                "max_supply": max_supply,
                "market_cap_fdv_ratio": _safe_divide(
                    market_cap,
                    fdv,
                ),
                "circulating_total_ratio": _safe_divide(
                    circulating_supply,
                    total_supply,
                ),
                "ath": np.nan,
                "ath_change_percentage": np.nan,
                "atl": np.nan,
                "price_change_24h": _safe_float(
                    quote.get(
                        "percent_change_24h"
                    )
                ),
                "price_change_7d": _safe_float(
                    quote.get(
                        "percent_change_7d"
                    )
                ),
                "price_change_30d": _safe_float(
                    quote.get(
                        "percent_change_30d"
                    )
                ),
                "last_updated": coin.get(
                    "last_updated"
                ),
                "market_source": "COINPAPRIKA",
            }
        )

    if not rows:
        return _empty_market_dataframe()

    df = pd.DataFrame(
        rows,
        columns=MARKET_COLUMNS,
    )

    df = (
        df
        .sort_values(
            "market_cap",
            ascending=False,
            na_position="last",
        )
        .head(
            config.UNIVERSE_MAX_ASSETS
        )
        .reset_index(drop=True)
    )

    print(
        "[market_data] "
        f"CoinPaprika: {len(df)} "
        "ativos após filtros."
    )

    return df


# ============================================================
# UNIVERSO COM FAILOVER
# ============================================================

def fetch_market_universe() -> pd.DataFrame:

    print(
        "[market_data] "
        "Tentando CoinGecko..."
    )

    universe = fetch_coingecko_universe()

    if not universe.empty:

        print(
            "[market_data] "
            f"CoinGecko OK: "
            f"{len(universe)} ativos."
        )

    else:

        print(
            "[market_data] "
            "CoinGecko indisponível. "
            "Tentando CoinPaprika."
        )

        universe = fetch_coinpaprika_universe()

    if universe.empty:

        print(
            "[market_data] "
            "Nenhuma fonte de market data "
            "retornou universo válido."
        )

        return _empty_market_dataframe()

    universe = (
        universe
        .drop_duplicates(
            subset=["coin_id"]
        )
        .sort_values(
            "market_cap",
            ascending=False,
            na_position="last",
        )
        .head(
            config.UNIVERSE_MAX_ASSETS
        )
        .reset_index(drop=True)
    )

    print(
        "[market_data] "
        f"Universo bruto: "
        f"{len(universe)} ativos."
    )

    return universe


# ============================================================
# HISTÓRICO COINGECKO
# ============================================================

def fetch_coingecko_price_history(
    coin_id: str,
    days: int = 365,
) -> pd.DataFrame:

    url = (
        f"{COINGECKO_BASE_URL}"
        f"/coins/{coin_id}/market_chart"
    )

    params = {
        "vs_currency":
            config.BASE_CURRENCY.lower(),
        "days":
            days,
        "interval":
            "daily",
    }

    data = _http_get(
        url=url,
        params=params,
        provider="CoinGecko-History",
    )

    if not isinstance(data, dict):
        return pd.DataFrame()

    prices = data.get(
        "prices",
        [],
    )

    if not prices:
        return pd.DataFrame()

    volumes = data.get(
        "total_volumes",
        [],
    )

    market_caps = data.get(
        "market_caps",
        [],
    )

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

        volume_df = volume_df[
            [
                "date",
                "volume",
            ]
        ]

        price_df = price_df.merge(
            volume_df,
            on="date",
            how="left",
        )

    else:
        price_df["volume"] = np.nan

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

        cap_df = cap_df[
            [
                "date",
                "historical_market_cap",
            ]
        ]

        price_df = price_df.merge(
            cap_df,
            on="date",
            how="left",
        )

    else:
        price_df[
            "historical_market_cap"
        ] = np.nan

    return (
        price_df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# HISTÓRICO COINPAPRIKA
# ============================================================

def fetch_coinpaprika_price_history(
    coin_id: str,
    days: int = 365,
) -> pd.DataFrame:

    end_date = datetime.now(
        timezone.utc
    ).date()

    start_date = (
        end_date
        - timedelta(days=days)
    )

    url = (
        f"{COINPAPRIKA_BASE_URL}"
        f"/coins/{coin_id}"
        "/ohlcv/historical"
    )

    params = {
        "start":
            start_date.isoformat(),
        "end":
            end_date.isoformat(),
        "limit":
            min(
                days + 1,
                366,
            ),
        "quote":
            config.BASE_CURRENCY.lower(),
    }

    data = _http_get(
        url=url,
        params=params,
        provider="CoinPaprika-History",
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []

    for candle in data:

        timestamp = (
            candle.get("time_open")
            or candle.get("time_close")
        )

        if timestamp is None:
            continue

        rows.append(
            {
                "date": pd.to_datetime(
                    timestamp,
                    utc=True,
                    errors="coerce",
                ),
                "price": _safe_float(
                    candle.get("close")
                ),
                "volume": _safe_float(
                    candle.get("volume")
                ),
                "historical_market_cap": _safe_float(
                    candle.get("market_cap")
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.dropna(
        subset=["date"]
    )

    return (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# HISTÓRICO COM FAILOVER
# ============================================================

def fetch_price_history(
    coin_id: str,
    days: int = 365,
    preferred_source: Optional[str] = None,
) -> pd.DataFrame:

    preferred_source = str(
        preferred_source or ""
    ).upper()

    if preferred_source == "COINPAPRIKA":

        history = (
            fetch_coinpaprika_price_history(
                coin_id=coin_id,
                days=days,
            )
        )

        if not history.empty:
            return history

        return pd.DataFrame()

    history = (
        fetch_coingecko_price_history(
            coin_id=coin_id,
            days=days,
        )
    )

    if not history.empty:
        return history

    return pd.DataFrame()


# ============================================================
# MÉTRICAS DE PREÇO
# ============================================================

def calculate_return(
    history: pd.DataFrame,
    days: int,
) -> float:

    if (
        history.empty
        or "price" not in history.columns
        or "date" not in history.columns
    ):
        return np.nan

    history = history.copy()

    history["price"] = pd.to_numeric(
        history["price"],
        errors="coerce",
    )

    history = history.dropna(
        subset=[
            "date",
            "price",
        ]
    )

    if len(history) < 2:
        return np.nan

    current_price = _safe_float(
        history["price"].iloc[-1]
    )

    if (
        pd.isna(current_price)
        or current_price <= 0
    ):
        return np.nan

    target_date = (
        history["date"].iloc[-1]
        - pd.Timedelta(days=days)
    )

    historical = history[
        history["date"] <= target_date
    ]

    if historical.empty:
        return np.nan

    old_price = _safe_float(
        historical["price"].iloc[-1]
    )

    if (
        pd.isna(old_price)
        or old_price <= 0
    ):
        return np.nan

    return (
        current_price
        / old_price
        - 1
    )


def calculate_drawdown(
    history: pd.DataFrame,
) -> float:

    if (
        history.empty
        or "price" not in history.columns
    ):
        return np.nan

    prices = pd.to_numeric(
        history["price"],
        errors="coerce",
    ).dropna()

    if prices.empty:
        return np.nan

    peak = prices.max()
    current = prices.iloc[-1]

    if (
        pd.isna(peak)
        or peak <= 0
    ):
        return np.nan

    return (
        current
        / peak
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
                history
            ),
    }

    if (
        "volume" in history.columns
        and len(history) >= 60
    ):

        volume_series = pd.to_numeric(
            history["volume"],
            errors="coerce",
        )

        recent = (
            volume_series
            .tail(30)
            .mean()
        )

        previous = (
            volume_series
            .iloc[-60:-30]
            .mean()
        )

        if (
            not pd.isna(recent)
            and not pd.isna(previous)
            and previous > 0
        ):

            metrics[
                "volume_growth_30d"
            ] = (
                recent
                / previous
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
# ENRIQUECIMENTO HISTÓRICO
# ============================================================

def enrich_with_price_history(
    universe: pd.DataFrame,
    top_n: Optional[int] = None,
    days: int = 365,
) -> pd.DataFrame:

    if universe.empty:

        result = universe.copy()

        for column in HISTORY_METRIC_COLUMNS:

            if column not in result.columns:

                result[column] = pd.Series(
                    dtype="float64"
                )

        return result

    df = universe.copy()

    if top_n is not None:
        working = df.head(top_n).copy()
    else:
        working = df.copy()

    metrics_records = []

    total = len(working)

    for position, (_, row) in enumerate(
        working.iterrows(),
        start=1,
    ):

        coin_id = row.get(
            "coin_id"
        )

        symbol = row.get(
            "symbol",
            "UNKNOWN",
        )

        source = row.get(
            "market_source",
            "",
        )

        print(
            "[market_data] "
            f"Histórico {symbol} "
            f"({position}/{total}) "
            f"[{source}]"
        )

        history = fetch_price_history(
            coin_id=coin_id,
            days=days,
            preferred_source=source,
        )

        metrics = calculate_price_metrics(
            history
        )

        metrics["coin_id"] = coin_id

        metrics_records.append(
            metrics
        )

        time.sleep(0.4)

    if not metrics_records:

        result = working.copy()

        for column in HISTORY_METRIC_COLUMNS:

            if column not in result.columns:
                result[column] = np.nan

        return result

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

        result = df.copy()

        result[
            "market_data_completeness"
        ] = pd.Series(
            dtype="float64"
        )

        result[
            "market_data_quality_pass"
        ] = pd.Series(
            dtype="bool"
        )

        return result

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
        "[market_data] "
        "Construindo universo..."
    )

    universe = fetch_market_universe()

    if universe.empty:

        print(
            "[market_data] "
            "Universo vazio após todas "
            "as fontes de dados."
        )

        return _empty_market_dataframe(
            include_history=include_history
        )

    universe = apply_data_quality(
        universe
    )

    passed = universe[
        universe[
            "market_data_quality_pass"
        ]
    ].copy()

    if passed.empty:

        print(
            "[market_data] "
            "Nenhum ativo passou pelo "
            "controle de qualidade."
        )

        print(
            "[market_data] "
            "Distribuição de completeness:"
        )

        print(
            universe[
                "market_data_completeness"
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

        return _empty_market_dataframe(
            include_history=include_history
        )

    universe = passed

    if include_history:

        universe = enrich_with_price_history(
            universe=universe,
            top_n=history_top_n,
            days=365,
        )

    universe = universe.reset_index(
        drop=True
    )

    source_counts = (
        universe[
            "market_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    print(
        "[market_data] "
        f"Universo elegível: "
        f"{len(universe)} ativos."
    )

    print(
        "[market_data] "
        f"Fontes: {source_counts}"
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

        print(
            "Colunas:"
        )

        print(
            list(dataset.columns)
        )

    else:

        columns = [
            "coin_id",
            "symbol",
            "name",
            "price",
            "market_cap",
            "fdv",
            "daily_volume",
            "circulating_supply",
            "total_supply",
            "circulating_total_ratio",
            "market_cap_fdv_ratio",
            "market_data_completeness",
            "market_data_quality_pass",
            "market_source",
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
