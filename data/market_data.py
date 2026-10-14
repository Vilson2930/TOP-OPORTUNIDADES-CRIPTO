"""
CRYPTO OPPORTUNITY ENGINE
Market Data Layer

Versão otimizada para GitHub Actions.

Estratégia:
1. CoinGecko continua como fonte principal do universo.
2. CoinPaprika funciona como fallback automático.
3. NÃO realiza centenas de chamadas históricas individuais.
4. Retornos 24h / 7d / 30d vêm diretamente do snapshot de mercado.
5. 90d / 180d / 365d ficam disponíveis quando fornecidos pela fonte.
6. Evita o gargalo de HTTP 429 que fazia o workflow levar dezenas de minutos.
7. Mantém as colunas esperadas pelos demais engines.

Objetivo:
MARKET DATA -> FILTROS -> QUALITY -> ENGINE
"""

from __future__ import annotations

import time
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

# Para market snapshot não precisamos de retries longos.
# Se CoinGecko bloquear, o engine muda rapidamente para CoinPaprika.
MAX_RETRIES = 2
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
# WRAPPED / DERIVATIVE ASSETS
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

    columns = list(
        dict.fromkeys(columns)
    )

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value) -> float:

    try:

        if value is None:
            return np.nan

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
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


def _percent_to_decimal(
    value,
) -> float:

    value = _safe_float(value)

    if pd.isna(value):
        return np.nan

    return value / 100.0


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

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

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
                        "[market_data] "
                        f"{provider}: JSON inválido."
                    )

                    return None

            if response.status_code in {
                401,
                403,
            }:

                print(
                    "[market_data] "
                    f"{provider}: "
                    f"HTTP {response.status_code}. "
                    "Fonte indisponível para este runner."
                )

                return None

            if response.status_code == 429:

                # Não vamos deixar o workflow parado
                # durante vários minutos.
                if attempt >= MAX_RETRIES:

                    print(
                        "[market_data] "
                        f"{provider}: HTTP 429. "
                        "Limite de requisições atingido."
                    )

                    return None

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    "[market_data] "
                    f"{provider}: HTTP 429. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)

                continue

            if 500 <= response.status_code < 600:

                if attempt >= MAX_RETRIES:

                    print(
                        "[market_data] "
                        f"{provider}: "
                        f"HTTP {response.status_code}."
                    )

                    return None

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                time.sleep(wait)
                continue

            print(
                "[market_data] "
                f"{provider}: "
                f"HTTP {response.status_code}. "
                f"{response.text[:200]}"
            )

            return None

        except requests.RequestException as exc:

            if attempt >= MAX_RETRIES:

                print(
                    "[market_data] "
                    f"{provider}: "
                    f"falha definitiva: {exc}"
                )

                return None

            wait = (
                RETRY_BACKOFF_SECONDS
                * attempt
            )

            time.sleep(wait)

    return None


# ============================================================
# FILTROS
# ============================================================

def is_excluded_symbol(
    symbol: str,
) -> bool:

    symbol = str(
        symbol
    ).upper().strip()

    return (
        symbol
        in config.EXCLUDED_SYMBOLS
    )


def is_stablecoin(
    symbol: str,
    name: str = "",
) -> bool:

    symbol = str(
        symbol
    ).upper().strip()

    name = str(
        name
    ).lower().strip()

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

    symbol = str(
        symbol
    ).upper().strip()

    name = str(
        name
    ).lower().strip()

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

    if is_excluded_symbol(
        symbol
    ):
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
        or market_cap
        < config.MIN_MARKET_CAP_USD
    ):
        return False

    if (
        pd.isna(daily_volume)
        or daily_volume
        < config.MIN_DAILY_VOLUME_USD
    ):
        return False

    return True


# ============================================================
# COINGECKO SNAPSHOT
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
        "vs_currency":
            config.BASE_CURRENCY.lower(),

        "order":
            "market_cap_desc",

        "per_page":
            per_page,

        "page":
            page,

        "sparkline":
            "false",

        "price_change_percentage":
            "24h,7d,30d",
    }

    data = _http_get(
        url=url,
        params=params,
        provider="CoinGecko",
    )

    if not isinstance(
        data,
        list,
    ):
        return []

    return data


def fetch_coingecko_universe() -> pd.DataFrame:

    target = (
        config.UNIVERSE_MAX_ASSETS
    )

    per_page = 250

    pages = int(
        np.ceil(
            target / per_page
        )
    )

    records: List[Dict] = []

    for page in range(
        1,
        pages + 1,
    ):

        print(
            "[market_data] "
            f"CoinGecko página "
            f"{page}/{pages}..."
        )

        data = (
            fetch_coingecko_market_page(
                page=page,
                per_page=per_page,
            )
        )

        if not data:
            break

        records.extend(data)

        if len(records) >= target:
            break

        # Pequena pausa apenas entre páginas.
        time.sleep(1.0)

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

        if (
            not coin_id
            or not symbol
        ):
            continue

        market_cap = _safe_float(
            coin.get(
                "market_cap"
            )
        )

        daily_volume = _safe_float(
            coin.get(
                "total_volume"
            )
        )

        if not _asset_allowed(
            symbol=symbol,
            name=name,
            market_cap=market_cap,
            daily_volume=daily_volume,
        ):
            continue

        circulating_supply = (
            _safe_float(
                coin.get(
                    "circulating_supply"
                )
            )
        )

        total_supply = (
            _safe_float(
                coin.get(
                    "total_supply"
                )
            )
        )

        max_supply = (
            _safe_float(
                coin.get(
                    "max_supply"
                )
            )
        )

        fdv = _safe_float(
            coin.get(
                "fully_diluted_valuation"
            )
        )

        rows.append(
            {
                "coin_id":
                    coin_id,

                "symbol":
                    symbol,

                "name":
                    name,

                "price":
                    _safe_float(
                        coin.get(
                            "current_price"
                        )
                    ),

                "market_cap":
                    market_cap,

                "market_cap_rank":
                    coin.get(
                        "market_cap_rank"
                    ),

                "fdv":
                    fdv,

                "daily_volume":
                    daily_volume,

                "circulating_supply":
                    circulating_supply,

                "total_supply":
                    total_supply,

                "max_supply":
                    max_supply,

                "market_cap_fdv_ratio":
                    _safe_divide(
                        market_cap,
                        fdv,
                    ),

                "circulating_total_ratio":
                    _safe_divide(
                        circulating_supply,
                        total_supply,
                    ),

                "ath":
                    _safe_float(
                        coin.get(
                            "ath"
                        )
                    ),

                "ath_change_percentage":
                    _safe_float(
                        coin.get(
                            "ath_change_percentage"
                        )
                    ),

                "atl":
                    _safe_float(
                        coin.get(
                            "atl"
                        )
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
                    coin.get(
                        "last_updated"
                    ),

                "market_source":
                    "COINGECKO",
            }
        )

    if not rows:
        return _empty_market_dataframe()

    return pd.DataFrame(
        rows,
        columns=MARKET_COLUMNS,
    )


# ============================================================
# COINPAPRIKA SNAPSHOT
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

    if not isinstance(
        data,
        list,
    ):
        return []

    return data


def fetch_coinpaprika_universe() -> pd.DataFrame:

    print(
        "[market_data] "
        "Ativando fallback CoinPaprika..."
    )

    records = (
        fetch_coinpaprika_tickers()
    )

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

        if not isinstance(
            quotes,
            dict,
        ):
            continue

        quote = quotes.get(
            quote_currency,
            {},
        )

        if not isinstance(
            quote,
            dict,
        ):
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

        if (
            not coin_id
            or not symbol
        ):
            continue

        market_cap = _safe_float(
            quote.get(
                "market_cap"
            )
        )

        daily_volume = _safe_float(
            quote.get(
                "volume_24h"
            )
        )

        if not _asset_allowed(
            symbol=symbol,
            name=name,
            market_cap=market_cap,
            daily_volume=daily_volume,
        ):
            continue

        circulating_supply = (
            _safe_float(
                coin.get(
                    "circulating_supply"
                )
            )
        )

        total_supply = (
            _safe_float(
                coin.get(
                    "total_supply"
                )
            )
        )

        max_supply = (
            _safe_float(
                coin.get(
                    "max_supply"
                )
            )
        )

        price = _safe_float(
            quote.get(
                "price"
            )
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
            fdv = (
                price
                * fdv_supply
            )

        else:
            fdv = np.nan

        rows.append(
            {
                "coin_id":
                    coin_id,

                "symbol":
                    symbol,

                "name":
                    name,

                "price":
                    price,

                "market_cap":
                    market_cap,

                "market_cap_rank":
                    coin.get(
                        "rank"
                    ),

                "fdv":
                    fdv,

                "daily_volume":
                    daily_volume,

                "circulating_supply":
                    circulating_supply,

                "total_supply":
                    total_supply,

                "max_supply":
                    max_supply,

                "market_cap_fdv_ratio":
                    _safe_divide(
                        market_cap,
                        fdv,
                    ),

                "circulating_total_ratio":
                    _safe_divide(
                        circulating_supply,
                        total_supply,
                    ),

                "ath":
                    np.nan,

                "ath_change_percentage":
                    np.nan,

                "atl":
                    np.nan,

                "price_change_24h":
                    _safe_float(
                        quote.get(
                            "percent_change_24h"
                        )
                    ),

                "price_change_7d":
                    _safe_float(
                        quote.get(
                            "percent_change_7d"
                        )
                    ),

                "price_change_30d":
                    _safe_float(
                        quote.get(
                            "percent_change_30d"
                        )
                    ),

                "last_updated":
                    coin.get(
                        "last_updated"
                    ),

                "market_source":
                    "COINPAPRIKA",
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
        .reset_index(
            drop=True
        )
    )

    print(
        "[market_data] "
        f"CoinPaprika: "
        f"{len(df)} ativos após filtros."
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

    universe = (
        fetch_coingecko_universe()
    )

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

        universe = (
            fetch_coinpaprika_universe()
        )

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
            subset=[
                "coin_id"
            ]
        )
        .sort_values(
            "market_cap",
            ascending=False,
            na_position="last",
        )
        .head(
            config.UNIVERSE_MAX_ASSETS
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "[market_data] "
        f"Universo bruto: "
        f"{len(universe)} ativos."
    )

    return universe


# ============================================================
# HISTÓRICO OTIMIZADO
# ============================================================

def calculate_snapshot_history_metrics(row: pd.Series) -> Dict[str, float]:
    """Métricas comprováveis do snapshot; ATH não é drawdown de 365 dias."""
    return {
        "return_30d": _percent_to_decimal(row.get("price_change_30d")),
        "return_90d": np.nan,
        "return_180d": np.nan,
        "return_365d": np.nan,
        "drawdown_365d": np.nan,
        "volume_growth_30d": np.nan,
    }


def _history_budget() -> int:
    """Limite explícito de requisições por execução para evitar rate limit."""
    import os
    try:
        return max(0, min(100, int(os.getenv("CRYPTO_HISTORY_MAX_REQUESTS", "40"))))
    except ValueError:
        return 40


def _coingecko_chart(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Uma consulta por moeda, sem fabricar preços ou trocar IDs entre APIs."""
    from urllib.parse import quote
    url = f"{COINGECKO_BASE_URL}/coins/{quote(str(coin_id), safe='')}/market_chart"
    data = _http_get(
        url,
        params={"vs_currency": config.BASE_CURRENCY.lower(), "days": days},
        provider=f"CoinGecko history {coin_id}",
    )
    if not isinstance(data, dict):
        return pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    prices = data.get("prices")
    if not isinstance(prices, list) or len(prices) < 2:
        return pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0].sort_values("date").drop_duplicates("date")
    volumes = data.get("total_volumes")
    if isinstance(volumes, list) and volumes:
        vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        vol["date"] = pd.to_datetime(vol["timestamp"], unit="ms", utc=True, errors="coerce")
        vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce")
        df = df.merge(vol[["date", "volume"]], on="date", how="left")
    else:
        df["volume"] = np.nan
    df["historical_market_cap"] = np.nan
    return df[["date", "price", "volume", "historical_market_cap"]].reset_index(drop=True)


# ============================================================
# KRAKEN OHLC - FALLBACK HISTORICO COM MAPEAMENTO EXPLICITO
# ============================================================

# Apenas pares identificados manualmente. NUNCA associar ativos por
# simbolo generico ou utilizar o ID CoinGecko como ID Kraken.
KRAKEN_USD_PAIRS = {
    "ethereum": ("ETH", "ETHUSD"),
    "ripple": ("XRP", "XRPUSD"),
    "solana": ("SOL", "SOLUSD"),
    "tron": ("TRX", "TRXUSD"),
    "zcash": ("ZEC", "ZECUSD"),
    "chainlink": ("LINK", "LINKUSD"),
    "avalanche-2": ("AVAX", "AVAXUSD"),
    "polkadot": ("DOT", "DOTUSD"),
    "dogecoin": ("DOGE", "DOGEUSD"),
    "litecoin": ("LTC", "LTCUSD"),
    "cardano": ("ADA", "ADAUSD"),
    "uniswap": ("UNI", "UNIUSD"),
    "aave": ("AAVE", "AAVEUSD"),
}


def _kraken_daily_chart(coin_id: str, symbol: str) -> pd.DataFrame:
    """OHLC diario publico; sem chaves. Limite da API: ~720 candles."""
    empty = pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    pair = KRAKEN_USD_PAIRS.get(str(coin_id).lower())
    if pair is None or str(symbol).upper().strip() != pair[0]:
        return empty
    data = _http_get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair[1], "interval": 1440},
        provider=f"Kraken history {coin_id}",
    )
    if not isinstance(data, dict) or data.get("error"):
        print(f"[market_data] Kraken {coin_id}: erro API {str(data.get('error') if isinstance(data, dict) else 'sem resposta')[:120]}")
        return empty
    payload = data.get("result")
    if not isinstance(payload, dict):
        return empty
    keys = [key for key in payload if key != "last"]
    if len(keys) != 1 or not isinstance(payload[keys[0]], list):
        return empty
    rows = payload[keys[0]]
    if len(rows) < 2:
        return empty
    try:
        df = pd.DataFrame(rows)
        if df.shape[1] < 7:
            return empty
        # Kraken OHLC: time, open, high, low, close, vwap, volume, count
        df = pd.DataFrame({
            "date": pd.to_datetime(pd.to_numeric(df.iloc[:, 0], errors="coerce"), unit="s", utc=True, errors="coerce"),
            "price": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "volume": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
        })
        df = df.dropna(subset=["date", "price"])
        df = df[df["price"] > 0].sort_values("date").drop_duplicates("date")
        # Excluir candle diario ainda em formacao (ultimo candle Kraken).
        if len(df) > 1:
            df = df.iloc[:-1].copy()
        df["historical_market_cap"] = np.nan
        return df.reset_index(drop=True)
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        print(f"[market_data] Kraken {coin_id}: OHLC invalido: {exc}")
        return empty


def _defillama_daily_chart(coin_id: str, days: int = 390) -> pd.DataFrame:
    """DefiLlama Coins chart, identificador CoinGecko explícito.

    Não procura por ticker: tickers iguais podem representar projetos diferentes.
    A API deve confirmar a chave consultada no objeto coins.
    """
    from urllib.parse import quote
    import time as _time

    empty = pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    if not coin_id or not isinstance(coin_id, str):
        return empty
    key = f"coingecko:{coin_id}"
    now = int(_time.time())
    start = now - max(390, int(days) + 10) * 86400
    url = "https://coins.llama.fi/chart/" + quote(key, safe=":")
    data = _http_get(
        url,
        params={"start": start, "span": max(390, int(days) + 10), "period": "1d"},
        provider=f"DefiLlama history {coin_id}",
    )
    if not isinstance(data, dict):
        return empty
    coins = data.get("coins")
    if not isinstance(coins, dict) or key not in coins:
        return empty
    payload = coins[key]
    if not isinstance(payload, dict) or not isinstance(payload.get("prices"), list):
        return empty
    rows = payload["prices"]
    if len(rows) < 2:
        return empty
    try:
        df = pd.DataFrame(rows)
        if not {"timestamp", "price"}.issubset(df.columns):
            return empty
        df = pd.DataFrame({
            "date": pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True, errors="coerce"),
            "price": pd.to_numeric(df["price"], errors="coerce"),
        })
        df = df.dropna(subset=["date", "price"])
        df = df[df["price"] > 0].sort_values("date").drop_duplicates("date")
        if df.empty:
            return empty
        # Rejeitar série desatualizada (não confundir preço antigo com atual).
        if (pd.Timestamp.now(tz="UTC") - df["date"].iloc[-1]).total_seconds() > 3 * 86400:
            return empty
        df["volume"] = np.nan
        df["historical_market_cap"] = np.nan
        return df.reset_index(drop=True)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        print(f"[market_data] DefiLlama {coin_id}: série inválida: {exc}")
        return empty


def _coinpaprika_id_map() -> dict:
    """Mapeia (símbolo, nome) para ID Paprika somente quando único e ativo.

    Não vincula projetos por ticker isolado; falha fechada em ambiguidades.
    """
    from functools import lru_cache
    return _coinpaprika_id_map_cached()


from functools import lru_cache


@lru_cache(maxsize=1)
def _coinpaprika_id_map_cached() -> dict:
    data = _http_get(f"{COINPAPRIKA_BASE_URL}/coins", provider="CoinPaprika coin mapping")
    if not isinstance(data, list):
        return {}
    matches = {}
    ambiguous = set()
    for item in data:
        if not isinstance(item, dict) or item.get("is_active") is not True:
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        name = " ".join(str(item.get("name") or "").casefold().split())
        paprika_id = str(item.get("id") or "").strip()
        if not symbol or not name or not paprika_id:
            continue
        key = (symbol, name)
        if key in matches and matches[key] != paprika_id:
            ambiguous.add(key)
        else:
            matches[key] = paprika_id
    for key in ambiguous:
        matches.pop(key, None)
    print(f"[market_data] CoinPaprika: {len(matches)} IDs únicos por nome+símbolo; {len(ambiguous)} ambiguidades descartadas.")
    return matches


def _coinpaprika_daily_chart(coin_id: str, symbol: str, name: str, days: int = 390,
                            market_source: str = "COINGECKO") -> pd.DataFrame:
    """Consulta endpoint histórico público; disponibilidade depende do plano/API.

    Para universo CoinPaprika, o ID já é nativo; para CoinGecko, exige
    correspondência exata e não ambígua de nome + símbolo.
    """
    from datetime import datetime, timedelta, timezone
    from urllib.parse import quote
    empty = pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    if market_source == "COINPAPRIKA":
        paprika_id = str(coin_id).strip()
    else:
        key = (str(symbol).strip().upper(), " ".join(str(name).casefold().split()))
        paprika_id = _coinpaprika_id_map().get(key)
    if not paprika_id:
        return empty
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(390, int(days) + 10))
    data = _http_get(
        f"{COINPAPRIKA_BASE_URL}/tickers/{quote(paprika_id, safe='')}/historical",
        params={"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "interval": "24h"},
        provider=f"CoinPaprika history {paprika_id}",
    )
    if not isinstance(data, list) or len(data) < 2:
        return empty
    try:
        df = pd.DataFrame(data)
        if not {"timestamp", "price"}.issubset(df.columns):
            return empty
        result = pd.DataFrame({
            "date": pd.to_datetime(df["timestamp"], utc=True, errors="coerce"),
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "volume": pd.to_numeric(df["volume_24h"], errors="coerce") if "volume_24h" in df else np.nan,
            "historical_market_cap": pd.to_numeric(df["market_cap"], errors="coerce") if "market_cap" in df else np.nan,
        })
        result = result.dropna(subset=["date", "price"])
        result = result[result["price"] > 0].sort_values("date").drop_duplicates("date")
        if result.empty or (pd.Timestamp.now(tz="UTC") - result["date"].iloc[-1]).total_seconds() > 3 * 86400:
            return empty
        return result.reset_index(drop=True)
    except (TypeError, ValueError, KeyError) as exc:
        print(f"[market_data] CoinPaprika {paprika_id}: dados inválidos: {exc}")
        return empty


def _historical_chart_with_fallback(coin_id: str, symbol: str, days: int,
                                    name: str = "", market_source: str = "COINGECKO") -> tuple:
    """Fontes gratuitas em cascata, preservando identidade do ativo."""
    if market_source == "COINGECKO":
        chart = _defillama_daily_chart(coin_id, days=max(390, days))
        if not chart.empty:
            return chart, "DEFILLAMA"
    chart = _coinpaprika_daily_chart(coin_id, symbol, name, days=max(390, days), market_source=market_source)
    if not chart.empty:
        return chart, "COINPAPRIKA"
    if market_source == "COINGECKO":
        if str(coin_id).lower() in KRAKEN_USD_PAIRS:
            chart = _kraken_daily_chart(coin_id, symbol)
            if not chart.empty:
                return chart, "KRAKEN"
        chart = _coingecko_chart(coin_id, days=min(365, max(1, int(days))))
        if not chart.empty:
            return chart, "COINGECKO"
    return pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"]), "UNAVAILABLE"


def enrich_with_price_history(
    universe: pd.DataFrame,
    top_n: Optional[int] = None,
    days: int = 365,
) -> pd.DataFrame:
    """Enriquece até N ativos, preservando TODAS as linhas do universo.

    IDs CoinPaprika e CoinGecko não são intercambiáveis; usa mapeamento
    validado e registra a fonte do histórico.
    """
    result = universe.copy()
    for column in HISTORY_METRIC_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    if result.empty:
        return result

    for idx, row in result.iterrows():
        for key, value in calculate_snapshot_history_metrics(row).items():
            result.at[idx, key] = value

    eligible = result.index if top_n is None else result.index[:max(0, int(top_n))]
    budget = _history_budget()
    result["history_source"] = "NOT_REQUESTED"
    requests_made = 0
    print(f"[market_data] Histórico real: orçamento máximo {budget} requisições; elegíveis {len(eligible)}.")
    for idx in eligible:
        if requests_made >= budget:
            break
        row = result.loc[idx]
        coin_id = str(row.get("coin_id", "")).strip()
        if not coin_id:
            continue
        requests_made += 1
        history, history_provider = _historical_chart_with_fallback(
            coin_id, str(row.get("symbol", "")), days,
            name=str(row.get("name", "")),
            market_source=str(row.get("market_source", "COINGECKO")).upper(),
        )
        result.at[idx, "history_source"] = history_provider
        if history.empty:
            print(f"[market_data] Histórico {coin_id}: sem dados; verifique HTTP/API no log.")
            continue
        metrics = calculate_price_metrics(history)
        for key in HISTORY_METRIC_COLUMNS:
            value = metrics.get(key, np.nan)
            if pd.notna(value):
                result.at[idx, key] = value
        print(f"[market_data] Histórico {coin_id} ({history_provider}): {len(history)} observações; retornos 90d={metrics.get('return_90d')}, 180d={metrics.get('return_180d')}, 365d={metrics.get('return_365d')}.")
        time.sleep(1.0)
    print(f"[market_data] Histórico: {requests_made} ativos consultados; {len(result)} ativos preservados.")
    return result


# ============================================================
# COMPATIBILIDADE
# ============================================================

def fetch_price_history(
    coin_id: str,
    days: int = 365,
    preferred_source: Optional[str] = None,
) -> pd.DataFrame:
    """Histórico real por ID CoinGecko, com fallback auditável."""
    source = str(preferred_source or "COINGECKO").upper()
    if source not in {"COINGECKO", "COINPAPRIKA"}:
        return pd.DataFrame(columns=["date", "price", "volume", "historical_market_cap"])
    history, _ = _historical_chart_with_fallback(coin_id, "", days, market_source=source)
    return history


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

    history[
        "price"
    ] = pd.to_numeric(
        history[
            "price"
        ],
        errors="coerce",
    )

    history[
        "date"
    ] = pd.to_datetime(
        history[
            "date"
        ],
        utc=True,
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

    history = history.sort_values(
        "date"
    )

    current_price = _safe_float(
        history[
            "price"
        ].iloc[-1]
    )

    if (
        pd.isna(current_price)
        or current_price <= 0
    ):
        return np.nan

    target_date = (
        history[
            "date"
        ].iloc[-1]
        - pd.Timedelta(
            days=days
        )
    )

    historical = history[
        history[
            "date"
        ]
        <= target_date
    ]

    if historical.empty:
        # Candles diários podem estar deslocados algumas horas do alvo exato.
        # Aceita apenas ponto a até 48h do alvo; não inventa retorno sem histórico.
        near = history[(history["date"] > target_date) & (history["date"] <= target_date + pd.Timedelta(hours=48))]
        if near.empty:
            return np.nan
        historical = near.iloc[:1]

    old_price = _safe_float(
        historical[
            "price"
        ].iloc[-1]
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
        history[
            "price"
        ],
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
            "return_30d":
                np.nan,

            "return_90d":
                np.nan,

            "return_180d":
                np.nan,

            "return_365d":
                np.nan,

            "drawdown_365d":
                np.nan,

            "volume_growth_30d":
                np.nan,
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

        "volume_growth_30d":
            np.nan,
    }

    if (
        "volume" in history.columns
        and len(history) >= 60
    ):

        volume_series = pd.to_numeric(
            history[
                "volume"
            ],
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

        value = row.get(
            field
        )

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

    universe = (
        fetch_market_universe()
    )

    if universe.empty:

        print(
            "[market_data] "
            "Universo vazio após todas "
            "as fontes de dados."
        )

        return _empty_market_dataframe(
            include_history=include_history
        )

    universe = (
        apply_data_quality(
            universe
        )
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

    print(
        "[market_data] "
        f"Ativos após quality filter: "
        f"{len(universe)}"
    )

    if include_history:

        universe = (
            enrich_with_price_history(
                universe=universe,
                top_n=history_top_n,
                days=365,
            )
        )

    universe = (
        universe
        .reset_index(
            drop=True
        )
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

    if include_history:

        print(
            "[market_data] "
            "Histórico: consultas limitadas e sem valores inventados."
        )

    return universe


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    dataset = (
        build_market_dataset(
            include_history=True
        )
    )

    if dataset.empty:

        print(
            "Nenhum ativo encontrado."
        )

        print(
            "Colunas:"
        )

        print(
            list(
                dataset.columns
            )
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
            "price_change_24h",
            "price_change_7d",
            "price_change_30d",
            "return_30d",
            "return_90d",
            "return_180d",
            "return_365d",
            "drawdown_365d",
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
