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

def calculate_snapshot_history_metrics(
    row: pd.Series,
) -> Dict[str, float]:

    """
    Constrói as métricas históricas disponíveis diretamente
    no snapshot de mercado.

    Isso elimina centenas de chamadas individuais à API.

    Importante:
    - return_30d vem diretamente da variação de 30 dias.
    - métricas não fornecidas pela fonte ficam NaN.
    - nenhum valor histórico é inventado.
    """

    return_30d = _percent_to_decimal(
        row.get(
            "price_change_30d"
        )
    )

    ath_change = _safe_float(
        row.get(
            "ath_change_percentage"
        )
    )

    if pd.isna(ath_change):
        drawdown = np.nan
    else:
        drawdown = (
            ath_change
            / 100.0
        )

    return {
        "return_30d":
            return_30d,

        "return_90d":
            np.nan,

        "return_180d":
            np.nan,

        "return_365d":
            np.nan,

        "drawdown_365d":
            drawdown,

        "volume_growth_30d":
            np.nan,
    }


def enrich_with_price_history(
    universe: pd.DataFrame,
    top_n: Optional[int] = None,
    days: int = 365,
) -> pd.DataFrame:

    """
    Mantém compatibilidade com a interface anterior.

    Não realiza chamadas históricas individuais.

    O parâmetro days permanece por compatibilidade com main.py.
    """

    del days

    if universe.empty:

        result = universe.copy()

        for column in HISTORY_METRIC_COLUMNS:

            if column not in result.columns:

                result[
                    column
                ] = pd.Series(
                    dtype="float64"
                )

        return result

    working = universe.copy()

    if top_n is not None:

        working = (
            working
            .head(top_n)
            .copy()
        )

    print(
        "[market_data] "
        "Modo histórico otimizado ativo."
    )

    print(
        "[market_data] "
        "Usando métricas disponíveis "
        "no snapshot; sem consultas "
        "históricas individuais."
    )

    metric_rows = []

    for _, row in working.iterrows():

        metrics = (
            calculate_snapshot_history_metrics(
                row
            )
        )

        metrics[
            "coin_id"
        ] = row.get(
            "coin_id"
        )

        metric_rows.append(
            metrics
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    if metrics_df.empty:

        result = working.copy()

        for column in HISTORY_METRIC_COLUMNS:

            if column not in result.columns:
                result[column] = np.nan

        return result

    result = working.merge(
        metrics_df,
        on="coin_id",
        how="left",
    )

    return result


# ============================================================
# COMPATIBILIDADE
# ============================================================

def fetch_price_history(
    coin_id: str,
    days: int = 365,
    preferred_source: Optional[str] = None,
) -> pd.DataFrame:

    """
    Função mantida para compatibilidade com imports existentes.

    O pipeline principal não usa mais chamadas históricas
    individuais devido ao rate limit das APIs públicas.
    """

    del coin_id
    del days
    del preferred_source

    return pd.DataFrame(
        columns=[
            "date",
            "price",
            "volume",
            "historical_market_cap",
        ]
    )


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
        return np.nan

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
            "Histórico: modo otimizado "
            "sem requisições individuais."
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
