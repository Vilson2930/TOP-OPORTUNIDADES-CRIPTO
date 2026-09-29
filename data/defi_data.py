"""
CRYPTO OPPORTUNITY ENGINE
DeFi / TVL Data Layer

OBJETIVO:
Coletar TVL e métricas DeFi necessárias para identificar OPORTUNIDADES.

O foco não é premiar TVL alto.
O foco é detectar:
- crescimento de TVL;
- aceleração de TVL;
- capital entrando antes de forte reprecificação;
- eficiência econômica do capital;
- confirmação de adoção real.

Fonte principal:
DefiLlama Public API.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config


DEFILLAMA_BASE_URL = "https://api.llama.fi"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


# ============================================================
# HTTP
# ============================================================

def _request(
    endpoint: str,
    params: Optional[Dict] = None,
) -> Optional[object]:

    url = f"{DEFILLAMA_BASE_URL}{endpoint}"

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
                    f"[defi_data] Rate limit. "
                    f"Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:

                wait = RETRY_BACKOFF_SECONDS * attempt

                print(
                    f"[defi_data] HTTP "
                    f"{response.status_code}. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            print(
                f"[defi_data] HTTP "
                f"{response.status_code}: "
                f"{response.text[:200]}"
            )

            return None

        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:

                print(
                    "[defi_data] Falha definitiva: "
                    f"{exc}"
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


def _normalize_symbol(symbol: str) -> str:

    return str(
        symbol or ""
    ).upper().strip()


def _normalize_text(value: str) -> str:

    return str(
        value or ""
    ).lower().strip()


# ============================================================
# LISTA DE PROTOCOLOS
# ============================================================

def fetch_protocols() -> pd.DataFrame:

    data = _request("/protocols")

    if not isinstance(data, list):
        return pd.DataFrame()

    rows: List[Dict] = []

    for protocol in data:

        symbol = _normalize_symbol(
            protocol.get("symbol")
        )

        name = str(
            protocol.get("name", "")
        ).strip()

        slug = str(
            protocol.get("slug", "")
        ).strip()

        if not slug:
            continue

        rows.append(
            {
                "defillama_slug": slug,
                "defillama_name": name,
                "defillama_symbol": symbol,

                "tvl_current":
                    _safe_float(
                        protocol.get("tvl")
                    ),

                "defillama_category":
                    protocol.get("category"),

                "defillama_chain":
                    protocol.get("chain"),

                "defillama_mcap":
                    _safe_float(
                        protocol.get("mcap")
                    ),

                "defillama_change_1d":
                    _safe_float(
                        protocol.get("change_1d")
                    ),

                "defillama_change_7d":
                    _safe_float(
                        protocol.get("change_7d")
                    ),

                "defillama_change_1m":
                    _safe_float(
                        protocol.get("change_1m")
                    ),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.drop_duplicates(
        subset=["defillama_slug"]
    ).reset_index(drop=True)


# ============================================================
# MATCH PROJETO ↔ DEFILLAMA
# ============================================================

def match_protocol(
    symbol: str,
    name: str,
    protocols: pd.DataFrame,
) -> Optional[pd.Series]:

    if protocols.empty:
        return None

    symbol = _normalize_symbol(symbol)
    name_normalized = _normalize_text(name)

    # --------------------------------------------------------
    # Primeiro tenta símbolo + nome.
    # --------------------------------------------------------

    symbol_matches = protocols[
        protocols["defillama_symbol"] == symbol
    ]

    if len(symbol_matches) == 1:
        return symbol_matches.iloc[0]

    if len(symbol_matches) > 1:

        exact_name = symbol_matches[
            symbol_matches[
                "defillama_name"
            ].str.lower() == name_normalized
        ]

        if len(exact_name) == 1:
            return exact_name.iloc[0]

        contains_name = symbol_matches[
            symbol_matches[
                "defillama_name"
            ].str.lower().str.contains(
                name_normalized,
                regex=False,
                na=False,
            )
        ]

        if len(contains_name) == 1:
            return contains_name.iloc[0]

    # --------------------------------------------------------
    # Depois tenta nome exato.
    # --------------------------------------------------------

    exact_name = protocols[
        protocols[
            "defillama_name"
        ].str.lower() == name_normalized
    ]

    if len(exact_name) == 1:
        return exact_name.iloc[0]

    return None


# ============================================================
# HISTÓRICO DO PROTOCOLO
# ============================================================

def fetch_protocol_history(
    slug: str,
) -> pd.DataFrame:

    data = _request(
        f"/protocol/{slug}"
    )

    if not isinstance(data, dict):
        return pd.DataFrame()

    tvl_data = data.get("tvl", [])

    if not isinstance(tvl_data, list):
        return pd.DataFrame()

    rows = []

    for item in tvl_data:

        timestamp = item.get("date")

        tvl = _safe_float(
            item.get("totalLiquidityUSD")
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
                "tvl": tvl,
            }
        )

    history = pd.DataFrame(rows)

    if history.empty:
        return history

    history = (
        history
        .dropna(subset=["tvl"])
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return history


# ============================================================
# TVL EM DATA PASSADA
# ============================================================

def _value_at_days_ago(
    history: pd.DataFrame,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    latest_date = history[
        "date"
    ].iloc[-1]

    target_date = (
        latest_date
        - pd.Timedelta(days=days)
    )

    historical = history[
        history["date"] <= target_date
    ]

    if historical.empty:
        return np.nan

    return _safe_float(
        historical["tvl"].iloc[-1]
    )


def calculate_tvl_growth(
    history: pd.DataFrame,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    current = _safe_float(
        history["tvl"].iloc[-1]
    )

    previous = _value_at_days_ago(
        history,
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


# ============================================================
# ACELERAÇÃO DO TVL
# ============================================================

def calculate_tvl_acceleration(
    history: pd.DataFrame,
) -> float:

    """
    Compara crescimento dos últimos 30 dias
    com crescimento dos 30 dias anteriores.

    Positivo:
    entrada de capital está acelerando.

    Negativo:
    crescimento está desacelerando.
    """

    if history.empty:
        return np.nan

    current = _safe_float(
        history["tvl"].iloc[-1]
    )

    tvl_30d = _value_at_days_ago(
        history,
        30,
    )

    tvl_60d = _value_at_days_ago(
        history,
        60,
    )

    if any(
        pd.isna(value)
        for value in [
            current,
            tvl_30d,
            tvl_60d,
        ]
    ):
        return np.nan

    if tvl_30d <= 0 or tvl_60d <= 0:
        return np.nan

    recent_growth = (
        current / tvl_30d
        - 1
    )

    previous_growth = (
        tvl_30d / tvl_60d
        - 1
    )

    return (
        recent_growth
        - previous_growth
    )


# ============================================================
# ESTABILIDADE / QUALIDADE DO TVL
# ============================================================

def calculate_tvl_stability(
    history: pd.DataFrame,
    window_days: int = 90,
) -> float:

    """
    Mede estabilidade aproximada do TVL.

    1.0 = muito estável.
    Valores menores = maior volatilidade.
    """

    if history.empty:
        return np.nan

    cutoff = (
        history["date"].iloc[-1]
        - pd.Timedelta(
            days=window_days
        )
    )

    sample = history[
        history["date"] >= cutoff
    ].copy()

    if len(sample) < 10:
        return np.nan

    tvl = sample[
        "tvl"
    ].replace(
        0,
        np.nan,
    ).dropna()

    if len(tvl) < 10:
        return np.nan

    returns = tvl.pct_change().dropna()

    if returns.empty:
        return np.nan

    volatility = returns.std()

    if pd.isna(volatility):
        return np.nan

    stability = (
        1.0
        / (
            1.0
            + volatility * 100
        )
    )

    return float(
        np.clip(
            stability,
            0,
            1,
        )
    )


# ============================================================
# TVL DRAWDOWN
# ============================================================

def calculate_tvl_drawdown(
    history: pd.DataFrame,
    days: int = 365,
) -> float:

    if history.empty:
        return np.nan

    cutoff = (
        history["date"].iloc[-1]
        - pd.Timedelta(days=days)
    )

    sample = history[
        history["date"] >= cutoff
    ]

    if sample.empty:
        return np.nan

    tvl = sample[
        "tvl"
    ].dropna()

    if tvl.empty:
        return np.nan

    peak = tvl.max()

    current = tvl.iloc[-1]

    if peak <= 0:
        return np.nan

    return (
        current / peak
        - 1
    )


# ============================================================
# MÉTRICAS DE OPORTUNIDADE TVL
# ============================================================

def calculate_tvl_metrics(
    history: pd.DataFrame,
) -> Dict[str, float]:

    if history.empty:

        return {
            "tvl_current": np.nan,
            "tvl_growth_30d": np.nan,
            "tvl_growth_90d": np.nan,
            "tvl_growth_180d": np.nan,
            "tvl_growth_365d": np.nan,
            "tvl_acceleration": np.nan,
            "tvl_stability": np.nan,
            "tvl_drawdown_365d": np.nan,
        }

    return {
        "tvl_current":
            _safe_float(
                history[
                    "tvl"
                ].iloc[-1]
            ),

        "tvl_growth_30d":
            calculate_tvl_growth(
                history,
                30,
            ),

        "tvl_growth_90d":
            calculate_tvl_growth(
                history,
                90,
            ),

        "tvl_growth_180d":
            calculate_tvl_growth(
                history,
                180,
            ),

        "tvl_growth_365d":
            calculate_tvl_growth(
                history,
                365,
            ),

        "tvl_acceleration":
            calculate_tvl_acceleration(
                history
            ),

        "tvl_stability":
            calculate_tvl_stability(
                history
            ),

        "tvl_drawdown_365d":
            calculate_tvl_drawdown(
                history,
                365,
            ),
    }


# ============================================================
# VALUATION RELACIONADO AO TVL
# ============================================================

def calculate_tvl_valuation(
    market_cap: float,
    fdv: float,
    tvl: float,
) -> Dict[str, float]:

    return {
        "market_cap_tvl":
            _safe_divide(
                market_cap,
                tvl,
            ),

        "fdv_tvl":
            _safe_divide(
                fdv,
                tvl,
            ),

        "tvl_market_cap":
            _safe_divide(
                tvl,
                market_cap,
            ),
    }


# ============================================================
# DIVERGÊNCIA TVL × PREÇO
# ============================================================

def calculate_tvl_price_divergence(
    tvl_growth: float,
    price_return: float,
) -> float:

    """
    Sinal central de oportunidade.

    Exemplo:
    TVL +60%
    Preço +10%

    Divergência = +50 pontos percentuais.

    Fundamento avançando mais que preço.
    """

    if (
        pd.isna(tvl_growth)
        or pd.isna(price_return)
    ):
        return np.nan

    return (
        tvl_growth
        - price_return
    )


# ============================================================
# SCORE TVL
# ============================================================

def _growth_score(
    growth: float,
) -> float:

    if pd.isna(growth):
        return np.nan

    if growth <= -0.50:
        return 0.0

    if growth >= 1.00:
        return 100.0

    return float(
        np.clip(
            (
                (growth + 0.50)
                / 1.50
            )
            * 100,
            0,
            100,
        )
    )


def _acceleration_score(
    acceleration: float,
) -> float:

    if pd.isna(acceleration):
        return np.nan

    if acceleration <= -0.30:
        return 0.0

    if acceleration >= 0.50:
        return 100.0

    return float(
        np.clip(
            (
                (acceleration + 0.30)
                / 0.80
            )
            * 100,
            0,
            100,
        )
    )


def calculate_tvl_opportunity_score(
    metrics: Dict[str, float],
) -> float:

    components = {
        "growth_30d":
            (
                _growth_score(
                    metrics.get(
                        "tvl_growth_30d"
                    )
                ),
                0.20,
            ),

        "growth_90d":
            (
                _growth_score(
                    metrics.get(
                        "tvl_growth_90d"
                    )
                ),
                0.30,
            ),

        "growth_180d":
            (
                _growth_score(
                    metrics.get(
                        "tvl_growth_180d"
                    )
                ),
                0.20,
            ),

        "acceleration":
            (
                _acceleration_score(
                    metrics.get(
                        "tvl_acceleration"
                    )
                ),
                0.20,
            ),

        "stability":
            (
                (
                    metrics.get(
                        "tvl_stability"
                    )
                    * 100
                    if not pd.isna(
                        metrics.get(
                            "tvl_stability"
                        )
                    )
                    else np.nan
                ),
                0.10,
            ),
    }

    weighted_sum = 0.0
    available_weight = 0.0

    for score, weight in components.values():

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
# ENRIQUECIMENTO DO DATASET
# ============================================================

def enrich_with_defi_data(
    market_df: pd.DataFrame,
    sleep_seconds: float = 0.7,
) -> pd.DataFrame:

    if market_df.empty:
        return market_df.copy()

    protocols = fetch_protocols()

    if protocols.empty:

        result = market_df.copy()

        result[
            "defillama_match"
        ] = False

        return result

    records = []

    total = len(market_df)

    for position, (_, row) in enumerate(
        market_df.iterrows(),
        start=1,
    ):

        coin_id = row.get("coin_id")

        symbol = row.get("symbol")

        name = row.get("name")

        print(
            f"[defi_data] "
            f"{symbol} "
            f"({position}/{total})"
        )

        match = match_protocol(
            symbol=symbol,
            name=name,
            protocols=protocols,
        )

        record = {
            "coin_id": coin_id,
            "defillama_match": False,
            "defillama_slug": None,
            "defillama_category": None,
            "tvl_current": np.nan,
            "tvl_growth_30d": np.nan,
            "tvl_growth_90d": np.nan,
            "tvl_growth_180d": np.nan,
            "tvl_growth_365d": np.nan,
            "tvl_acceleration": np.nan,
            "tvl_stability": np.nan,
            "tvl_drawdown_365d": np.nan,
            "market_cap_tvl": np.nan,
            "fdv_tvl": np.nan,
            "tvl_market_cap": np.nan,
            "tvl_price_divergence_30d":
                np.nan,
            "tvl_price_divergence_90d":
                np.nan,
            "tvl_opportunity_score":
                np.nan,
        }

        if match is None:

            records.append(record)
            continue

        slug = match[
            "defillama_slug"
        ]

        history = fetch_protocol_history(
            slug
        )

        metrics = calculate_tvl_metrics(
            history
        )

        valuation = calculate_tvl_valuation(
            market_cap=_safe_float(
                row.get("market_cap")
            ),
            fdv=_safe_float(
                row.get("fdv")
            ),
            tvl=metrics[
                "tvl_current"
            ],
        )

        price_30d = _safe_float(
            row.get("return_30d")
        )

        price_90d = _safe_float(
            row.get("return_90d")
        )

        divergence_30d = (
            calculate_tvl_price_divergence(
                metrics[
                    "tvl_growth_30d"
                ],
                price_30d,
            )
        )

        divergence_90d = (
            calculate_tvl_price_divergence(
                metrics[
                    "tvl_growth_90d"
                ],
                price_90d,
            )
        )

        tvl_score = (
            calculate_tvl_opportunity_score(
                metrics
            )
        )

        record.update(
            {
                "defillama_match": True,

                "defillama_slug":
                    slug,

                "defillama_category":
                    match.get(
                        "defillama_category"
                    ),

                **metrics,

                **valuation,

                "tvl_price_divergence_30d":
                    divergence_30d,

                "tvl_price_divergence_90d":
                    divergence_90d,

                "tvl_opportunity_score":
                    tvl_score,
            }
        )

        records.append(record)

        time.sleep(
            sleep_seconds
        )

    defi_df = pd.DataFrame(
        records
    )

    result = market_df.merge(
        defi_df,
        on="coin_id",
        how="left",
    )

    return result


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

    result = enrich_with_defi_data(
        market
    )

    columns = [
        "symbol",
        "name",
        "tvl_current",
        "tvl_growth_30d",
        "tvl_growth_90d",
        "tvl_acceleration",
        "market_cap_tvl",
        "tvl_price_divergence_90d",
        "tvl_opportunity_score",
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
            "tvl_opportunity_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
