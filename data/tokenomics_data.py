"""
CRYPTO OPPORTUNITY ENGINE
Tokenomics / Dilution Data Layer

FOCO:
Encontrar oportunidades sem confundir preço baixo com token barato.

Analisa:
- circulating supply;
- total supply;
- max supply;
- market cap / FDV;
- supply ainda não circulante;
- diluição potencial;
- inflação aproximada;
- pressão estrutural de oferta;
- risco de grande quantidade de tokens ainda por liberar.

IMPORTANTE:
Diluição crítica pode bloquear uma oportunidade.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

import config


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

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

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    "[tokenomics_data] "
                    f"Rate limit. Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    "[tokenomics_data] "
                    f"HTTP {response.status_code}. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            return None

        except requests.RequestException:

            if attempt == MAX_RETRIES:
                return None

            time.sleep(
                RETRY_BACKOFF_SECONDS
                * attempt
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


def _clip_score(value: float) -> float:

    if pd.isna(value):
        return np.nan

    return float(
        np.clip(
            value,
            0,
            100,
        )
    )


# ============================================================
# COINGECKO TOKEN DETAIL
# ============================================================

def fetch_token_details(
    coin_id: str,
) -> Dict:

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }

    data = _request(
        f"/coins/{coin_id}",
        params=params,
    )

    if not isinstance(data, dict):
        return {}

    return data


# ============================================================
# SUPPLY METRICS
# ============================================================

def calculate_supply_metrics(
    circulating_supply: float,
    total_supply: float,
    max_supply: float,
) -> Dict[str, float]:

    circulating_ratio_total = (
        _safe_divide(
            circulating_supply,
            total_supply,
        )
    )

    circulating_ratio_max = (
        _safe_divide(
            circulating_supply,
            max_supply,
        )
    )

    if (
        not pd.isna(total_supply)
        and not pd.isna(circulating_supply)
    ):

        non_circulating_supply = max(
            total_supply
            - circulating_supply,
            0,
        )

    else:

        non_circulating_supply = np.nan

    non_circulating_ratio = (
        _safe_divide(
            non_circulating_supply,
            total_supply,
        )
    )

    return {
        "circulating_supply":
            circulating_supply,

        "total_supply":
            total_supply,

        "max_supply":
            max_supply,

        "circulating_supply_ratio":
            circulating_ratio_total,

        "circulating_max_supply_ratio":
            circulating_ratio_max,

        "non_circulating_supply":
            non_circulating_supply,

        "non_circulating_ratio":
            non_circulating_ratio,
    }


# ============================================================
# FDV
# ============================================================

def calculate_fdv_metrics(
    market_cap: float,
    fdv: float,
) -> Dict[str, float]:

    market_cap_fdv_ratio = (
        _safe_divide(
            market_cap,
            fdv,
        )
    )

    if (
        not pd.isna(fdv)
        and not pd.isna(market_cap)
        and market_cap > 0
    ):

        fdv_premium = (
            fdv / market_cap
            - 1
        )

    else:

        fdv_premium = np.nan

    return {
        "market_cap_fdv_ratio":
            market_cap_fdv_ratio,

        "fdv_premium":
            fdv_premium,
    }


# ============================================================
# SUPPLY HISTORY
# ============================================================

def fetch_market_cap_history(
    coin_id: str,
    days: int = 365,
) -> pd.DataFrame:

    params = {
        "vs_currency":
            config.BASE_CURRENCY.lower(),

        "days":
            days,

        "interval":
            "daily",
    }

    data = _request(
        f"/coins/{coin_id}/market_chart",
        params=params,
    )

    if not isinstance(data, dict):
        return pd.DataFrame()

    prices = data.get(
        "prices",
        [],
    )

    market_caps = data.get(
        "market_caps",
        [],
    )

    if (
        not prices
        or not market_caps
    ):
        return pd.DataFrame()

    price_df = pd.DataFrame(
        prices,
        columns=[
            "timestamp",
            "price",
        ],
    )

    cap_df = pd.DataFrame(
        market_caps,
        columns=[
            "timestamp",
            "market_cap",
        ],
    )

    history = price_df.merge(
        cap_df,
        on="timestamp",
        how="inner",
    )

    history["date"] = pd.to_datetime(
        history["timestamp"],
        unit="ms",
        utc=True,
    )

    history["estimated_supply"] = (
        history["market_cap"]
        / history["price"].replace(
            0,
            np.nan,
        )
    )

    return (
        history[
            [
                "date",
                "price",
                "market_cap",
                "estimated_supply",
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "estimated_supply",
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# SUPPLY EM DATA PASSADA
# ============================================================

def _supply_days_ago(
    history: pd.DataFrame,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    target = (
        history["date"].iloc[-1]
        - pd.Timedelta(days=days)
    )

    previous = history[
        history["date"] <= target
    ]

    if previous.empty:
        return np.nan

    return _safe_float(
        previous[
            "estimated_supply"
        ].iloc[-1]
    )


# ============================================================
# INFLAÇÃO APROXIMADA
# ============================================================

def calculate_supply_inflation(
    history: pd.DataFrame,
    days: int,
) -> float:

    if history.empty:
        return np.nan

    current_supply = _safe_float(
        history[
            "estimated_supply"
        ].iloc[-1]
    )

    old_supply = _supply_days_ago(
        history,
        days,
    )

    if (
        pd.isna(current_supply)
        or pd.isna(old_supply)
        or old_supply <= 0
    ):
        return np.nan

    return (
        current_supply
        / old_supply
        - 1
    )


# ============================================================
# PRESSÃO POTENCIAL DE OFERTA
# ============================================================

def calculate_supply_overhang(
    circulating_supply: float,
    total_supply: float,
) -> float:

    if (
        pd.isna(circulating_supply)
        or pd.isna(total_supply)
        or total_supply <= 0
    ):
        return np.nan

    future_supply = max(
        total_supply
        - circulating_supply,
        0,
    )

    return (
        future_supply
        / total_supply
    )


# ============================================================
# SCORES
# ============================================================

def score_circulating_ratio(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio >= 0.90:
        return 100.0

    if ratio >= 0.75:
        return 90.0

    if ratio >= 0.60:
        return 75.0

    if ratio >= 0.50:
        return 60.0

    if ratio >= 0.40:
        return 45.0

    if ratio >= 0.25:
        return 25.0

    return 5.0


def score_market_cap_fdv(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio >= 0.90:
        return 100.0

    if ratio >= 0.75:
        return 90.0

    if ratio >= 0.60:
        return 75.0

    if ratio >= 0.50:
        return 60.0

    if ratio >= 0.40:
        return 45.0

    if ratio >= 0.25:
        return 25.0

    return 5.0


def score_inflation(
    inflation: float,
) -> float:

    if pd.isna(inflation):
        return np.nan

    if inflation <= 0:
        return 100.0

    if inflation <= 0.02:
        return 95.0

    if inflation <= 0.05:
        return 85.0

    if inflation <= 0.10:
        return 70.0

    if inflation <= 0.15:
        return 55.0

    if inflation <= 0.20:
        return 40.0

    if inflation <= 0.30:
        return 20.0

    return 5.0


def score_supply_overhang(
    overhang: float,
) -> float:

    if pd.isna(overhang):
        return np.nan

    return _clip_score(
        (1 - overhang)
        * 100
    )


# ============================================================
# DILUTION SCORE
# ============================================================

def calculate_dilution_score(
    circulating_ratio: float,
    market_cap_fdv_ratio: float,
    inflation_90d: float,
    inflation_365d: float,
    supply_overhang: float,
) -> float:

    annualized_90d = np.nan

    if not pd.isna(
        inflation_90d
    ):

        try:

            annualized_90d = (
                (1 + inflation_90d)
                ** (365 / 90)
                - 1
            )

        except Exception:

            annualized_90d = np.nan

    components = [
        (
            score_circulating_ratio(
                circulating_ratio
            ),
            0.25,
        ),
        (
            score_market_cap_fdv(
                market_cap_fdv_ratio
            ),
            0.20,
        ),
        (
            score_inflation(
                inflation_365d
            ),
            0.25,
        ),
        (
            score_inflation(
                annualized_90d
            ),
            0.15,
        ),
        (
            score_supply_overhang(
                supply_overhang
            ),
            0.15,
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
# DILUTION RISK
# ============================================================

def classify_dilution_risk(
    circulating_ratio: float,
    market_cap_fdv_ratio: float,
    inflation_365d: float,
    dilution_score: float,
) -> str:

    critical_conditions = []

    if (
        not pd.isna(circulating_ratio)
        and circulating_ratio
        < config.MIN_CIRCULATING_SUPPLY_RATIO_CRITICAL
    ):
        critical_conditions.append(
            "LOW_CIRCULATING_SUPPLY"
        )

    if (
        not pd.isna(market_cap_fdv_ratio)
        and market_cap_fdv_ratio
        < config.MIN_MARKET_CAP_FDV_RATIO_CRITICAL
    ):
        critical_conditions.append(
            "LOW_MC_FDV"
        )

    if (
        not pd.isna(inflation_365d)
        and inflation_365d
        > config.MAX_ANNUAL_INFLATION_CRITICAL
    ):
        critical_conditions.append(
            "HIGH_INFLATION"
        )

    if (
        not pd.isna(dilution_score)
        and dilution_score < 25
    ):
        critical_conditions.append(
            "VERY_LOW_DILUTION_SCORE"
        )

    if critical_conditions:
        return "CRITICAL"

    warning_conditions = []

    if (
        not pd.isna(circulating_ratio)
        and circulating_ratio
        < config.MIN_CIRCULATING_SUPPLY_RATIO_WARNING
    ):
        warning_conditions.append(
            "CIRCULATING_WARNING"
        )

    if (
        not pd.isna(market_cap_fdv_ratio)
        and market_cap_fdv_ratio
        < config.MIN_MARKET_CAP_FDV_RATIO_WARNING
    ):
        warning_conditions.append(
            "MC_FDV_WARNING"
        )

    if (
        not pd.isna(inflation_365d)
        and inflation_365d
        > config.MAX_ANNUAL_INFLATION_WARNING
    ):
        warning_conditions.append(
            "INFLATION_WARNING"
        )

    if warning_conditions:
        return "HIGH"

    if (
        not pd.isna(dilution_score)
        and dilution_score < 60
    ):
        return "MODERATE"

    return "LOW"


# ============================================================
# HARD BLOCK
# ============================================================

def dilution_hard_block(
    dilution_risk: str,
) -> bool:

    return (
        dilution_risk
        == "CRITICAL"
    )


# ============================================================
# OPORTUNIDADE × DILUIÇÃO
# ============================================================

def calculate_dilution_opportunity_modifier(
    dilution_score: float,
) -> float:

    """
    Retorna multiplicador de 0 a 1.

    Não cria oportunidade.
    Apenas reduz oportunidade quando
    a estrutura de oferta é ruim.
    """

    if pd.isna(dilution_score):
        return np.nan

    if dilution_score >= 80:
        return 1.00

    if dilution_score >= 70:
        return 0.95

    if dilution_score >= 60:
        return 0.90

    if dilution_score >= 50:
        return 0.80

    if dilution_score >= 40:
        return 0.65

    if dilution_score >= 25:
        return 0.40

    return 0.0


# ============================================================
# ENRIQUECIMENTO
# ============================================================

def enrich_with_tokenomics_data(
    market_df: pd.DataFrame,
    fetch_history: bool = True,
    sleep_seconds: float = 1.2,
) -> pd.DataFrame:

    if market_df.empty:
        return market_df.copy()

    records = []

    total = len(
        market_df
    )

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

        print(
            "[tokenomics_data] "
            f"{symbol} "
            f"({position}/{total})"
        )

        circulating_supply = (
            _safe_float(
                row.get(
                    "circulating_supply"
                )
            )
        )

        total_supply = (
            _safe_float(
                row.get(
                    "total_supply"
                )
            )
        )

        max_supply = (
            _safe_float(
                row.get(
                    "max_supply"
                )
            )
        )

        market_cap = (
            _safe_float(
                row.get(
                    "market_cap"
                )
            )
        )

        fdv = (
            _safe_float(
                row.get(
                    "fdv"
                )
            )
        )

        # ----------------------------------------------------
        # Fallback para detalhes CoinGecko
        # ----------------------------------------------------

        if any(
            pd.isna(value)
            for value in [
                circulating_supply,
                total_supply,
            ]
        ):

            details = fetch_token_details(
                coin_id
            )

            market_data = (
                details.get(
                    "market_data",
                    {}
                )
                if details
                else {}
            )

            if pd.isna(
                circulating_supply
            ):

                circulating_supply = (
                    _safe_float(
                        market_data.get(
                            "circulating_supply"
                        )
                    )
                )

            if pd.isna(
                total_supply
            ):

                total_supply = (
                    _safe_float(
                        market_data.get(
                            "total_supply"
                        )
                    )
                )

            if pd.isna(
                max_supply
            ):

                max_supply = (
                    _safe_float(
                        market_data.get(
                            "max_supply"
                        )
                    )
                )

        supply_metrics = (
            calculate_supply_metrics(
                circulating_supply,
                total_supply,
                max_supply,
            )
        )

        fdv_metrics = (
            calculate_fdv_metrics(
                market_cap,
                fdv,
            )
        )

        supply_overhang = (
            calculate_supply_overhang(
                circulating_supply,
                total_supply,
            )
        )

        inflation_30d = np.nan
        inflation_90d = np.nan
        inflation_180d = np.nan
        inflation_365d = np.nan

        if fetch_history:

            history = (
                fetch_market_cap_history(
                    coin_id=coin_id,
                    days=365,
                )
            )

            inflation_30d = (
                calculate_supply_inflation(
                    history,
                    30,
                )
            )

            inflation_90d = (
                calculate_supply_inflation(
                    history,
                    90,
                )
            )

            inflation_180d = (
                calculate_supply_inflation(
                    history,
                    180,
                )
            )

            inflation_365d = (
                calculate_supply_inflation(
                    history,
                    365,
                )
            )

        dilution_score = (
            calculate_dilution_score(
                circulating_ratio=
                    supply_metrics[
                        "circulating_supply_ratio"
                    ],

                market_cap_fdv_ratio=
                    fdv_metrics[
                        "market_cap_fdv_ratio"
                    ],

                inflation_90d=
                    inflation_90d,

                inflation_365d=
                    inflation_365d,

                supply_overhang=
                    supply_overhang,
            )
        )

        dilution_risk = (
            classify_dilution_risk(
                circulating_ratio=
                    supply_metrics[
                        "circulating_supply_ratio"
                    ],

                market_cap_fdv_ratio=
                    fdv_metrics[
                        "market_cap_fdv_ratio"
                    ],

                inflation_365d=
                    inflation_365d,

                dilution_score=
                    dilution_score,
            )
        )

        hard_block = (
            dilution_hard_block(
                dilution_risk
            )
        )

        modifier = (
            calculate_dilution_opportunity_modifier(
                dilution_score
            )
        )

        records.append(
            {
                "coin_id":
                    coin_id,

                **supply_metrics,

                **fdv_metrics,

                "supply_overhang":
                    supply_overhang,

                "estimated_supply_inflation_30d":
                    inflation_30d,

                "estimated_supply_inflation_90d":
                    inflation_90d,

                "estimated_supply_inflation_180d":
                    inflation_180d,

                "estimated_supply_inflation_365d":
                    inflation_365d,

                "dilution_score":
                    dilution_score,

                "dilution_risk":
                    dilution_risk,

                "dilution_hard_block":
                    hard_block,

                "dilution_opportunity_modifier":
                    modifier,
            }
        )

        if fetch_history:

            time.sleep(
                sleep_seconds
            )

    tokenomics_df = (
        pd.DataFrame(
            records
        )
    )

    duplicate_columns = [
        column
        for column in [
            "circulating_supply",
            "total_supply",
            "max_supply",
            "market_cap_fdv_ratio",
        ]
        if column in market_df.columns
    ]

    base_df = (
        market_df.drop(
            columns=duplicate_columns,
            errors="ignore",
        )
    )

    result = base_df.merge(
        tokenomics_df,
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

    market = (
        build_market_dataset(
            include_history=False
        )
    )

    market = market.head(
        50
    )

    result = (
        enrich_with_tokenomics_data(
            market,
            fetch_history=True,
        )
    )

    columns = [
        "symbol",
        "name",
        "market_cap",
        "fdv",
        "circulating_supply_ratio",
        "market_cap_fdv_ratio",
        "supply_overhang",
        "estimated_supply_inflation_90d",
        "estimated_supply_inflation_365d",
        "dilution_score",
        "dilution_risk",
        "dilution_hard_block",
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
            "dilution_score",
            ascending=False,
            na_position="last",
        )
        .head(50)
        .to_string(
            index=False
        )
    )
