"""
CRYPTO OPPORTUNITY ENGINE
Dilution Engine

OBJETIVO:
Transformar os dados de tokenomics em uma avaliação institucional
de risco de diluição.

Princípio:
Um projeto pode apresentar bons fundamentos, receita e crescimento,
mas deixar de ser uma boa oportunidade quando existe grande pressão
futura de oferta.

Analisa:
- circulating supply / total supply;
- market cap / FDV;
- inflação;
- unlocks 30/90/180/365 dias;
- concentração de insiders;
- unlock versus liquidez;
- supply overhang;
- qualidade/completude dos dados.

REGRA:
Diluição crítica pode BLOQUEAR o ativo.

IMPORTANTE:
Dados ausentes nunca são inventados.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

import config


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


def _first_available(
    row: pd.Series,
    fields: Iterable[str],
) -> float:

    for field in fields:

        if field not in row.index:
            continue

        value = _safe_float(
            row.get(field)
        )

        if not pd.isna(value):
            return value

    return np.nan


def _normalize_ratio(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    if value > 1:
        value = (
            value / 100
        )

    return value


def _weighted_average(
    components: Dict[str, tuple],
) -> float:

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

    return (
        weighted_sum
        / available_weight
    )


# ============================================================
# CIRCULATING SUPPLY
# ============================================================

def extract_circulating_supply_ratio(
    row: pd.Series,
) -> float:

    ratio = _first_available(
        row,
        [
            "circulating_supply_ratio",
            "circulating_max_supply_ratio",
        ],
    )

    return _normalize_ratio(
        ratio
    )


def score_circulating_supply_ratio(
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


# ============================================================
# MARKET CAP / FDV
# ============================================================

def extract_market_cap_fdv_ratio(
    row: pd.Series,
) -> float:

    ratio = _first_available(
        row,
        [
            "market_cap_fdv_ratio",
        ],
    )

    if not pd.isna(ratio):
        return _normalize_ratio(
            ratio
        )

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    fdv = _safe_float(
        row.get(
            "fdv"
        )
    )

    if (
        pd.isna(market_cap)
        or pd.isna(fdv)
        or fdv <= 0
    ):
        return np.nan

    return (
        market_cap
        / fdv
    )


def score_market_cap_fdv_ratio(
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


# ============================================================
# INFLAÇÃO
# ============================================================

def extract_annual_inflation(
    row: pd.Series,
) -> float:

    inflation = _first_available(
        row,
        [
            "annual_inflation",
            "estimated_supply_inflation_365d",
            "supply_inflation_365d",
        ],
    )

    return _normalize_ratio(
        inflation
    )


def score_annual_inflation(
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


# ============================================================
# UNLOCKS
# ============================================================

def extract_unlock_ratio(
    row: pd.Series,
    days: int,
) -> float:

    fields = {
        30: [
            "unlock_30d",
            "unlock_ratio_30d",
            "tokens_unlock_30d_ratio",
        ],

        90: [
            "unlock_90d",
            "unlock_ratio_90d",
            "tokens_unlock_90d_ratio",
        ],

        180: [
            "unlock_180d",
            "unlock_ratio_180d",
            "tokens_unlock_180d_ratio",
        ],

        365: [
            "unlock_365d",
            "unlock_ratio_365d",
            "tokens_unlock_365d_ratio",
        ],
    }

    value = _first_available(
        row,
        fields.get(
            days,
            [],
        ),
    )

    return _normalize_ratio(
        value
    )


def score_unlock_ratio(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0:
        return 100.0

    if ratio <= 0.025:
        return 95.0

    if ratio <= 0.05:
        return 85.0

    if ratio <= 0.10:
        return 65.0

    if ratio <= 0.15:
        return 50.0

    if ratio <= 0.20:
        return 35.0

    if ratio <= 0.30:
        return 15.0

    return 0.0


# ============================================================
# INSIDER CONCENTRATION
# ============================================================

def extract_insider_concentration(
    row: pd.Series,
) -> float:

    value = _first_available(
        row,
        [
            "insider_concentration",
            "team_vc_allocation",
            "team_investor_allocation",
            "insider_supply_ratio",
        ],
    )

    return _normalize_ratio(
        value
    )


def score_insider_concentration(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0.10:
        return 100.0

    if ratio <= 0.20:
        return 85.0

    if ratio <= 0.30:
        return 70.0

    if ratio <= 0.40:
        return 50.0

    if ratio <= 0.50:
        return 30.0

    if ratio <= 0.60:
        return 15.0

    return 0.0


# ============================================================
# UNLOCK VS LIQUIDITY
# ============================================================

def extract_unlock_vs_liquidity(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "unlock_vs_liquidity",
            "unlock_liquidity_ratio",
        ],
    )

    if not pd.isna(explicit):
        return _normalize_ratio(
            explicit
        )

    unlock_value = _first_available(
        row,
        [
            "unlock_90d_usd",
            "tokens_unlock_90d_usd",
        ],
    )

    daily_volume = _safe_float(
        row.get(
            "daily_volume"
        )
    )

    if (
        pd.isna(unlock_value)
        or pd.isna(daily_volume)
        or daily_volume <= 0
    ):
        return np.nan

    ninety_day_liquidity = (
        daily_volume * 90
    )

    return (
        unlock_value
        / ninety_day_liquidity
    )


def score_unlock_vs_liquidity(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0.01:
        return 100.0

    if ratio <= 0.025:
        return 90.0

    if ratio <= 0.05:
        return 80.0

    if ratio <= 0.10:
        return 65.0

    if ratio <= 0.20:
        return 45.0

    if ratio <= 0.40:
        return 20.0

    return 0.0


# ============================================================
# SUPPLY OVERHANG
# ============================================================

def extract_supply_overhang(
    row: pd.Series,
) -> float:

    value = _first_available(
        row,
        [
            "supply_overhang",
            "non_circulating_ratio",
        ],
    )

    return _normalize_ratio(
        value
    )


# ============================================================
# DILUTION SCORE
# ============================================================

def calculate_dilution_score(
    row: pd.Series,
) -> float:

    circulating_ratio = (
        extract_circulating_supply_ratio(
            row
        )
    )

    market_cap_fdv_ratio = (
        extract_market_cap_fdv_ratio(
            row
        )
    )

    annual_inflation = (
        extract_annual_inflation(
            row
        )
    )

    unlock_90d = (
        extract_unlock_ratio(
            row,
            90,
        )
    )

    unlock_365d = (
        extract_unlock_ratio(
            row,
            365,
        )
    )

    insider_concentration = (
        extract_insider_concentration(
            row
        )
    )

    unlock_vs_liquidity = (
        extract_unlock_vs_liquidity(
            row
        )
    )

    components = {
        "circulating_supply_ratio":
            (
                score_circulating_supply_ratio(
                    circulating_ratio
                ),
                config.DILUTION_WEIGHTS[
                    "circulating_supply_ratio"
                ],
            ),

        "market_cap_fdv_ratio":
            (
                score_market_cap_fdv_ratio(
                    market_cap_fdv_ratio
                ),
                config.DILUTION_WEIGHTS[
                    "market_cap_fdv_ratio"
                ],
            ),

        "annual_inflation":
            (
                score_annual_inflation(
                    annual_inflation
                ),
                config.DILUTION_WEIGHTS[
                    "annual_inflation"
                ],
            ),

        "unlock_90d":
            (
                score_unlock_ratio(
                    unlock_90d
                ),
                config.DILUTION_WEIGHTS[
                    "unlock_90d"
                ],
            ),

        "unlock_365d":
            (
                score_unlock_ratio(
                    unlock_365d
                ),
                config.DILUTION_WEIGHTS[
                    "unlock_365d"
                ],
            ),

        "insider_concentration":
            (
                score_insider_concentration(
                    insider_concentration
                ),
                config.DILUTION_WEIGHTS[
                    "insider_concentration"
                ],
            ),

        "unlock_vs_liquidity":
            (
                score_unlock_vs_liquidity(
                    unlock_vs_liquidity
                ),
                config.DILUTION_WEIGHTS[
                    "unlock_vs_liquidity"
                ],
            ),
    }

    score = _weighted_average(
        components
    )

    if pd.isna(score):
        return np.nan

    return round(
        _clip_score(score),
        2,
    )


# ============================================================
# CRITICAL FLAGS
# ============================================================

def detect_critical_dilution(
    row: pd.Series,
) -> Dict[str, bool]:

    circulating_ratio = (
        extract_circulating_supply_ratio(
            row
        )
    )

    market_cap_fdv_ratio = (
        extract_market_cap_fdv_ratio(
            row
        )
    )

    annual_inflation = (
        extract_annual_inflation(
            row
        )
    )

    unlock_90d = (
        extract_unlock_ratio(
            row,
            90,
        )
    )

    unlock_365d = (
        extract_unlock_ratio(
            row,
            365,
        )
    )

    return {
        "critical_low_circulating_supply":
            (
                not pd.isna(
                    circulating_ratio
                )
                and circulating_ratio
                < config.MIN_CIRCULATING_SUPPLY_RATIO_CRITICAL
            ),

        "critical_low_market_cap_fdv":
            (
                not pd.isna(
                    market_cap_fdv_ratio
                )
                and market_cap_fdv_ratio
                < config.MIN_MARKET_CAP_FDV_RATIO_CRITICAL
            ),

        "critical_annual_inflation":
            (
                not pd.isna(
                    annual_inflation
                )
                and annual_inflation
                > config.MAX_ANNUAL_INFLATION_CRITICAL
            ),

        "critical_unlock_90d":
            (
                not pd.isna(
                    unlock_90d
                )
                and unlock_90d
                > config.MAX_UNLOCK_90D_CRITICAL
            ),

        "critical_unlock_365d":
            (
                not pd.isna(
                    unlock_365d
                )
                and unlock_365d
                > config.MAX_UNLOCK_365D_CRITICAL
            ),
    }


# ============================================================
# WARNING FLAGS
# ============================================================

def detect_dilution_warnings(
    row: pd.Series,
) -> Dict[str, bool]:

    circulating_ratio = (
        extract_circulating_supply_ratio(
            row
        )
    )

    market_cap_fdv_ratio = (
        extract_market_cap_fdv_ratio(
            row
        )
    )

    annual_inflation = (
        extract_annual_inflation(
            row
        )
    )

    unlock_90d = (
        extract_unlock_ratio(
            row,
            90,
        )
    )

    unlock_365d = (
        extract_unlock_ratio(
            row,
            365,
        )
    )

    return {
        "warning_low_circulating_supply":
            (
                not pd.isna(
                    circulating_ratio
                )
                and circulating_ratio
                < config.MIN_CIRCULATING_SUPPLY_RATIO_WARNING
            ),

        "warning_low_market_cap_fdv":
            (
                not pd.isna(
                    market_cap_fdv_ratio
                )
                and market_cap_fdv_ratio
                < config.MIN_MARKET_CAP_FDV_RATIO_WARNING
            ),

        "warning_annual_inflation":
            (
                not pd.isna(
                    annual_inflation
                )
                and annual_inflation
                > config.MAX_ANNUAL_INFLATION_WARNING
            ),

        "warning_unlock_90d":
            (
                not pd.isna(
                    unlock_90d
                )
                and unlock_90d
                > config.MAX_UNLOCK_90D_WARNING
            ),

        "warning_unlock_365d":
            (
                not pd.isna(
                    unlock_365d
                )
                and unlock_365d
                > config.MAX_UNLOCK_365D_WARNING
            ),
    }


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_dilution_risk(
    row: pd.Series,
    dilution_score: float,
) -> str:

    critical_flags = (
        detect_critical_dilution(
            row
        )
    )

    if any(
        critical_flags.values()
    ):
        return "CRITICAL"

    warnings = (
        detect_dilution_warnings(
            row
        )
    )

    warning_count = sum(
        1
        for value in warnings.values()
        if value
    )

    if (
        not pd.isna(dilution_score)
        and dilution_score < 25
    ):
        return "CRITICAL"

    if (
        warning_count >= 2
        or (
            not pd.isna(dilution_score)
            and dilution_score < 40
        )
    ):
        return "HIGH"

    if (
        warning_count == 1
        or (
            not pd.isna(dilution_score)
            and dilution_score < 60
        )
    ):
        return "MODERATE"

    if pd.isna(
        dilution_score
    ):
        return "DATA_INSUFFICIENT"

    return "LOW"


# ============================================================
# HARD BLOCK
# ============================================================

def dilution_hard_block(
    risk_level: str,
) -> bool:

    if not config.ENGINE_RULES.get(
        "critical_dilution_can_block_asset",
        True,
    ):
        return False

    return (
        risk_level == "CRITICAL"
    )


# ============================================================
# QUALITY GATE
# ============================================================

def dilution_quality_gate(
    dilution_score: float,
    hard_block: bool,
) -> bool:

    if hard_block:
        return False

    if pd.isna(
        dilution_score
    ):
        return False

    return (
        dilution_score
        >= config.MIN_DILUTION_SCORE
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_dilution_data_completeness(
    row: pd.Series,
) -> float:

    values = [
        extract_circulating_supply_ratio(
            row
        ),

        extract_market_cap_fdv_ratio(
            row
        ),

        extract_annual_inflation(
            row
        ),

        extract_unlock_ratio(
            row,
            90,
        ),

        extract_unlock_ratio(
            row,
            365,
        ),

        extract_insider_concentration(
            row
        ),

        extract_unlock_vs_liquidity(
            row
        ),
    ]

    available = sum(
        1
        for value in values
        if not pd.isna(value)
    )

    return round(
        available
        / len(values),
        4,
    )


# ============================================================
# STATUS
# ============================================================

def classify_dilution_status(
    score: float,
    risk: str,
) -> str:

    if risk == "CRITICAL":
        return "BLOCKED"

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 85:
        return "VERY_STRONG"

    if score >= 70:
        return "STRONG"

    if score >= config.MIN_DILUTION_SCORE:
        return "ACCEPTABLE"

    return "WEAK"


# ============================================================
# AVALIAÇÃO INDIVIDUAL
# ============================================================

def evaluate_dilution(
    row: pd.Series,
) -> Dict:

    circulating_ratio = (
        extract_circulating_supply_ratio(
            row
        )
    )

    market_cap_fdv_ratio = (
        extract_market_cap_fdv_ratio(
            row
        )
    )

    annual_inflation = (
        extract_annual_inflation(
            row
        )
    )

    unlock_30d = (
        extract_unlock_ratio(
            row,
            30,
        )
    )

    unlock_90d = (
        extract_unlock_ratio(
            row,
            90,
        )
    )

    unlock_180d = (
        extract_unlock_ratio(
            row,
            180,
        )
    )

    unlock_365d = (
        extract_unlock_ratio(
            row,
            365,
        )
    )

    insider_concentration = (
        extract_insider_concentration(
            row
        )
    )

    unlock_vs_liquidity = (
        extract_unlock_vs_liquidity(
            row
        )
    )

    supply_overhang = (
        extract_supply_overhang(
            row
        )
    )

    dilution_score = (
        calculate_dilution_score(
            row
        )
    )

    risk = (
        classify_dilution_risk(
            row,
            dilution_score,
        )
    )

    hard_block = (
        dilution_hard_block(
            risk
        )
    )

    quality_gate = (
        dilution_quality_gate(
            dilution_score,
            hard_block,
        )
    )

    critical_flags = (
        detect_critical_dilution(
            row
        )
    )

    warning_flags = (
        detect_dilution_warnings(
            row
        )
    )

    critical_reasons = [
        key
        for key, value
        in critical_flags.items()
        if value
    ]

    warning_reasons = [
        key
        for key, value
        in warning_flags.items()
        if value
    ]

    return {
        "circulating_supply_ratio":
            circulating_ratio,

        "market_cap_fdv_ratio":
            market_cap_fdv_ratio,

        "annual_inflation":
            annual_inflation,

        "unlock_30d":
            unlock_30d,

        "unlock_90d":
            unlock_90d,

        "unlock_180d":
            unlock_180d,

        "unlock_365d":
            unlock_365d,

        "insider_concentration":
            insider_concentration,

        "unlock_vs_liquidity":
            unlock_vs_liquidity,

        "supply_overhang":
            supply_overhang,

        "dilution_score":
            dilution_score,

        "dilution_risk":
            risk,

        "dilution_status":
            classify_dilution_status(
                dilution_score,
                risk,
            ),

        "dilution_quality_gate":
            quality_gate,

        "dilution_hard_block":
            hard_block,

        "dilution_critical_reasons":
            "|".join(
                critical_reasons
            ),

        "dilution_warning_reasons":
            "|".join(
                warning_reasons
            ),

        "dilution_data_completeness":
            calculate_dilution_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_dilution_engine(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    evaluations = []

    total = len(
        result
    )

    for position, (_, row) in enumerate(
        result.iterrows(),
        start=1,
    ):

        symbol = row.get(
            "symbol",
            "UNKNOWN",
        )

        print(
            "[dilution_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_dilution(
                row
            )
        )

    evaluation_df = pd.DataFrame(
        evaluations,
        index=result.index,
    )

    for column in evaluation_df.columns:

        result[
            column
        ] = evaluation_df[
            column
        ]

    return result


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    from data.market_data import (
        build_market_dataset,
    )

    from data.tokenomics_data import (
        enrich_with_tokenomics_data,
    )

    market = build_market_dataset(
        include_history=False
    )

    market = market.head(
        50
    )

    market = enrich_with_tokenomics_data(
        market,
        fetch_history=True,
    )

    result = run_dilution_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "circulating_supply_ratio",
        "market_cap_fdv_ratio",
        "annual_inflation",
        "unlock_90d",
        "unlock_365d",
        "insider_concentration",
        "unlock_vs_liquidity",
        "supply_overhang",
        "dilution_score",
        "dilution_risk",
        "dilution_status",
        "dilution_quality_gate",
        "dilution_hard_block",
        "dilution_critical_reasons",
        "dilution_warning_reasons",
        "dilution_data_completeness",
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
