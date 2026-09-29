"""
CRYPTO OPPORTUNITY ENGINE
Timing Engine

OBJETIVO:
Separar QUALIDADE DO PROJETO do MOMENTO DE ENTRADA.

Princípio:
Um excelente projeto pode estar em um momento ruim de compra.
Um ativo tecnicamente forte não pode compensar fundamentos ruins.

O Timing Engine NÃO altera o Opportunity Score.
Ele apenas determina se o momento de entrada é favorável.

Analisa:
- momentum 3 meses;
- momentum 6 meses;
- força relativa;
- confirmação por volume;
- estrutura de tendência;
- posição dentro do drawdown;
- funding;
- open interest.

REGRA:
Timing nunca pode resgatar um projeto que falhou nos Quality Gates.
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
# MOMENTUM
# ============================================================

def extract_momentum_3m(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "return_90d",
            "price_change_90d",
            "momentum_3m",
        ],
    )


def extract_momentum_6m(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "return_180d",
            "price_change_180d",
            "momentum_6m",
        ],
    )


def score_momentum(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    if value <= -0.50:
        return 5.0

    if value <= -0.30:
        return 15.0

    if value <= -0.15:
        return 30.0

    if value < 0:
        return 45.0

    if value <= 0.10:
        return 60.0

    if value <= 0.25:
        return 75.0

    if value <= 0.50:
        return 90.0

    if value <= 1.00:
        return 100.0

    # Momentum extremamente estendido recebe pequena penalização.
    if value <= 1.50:
        return 85.0

    return 70.0


# ============================================================
# RELATIVE STRENGTH
# ============================================================

def extract_relative_strength(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "relative_strength",
            "relative_strength_90d",
            "relative_strength_score",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    asset_return = extract_momentum_3m(
        row
    )

    benchmark_return = _first_available(
        row,
        [
            "benchmark_return_90d",
            "crypto_market_return_90d",
            "total_market_return_90d",
        ],
    )

    if (
        pd.isna(asset_return)
        or pd.isna(benchmark_return)
    ):
        return np.nan

    return (
        asset_return
        - benchmark_return
    )


def score_relative_strength(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    # Se já vier como score 0-100.
    if value > 2:
        return _clip_score(
            value
        )

    lower = -0.40
    upper = 0.50

    if value <= lower:
        return 0.0

    if value >= upper:
        return 100.0

    return _clip_score(
        (
            value - lower
        )
        / (
            upper - lower
        )
        * 100
    )


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def extract_volume_confirmation(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "volume_confirmation",
            "volume_confirmation_score",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    volume_growth = _first_available(
        row,
        [
            "volume_growth_30d",
            "daily_volume_growth",
            "volume_growth_90d",
        ],
    )

    if not pd.isna(volume_growth):
        return volume_growth

    volume = _safe_float(
        row.get(
            "daily_volume"
        )
    )

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    if (
        pd.isna(volume)
        or pd.isna(market_cap)
        or market_cap <= 0
    ):
        return np.nan

    return (
        volume
        / market_cap
    )


def score_volume_confirmation(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    # Score já normalizado.
    if value > 2:
        return _clip_score(
            value
        )

    if value < 0:
        lower = -0.50
        upper = 0.50

        return _clip_score(
            (
                value - lower
            )
            / (
                upper - lower
            )
            * 100
        )

    if value <= 0.01:
        return 20.0

    if value <= 0.02:
        return 35.0

    if value <= 0.05:
        return 55.0

    if value <= 0.10:
        return 75.0

    if value <= 0.20:
        return 90.0

    return 100.0


# ============================================================
# TREND STRUCTURE
# ============================================================

def calculate_trend_structure(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "trend_structure_score",
            "trend_structure",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    momentum_3m = extract_momentum_3m(
        row
    )

    momentum_6m = extract_momentum_6m(
        row
    )

    return_30d = _first_available(
        row,
        [
            "return_30d",
            "price_change_30d",
        ],
    )

    signals = []

    if not pd.isna(return_30d):
        signals.append(
            1 if return_30d > 0 else 0
        )

    if not pd.isna(momentum_3m):
        signals.append(
            1 if momentum_3m > 0 else 0
        )

    if not pd.isna(momentum_6m):
        signals.append(
            1 if momentum_6m > 0 else 0
        )

    if not signals:
        return np.nan

    positive_ratio = (
        sum(signals)
        / len(signals)
    )

    return round(
        positive_ratio
        * 100,
        2,
    )


# ============================================================
# DRAWDOWN POSITION
# ============================================================

def extract_drawdown(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "drawdown",
            "drawdown_from_ath",
            "current_drawdown",
        ],
    )


def score_drawdown_position(
    drawdown: float,
) -> float:

    if pd.isna(drawdown):
        return np.nan

    if drawdown > 0:
        drawdown = -drawdown

    # Muito próximo do ATH:
    # menos assimetria de entrada.
    if drawdown >= -0.05:
        return 45.0

    if drawdown >= -0.15:
        return 65.0

    if drawdown >= -0.30:
        return 85.0

    if drawdown >= -0.45:
        return 100.0

    if drawdown >= -0.60:
        return 85.0

    if drawdown >= -0.75:
        return 60.0

    return 35.0


# ============================================================
# FUNDING
# ============================================================

def extract_funding(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "funding_7d",
            "funding_rate_7d",
            "funding_rate",
            "funding",
        ],
    )


def score_funding(
    funding: float,
) -> float:

    if pd.isna(funding):
        return np.nan

    # Aceita percentual ou decimal.
    if abs(funding) > 1:
        funding /= 100

    if funding <= -0.02:
        return 80.0

    if funding <= -0.005:
        return 90.0

    if funding <= 0.005:
        return 100.0

    if funding <= 0.015:
        return 80.0

    if funding <= 0.03:
        return 60.0

    if funding <= 0.05:
        return 35.0

    return 10.0


# ============================================================
# OPEN INTEREST
# ============================================================

def extract_open_interest_change(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "open_interest_change_30d",
            "oi_change_30d",
            "delta_oi_30d",
            "open_interest",
        ],
    )


def score_open_interest(
    oi_change: float,
    price_momentum: float,
) -> float:

    if pd.isna(oi_change):
        return np.nan

    if abs(oi_change) > 5:
        oi_change /= 100

    if pd.isna(price_momentum):

        if oi_change <= -0.30:
            return 45.0

        if oi_change <= 0:
            return 60.0

        if oi_change <= 0.20:
            return 75.0

        if oi_change <= 0.50:
            return 65.0

        return 35.0

    # Preço e OI subindo de forma controlada.
    if (
        price_momentum > 0
        and 0 < oi_change <= 0.30
    ):
        return 90.0

    # Preço subindo, mas OI explodindo:
    # risco de excesso de alavancagem.
    if (
        price_momentum > 0
        and oi_change > 0.60
    ):
        return 35.0

    if (
        price_momentum > 0
        and oi_change > 0.30
    ):
        return 65.0

    # Preço sobe com OI estável/caindo:
    # movimento potencialmente mais spot-driven.
    if (
        price_momentum > 0
        and oi_change <= 0
    ):
        return 80.0

    # Preço cai e OI cresce:
    # pressão especulativa / risco.
    if (
        price_momentum < 0
        and oi_change > 0.20
    ):
        return 25.0

    # Desalavancagem.
    if (
        price_momentum < 0
        and oi_change < 0
    ):
        return 60.0

    return 50.0


# ============================================================
# TIMING COMPONENTS
# ============================================================

def calculate_timing_components(
    row: pd.Series,
) -> Dict[str, float]:

    momentum_3m = (
        extract_momentum_3m(
            row
        )
    )

    momentum_6m = (
        extract_momentum_6m(
            row
        )
    )

    relative_strength = (
        extract_relative_strength(
            row
        )
    )

    volume_confirmation = (
        extract_volume_confirmation(
            row
        )
    )

    trend_structure = (
        calculate_trend_structure(
            row
        )
    )

    drawdown = (
        extract_drawdown(
            row
        )
    )

    funding = (
        extract_funding(
            row
        )
    )

    open_interest = (
        extract_open_interest_change(
            row
        )
    )

    return {
        "momentum_3m":
            momentum_3m,

        "momentum_6m":
            momentum_6m,

        "relative_strength":
            relative_strength,

        "volume_confirmation":
            volume_confirmation,

        "trend_structure":
            trend_structure,

        "drawdown_position":
            drawdown,

        "funding":
            funding,

        "open_interest":
            open_interest,

        "momentum_3m_score":
            score_momentum(
                momentum_3m
            ),

        "momentum_6m_score":
            score_momentum(
                momentum_6m
            ),

        "relative_strength_score":
            score_relative_strength(
                relative_strength
            ),

        "volume_confirmation_score":
            score_volume_confirmation(
                volume_confirmation
            ),

        "trend_structure_score":
            trend_structure,

        "drawdown_position_score":
            score_drawdown_position(
                drawdown
            ),

        "funding_score":
            score_funding(
                funding
            ),

        "open_interest_score":
            score_open_interest(
                open_interest,
                momentum_3m,
            ),
    }


# ============================================================
# TIMING SCORE
# ============================================================

def calculate_timing_score(
    row: pd.Series,
) -> float:

    components = (
        calculate_timing_components(
            row
        )
    )

    weighted_components = {
        "momentum_3m":
            (
                components[
                    "momentum_3m_score"
                ],
                config.TIMING_WEIGHTS[
                    "momentum_3m"
                ],
            ),

        "momentum_6m":
            (
                components[
                    "momentum_6m_score"
                ],
                config.TIMING_WEIGHTS[
                    "momentum_6m"
                ],
            ),

        "relative_strength":
            (
                components[
                    "relative_strength_score"
                ],
                config.TIMING_WEIGHTS[
                    "relative_strength"
                ],
            ),

        "volume_confirmation":
            (
                components[
                    "volume_confirmation_score"
                ],
                config.TIMING_WEIGHTS[
                    "volume_confirmation"
                ],
            ),

        "trend_structure":
            (
                components[
                    "trend_structure_score"
                ],
                config.TIMING_WEIGHTS[
                    "trend_structure"
                ],
            ),

        "drawdown_position":
            (
                components[
                    "drawdown_position_score"
                ],
                config.TIMING_WEIGHTS[
                    "drawdown_position"
                ],
            ),

        "funding":
            (
                components[
                    "funding_score"
                ],
                config.TIMING_WEIGHTS[
                    "funding"
                ],
            ),

        "open_interest":
            (
                components[
                    "open_interest_score"
                ],
                config.TIMING_WEIGHTS[
                    "open_interest"
                ],
            ),
    }

    score = _weighted_average(
        weighted_components
    )

    if pd.isna(score):
        return np.nan

    return round(
        _clip_score(score),
        2,
    )


# ============================================================
# TIMING STATUS
# ============================================================

def classify_timing(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= config.TIMING_STRONG_ENTRY:
        return "STRONG"

    if score >= config.TIMING_ENTRY:
        return "POSITIVE"

    if score >= config.TIMING_WATCH:
        return "NEUTRAL"

    return "UNFAVORABLE"


# ============================================================
# OVERHEATING
# ============================================================

def detect_timing_overheating(
    row: pd.Series,
) -> bool:

    momentum_3m = (
        extract_momentum_3m(
            row
        )
    )

    funding = (
        extract_funding(
            row
        )
    )

    oi_change = (
        extract_open_interest_change(
            row
        )
    )

    conditions = []

    if not pd.isna(momentum_3m):
        conditions.append(
            momentum_3m > 1.0
        )

    if not pd.isna(funding):

        normalized_funding = funding

        if abs(
            normalized_funding
        ) > 1:
            normalized_funding /= 100

        conditions.append(
            normalized_funding > 0.03
        )

    if not pd.isna(oi_change):

        normalized_oi = oi_change

        if abs(
            normalized_oi
        ) > 5:
            normalized_oi /= 100

        conditions.append(
            normalized_oi > 0.60
        )

    if len(conditions) < 2:
        return False

    return (
        sum(
            1
            for condition in conditions
            if condition
        )
        >= 2
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_timing_data_completeness(
    row: pd.Series,
) -> float:

    components = (
        calculate_timing_components(
            row
        )
    )

    score_fields = [
        "momentum_3m_score",
        "momentum_6m_score",
        "relative_strength_score",
        "volume_confirmation_score",
        "trend_structure_score",
        "drawdown_position_score",
        "funding_score",
        "open_interest_score",
    ]

    available = sum(
        1
        for field in score_fields
        if not pd.isna(
            components.get(
                field,
                np.nan,
            )
        )
    )

    return round(
        available
        / len(score_fields),
        4,
    )


# ============================================================
# DATA QUALITY
# ============================================================

def classify_timing_data_quality(
    completeness: float,
) -> str:

    if pd.isna(completeness):
        return "DATA_INSUFFICIENT"

    if completeness >= 0.80:
        return "HIGH"

    if completeness >= 0.60:
        return "GOOD"

    if completeness >= 0.40:
        return "PARTIAL"

    if completeness > 0:
        return "LOW"

    return "DATA_INSUFFICIENT"


# ============================================================
# INDIVIDUAL EVALUATION
# ============================================================

def evaluate_timing(
    row: pd.Series,
) -> Dict:

    components = (
        calculate_timing_components(
            row
        )
    )

    timing_score = (
        calculate_timing_score(
            row
        )
    )

    completeness = (
        calculate_timing_data_completeness(
            row
        )
    )

    return {
        "momentum_3m":
            components[
                "momentum_3m"
            ],

        "momentum_6m":
            components[
                "momentum_6m"
            ],

        "relative_strength":
            components[
                "relative_strength"
            ],

        "volume_confirmation":
            components[
                "volume_confirmation"
            ],

        "trend_structure_score":
            components[
                "trend_structure_score"
            ],

        "drawdown_position":
            components[
                "drawdown_position"
            ],

        "funding":
            components[
                "funding"
            ],

        "open_interest_change":
            components[
                "open_interest"
            ],

        "momentum_3m_score":
            components[
                "momentum_3m_score"
            ],

        "momentum_6m_score":
            components[
                "momentum_6m_score"
            ],

        "relative_strength_score":
            components[
                "relative_strength_score"
            ],

        "volume_confirmation_score":
            components[
                "volume_confirmation_score"
            ],

        "drawdown_position_score":
            components[
                "drawdown_position_score"
            ],

        "funding_score":
            components[
                "funding_score"
            ],

        "open_interest_score":
            components[
                "open_interest_score"
            ],

        "timing_score":
            timing_score,

        "timing_status":
            classify_timing(
                timing_score
            ),

        "timing_overheating":
            detect_timing_overheating(
                row
            ),

        "timing_data_completeness":
            completeness,

        "timing_data_quality":
            classify_timing_data_quality(
                completeness
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_timing_engine(
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
            "[timing_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_timing(
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

    market = build_market_dataset(
        include_history=True,
        history_top_n=50,
    )

    result = run_timing_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "momentum_3m",
        "momentum_6m",
        "relative_strength",
        "volume_confirmation",
        "trend_structure_score",
        "drawdown_position",
        "funding",
        "open_interest_change",
        "timing_score",
        "timing_status",
        "timing_overheating",
        "timing_data_completeness",
        "timing_data_quality",
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
            "timing_score",
            ascending=False,
            na_position="last",
        )
        .head(50)
        .to_string(
            index=False
        )
    )
