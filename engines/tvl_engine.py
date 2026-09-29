"""
CRYPTO OPPORTUNITY ENGINE
TVL Engine

OBJETIVO:
Avaliar crescimento, qualidade e eficiência econômica do TVL.

Princípio:
TVL alto sozinho NÃO representa oportunidade.

O motor analisa:
- crescimento de TVL em 30/90/180 dias;
- receita / TVL;
- fees / TVL;
- market cap / TVL;
- qualidade e estabilidade do TVL;
- relevância do TVL para cada categoria.

A importância do TVL é ajustada pela categoria do projeto.
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
# CATEGORY
# ============================================================

def extract_category(
    row: pd.Series,
) -> str:

    category = str(
        row.get(
            "category",
            "OTHER",
        )
    ).upper().strip()

    if not category:
        return "OTHER"

    return category


def get_tvl_category_relevance(
    row: pd.Series,
) -> float:

    category = extract_category(
        row
    )

    return float(
        config.TVL_CATEGORY_RELEVANCE.get(
            category,
            config.TVL_CATEGORY_RELEVANCE.get(
                "OTHER",
                0.30,
            ),
        )
    )


# ============================================================
# TVL
# ============================================================

def extract_tvl(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "tvl_current",
            "chain_tvl_current",
            "tvl",
        ],
    )


def extract_tvl_growth(
    row: pd.Series,
    days: int,
) -> float:

    fields = {
        30: [
            "tvl_growth_30d",
            "chain_tvl_growth_30d",
        ],

        90: [
            "tvl_growth_90d",
            "chain_tvl_growth_90d",
        ],

        180: [
            "tvl_growth_180d",
            "chain_tvl_growth_180d",
        ],

        365: [
            "tvl_growth_365d",
            "chain_tvl_growth_365d",
        ],
    }

    return _first_available(
        row,
        fields.get(
            days,
            [],
        ),
    )


# ============================================================
# GROWTH SCORE
# ============================================================

def score_tvl_growth(
    growth: float,
) -> float:

    if pd.isna(growth):
        return np.nan

    lower = -0.50
    upper = 1.00

    if growth <= lower:
        return 0.0

    if growth >= upper:
        return 100.0

    return _clip_score(
        (
            (growth - lower)
            / (upper - lower)
        )
        * 100
    )


# ============================================================
# REVENUE / TVL
# ============================================================

def extract_revenue_per_tvl(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "revenue_per_tvl",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    revenue = _first_available(
        row,
        [
            "protocol_revenue",
            "annualized_revenue",
            "revenue_annualized",
        ],
    )

    tvl = extract_tvl(
        row
    )

    if (
        pd.isna(revenue)
        or pd.isna(tvl)
        or tvl <= 0
    ):
        return np.nan

    return (
        revenue
        / tvl
    )


def score_revenue_per_tvl(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0:
        return 0.0

    if ratio >= 0.30:
        return 100.0

    return _clip_score(
        ratio
        / 0.30
        * 100
    )


# ============================================================
# FEES / TVL
# ============================================================

def extract_fees_per_tvl(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "fees_per_tvl",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    fees = _first_available(
        row,
        [
            "protocol_fees",
            "annualized_fees",
            "fees_annualized",
        ],
    )

    tvl = extract_tvl(
        row
    )

    if (
        pd.isna(fees)
        or pd.isna(tvl)
        or tvl <= 0
    ):
        return np.nan

    return (
        fees
        / tvl
    )


def score_fees_per_tvl(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0:
        return 0.0

    if ratio >= 0.50:
        return 100.0

    return _clip_score(
        ratio
        / 0.50
        * 100
    )


# ============================================================
# MARKET CAP / TVL
# ============================================================

def extract_market_cap_tvl(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "market_cap_tvl",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    tvl = extract_tvl(
        row
    )

    if (
        pd.isna(market_cap)
        or pd.isna(tvl)
        or tvl <= 0
    ):
        return np.nan

    return (
        market_cap
        / tvl
    )


def score_market_cap_tvl(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0:
        return np.nan

    if ratio <= 0.50:
        return 100.0

    if ratio <= 1.00:
        return 90.0

    if ratio <= 2.00:
        return 75.0

    if ratio <= 3.00:
        return 60.0

    if ratio <= 5.00:
        return 45.0

    if ratio <= 10.00:
        return 25.0

    return 10.0


# ============================================================
# TVL QUALITY
# ============================================================

def calculate_tvl_quality(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "tvl_quality",
            "tvl_quality_score",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    stability = _first_available(
        row,
        [
            "tvl_stability",
        ],
    )

    drawdown = _first_available(
        row,
        [
            "tvl_drawdown",
        ],
    )

    acceleration = _first_available(
        row,
        [
            "tvl_acceleration",
            "chain_tvl_acceleration",
        ],
    )

    components = []

    if not pd.isna(stability):

        if stability <= 1:
            stability *= 100

        components.append(
            _clip_score(
                stability
            )
        )

    if not pd.isna(drawdown):

        drawdown_score = _clip_score(
            (
                1
                + max(
                    drawdown,
                    -1,
                )
            )
            * 100
        )

        components.append(
            drawdown_score
        )

    if not pd.isna(acceleration):

        acceleration_score = _clip_score(
            (
                acceleration
                + 0.25
            )
            / 0.75
            * 100
        )

        components.append(
            acceleration_score
        )

    if not components:
        return np.nan

    return round(
        float(
            np.mean(
                components
            )
        ),
        2,
    )


# ============================================================
# RAW TVL SCORE
# ============================================================

def calculate_raw_tvl_score(
    row: pd.Series,
) -> float:

    growth_30d = (
        extract_tvl_growth(
            row,
            30,
        )
    )

    growth_90d = (
        extract_tvl_growth(
            row,
            90,
        )
    )

    growth_180d = (
        extract_tvl_growth(
            row,
            180,
        )
    )

    revenue_per_tvl = (
        extract_revenue_per_tvl(
            row
        )
    )

    fees_per_tvl = (
        extract_fees_per_tvl(
            row
        )
    )

    market_cap_tvl = (
        extract_market_cap_tvl(
            row
        )
    )

    tvl_quality = (
        calculate_tvl_quality(
            row
        )
    )

    components = {
        "tvl_growth_30d":
            (
                score_tvl_growth(
                    growth_30d
                ),
                config.TVL_WEIGHTS[
                    "tvl_growth_30d"
                ],
            ),

        "tvl_growth_90d":
            (
                score_tvl_growth(
                    growth_90d
                ),
                config.TVL_WEIGHTS[
                    "tvl_growth_90d"
                ],
            ),

        "tvl_growth_180d":
            (
                score_tvl_growth(
                    growth_180d
                ),
                config.TVL_WEIGHTS[
                    "tvl_growth_180d"
                ],
            ),

        "revenue_per_tvl":
            (
                score_revenue_per_tvl(
                    revenue_per_tvl
                ),
                config.TVL_WEIGHTS[
                    "revenue_per_tvl"
                ],
            ),

        "fees_per_tvl":
            (
                score_fees_per_tvl(
                    fees_per_tvl
                ),
                config.TVL_WEIGHTS[
                    "fees_per_tvl"
                ],
            ),

        "market_cap_tvl":
            (
                score_market_cap_tvl(
                    market_cap_tvl
                ),
                config.TVL_WEIGHTS[
                    "market_cap_tvl"
                ],
            ),

        "tvl_quality":
            (
                tvl_quality,
                config.TVL_WEIGHTS[
                    "tvl_quality"
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
# CATEGORY-ADJUSTED TVL SCORE
# ============================================================

def calculate_tvl_score(
    row: pd.Series,
) -> float:

    raw_score = (
        calculate_raw_tvl_score(
            row
        )
    )

    if pd.isna(raw_score):
        return np.nan

    relevance = (
        get_tvl_category_relevance(
            row
        )
    )

    neutral_score = 50.0

    adjusted = (
        raw_score
        * relevance
        + neutral_score
        * (1 - relevance)
    )

    return round(
        _clip_score(
            adjusted
        ),
        2,
    )


# ============================================================
# TVL × PRICE DIVERGENCE
# ============================================================

def calculate_tvl_price_divergence(
    row: pd.Series,
) -> float:

    tvl_growth = (
        extract_tvl_growth(
            row,
            90,
        )
    )

    price_return = _first_available(
        row,
        [
            "return_90d",
            "price_change_90d",
        ],
    )

    if (
        pd.isna(tvl_growth)
        or pd.isna(price_return)
    ):
        return np.nan

    return (
        tvl_growth
        - price_return
    )


def calculate_tvl_divergence_score(
    divergence: float,
) -> float:

    if pd.isna(divergence):
        return np.nan

    lower = -0.50
    upper = 1.00

    if divergence <= lower:
        return 0.0

    if divergence >= upper:
        return 100.0

    return round(
        _clip_score(
            (
                divergence
                - lower
            )
            / (
                upper
                - lower
            )
            * 100
        ),
        2,
    )


# ============================================================
# STATUS
# ============================================================

def classify_tvl_status(
    score: float,
    relevance: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if relevance <= 0.30:
        return "LOW_CATEGORY_RELEVANCE"

    if score >= 85:
        return "VERY_STRONG"

    if score >= 70:
        return "STRONG"

    if score >= 55:
        return "ACCEPTABLE"

    if score >= 40:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_tvl_data_completeness(
    row: pd.Series,
) -> float:

    values = [
        extract_tvl_growth(
            row,
            30,
        ),

        extract_tvl_growth(
            row,
            90,
        ),

        extract_tvl_growth(
            row,
            180,
        ),

        extract_revenue_per_tvl(
            row
        ),

        extract_fees_per_tvl(
            row
        ),

        extract_market_cap_tvl(
            row
        ),

        calculate_tvl_quality(
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
# AVALIAÇÃO INDIVIDUAL
# ============================================================

def evaluate_tvl(
    row: pd.Series,
) -> Dict:

    tvl = extract_tvl(
        row
    )

    growth_30d = (
        extract_tvl_growth(
            row,
            30,
        )
    )

    growth_90d = (
        extract_tvl_growth(
            row,
            90,
        )
    )

    growth_180d = (
        extract_tvl_growth(
            row,
            180,
        )
    )

    revenue_per_tvl = (
        extract_revenue_per_tvl(
            row
        )
    )

    fees_per_tvl = (
        extract_fees_per_tvl(
            row
        )
    )

    market_cap_tvl = (
        extract_market_cap_tvl(
            row
        )
    )

    tvl_quality = (
        calculate_tvl_quality(
            row
        )
    )

    raw_score = (
        calculate_raw_tvl_score(
            row
        )
    )

    adjusted_score = (
        calculate_tvl_score(
            row
        )
    )

    relevance = (
        get_tvl_category_relevance(
            row
        )
    )

    divergence = (
        calculate_tvl_price_divergence(
            row
        )
    )

    return {
        "tvl_current":
            tvl,

        "tvl_growth_30d":
            growth_30d,

        "tvl_growth_90d":
            growth_90d,

        "tvl_growth_180d":
            growth_180d,

        "revenue_per_tvl":
            revenue_per_tvl,

        "fees_per_tvl":
            fees_per_tvl,

        "market_cap_tvl":
            market_cap_tvl,

        "tvl_quality":
            tvl_quality,

        "tvl_raw_score":
            raw_score,

        "tvl_category_relevance":
            relevance,

        "tvl_score":
            adjusted_score,

        "tvl_price_divergence":
            divergence,

        "tvl_price_divergence_score":
            calculate_tvl_divergence_score(
                divergence
            ),

        "tvl_status":
            classify_tvl_status(
                adjusted_score,
                relevance,
            ),

        "tvl_data_completeness":
            calculate_tvl_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_tvl_engine(
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
            "[tvl_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_tvl(
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

    from data.defi_data import (
        enrich_with_defi_data,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=30,
    )

    market = enrich_with_defi_data(
        market
    )

    result = run_tvl_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "category",
        "tvl_current",
        "tvl_growth_30d",
        "tvl_growth_90d",
        "tvl_growth_180d",
        "revenue_per_tvl",
        "fees_per_tvl",
        "market_cap_tvl",
        "tvl_quality",
        "tvl_category_relevance",
        "tvl_raw_score",
        "tvl_score",
        "tvl_price_divergence",
        "tvl_status",
        "tvl_data_completeness",
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
            "tvl_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
