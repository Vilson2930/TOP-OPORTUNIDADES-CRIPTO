"""
CRYPTO OPPORTUNITY ENGINE
Revenue Engine

OBJETIVO:
Medir a qualidade econômica do protocolo.

Princípio:
Receita e fees só são relevantes quando representam
atividade econômica real e sustentável.

O motor analisa:
- crescimento de receita;
- crescimento de fees;
- qualidade da receita;
- consistência;
- atividade econômica;
- eficiência sobre TVL;
- divergência receita × preço.

IMPORTANTE:
Receita alta sem captura de valor pelo token
não é suficiente para caracterizar oportunidade.
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
# EXTRAÇÃO DE MÉTRICAS
# ============================================================

def extract_revenue_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "revenue_growth_90d",
            "protocol_revenue_growth_90d",
            "revenue_growth",
            "protocol_revenue_growth",
        ],
    )


def extract_fees_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "fees_growth_90d",
            "protocol_fees_growth_90d",
            "fees_growth",
            "protocol_fees_growth",
        ],
    )


def extract_revenue(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "annualized_revenue",
            "revenue_annualized",
            "protocol_revenue_annualized",
            "revenue_365d",
            "protocol_revenue",
        ],
    )


def extract_fees(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "annualized_fees",
            "fees_annualized",
            "protocol_fees_annualized",
            "fees_365d",
            "protocol_fees",
        ],
    )


def extract_economic_activity_growth(
    row: pd.Series,
) -> float:

    values = []

    fields = [
        "revenue_growth_90d",
        "fees_growth_90d",
        "transactions_growth_90d",
        "transaction_growth_90d",
        "users_growth_90d",
        "active_addresses_growth_90d",
        "tvl_growth_90d",
    ]

    for field in fields:

        if field not in row.index:
            continue

        value = _safe_float(
            row.get(field)
        )

        if not pd.isna(value):
            values.append(
                value
            )

    if not values:
        return np.nan

    return float(
        np.median(
            values
        )
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def score_growth(
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
# QUALIDADE DA RECEITA
# ============================================================

def calculate_revenue_quality(
    revenue: float,
    fees: float,
) -> float:

    if (
        pd.isna(revenue)
        or revenue < 0
    ):
        return np.nan

    if (
        pd.isna(fees)
        or fees <= 0
    ):

        if revenue > 0:
            return 60.0

        return 0.0

    ratio = (
        revenue
        / fees
    )

    if ratio >= 0.80:
        return 100.0

    if ratio >= 0.60:
        return 90.0

    if ratio >= 0.40:
        return 80.0

    if ratio >= 0.25:
        return 70.0

    if ratio >= 0.10:
        return 55.0

    if ratio > 0:
        return 35.0

    return 0.0


# ============================================================
# CONSISTÊNCIA
# ============================================================

def calculate_revenue_consistency(
    row: pd.Series,
) -> float:

    values = []

    fields = [
        "revenue_growth_30d",
        "revenue_growth_90d",
        "revenue_growth_180d",
        "revenue_growth_365d",
    ]

    for field in fields:

        if field not in row.index:
            continue

        value = _safe_float(
            row.get(field)
        )

        if not pd.isna(value):
            values.append(
                value
            )

    if not values:

        revenue_growth = (
            extract_revenue_growth(
                row
            )
        )

        if pd.isna(
            revenue_growth
        ):
            return np.nan

        return score_growth(
            revenue_growth
        )

    positive_ratio = (
        sum(
            1
            for value in values
            if value > 0
        )
        / len(values)
    )

    median_growth = float(
        np.median(
            values
        )
    )

    growth_component = (
        score_growth(
            median_growth
        )
    )

    consistency_component = (
        positive_ratio
        * 100
    )

    return round(
        (
            consistency_component
            * 0.60
            + growth_component
            * 0.40
        ),
        2,
    )


# ============================================================
# RECEITA / TVL
# ============================================================

def calculate_revenue_per_tvl(
    revenue: float,
    tvl: float,
) -> float:

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


def calculate_fees_per_tvl(
    fees: float,
    tvl: float,
) -> float:

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


# ============================================================
# MARKET CAP / REVENUE
# ============================================================

def calculate_market_cap_revenue(
    market_cap: float,
    revenue: float,
) -> float:

    if (
        pd.isna(market_cap)
        or pd.isna(revenue)
        or revenue <= 0
    ):
        return np.nan

    return (
        market_cap
        / revenue
    )


def calculate_fdv_revenue(
    fdv: float,
    revenue: float,
) -> float:

    if (
        pd.isna(fdv)
        or pd.isna(revenue)
        or revenue <= 0
    ):
        return np.nan

    return (
        fdv
        / revenue
    )


def calculate_market_cap_fees(
    market_cap: float,
    fees: float,
) -> float:

    if (
        pd.isna(market_cap)
        or pd.isna(fees)
        or fees <= 0
    ):
        return np.nan

    return (
        market_cap
        / fees
    )


# ============================================================
# REVENUE SCORE
# ============================================================

def calculate_revenue_score(
    row: pd.Series,
) -> float:

    revenue_growth = (
        extract_revenue_growth(
            row
        )
    )

    fees_growth = (
        extract_fees_growth(
            row
        )
    )

    revenue = (
        extract_revenue(
            row
        )
    )

    fees = (
        extract_fees(
            row
        )
    )

    economic_growth = (
        extract_economic_activity_growth(
            row
        )
    )

    revenue_quality = (
        calculate_revenue_quality(
            revenue,
            fees,
        )
    )

    consistency = (
        calculate_revenue_consistency(
            row
        )
    )

    components = {
        "revenue_growth":
            (
                score_growth(
                    revenue_growth
                ),
                config.REVENUE_WEIGHTS[
                    "revenue_growth"
                ],
            ),

        "fees_growth":
            (
                score_growth(
                    fees_growth
                ),
                config.REVENUE_WEIGHTS[
                    "fees_growth"
                ],
            ),

        "revenue_quality":
            (
                revenue_quality,
                config.REVENUE_WEIGHTS[
                    "revenue_quality"
                ],
            ),

        "revenue_consistency":
            (
                consistency,
                config.REVENUE_WEIGHTS[
                    "revenue_consistency"
                ],
            ),

        "economic_activity_growth":
            (
                score_growth(
                    economic_growth
                ),
                config.REVENUE_WEIGHTS[
                    "economic_activity_growth"
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
# DIVERGÊNCIA RECEITA × PREÇO
# ============================================================

def calculate_revenue_price_divergence(
    row: pd.Series,
) -> float:

    revenue_growth = (
        extract_revenue_growth(
            row
        )
    )

    price_return = (
        _first_available(
            row,
            [
                "return_90d",
                "price_change_90d",
            ],
        )
    )

    if (
        pd.isna(revenue_growth)
        or pd.isna(price_return)
    ):
        return np.nan

    return (
        revenue_growth
        - price_return
    )


def calculate_revenue_divergence_score(
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
                (divergence - lower)
                / (upper - lower)
            )
            * 100
        ),
        2,
    )


# ============================================================
# STATUS
# ============================================================

def classify_revenue_status(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 85:
        return "EXCELLENT"

    if score >= 70:
        return "STRONG"

    if score >= config.MIN_REVENUE_SCORE:
        return "ACCEPTABLE"

    if score >= 30:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# QUALITY GATE
# ============================================================

def revenue_quality_gate(
    score: float,
) -> bool:

    if pd.isna(score):
        return False

    return (
        score
        >= config.MIN_REVENUE_SCORE
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_revenue_data_completeness(
    row: pd.Series,
) -> float:

    metrics = [
        extract_revenue_growth(
            row
        ),
        extract_fees_growth(
            row
        ),
        extract_revenue(
            row
        ),
        extract_fees(
            row
        ),
        extract_economic_activity_growth(
            row
        ),
    ]

    available = sum(
        1
        for value in metrics
        if not pd.isna(value)
    )

    return round(
        available
        / len(metrics),
        4,
    )


# ============================================================
# AVALIAÇÃO INDIVIDUAL
# ============================================================

def evaluate_revenue(
    row: pd.Series,
) -> Dict:

    revenue = (
        extract_revenue(
            row
        )
    )

    fees = (
        extract_fees(
            row
        )
    )

    revenue_growth = (
        extract_revenue_growth(
            row
        )
    )

    fees_growth = (
        extract_fees_growth(
            row
        )
    )

    economic_growth = (
        extract_economic_activity_growth(
            row
        )
    )

    revenue_quality = (
        calculate_revenue_quality(
            revenue,
            fees,
        )
    )

    consistency = (
        calculate_revenue_consistency(
            row
        )
    )

    score = (
        calculate_revenue_score(
            row
        )
    )

    divergence = (
        calculate_revenue_price_divergence(
            row
        )
    )

    divergence_score = (
        calculate_revenue_divergence_score(
            divergence
        )
    )

    tvl = _first_available(
        row,
        [
            "tvl_current",
            "chain_tvl_current",
        ],
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

    return {
        "protocol_revenue":
            revenue,

        "protocol_fees":
            fees,

        "revenue_growth":
            revenue_growth,

        "fees_growth":
            fees_growth,

        "economic_activity_growth":
            economic_growth,

        "revenue_quality_score":
            revenue_quality,

        "revenue_consistency_score":
            consistency,

        "revenue_per_tvl":
            calculate_revenue_per_tvl(
                revenue,
                tvl,
            ),

        "fees_per_tvl":
            calculate_fees_per_tvl(
                fees,
                tvl,
            ),

        "market_cap_revenue":
            calculate_market_cap_revenue(
                market_cap,
                revenue,
            ),

        "fdv_revenue":
            calculate_fdv_revenue(
                fdv,
                revenue,
            ),

        "market_cap_fees":
            calculate_market_cap_fees(
                market_cap,
                fees,
            ),

        "revenue_score":
            score,

        "revenue_price_divergence":
            divergence,

        "revenue_price_divergence_score":
            divergence_score,

        "revenue_status":
            classify_revenue_status(
                score
            ),

        "revenue_quality_gate":
            revenue_quality_gate(
                score
            ),

        "revenue_data_completeness":
            calculate_revenue_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_revenue_engine(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    evaluations = []

    total = len(result)

    for position, (_, row) in enumerate(
        result.iterrows(),
        start=1,
    ):

        symbol = row.get(
            "symbol",
            "UNKNOWN",
        )

        print(
            "[revenue_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_revenue(
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
        history_top_n=30,
    )

    result = run_revenue_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "protocol_revenue",
        "protocol_fees",
        "revenue_growth",
        "fees_growth",
        "revenue_quality_score",
        "revenue_consistency_score",
        "revenue_score",
        "revenue_price_divergence",
        "revenue_status",
        "revenue_quality_gate",
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
            "revenue_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
