"""
CRYPTO OPPORTUNITY ENGINE
Fundamental Engine

OBJETIVO:
Medir a ACELERAÇÃO FUNDAMENTAL do projeto.

Princípio central:
O motor procura projetos cuja atividade econômica e utilização
estejam melhorando mais rapidamente do que o mercado está precificando.

Métricas principais:
- crescimento de usuários;
- crescimento de transações;
- crescimento de fees;
- crescimento de receita;
- crescimento de TVL;
- desenvolvimento;
- aceleração;
- divergência fundamentos × preço.

IMPORTANTE:
Fundamentos fortes não significam automaticamente entrada.
Timing é analisado separadamente.
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
# NORMALIZAÇÃO DE CRESCIMENTO
# ============================================================

def score_growth(
    growth: float,
    lower: float = -0.30,
    upper: float = 0.70,
) -> float:

    if pd.isna(growth):
        return np.nan

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


def score_high_growth(
    growth: float,
) -> float:

    return score_growth(
        growth,
        lower=-0.50,
        upper=1.00,
    )


def score_acceleration(
    acceleration: float,
) -> float:

    return score_growth(
        acceleration,
        lower=-0.25,
        upper=0.50,
    )


# ============================================================
# ACELERAÇÃO ENTRE JANELAS
# ============================================================

def calculate_growth_acceleration(
    growth_30d: float,
    growth_90d: float,
) -> float:

    if (
        pd.isna(growth_30d)
        or pd.isna(growth_90d)
    ):
        return np.nan

    monthly_equivalent_90d = (
        (1 + growth_90d) ** (30 / 90)
        - 1
        if growth_90d > -1
        else np.nan
    )

    if pd.isna(
        monthly_equivalent_90d
    ):
        return np.nan

    return (
        growth_30d
        - monthly_equivalent_90d
    )


# ============================================================
# USERS
# ============================================================

def extract_users_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "users_growth_90d",
            "active_addresses_growth_90d",
            "new_holders_growth_90d",
            "users_growth",
            "active_addresses_growth",
        ],
    )


# ============================================================
# TRANSACTIONS
# ============================================================

def extract_transactions_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "transactions_growth_90d",
            "transaction_growth_90d",
            "network_usage_growth_90d",
            "transactions_growth",
            "transaction_growth",
            "network_usage_growth",
        ],
    )


# ============================================================
# FEES
# ============================================================

def extract_fees_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "fees_growth_90d",
            "fees_growth",
            "protocol_fees_growth_90d",
            "protocol_fees_growth",
        ],
    )


# ============================================================
# REVENUE
# ============================================================

def extract_revenue_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "revenue_growth_90d",
            "revenue_growth",
            "protocol_revenue_growth_90d",
            "protocol_revenue_growth",
        ],
    )


# ============================================================
# TVL
# ============================================================

def extract_tvl_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "tvl_growth_90d",
            "chain_tvl_growth_90d",
            "tvl_growth_180d",
            "chain_tvl_growth_180d",
        ],
    )


# ============================================================
# DEVELOPMENT
# ============================================================

def extract_developer_score(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "developer_score",
        ],
    )


# ============================================================
# COMPONENT SCORES
# ============================================================

def calculate_component_scores(
    row: pd.Series,
) -> Dict[str, float]:

    users_growth = (
        extract_users_growth(
            row
        )
    )

    transactions_growth = (
        extract_transactions_growth(
            row
        )
    )

    fees_growth = (
        extract_fees_growth(
            row
        )
    )

    revenue_growth = (
        extract_revenue_growth(
            row
        )
    )

    tvl_growth = (
        extract_tvl_growth(
            row
        )
    )

    developer_score = (
        extract_developer_score(
            row
        )
    )

    return {
        "users_growth":
            users_growth,

        "transactions_growth":
            transactions_growth,

        "fees_growth":
            fees_growth,

        "revenue_growth":
            revenue_growth,

        "tvl_growth":
            tvl_growth,

        "developer_score":
            developer_score,

        "users_score":
            score_growth(
                users_growth
            ),

        "transactions_score":
            score_growth(
                transactions_growth
            ),

        "fees_score":
            score_high_growth(
                fees_growth
            ),

        "revenue_score_component":
            score_high_growth(
                revenue_growth
            ),

        "tvl_score_component":
            score_high_growth(
                tvl_growth
            ),

        "developer_score_component":
            _clip_score(
                developer_score
            ),
    }


# ============================================================
# FUNDAMENTAL ACCELERATION SCORE
# ============================================================

def calculate_fundamental_acceleration_score(
    row: pd.Series,
) -> float:

    components = (
        calculate_component_scores(
            row
        )
    )

    weighted_components = {
        "users_growth":
            (
                components[
                    "users_score"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "users_growth"
                ],
            ),

        "transactions_growth":
            (
                components[
                    "transactions_score"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "transactions_growth"
                ],
            ),

        "fees_growth":
            (
                components[
                    "fees_score"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "fees_growth"
                ],
            ),

        "revenue_growth":
            (
                components[
                    "revenue_score_component"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "revenue_growth"
                ],
            ),

        "tvl_growth":
            (
                components[
                    "tvl_score_component"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "tvl_growth"
                ],
            ),

        "developer_growth":
            (
                components[
                    "developer_score_component"
                ],
                config.FUNDAMENTAL_ACCELERATION_WEIGHTS[
                    "developer_growth"
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
# FUNDAMENTAL ACCELERATION BONUS
# ============================================================

def calculate_acceleration_confirmation(
    row: pd.Series,
) -> float:

    acceleration_values = []

    pairs = [
        (
            "users_growth_30d",
            "users_growth_90d",
        ),
        (
            "transactions_growth_30d",
            "transactions_growth_90d",
        ),
        (
            "fees_growth_30d",
            "fees_growth_90d",
        ),
        (
            "revenue_growth_30d",
            "revenue_growth_90d",
        ),
        (
            "tvl_growth_30d",
            "tvl_growth_90d",
        ),
        (
            "chain_tvl_growth_30d",
            "chain_tvl_growth_90d",
        ),
        (
            "stablecoin_growth_30d",
            "stablecoin_growth_90d",
        ),
    ]

    for field_30d, field_90d in pairs:

        if (
            field_30d not in row.index
            or field_90d not in row.index
        ):
            continue

        growth_30d = _safe_float(
            row.get(
                field_30d
            )
        )

        growth_90d = _safe_float(
            row.get(
                field_90d
            )
        )

        acceleration = (
            calculate_growth_acceleration(
                growth_30d,
                growth_90d,
            )
        )

        if not pd.isna(
            acceleration
        ):

            acceleration_values.append(
                acceleration
            )

    if not acceleration_values:
        return np.nan

    median_acceleration = float(
        np.median(
            acceleration_values
        )
    )

    return round(
        score_acceleration(
            median_acceleration
        ),
        2,
    )


# ============================================================
# FUNDAMENTAL CHANGE COMPOSITE
# ============================================================

def calculate_fundamental_change(
    row: pd.Series,
) -> float:

    growth_values = []

    fields = [
        "users_growth_90d",
        "active_addresses_growth_90d",
        "transactions_growth_90d",
        "transaction_growth_90d",
        "fees_growth_90d",
        "revenue_growth_90d",
        "tvl_growth_90d",
        "chain_tvl_growth_90d",
        "stablecoin_growth_90d",
    ]

    for field in fields:

        if field not in row.index:
            continue

        value = _safe_float(
            row.get(field)
        )

        if not pd.isna(value):
            growth_values.append(
                value
            )

    if not growth_values:
        return np.nan

    return float(
        np.median(
            growth_values
        )
    )


# ============================================================
# FUNDAMENTAL × PRICE DIVERGENCE
# ============================================================

def calculate_fundamental_price_divergence(
    row: pd.Series,
) -> float:

    fundamental_change = (
        calculate_fundamental_change(
            row
        )
    )

    price_change = (
        _first_available(
            row,
            [
                "return_90d",
                "price_change_90d",
            ],
        )
    )

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
# DIVERGENCE SCORE
# ============================================================

def calculate_divergence_score(
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
# FUNDAMENTAL STATUS
# ============================================================

def classify_fundamental_status(
    score: float,
    acceleration_score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if (
        score >= 80
        and (
            pd.isna(acceleration_score)
            or acceleration_score >= 60
        )
    ):
        return "VERY_STRONG"

    if score >= 70:
        return "STRONG"

    if score >= config.MIN_FUNDAMENTAL_SCORE:
        return "ACCEPTABLE"

    if score >= 40:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# FUNDAMENTAL QUALITY GATE
# ============================================================

def fundamental_quality_gate(
    score: float,
) -> bool:

    if pd.isna(score):
        return False

    return (
        score
        >= config.MIN_FUNDAMENTAL_SCORE
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_fundamental_data_completeness(
    components: Dict[str, float],
) -> float:

    fields = [
        "users_growth",
        "transactions_growth",
        "fees_growth",
        "revenue_growth",
        "tvl_growth",
        "developer_score",
    ]

    available = sum(
        1
        for field in fields
        if not pd.isna(
            components.get(
                field,
                np.nan,
            )
        )
    )

    return round(
        available
        / len(fields),
        4,
    )


# ============================================================
# AVALIAÇÃO INDIVIDUAL
# ============================================================

def evaluate_fundamentals(
    row: pd.Series,
) -> Dict:

    components = (
        calculate_component_scores(
            row
        )
    )

    fundamental_score = (
        calculate_fundamental_acceleration_score(
            row
        )
    )

    acceleration_score = (
        calculate_acceleration_confirmation(
            row
        )
    )

    fundamental_change = (
        calculate_fundamental_change(
            row
        )
    )

    divergence = (
        calculate_fundamental_price_divergence(
            row
        )
    )

    divergence_score = (
        calculate_divergence_score(
            divergence
        )
    )

    status = (
        classify_fundamental_status(
            fundamental_score,
            acceleration_score,
        )
    )

    quality_gate = (
        fundamental_quality_gate(
            fundamental_score
        )
    )

    completeness = (
        calculate_fundamental_data_completeness(
            components
        )
    )

    return {
        "fundamental_acceleration_score":
            fundamental_score,

        "fundamental_acceleration_confirmation":
            acceleration_score,

        "fundamental_change_90d":
            fundamental_change,

        "fundamental_price_divergence":
            divergence,

        "fundamental_price_divergence_score":
            divergence_score,

        "fundamental_status":
            status,

        "fundamental_quality_gate":
            quality_gate,

        "fundamental_data_completeness":
            completeness,

        "fundamental_users_growth":
            components[
                "users_growth"
            ],

        "fundamental_transactions_growth":
            components[
                "transactions_growth"
            ],

        "fundamental_fees_growth":
            components[
                "fees_growth"
            ],

        "fundamental_revenue_growth":
            components[
                "revenue_growth"
            ],

        "fundamental_tvl_growth":
            components[
                "tvl_growth"
            ],

        "fundamental_developer_score":
            components[
                "developer_score"
            ],
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_fundamental_engine(
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
            "[fundamental_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_fundamentals(
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

    from data.onchain_data import (
        enrich_with_onchain_data,
    )

    from data.developer_data import (
        enrich_with_developer_data,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=30,
    )

    market = enrich_with_defi_data(
        market
    )

    market = enrich_with_onchain_data(
        market
    )

    market = enrich_with_developer_data(
        market
    )

    result = run_fundamental_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "fundamental_acceleration_score",
        "fundamental_acceleration_confirmation",
        "fundamental_change_90d",
        "fundamental_price_divergence",
        "fundamental_price_divergence_score",
        "fundamental_status",
        "fundamental_quality_gate",
        "fundamental_data_completeness",
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
            "fundamental_acceleration_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
