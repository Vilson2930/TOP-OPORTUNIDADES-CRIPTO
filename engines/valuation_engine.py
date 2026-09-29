"""
CRYPTO OPPORTUNITY ENGINE
Valuation Engine

OBJETIVO:
Medir se o token está barato ou caro em relação à atividade econômica
real do protocolo e aos seus próprios fundamentos.

Princípio:
Preço baixo NÃO significa valuation barato.

O motor analisa:
- Market Cap / Revenue;
- FDV / Revenue;
- Market Cap / Fees;
- Market Cap / TVL;
- valuation histórico;
- valuation relativo aos pares.

Quanto menor o múltiplo econômico, melhor, desde que existam
fundamentos reais.

Valuation sozinho NÃO cria oportunidade.
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

        weighted_sum += score * weight
        available_weight += weight

    if available_weight == 0:
        return np.nan

    return weighted_sum / available_weight


# ============================================================
# ECONOMIC DATA
# ============================================================

def extract_revenue(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "protocol_revenue",
            "annualized_revenue",
            "revenue_annualized",
            "protocol_revenue_annualized",
            "revenue_365d",
        ],
    )


def extract_fees(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "protocol_fees",
            "annualized_fees",
            "fees_annualized",
            "protocol_fees_annualized",
            "fees_365d",
        ],
    )


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


# ============================================================
# MULTIPLES
# ============================================================

def calculate_market_cap_revenue(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "market_cap_revenue",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    market_cap = _safe_float(
        row.get("market_cap")
    )

    revenue = extract_revenue(
        row
    )

    if (
        pd.isna(market_cap)
        or pd.isna(revenue)
        or revenue <= 0
    ):
        return np.nan

    return market_cap / revenue


def calculate_fdv_revenue(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "fdv_revenue",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    fdv = _safe_float(
        row.get("fdv")
    )

    revenue = extract_revenue(
        row
    )

    if (
        pd.isna(fdv)
        or pd.isna(revenue)
        or revenue <= 0
    ):
        return np.nan

    return fdv / revenue


def calculate_market_cap_fees(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "market_cap_fees",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    market_cap = _safe_float(
        row.get("market_cap")
    )

    fees = extract_fees(
        row
    )

    if (
        pd.isna(market_cap)
        or pd.isna(fees)
        or fees <= 0
    ):
        return np.nan

    return market_cap / fees


def calculate_market_cap_tvl(
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
        row.get("market_cap")
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

    return market_cap / tvl


# ============================================================
# MULTIPLE SCORE
# ============================================================

def score_revenue_multiple(
    multiple: float,
) -> float:

    if (
        pd.isna(multiple)
        or multiple <= 0
    ):
        return np.nan

    if multiple <= 2:
        return 100.0

    if multiple <= 5:
        return 90.0

    if multiple <= 10:
        return 80.0

    if multiple <= 20:
        return 65.0

    if multiple <= 30:
        return 50.0

    if multiple <= 50:
        return 35.0

    if multiple <= 100:
        return 20.0

    return 5.0


def score_fees_multiple(
    multiple: float,
) -> float:

    if (
        pd.isna(multiple)
        or multiple <= 0
    ):
        return np.nan

    if multiple <= 2:
        return 100.0

    if multiple <= 5:
        return 90.0

    if multiple <= 10:
        return 80.0

    if multiple <= 20:
        return 70.0

    if multiple <= 40:
        return 55.0

    if multiple <= 75:
        return 35.0

    if multiple <= 150:
        return 20.0

    return 5.0


def score_tvl_multiple(
    multiple: float,
) -> float:

    if (
        pd.isna(multiple)
        or multiple <= 0
    ):
        return np.nan

    if multiple <= 0.50:
        return 100.0

    if multiple <= 1.00:
        return 90.0

    if multiple <= 2.00:
        return 75.0

    if multiple <= 3.00:
        return 60.0

    if multiple <= 5.00:
        return 45.0

    if multiple <= 10.00:
        return 25.0

    return 10.0


# ============================================================
# HISTORICAL VALUATION
# ============================================================

def calculate_historical_valuation_score(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "historical_valuation_score",
            "valuation_historical_score",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    current_multiple = (
        calculate_market_cap_revenue(
            row
        )
    )

    historical_median = _first_available(
        row,
        [
            "historical_market_cap_revenue_median",
            "market_cap_revenue_historical_median",
        ],
    )

    if (
        pd.isna(current_multiple)
        or pd.isna(historical_median)
        or historical_median <= 0
    ):
        return np.nan

    ratio = (
        current_multiple
        / historical_median
    )

    if ratio <= 0.50:
        return 100.0

    if ratio <= 0.70:
        return 90.0

    if ratio <= 0.85:
        return 80.0

    if ratio <= 1.00:
        return 70.0

    if ratio <= 1.25:
        return 50.0

    if ratio <= 1.50:
        return 35.0

    if ratio <= 2.00:
        return 20.0

    return 5.0


# ============================================================
# PEER VALUATION
# ============================================================

def calculate_peer_relative_score(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "peer_relative_valuation_score",
            "peer_valuation_score",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    current_multiple = (
        calculate_market_cap_revenue(
            row
        )
    )

    peer_median = _first_available(
        row,
        [
            "peer_market_cap_revenue_median",
            "category_market_cap_revenue_median",
        ],
    )

    if (
        pd.isna(current_multiple)
        or pd.isna(peer_median)
        or peer_median <= 0
    ):
        return np.nan

    ratio = (
        current_multiple
        / peer_median
    )

    if ratio <= 0.50:
        return 100.0

    if ratio <= 0.70:
        return 90.0

    if ratio <= 0.85:
        return 80.0

    if ratio <= 1.00:
        return 70.0

    if ratio <= 1.25:
        return 50.0

    if ratio <= 1.50:
        return 35.0

    if ratio <= 2.00:
        return 20.0

    return 5.0


# ============================================================
# CATEGORY PEER CALCULATION
# ============================================================

def add_peer_valuation_metrics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    if "category" not in result.columns:
        result["category"] = "OTHER"

    result["category"] = (
        result["category"]
        .fillna("OTHER")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if "market_cap_revenue" not in result.columns:

        result["market_cap_revenue"] = (
            result.apply(
                calculate_market_cap_revenue,
                axis=1,
            )
        )

    category_medians = (
        result.groupby(
            "category"
        )["market_cap_revenue"]
        .transform("median")
    )

    result[
        "category_market_cap_revenue_median"
    ] = category_medians

    return result


# ============================================================
# VALUATION SCORE
# ============================================================

def calculate_valuation_score(
    row: pd.Series,
) -> float:

    market_cap_revenue = (
        calculate_market_cap_revenue(
            row
        )
    )

    fdv_revenue = (
        calculate_fdv_revenue(
            row
        )
    )

    market_cap_fees = (
        calculate_market_cap_fees(
            row
        )
    )

    market_cap_tvl = (
        calculate_market_cap_tvl(
            row
        )
    )

    historical_score = (
        calculate_historical_valuation_score(
            row
        )
    )

    peer_score = (
        calculate_peer_relative_score(
            row
        )
    )

    components = {
        "market_cap_revenue":
            (
                score_revenue_multiple(
                    market_cap_revenue
                ),
                config.VALUATION_WEIGHTS[
                    "market_cap_revenue"
                ],
            ),

        "fdv_revenue":
            (
                score_revenue_multiple(
                    fdv_revenue
                ),
                config.VALUATION_WEIGHTS[
                    "fdv_revenue"
                ],
            ),

        "market_cap_fees":
            (
                score_fees_multiple(
                    market_cap_fees
                ),
                config.VALUATION_WEIGHTS[
                    "market_cap_fees"
                ],
            ),

        "market_cap_tvl":
            (
                score_tvl_multiple(
                    market_cap_tvl
                ),
                config.VALUATION_WEIGHTS[
                    "market_cap_tvl"
                ],
            ),

        "historical_valuation":
            (
                historical_score,
                config.VALUATION_WEIGHTS[
                    "historical_valuation"
                ],
            ),

        "peer_relative_valuation":
            (
                peer_score,
                config.VALUATION_WEIGHTS[
                    "peer_relative_valuation"
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
# VALUATION VS FUNDAMENTALS
# ============================================================

def calculate_valuation_fundamental_alignment(
    row: pd.Series,
    valuation_score: float,
) -> float:

    fundamental_score = _first_available(
        row,
        [
            "fundamental_acceleration_score",
        ],
    )

    revenue_score = _first_available(
        row,
        [
            "revenue_score",
        ],
    )

    quality_values = []

    if not pd.isna(fundamental_score):
        quality_values.append(
            fundamental_score
        )

    if not pd.isna(revenue_score):
        quality_values.append(
            revenue_score
        )

    if (
        not quality_values
        or pd.isna(valuation_score)
    ):
        return np.nan

    quality_score = float(
        np.mean(
            quality_values
        )
    )

    return round(
        _clip_score(
            valuation_score
            * 0.50
            + quality_score
            * 0.50
        ),
        2,
    )


# ============================================================
# STATUS
# ============================================================

def classify_valuation(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 85:
        return "VERY_CHEAP"

    if score >= 70:
        return "ATTRACTIVE"

    if score >= 55:
        return "FAIR"

    if score >= 40:
        return "EXPENSIVE"

    return "VERY_EXPENSIVE"


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_valuation_data_completeness(
    row: pd.Series,
) -> float:

    values = [
        calculate_market_cap_revenue(
            row
        ),

        calculate_fdv_revenue(
            row
        ),

        calculate_market_cap_fees(
            row
        ),

        calculate_market_cap_tvl(
            row
        ),

        calculate_historical_valuation_score(
            row
        ),

        calculate_peer_relative_score(
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
# INDIVIDUAL EVALUATION
# ============================================================

def evaluate_valuation(
    row: pd.Series,
) -> Dict:

    market_cap_revenue = (
        calculate_market_cap_revenue(
            row
        )
    )

    fdv_revenue = (
        calculate_fdv_revenue(
            row
        )
    )

    market_cap_fees = (
        calculate_market_cap_fees(
            row
        )
    )

    market_cap_tvl = (
        calculate_market_cap_tvl(
            row
        )
    )

    historical_score = (
        calculate_historical_valuation_score(
            row
        )
    )

    peer_score = (
        calculate_peer_relative_score(
            row
        )
    )

    valuation_score = (
        calculate_valuation_score(
            row
        )
    )

    alignment = (
        calculate_valuation_fundamental_alignment(
            row,
            valuation_score,
        )
    )

    return {
        "market_cap_revenue":
            market_cap_revenue,

        "fdv_revenue":
            fdv_revenue,

        "market_cap_fees":
            market_cap_fees,

        "market_cap_tvl":
            market_cap_tvl,

        "historical_valuation_score":
            historical_score,

        "peer_relative_valuation_score":
            peer_score,

        "valuation_score":
            valuation_score,

        "valuation_fundamental_alignment":
            alignment,

        "valuation_status":
            classify_valuation(
                valuation_score
            ),

        "valuation_data_completeness":
            calculate_valuation_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_valuation_engine(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    result = (
        add_peer_valuation_metrics(
            dataset
        )
    )

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
            "[valuation_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_valuation(
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

    from engines.revenue_engine import (
        run_revenue_engine,
    )

    from engines.fundamental_engine import (
        run_fundamental_engine,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=30,
    )

    market = enrich_with_defi_data(
        market
    )

    market = run_fundamental_engine(
        market
    )

    market = run_revenue_engine(
        market
    )

    result = run_valuation_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "category",
        "market_cap_revenue",
        "fdv_revenue",
        "market_cap_fees",
        "market_cap_tvl",
        "historical_valuation_score",
        "peer_relative_valuation_score",
        "valuation_score",
        "valuation_fundamental_alignment",
        "valuation_status",
        "valuation_data_completeness",
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
            "valuation_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
