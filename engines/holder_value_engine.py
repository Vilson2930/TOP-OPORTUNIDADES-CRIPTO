"""
CRYPTO OPPORTUNITY ENGINE
Holder Value Engine

OBJETIVO:
Medir quanto do valor econômico criado pelo protocolo
é efetivamente capturado pelo TOKEN e pelo HOLDER.

Princípio:
PROTOCOLO BOM != TOKEN BOM.

Um protocolo pode:
- gerar receita;
- ter usuários;
- crescer TVL;
- ter grande adoção;

e ainda assim o token pode capturar pouco ou nenhum valor.

O motor analisa:
- real yield;
- distribuição de fees;
- buyback;
- burn;
- necessidade econômica do token;
- staking econômico.

REGRA CENTRAL:
Yield criado apenas por emissão inflacionária
NÃO é considerado real yield.
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


def _first_boolean(
    row: pd.Series,
    fields: Iterable[str],
):

    for field in fields:

        if field not in row.index:
            continue

        value = row.get(field)

        if value is None:
            continue

        if isinstance(value, bool):
            return value

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):

            if pd.isna(value):
                continue

            return bool(value)

        text = str(value).strip().lower()

        if text in {
            "true",
            "yes",
            "sim",
            "1",
            "active",
            "enabled",
        }:
            return True

        if text in {
            "false",
            "no",
            "nao",
            "não",
            "0",
            "inactive",
            "disabled",
        }:
            return False

    return None


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
# REVENUE
# ============================================================

def extract_protocol_revenue(
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


# ============================================================
# HOLDER REVENUE
# ============================================================

def extract_holder_revenue(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "holders_revenue",
            "holder_revenue",
            "tokenholder_revenue",
            "token_holders_revenue",
            "annualized_holder_revenue",
        ],
    )


# ============================================================
# FEE DISTRIBUTION
# ============================================================

def extract_fee_distribution_ratio(
    row: pd.Series,
) -> float:

    explicit_ratio = _first_available(
        row,
        [
            "fee_distribution_ratio",
            "holder_fee_share",
            "tokenholder_fee_share",
            "revenue_share_to_holders",
        ],
    )

    if not pd.isna(
        explicit_ratio
    ):

        if explicit_ratio > 1:
            explicit_ratio = (
                explicit_ratio
                / 100
            )

        return float(
            np.clip(
                explicit_ratio,
                0,
                1,
            )
        )

    holder_revenue = (
        extract_holder_revenue(
            row
        )
    )

    protocol_revenue = (
        extract_protocol_revenue(
            row
        )
    )

    if (
        pd.isna(holder_revenue)
        or pd.isna(protocol_revenue)
        or protocol_revenue <= 0
    ):
        return np.nan

    return float(
        np.clip(
            holder_revenue
            / protocol_revenue,
            0,
            1,
        )
    )


# ============================================================
# REAL YIELD
# ============================================================

def extract_staking_yield(
    row: pd.Series,
) -> float:

    yield_value = _first_available(
        row,
        [
            "staking_yield",
            "staking_apr",
            "staking_apy",
            "holder_yield",
            "token_yield",
        ],
    )

    if pd.isna(
        yield_value
    ):
        return np.nan

    if yield_value > 1:
        yield_value = (
            yield_value
            / 100
        )

    return yield_value


def extract_emission_yield(
    row: pd.Series,
) -> float:

    value = _first_available(
        row,
        [
            "emission_yield",
            "inflationary_yield",
            "token_emission_yield",
            "staking_emission_apr",
        ],
    )

    if pd.isna(value):
        return np.nan

    if value > 1:
        value = (
            value / 100
        )

    return value


def calculate_real_yield(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "real_yield",
            "real_yield_rate",
            "holder_real_yield",
        ],
    )

    if not pd.isna(explicit):

        if explicit > 1:
            explicit = (
                explicit / 100
            )

        return explicit

    staking_yield = (
        extract_staking_yield(
            row
        )
    )

    emission_yield = (
        extract_emission_yield(
            row
        )
    )

    if pd.isna(
        staking_yield
    ):
        return np.nan

    if pd.isna(
        emission_yield
    ):
        return staking_yield

    return (
        staking_yield
        - emission_yield
    )


# ============================================================
# REAL YIELD SCORE
# ============================================================

def score_real_yield(
    real_yield: float,
) -> float:

    if pd.isna(real_yield):
        return np.nan

    if real_yield <= 0:
        return 0.0

    if real_yield >= 0.15:
        return 100.0

    return _clip_score(
        real_yield
        / 0.15
        * 100
    )


# ============================================================
# FEE DISTRIBUTION SCORE
# ============================================================

def score_fee_distribution(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    if ratio <= 0:
        return 0.0

    if ratio >= 0.75:
        return 100.0

    return _clip_score(
        ratio
        / 0.75
        * 100
    )


# ============================================================
# BUYBACK
# ============================================================

def calculate_buyback_score(
    row: pd.Series,
) -> float:

    buyback_yield = _first_available(
        row,
        [
            "buyback_yield",
            "annual_buyback_yield",
            "buyback_market_cap_ratio",
        ],
    )

    if not pd.isna(
        buyback_yield
    ):

        if buyback_yield > 1:
            buyback_yield = (
                buyback_yield / 100
            )

        if buyback_yield <= 0:
            return 0.0

        if buyback_yield >= 0.10:
            return 100.0

        return _clip_score(
            buyback_yield
            / 0.10
            * 100
        )

    buyback_value = _first_available(
        row,
        [
            "annual_buyback_usd",
            "buyback_annualized",
            "token_buyback_usd",
        ],
    )

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    if (
        not pd.isna(buyback_value)
        and not pd.isna(market_cap)
        and market_cap > 0
    ):

        ratio = (
            buyback_value
            / market_cap
        )

        return _clip_score(
            ratio
            / 0.10
            * 100
        )

    active = _first_boolean(
        row,
        [
            "buyback_active",
            "token_buyback",
            "has_buyback",
        ],
    )

    if active is True:
        return 60.0

    if active is False:
        return 0.0

    return np.nan


# ============================================================
# BURN
# ============================================================

def calculate_burn_score(
    row: pd.Series,
) -> float:

    burn_yield = _first_available(
        row,
        [
            "burn_yield",
            "annual_burn_yield",
            "burn_market_cap_ratio",
        ],
    )

    if not pd.isna(
        burn_yield
    ):

        if burn_yield > 1:
            burn_yield = (
                burn_yield / 100
            )

        if burn_yield <= 0:
            return 0.0

        if burn_yield >= 0.05:
            return 100.0

        return _clip_score(
            burn_yield
            / 0.05
            * 100
        )

    burn_value = _first_available(
        row,
        [
            "annual_burn_usd",
            "burn_annualized",
            "token_burn_usd",
        ],
    )

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    if (
        not pd.isna(burn_value)
        and not pd.isna(market_cap)
        and market_cap > 0
    ):

        ratio = (
            burn_value
            / market_cap
        )

        return _clip_score(
            ratio
            / 0.05
            * 100
        )

    active = _first_boolean(
        row,
        [
            "burn_active",
            "token_burn",
            "has_burn",
        ],
    )

    if active is True:
        return 60.0

    if active is False:
        return 0.0

    return np.nan


# ============================================================
# TOKEN REQUIRED FOR USAGE
# ============================================================

def calculate_token_usage_score(
    row: pd.Series,
) -> float:

    explicit_score = _first_available(
        row,
        [
            "token_usage_score",
            "token_utility_score",
        ],
    )

    if not pd.isna(
        explicit_score
    ):

        if explicit_score <= 1:
            explicit_score = (
                explicit_score * 100
            )

        return _clip_score(
            explicit_score
        )

    required = _first_boolean(
        row,
        [
            "token_required_for_usage",
            "token_required",
            "native_token_required",
        ],
    )

    gas_token = _first_boolean(
        row,
        [
            "gas_token",
            "native_gas_token",
        ],
    )

    collateral_token = _first_boolean(
        row,
        [
            "token_used_as_collateral",
            "economic_collateral_token",
        ],
    )

    scores = []

    if required is True:
        scores.append(
            100.0
        )

    elif required is False:
        scores.append(
            0.0
        )

    if gas_token is True:
        scores.append(
            100.0
        )

    if collateral_token is True:
        scores.append(
            80.0
        )

    if not scores:
        return np.nan

    return float(
        max(scores)
    )


# ============================================================
# ECONOMIC STAKING
# ============================================================

def calculate_economic_staking_score(
    row: pd.Series,
) -> float:

    explicit_score = _first_available(
        row,
        [
            "economic_staking_score",
            "staking_utility_score",
        ],
    )

    if not pd.isna(
        explicit_score
    ):

        if explicit_score <= 1:
            explicit_score *= 100

        return _clip_score(
            explicit_score
        )

    staking_required = _first_boolean(
        row,
        [
            "staking_required_for_security",
            "economic_staking",
            "staking_required",
        ],
    )

    slashing = _first_boolean(
        row,
        [
            "staking_slashing",
            "slashing_enabled",
        ],
    )

    validator_token = _first_boolean(
        row,
        [
            "validator_token_required",
            "token_required_for_validation",
        ],
    )

    if (
        staking_required is True
        and (
            slashing is True
            or validator_token is True
        )
    ):
        return 100.0

    if staking_required is True:
        return 80.0

    if validator_token is True:
        return 80.0

    if staking_required is False:
        return 0.0

    return np.nan


# ============================================================
# HOLDER VALUE SCORE
# ============================================================

def calculate_holder_value_score(
    row: pd.Series,
) -> float:

    real_yield = (
        calculate_real_yield(
            row
        )
    )

    fee_distribution = (
        extract_fee_distribution_ratio(
            row
        )
    )

    components = {
        "real_yield":
            (
                score_real_yield(
                    real_yield
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "real_yield"
                ],
            ),

        "fee_distribution":
            (
                score_fee_distribution(
                    fee_distribution
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "fee_distribution"
                ],
            ),

        "buyback":
            (
                calculate_buyback_score(
                    row
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "buyback"
                ],
            ),

        "burn":
            (
                calculate_burn_score(
                    row
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "burn"
                ],
            ),

        "token_required_for_usage":
            (
                calculate_token_usage_score(
                    row
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "token_required_for_usage"
                ],
            ),

        "economic_staking":
            (
                calculate_economic_staking_score(
                    row
                ),
                config.HOLDER_VALUE_WEIGHTS[
                    "economic_staking"
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
# VALUE CAPTURE RATIO
# ============================================================

def calculate_holder_capture_ratio(
    row: pd.Series,
) -> float:

    holder_revenue = (
        extract_holder_revenue(
            row
        )
    )

    protocol_revenue = (
        extract_protocol_revenue(
            row
        )
    )

    if (
        pd.isna(holder_revenue)
        or pd.isna(protocol_revenue)
        or protocol_revenue <= 0
    ):
        return np.nan

    return float(
        np.clip(
            holder_revenue
            / protocol_revenue,
            0,
            1,
        )
    )


# ============================================================
# TOKEN VALUE CAPTURE CLASSIFICATION
# ============================================================

def classify_holder_value(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 85:
        return "VERY_STRONG"

    if score >= 70:
        return "STRONG"

    if score >= config.MIN_HOLDER_VALUE_SCORE:
        return "ACCEPTABLE"

    if score >= 25:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# QUALITY GATE
# ============================================================

def holder_value_quality_gate(
    score: float,
) -> bool:

    if pd.isna(score):
        return False

    return (
        score
        >= config.MIN_HOLDER_VALUE_SCORE
    )


# ============================================================
# INFLATIONARY YIELD WARNING
# ============================================================

def detect_inflationary_yield(
    row: pd.Series,
) -> bool:

    staking_yield = (
        extract_staking_yield(
            row
        )
    )

    emission_yield = (
        extract_emission_yield(
            row
        )
    )

    if (
        pd.isna(staking_yield)
        or staking_yield <= 0
    ):
        return False

    if pd.isna(
        emission_yield
    ):
        return False

    return (
        emission_yield
        >= staking_yield * 0.80
    )


# ============================================================
# PROTOCOL GOOD / TOKEN BAD
# ============================================================

def detect_protocol_token_disconnect(
    row: pd.Series,
    holder_score: float,
) -> bool:

    revenue_score = _safe_float(
        row.get(
            "revenue_score"
        )
    )

    fundamental_score = _safe_float(
        row.get(
            "fundamental_acceleration_score"
        )
    )

    protocol_strength = []

    if not pd.isna(
        revenue_score
    ):
        protocol_strength.append(
            revenue_score
        )

    if not pd.isna(
        fundamental_score
    ):
        protocol_strength.append(
            fundamental_score
        )

    if not protocol_strength:
        return False

    protocol_score = float(
        np.mean(
            protocol_strength
        )
    )

    return (
        protocol_score >= 70
        and not pd.isna(holder_score)
        and holder_score
        < config.MIN_HOLDER_VALUE_SCORE
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_holder_data_completeness(
    row: pd.Series,
) -> float:

    values = [
        calculate_real_yield(
            row
        ),

        extract_fee_distribution_ratio(
            row
        ),

        calculate_buyback_score(
            row
        ),

        calculate_burn_score(
            row
        ),

        calculate_token_usage_score(
            row
        ),

        calculate_economic_staking_score(
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

def evaluate_holder_value(
    row: pd.Series,
) -> Dict:

    real_yield = (
        calculate_real_yield(
            row
        )
    )

    fee_distribution = (
        extract_fee_distribution_ratio(
            row
        )
    )

    buyback_score = (
        calculate_buyback_score(
            row
        )
    )

    burn_score = (
        calculate_burn_score(
            row
        )
    )

    token_usage_score = (
        calculate_token_usage_score(
            row
        )
    )

    staking_score = (
        calculate_economic_staking_score(
            row
        )
    )

    holder_score = (
        calculate_holder_value_score(
            row
        )
    )

    capture_ratio = (
        calculate_holder_capture_ratio(
            row
        )
    )

    inflationary_yield = (
        detect_inflationary_yield(
            row
        )
    )

    disconnect = (
        detect_protocol_token_disconnect(
            row,
            holder_score,
        )
    )

    return {
        "real_yield":
            real_yield,

        "real_yield_score":
            score_real_yield(
                real_yield
            ),

        "fee_distribution_ratio":
            fee_distribution,

        "fee_distribution_score":
            score_fee_distribution(
                fee_distribution
            ),

        "buyback_score":
            buyback_score,

        "burn_score":
            burn_score,

        "token_usage_score":
            token_usage_score,

        "economic_staking_score":
            staking_score,

        "holder_capture_ratio":
            capture_ratio,

        "holder_value_score":
            holder_score,

        "holder_value_status":
            classify_holder_value(
                holder_score
            ),

        "holder_value_quality_gate":
            holder_value_quality_gate(
                holder_score
            ),

        "inflationary_yield_warning":
            inflationary_yield,

        "protocol_token_disconnect":
            disconnect,

        "holder_value_data_completeness":
            calculate_holder_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_holder_value_engine(
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
            "[holder_value_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_holder_value(
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

    from engines.revenue_engine import (
        run_revenue_engine,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=30,
    )

    market = run_revenue_engine(
        market
    )

    result = run_holder_value_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "real_yield",
        "fee_distribution_ratio",
        "buyback_score",
        "burn_score",
        "token_usage_score",
        "economic_staking_score",
        "holder_capture_ratio",
        "holder_value_score",
        "holder_value_status",
        "holder_value_quality_gate",
        "inflationary_yield_warning",
        "protocol_token_disconnect",
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
            "holder_value_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
