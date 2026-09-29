"""
CRYPTO OPPORTUNITY ENGINE
On-Chain Engine

OBJETIVO:
Medir a confirmação on-chain da oportunidade.

O motor procura crescimento real de atividade econômica e utilização
da rede/protocolo.

Métricas principais:
- crescimento de endereços ativos;
- crescimento de transações;
- crescimento de novos holders;
- entrada de stablecoins;
- fluxo líquido em exchanges;
- acumulação por grandes holders;
- crescimento de utilização da rede.

Princípio:
Preço pode subir sem melhora on-chain.
Atividade on-chain crescente fornece confirmação adicional de que
a melhora fundamental possui utilização econômica real.

Dados ausentes NÃO são inventados.

Quando métricas on-chain canônicas não estiverem disponíveis,
o engine pode utilizar o onchain_opportunity_score produzido pela
camada data/onchain_data.py como fallback.
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
            score
            * weight
        )

        available_weight += weight

    if available_weight == 0:
        return np.nan

    return (
        weighted_sum
        / available_weight
    )


def _normalize_ratio(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    if abs(value) > 2:
        return value / 100.0

    return value


# ============================================================
# GENERIC GROWTH SCORE
# ============================================================

def score_growth(
    growth: float,
) -> float:

    if pd.isna(growth):
        return np.nan

    growth = _normalize_ratio(
        growth
    )

    lower = -0.40
    upper = 0.80

    score = (
        (
            growth
            - lower
        )
        / (
            upper
            - lower
        )
        * 100
    )

    return _clip_score(
        score
    )


def score_flow(
    flow: float,
) -> float:

    if pd.isna(flow):
        return np.nan

    flow = _normalize_ratio(
        flow
    )

    lower = -0.20
    upper = 0.20

    score = (
        (
            flow
            - lower
        )
        / (
            upper
            - lower
        )
        * 100
    )

    return _clip_score(
        score
    )


# ============================================================
# ACTIVE ADDRESSES
# ============================================================

def extract_active_addresses_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "active_addresses_growth",
            "active_addresses_growth_90d",
            "active_addresses_90d_growth",
            "active_address_growth_90d",
            "active_users_growth_90d",
            "active_users_growth",
            "users_growth_90d",
            "users_growth",
        ],
    )


# ============================================================
# TRANSACTIONS
# ============================================================

def extract_transaction_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "transaction_growth",
            "transactions_growth",
            "transaction_growth_90d",
            "transactions_growth_90d",
            "transactions_90d_growth",
            "tx_growth_90d",
            "tx_growth",
        ],
    )


# ============================================================
# NEW HOLDERS
# ============================================================

def extract_new_holders_growth(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "new_holders_growth",
            "new_holders_growth_90d",
            "holders_growth",
            "holders_growth_90d",
            "holder_growth_90d",
            "new_wallets_growth",
            "new_wallets_growth_90d",
        ],
    )


# ============================================================
# STABLECOIN INFLOWS
# ============================================================

def extract_stablecoin_inflows(
    row: pd.Series,
) -> float:

    direct = _first_available(
        row,
        [
            "stablecoin_inflows",
            "stablecoin_inflow",
            "stablecoin_inflow_30d",
            "stablecoin_inflow_90d",
            "stablecoin_netflow",
            "stablecoin_netflow_30d",
            "stablecoin_netflow_90d",
        ],
    )

    if not pd.isna(direct):
        return direct

    return _first_available(
        row,
        [
            "stablecoin_supply_growth_30d",
            "stablecoin_supply_growth_90d",
            "stablecoin_growth_30d",
            "stablecoin_growth_90d",
            "stablecoin_supply_acceleration",
        ],
    )


# ============================================================
# EXCHANGE NETFLOW
# ============================================================

def extract_exchange_netflow(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "exchange_netflow",
            "exchange_netflow_7d",
            "exchange_netflow_30d",
            "exchange_netflow_90d",
            "exchange_flow",
            "exchange_flow_30d",
        ],
    )


def calculate_exchange_netflow_score(
    row: pd.Series,
) -> float:

    netflow = extract_exchange_netflow(
        row
    )

    if pd.isna(netflow):
        return np.nan

    # Saída líquida de exchanges é interpretada
    # como confirmação positiva.
    return score_flow(
        -netflow
    )


# ============================================================
# WHALE ACCUMULATION
# ============================================================

def extract_whale_accumulation(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "whale_accumulation",
            "whale_accumulation_30d",
            "whale_accumulation_90d",
            "large_holder_accumulation",
            "large_holder_netflow",
            "large_holder_netflow_30d",
            "large_holder_netflow_90d",
        ],
    )


def calculate_whale_accumulation_score(
    row: pd.Series,
) -> float:

    value = extract_whale_accumulation(
        row
    )

    if pd.isna(value):
        return np.nan

    # Alguns provedores retornam score 0-1.
    if 0 <= value <= 1:

        return _clip_score(
            value * 100
        )

    return score_flow(
        value
    )


# ============================================================
# NETWORK USAGE
# ============================================================

def extract_network_usage_growth(
    row: pd.Series,
) -> float:

    direct = _first_available(
        row,
        [
            "network_usage_growth",
            "network_usage_growth_90d",
            "network_activity_growth",
            "network_activity_growth_90d",
            "usage_growth",
            "usage_growth_90d",
        ],
    )

    if not pd.isna(direct):
        return direct

    candidates = [
        extract_transaction_growth(
            row
        ),

        _first_available(
            row,
            [
                "chain_tvl_growth_90d",
                "tvl_growth_90d",
            ],
        ),

        extract_active_addresses_growth(
            row
        ),
    ]

    available = [
        value
        for value in candidates
        if not pd.isna(value)
    ]

    if not available:
        return np.nan

    return float(
        np.median(
            available
        )
    )


# ============================================================
# COMPONENT SCORES
# ============================================================

def calculate_onchain_components(
    row: pd.Series,
) -> Dict[str, float]:

    active_addresses_growth = (
        extract_active_addresses_growth(
            row
        )
    )

    transaction_growth = (
        extract_transaction_growth(
            row
        )
    )

    new_holders_growth = (
        extract_new_holders_growth(
            row
        )
    )

    stablecoin_inflows = (
        extract_stablecoin_inflows(
            row
        )
    )

    network_usage_growth = (
        extract_network_usage_growth(
            row
        )
    )

    return {
        "active_addresses_growth":
            score_growth(
                active_addresses_growth
            ),

        "transaction_growth":
            score_growth(
                transaction_growth
            ),

        "new_holders_growth":
            score_growth(
                new_holders_growth
            ),

        "stablecoin_inflows":
            score_growth(
                stablecoin_inflows
            ),

        "exchange_netflow":
            calculate_exchange_netflow_score(
                row
            ),

        "whale_accumulation":
            calculate_whale_accumulation_score(
                row
            ),

        "network_usage_growth":
            score_growth(
                network_usage_growth
            ),
    }


# ============================================================
# ON-CHAIN SCORE
# ============================================================

def calculate_onchain_score(
    row: pd.Series,
) -> float:

    component_scores = (
        calculate_onchain_components(
            row
        )
    )

    components = {}

    for component_name, score in (
        component_scores.items()
    ):

        weight = (
            config.ONCHAIN_WEIGHTS.get(
                component_name
            )
        )

        if weight is None:
            continue

        components[
            component_name
        ] = (
            score,
            weight,
        )

    score = _weighted_average(
        components
    )

    if not pd.isna(score):

        return round(
            _clip_score(
                score
            ),
            2,
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    # data/onchain_data.py já produz um score baseado em
    # crescimento de TVL da chain e stablecoin supply.
    # Só é usado quando nenhuma métrica canônica está disponível.
    # --------------------------------------------------------

    fallback = _first_available(
        row,
        [
            "onchain_opportunity_score",
        ],
    )

    if pd.isna(fallback):
        return np.nan

    if fallback <= 1:
        fallback *= 100

    return round(
        _clip_score(
            fallback
        ),
        2,
    )


# ============================================================
# ON-CHAIN CHANGE
# ============================================================

def calculate_onchain_change(
    row: pd.Series,
) -> float:

    values = [
        extract_active_addresses_growth(
            row
        ),

        extract_transaction_growth(
            row
        ),

        extract_new_holders_growth(
            row
        ),

        extract_stablecoin_inflows(
            row
        ),

        extract_network_usage_growth(
            row
        ),
    ]

    available = []

    for value in values:

        if pd.isna(value):
            continue

        available.append(
            _normalize_ratio(
                value
            )
        )

    if not available:
        return np.nan

    return float(
        np.median(
            available
        )
    )


# ============================================================
# PRICE RETURN
# ============================================================

def extract_price_return_90d(
    row: pd.Series,
) -> float:

    value = _first_available(
        row,
        [
            "return_90d",
            "price_return_90d",
            "price_change_90d",
            "price_change_percentage_90d",
        ],
    )

    if pd.isna(value):
        return np.nan

    return _normalize_ratio(
        value
    )


# ============================================================
# ON-CHAIN / PRICE DIVERGENCE
# ============================================================

def calculate_onchain_price_divergence(
    row: pd.Series,
) -> float:

    onchain_change = (
        calculate_onchain_change(
            row
        )
    )

    price_return = (
        extract_price_return_90d(
            row
        )
    )

    if (
        pd.isna(onchain_change)
        or pd.isna(price_return)
    ):
        return np.nan

    return (
        onchain_change
        - price_return
    )


def calculate_onchain_divergence_score(
    row: pd.Series,
) -> float:

    divergence = (
        calculate_onchain_price_divergence(
            row
        )
    )

    if pd.isna(divergence):
        return np.nan

    lower = -0.50
    upper = 0.80

    score = (
        (
            divergence
            - lower
        )
        / (
            upper
            - lower
        )
        * 100
    )

    return round(
        _clip_score(
            score
        ),
        2,
    )


# ============================================================
# CAPITAL CONFIRMATION
# ============================================================

def calculate_capital_confirmation(
    row: pd.Series,
) -> float:

    stablecoin_growth = (
        extract_stablecoin_inflows(
            row
        )
    )

    chain_tvl_growth = (
        _first_available(
            row,
            [
                "chain_tvl_growth_90d",
                "tvl_growth_90d",
            ],
        )
    )

    values = []

    if not pd.isna(
        stablecoin_growth
    ):

        values.append(
            score_growth(
                stablecoin_growth
            )
        )

    if not pd.isna(
        chain_tvl_growth
    ):

        values.append(
            score_growth(
                chain_tvl_growth
            )
        )

    values = [
        value
        for value in values
        if not pd.isna(value)
    ]

    if not values:
        return np.nan

    return round(
        float(
            np.mean(
                values
            )
        ),
        2,
    )


# ============================================================
# STATUS
# ============================================================

def classify_onchain_status(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 85:
        return "VERY_STRONG"

    if score >= 75:
        return "STRONG"

    if score >= 65:
        return "POSITIVE"

    if score >= 50:
        return "NEUTRAL"

    if score >= 35:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_onchain_data_completeness(
    row: pd.Series,
) -> float:

    raw_values = [
        extract_active_addresses_growth(
            row
        ),

        extract_transaction_growth(
            row
        ),

        extract_new_holders_growth(
            row
        ),

        extract_stablecoin_inflows(
            row
        ),

        extract_exchange_netflow(
            row
        ),

        extract_whale_accumulation(
            row
        ),

        extract_network_usage_growth(
            row
        ),
    ]

    available = sum(
        1
        for value in raw_values
        if not pd.isna(value)
    )

    return round(
        available
        / len(
            raw_values
        ),
        4,
    )


def classify_onchain_data_quality(
    completeness: float,
) -> str:

    if pd.isna(completeness):
        return "DATA_INSUFFICIENT"

    if completeness >= 0.85:
        return "HIGH"

    if completeness >= 0.60:
        return "GOOD"

    if completeness >= 0.35:
        return "LIMITED"

    if completeness > 0:
        return "LOW"

    return "DATA_INSUFFICIENT"


# ============================================================
# INDIVIDUAL EVALUATION
# ============================================================

def evaluate_onchain(
    row: pd.Series,
) -> Dict:

    component_scores = (
        calculate_onchain_components(
            row
        )
    )

    onchain_score = (
        calculate_onchain_score(
            row
        )
    )

    onchain_change = (
        calculate_onchain_change(
            row
        )
    )

    divergence = (
        calculate_onchain_price_divergence(
            row
        )
    )

    divergence_score = (
        calculate_onchain_divergence_score(
            row
        )
    )

    capital_confirmation = (
        calculate_capital_confirmation(
            row
        )
    )

    completeness = (
        calculate_onchain_data_completeness(
            row
        )
    )

    return {
        "active_addresses_growth_score":
            component_scores[
                "active_addresses_growth"
            ],

        "transaction_growth_score":
            component_scores[
                "transaction_growth"
            ],

        "new_holders_growth_score":
            component_scores[
                "new_holders_growth"
            ],

        "stablecoin_inflows_score":
            component_scores[
                "stablecoin_inflows"
            ],

        "exchange_netflow_score":
            component_scores[
                "exchange_netflow"
            ],

        "whale_accumulation_score":
            component_scores[
                "whale_accumulation"
            ],

        "network_usage_growth_score":
            component_scores[
                "network_usage_growth"
            ],

        "onchain_change":
            onchain_change,

        "onchain_score":
            onchain_score,

        "onchain_price_divergence":
            divergence,

        "onchain_price_divergence_score":
            divergence_score,

        "onchain_capital_confirmation":
            capital_confirmation,

        "onchain_status":
            classify_onchain_status(
                onchain_score
            ),

        "onchain_data_completeness":
            completeness,

        "onchain_data_quality":
            classify_onchain_data_quality(
                completeness
            ),
    }


# ============================================================
# DATAFRAME ENGINE
# ============================================================

def run_onchain_engine(
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
            "[onchain_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_onchain(
                row
            )
        )

    evaluation_df = pd.DataFrame(
        evaluations,
        index=result.index,
    )

    for column in (
        evaluation_df.columns
    ):

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

    from data.onchain_data import (
        enrich_with_onchain_data,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=30,
    )

    market = enrich_with_onchain_data(
        market
    )

    result = run_onchain_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "active_addresses_growth_score",
        "transaction_growth_score",
        "new_holders_growth_score",
        "stablecoin_inflows_score",
        "exchange_netflow_score",
        "whale_accumulation_score",
        "network_usage_growth_score",
        "onchain_change",
        "onchain_score",
        "onchain_price_divergence",
        "onchain_price_divergence_score",
        "onchain_capital_confirmation",
        "onchain_status",
        "onchain_data_completeness",
        "onchain_data_quality",
    ]

    available = [
        column
        for column in columns
        if column in result.columns
    ]

    if "onchain_score" in result.columns:

        result = result.sort_values(
            "onchain_score",
            ascending=False,
            na_position="last",
        )

    print(
        result[
            available
        ]
        .head(30)
        .to_string(
            index=False
        )
    )
