"""
CRYPTO OPPORTUNITY ENGINE
Narrative Engine

OBJETIVO:
Medir narrativas que possuem confirmação econômica e adoção real.

Princípio:
Narrativa é apenas um bônus de 5% no Opportunity Score.

Narrativa NÃO pode:
- aprovar projeto ruim;
- compensar fundamentos fracos;
- compensar receita inexistente;
- compensar diluição crítica;
- compensar risco crítico.

O motor prioriza:
- adoção real;
- crescimento de capital;
- adoção institucional;
- crescimento do setor.

Narrativas monitoradas:
- RWA;
- Tokenization;
- Tokenized Stocks;
- Tokenized Treasuries;
- DeFi;
- L1;
- L2;
- DePIN;
- AI;
- Oracle;
- Interoperability.

RWA é medido por atividade real, não por hype.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

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


def _normalize_text(
    value,
) -> str:

    if value is None:
        return ""

    return (
        str(value)
        .upper()
        .strip()
    )


# ============================================================
# GROWTH SCORE
# ============================================================

def score_growth(
    growth: float,
    lower: float = -0.30,
    upper: float = 1.00,
) -> float:

    if pd.isna(growth):
        return np.nan

    if growth <= lower:
        return 0.0

    if growth >= upper:
        return 100.0

    return _clip_score(
        (
            growth - lower
        )
        / (
            upper - lower
        )
        * 100
    )


# ============================================================
# IDENTIFICAÇÃO DE NARRATIVA
# ============================================================

def detect_narratives(
    row: pd.Series,
) -> List[str]:

    detected = []

    category = _normalize_text(
        row.get(
            "category"
        )
    )

    narrative = _normalize_text(
        row.get(
            "narrative"
        )
    )

    sector = _normalize_text(
        row.get(
            "sector"
        )
    )

    tags = _normalize_text(
        row.get(
            "tags"
        )
    )

    combined = " ".join(
        [
            category,
            narrative,
            sector,
            tags,
        ]
    )

    mapping = {
        "RWA": [
            "RWA",
            "REAL WORLD ASSET",
            "REAL-WORLD ASSET",
        ],

        "TOKENIZATION": [
            "TOKENIZATION",
            "TOKENISATION",
            "TOKENIZED ASSET",
            "TOKENISED ASSET",
        ],

        "TOKENIZED_STOCKS": [
            "TOKENIZED STOCK",
            "TOKENIZED EQUITY",
            "TOKENISED STOCK",
            "TOKENISED EQUITY",
        ],

        "TOKENIZED_TREASURIES": [
            "TOKENIZED TREASURY",
            "TOKENIZED TREASURIES",
            "TOKENISED TREASURY",
            "TREASURY TOKENIZATION",
        ],

        "DEFI": [
            "DEFI",
            "DEX",
            "LENDING",
            "LIQUID STAKING",
            "YIELD",
        ],

        "L1": [
            "L1",
            "LAYER 1",
            "LAYER-1",
            "LAYER1",
        ],

        "L2": [
            "L2",
            "LAYER 2",
            "LAYER-2",
            "LAYER2",
            "ROLLUP",
        ],

        "DEPIN": [
            "DEPIN",
            "DECENTRALIZED PHYSICAL",
        ],

        "AI": [
            "AI",
            "ARTIFICIAL INTELLIGENCE",
            "MACHINE LEARNING",
        ],

        "ORACLE": [
            "ORACLE",
        ],

        "INTEROPERABILITY": [
            "INTEROPERABILITY",
            "CROSS-CHAIN",
            "CROSS CHAIN",
        ],
    }

    for narrative_name, keywords in mapping.items():

        if any(
            keyword in combined
            for keyword in keywords
        ):

            detected.append(
                narrative_name
            )

    if (
        "RWA" in detected
        and "TOKENIZATION"
        not in detected
    ):
        detected.append(
            "TOKENIZATION"
        )

    valid = []

    for item in detected:

        if (
            item
            in config.TRACKED_NARRATIVES
            and item not in valid
        ):
            valid.append(
                item
            )

    return valid


# ============================================================
# REAL ADOPTION
# ============================================================

def calculate_real_adoption(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "narrative_real_adoption_score",
            "real_adoption_score",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    values = []

    metrics = [
        _first_available(
            row,
            [
                "users_growth_90d",
                "active_addresses_growth_90d",
                "active_addresses_growth",
            ],
        ),

        _first_available(
            row,
            [
                "transactions_growth_90d",
                "transaction_growth_90d",
                "transaction_growth",
            ],
        ),

        _first_available(
            row,
            [
                "revenue_growth_90d",
                "revenue_growth",
            ],
        ),

        _first_available(
            row,
            [
                "fees_growth_90d",
                "fees_growth",
            ],
        ),

        _first_available(
            row,
            [
                "tvl_growth_90d",
                "chain_tvl_growth_90d",
            ],
        ),
    ]

    for metric in metrics:

        if not pd.isna(metric):

            values.append(
                score_growth(
                    metric
                )
            )

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
# CAPITAL GROWTH
# ============================================================

def calculate_capital_growth(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "narrative_capital_growth_score",
            "capital_growth_score",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    values = []

    metrics = [
        _first_available(
            row,
            [
                "stablecoin_growth_90d",
                "stablecoin_supply_growth_90d",
                "stablecoin_inflows",
            ],
        ),

        _first_available(
            row,
            [
                "tvl_growth_90d",
                "chain_tvl_growth_90d",
            ],
        ),

        _first_available(
            row,
            [
                "daily_volume_growth",
                "volume_growth_30d",
                "volume_growth_90d",
            ],
        ),
    ]

    for metric in metrics:

        if not pd.isna(metric):

            values.append(
                score_growth(
                    metric
                )
            )

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
# INSTITUTIONAL ADOPTION
# ============================================================

def calculate_institutional_adoption(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "institutional_adoption_score",
            "institutional_adoption",
        ],
    )

    if not pd.isna(explicit):

        if explicit <= 1:
            explicit *= 100

        return _clip_score(
            explicit
        )

    integrations = _first_available(
        row,
        [
            "institutional_integrations",
            "institutional_partnerships",
        ],
    )

    if pd.isna(integrations):
        return np.nan

    if integrations <= 0:
        return 0.0

    if integrations >= 10:
        return 100.0

    return _clip_score(
        integrations
        / 10
        * 100
    )


# ============================================================
# SECTOR GROWTH
# ============================================================

def calculate_sector_growth(
    row: pd.Series,
) -> float:

    explicit_score = _first_available(
        row,
        [
            "sector_growth_score",
            "narrative_sector_growth_score",
        ],
    )

    if not pd.isna(explicit_score):

        if explicit_score <= 1:
            explicit_score *= 100

        return _clip_score(
            explicit_score
        )

    growth = _first_available(
        row,
        [
            "sector_growth_90d",
            "category_growth_90d",
            "narrative_growth_90d",
        ],
    )

    if pd.isna(growth):
        return np.nan

    return score_growth(
        growth
    )


# ============================================================
# RWA
# ============================================================

def calculate_rwa_score(
    row: pd.Series,
) -> float:

    metrics = {
        "tokenized_assets_growth":
            _first_available(
                row,
                [
                    "tokenized_assets_growth",
                    "tokenized_assets_growth_90d",
                ],
            ),

        "tokenized_equities_growth":
            _first_available(
                row,
                [
                    "tokenized_equities_growth",
                    "tokenized_equities_growth_90d",
                ],
            ),

        "tokenized_treasuries_growth":
            _first_available(
                row,
                [
                    "tokenized_treasuries_growth",
                    "tokenized_treasuries_growth_90d",
                ],
            ),

        "rwa_tvl_growth":
            _first_available(
                row,
                [
                    "rwa_tvl_growth",
                    "rwa_tvl_growth_90d",
                ],
            ),

        "rwa_volume_growth":
            _first_available(
                row,
                [
                    "rwa_volume_growth",
                    "rwa_volume_growth_90d",
                ],
            ),

        "rwa_users_growth":
            _first_available(
                row,
                [
                    "rwa_users_growth",
                    "rwa_users_growth_90d",
                ],
            ),

        "institutional_integrations":
            _first_available(
                row,
                [
                    "institutional_integrations",
                ],
            ),

        "protocol_revenue_from_rwa":
            _first_available(
                row,
                [
                    "protocol_revenue_from_rwa",
                    "rwa_revenue_growth",
                ],
            ),
    }

    scores = []

    for name, value in metrics.items():

        if pd.isna(value):
            continue

        if name == "institutional_integrations":

            if value <= 0:
                score = 0.0

            elif value >= 10:
                score = 100.0

            else:
                score = (
                    value
                    / 10
                    * 100
                )

        elif (
            name
            == "protocol_revenue_from_rwa"
            and value > 1
        ):

            score = min(
                100.0,
                50.0
                + np.log10(
                    max(
                        value,
                        1,
                    )
                )
                * 10,
            )

        else:

            score = score_growth(
                value
            )

        if not pd.isna(score):
            scores.append(
                score
            )

    if not scores:
        return np.nan

    return round(
        float(
            np.mean(
                scores
            )
        ),
        2,
    )


# ============================================================
# NARRATIVE SCORE
# ============================================================

def calculate_narrative_score(
    row: pd.Series,
) -> float:

    narratives = (
        detect_narratives(
            row
        )
    )

    if not narratives:
        return 50.0

    real_adoption = (
        calculate_real_adoption(
            row
        )
    )

    capital_growth = (
        calculate_capital_growth(
            row
        )
    )

    institutional = (
        calculate_institutional_adoption(
            row
        )
    )

    sector_growth = (
        calculate_sector_growth(
            row
        )
    )

    components = {
        "real_adoption":
            (
                real_adoption,
                config.NARRATIVE_WEIGHTS[
                    "real_adoption"
                ],
            ),

        "capital_growth":
            (
                capital_growth,
                config.NARRATIVE_WEIGHTS[
                    "capital_growth"
                ],
            ),

        "institutional_adoption":
            (
                institutional,
                config.NARRATIVE_WEIGHTS[
                    "institutional_adoption"
                ],
            ),

        "sector_growth":
            (
                sector_growth,
                config.NARRATIVE_WEIGHTS[
                    "sector_growth"
                ],
            ),
    }

    base_score = (
        _weighted_average(
            components
        )
    )

    rwa_score = np.nan

    if any(
        narrative in {
            "RWA",
            "TOKENIZATION",
            "TOKENIZED_STOCKS",
            "TOKENIZED_TREASURIES",
        }
        for narrative in narratives
    ):

        rwa_score = (
            calculate_rwa_score(
                row
            )
        )

    if (
        not pd.isna(rwa_score)
        and not pd.isna(base_score)
    ):

        final_score = (
            base_score
            * 0.70
            + rwa_score
            * 0.30
        )

    elif not pd.isna(rwa_score):

        final_score = rwa_score

    elif not pd.isna(base_score):

        final_score = base_score

    else:

        return 50.0

    return round(
        _clip_score(
            final_score
        ),
        2,
    )


# ============================================================
# NARRATIVE CONFIRMATION
# ============================================================

def narrative_has_real_confirmation(
    row: pd.Series,
    score: float,
) -> bool:

    if pd.isna(score):
        return False

    real_adoption = (
        calculate_real_adoption(
            row
        )
    )

    capital_growth = (
        calculate_capital_growth(
            row
        )
    )

    confirmations = []

    if not pd.isna(
        real_adoption
    ):
        confirmations.append(
            real_adoption >= 55
        )

    if not pd.isna(
        capital_growth
    ):
        confirmations.append(
            capital_growth >= 55
        )

    if not confirmations:
        return False

    return (
        score >= 55
        and any(confirmations)
    )


# ============================================================
# STATUS
# ============================================================

def classify_narrative_status(
    score: float,
    confirmed: bool,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= 80 and confirmed:
        return "STRONG_CONFIRMED"

    if score >= 65 and confirmed:
        return "CONFIRMED"

    if score >= 55:
        return "EMERGING"

    if score >= 40:
        return "NEUTRAL"

    return "WEAK"


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_narrative_data_completeness(
    row: pd.Series,
) -> float:

    values = [
        calculate_real_adoption(
            row
        ),

        calculate_capital_growth(
            row
        ),

        calculate_institutional_adoption(
            row
        ),

        calculate_sector_growth(
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

def evaluate_narrative(
    row: pd.Series,
) -> Dict:

    narratives = (
        detect_narratives(
            row
        )
    )

    real_adoption = (
        calculate_real_adoption(
            row
        )
    )

    capital_growth = (
        calculate_capital_growth(
            row
        )
    )

    institutional = (
        calculate_institutional_adoption(
            row
        )
    )

    sector_growth = (
        calculate_sector_growth(
            row
        )
    )

    rwa_score = np.nan

    if any(
        narrative in {
            "RWA",
            "TOKENIZATION",
            "TOKENIZED_STOCKS",
            "TOKENIZED_TREASURIES",
        }
        for narrative in narratives
    ):

        rwa_score = (
            calculate_rwa_score(
                row
            )
        )

    score = (
        calculate_narrative_score(
            row
        )
    )

    confirmed = (
        narrative_has_real_confirmation(
            row,
            score,
        )
    )

    return {
        "detected_narratives":
            "|".join(
                narratives
            ),

        "narrative_real_adoption_score":
            real_adoption,

        "narrative_capital_growth_score":
            capital_growth,

        "institutional_adoption_score":
            institutional,

        "sector_growth_score":
            sector_growth,

        "rwa_narrative_score":
            rwa_score,

        "narrative_score":
            score,

        "narrative_confirmed":
            confirmed,

        "narrative_status":
            classify_narrative_status(
                score,
                confirmed,
            ),

        "narrative_data_completeness":
            calculate_narrative_data_completeness(
                row
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_narrative_engine(
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
            "[narrative_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_narrative(
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

    result = run_narrative_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "category",
        "detected_narratives",
        "narrative_real_adoption_score",
        "narrative_capital_growth_score",
        "institutional_adoption_score",
        "sector_growth_score",
        "rwa_narrative_score",
        "narrative_score",
        "narrative_confirmed",
        "narrative_status",
        "narrative_data_completeness",
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
            "narrative_score",
            ascending=False,
            na_position="last",
        )
        .head(30)
        .to_string(
            index=False
        )
    )
