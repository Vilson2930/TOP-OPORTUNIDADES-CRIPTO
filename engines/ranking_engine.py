"""
CRYPTO OPPORTUNITY ENGINE
Ranking Engine

OBJETIVO:
Consolidar todas as camadas do sistema e produzir o ranking final
de oportunidades.

Princípio central:

PROJECT REAL
→ USO REAL
→ RECEITA REAL
→ VALUE CAPTURE
→ DILUIÇÃO CONTROLADA
→ VALUATION ATRATIVO
→ ON-CHAIN CONFIRMADO
→ DIVERGÊNCIA FUNDAMENTOS/PREÇO
→ TIMING
→ DECISÃO

O Opportunity Score mede QUALIDADE + ASSIMETRIA.

Timing é separado.
Risk é separado.
Quality Gates são obrigatórios.

Narrativa nunca pode aprovar um projeto ruim.
Timing nunca pode resgatar um projeto ruim.
Risco crítico pode bloquear o ativo.
Diluição crítica pode bloquear o ativo.
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


def _safe_bool(
    value,
    default: bool = False,
) -> bool:

    if value is None:
        return default

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    if isinstance(
        value,
        (int, float),
    ):

        if pd.isna(value):
            return default

        return bool(value)

    text = (
        str(value)
        .strip()
        .upper()
    )

    if text in {
        "TRUE",
        "YES",
        "Y",
        "1",
        "PASS",
        "PASSED",
    }:
        return True

    if text in {
        "FALSE",
        "NO",
        "N",
        "0",
        "FAIL",
        "FAILED",
    }:
        return False

    return default


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
# SCORE EXTRACTION
# ============================================================

def extract_component_scores(
    row: pd.Series,
) -> Dict[str, float]:

    return {
        "fundamental_acceleration":
            _first_available(
                row,
                [
                    "fundamental_acceleration_score",
                ],
            ),

        "revenue_economics":
            _first_available(
                row,
                [
                    "revenue_score",
                ],
            ),

        "holder_value_capture":
            _first_available(
                row,
                [
                    "holder_value_score",
                ],
            ),

        "valuation":
            _first_available(
                row,
                [
                    "valuation_score",
                ],
            ),

        "tokenomics_dilution":
            _first_available(
                row,
                [
                    "dilution_score",
                ],
            ),

        "onchain_growth":
            _first_available(
                row,
                [
                    "onchain_score",
                    "onchain_opportunity_score",
                ],
            ),

        "tvl_utilization":
            _first_available(
                row,
                [
                    "tvl_score",
                    "tvl_opportunity_score",
                ],
            ),

        "narrative":
            _first_available(
                row,
                [
                    "narrative_score",
                ],
            ),
    }


# ============================================================
# BASE OPPORTUNITY SCORE
# ============================================================

def calculate_base_opportunity_score(
    row: pd.Series,
) -> float:

    scores = extract_component_scores(
        row
    )

    components = {}

    for name, weight in (
        config.OPPORTUNITY_WEIGHTS.items()
    ):

        components[name] = (
            scores.get(
                name,
                np.nan,
            ),
            weight,
        )

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
# FUNDAMENTAL PRICE DIVERGENCE
# ============================================================

def extract_fundamental_change(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "fundamental_change",
            "fundamental_change_90d",
        ],
    )


def extract_revenue_change(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "revenue_change",
            "revenue_growth_90d",
            "revenue_growth",
        ],
    )


def extract_usage_change(
    row: pd.Series,
) -> float:

    values = [
        _first_available(
            row,
            [
                "users_growth_90d",
                "active_addresses_growth",
            ],
        ),

        _first_available(
            row,
            [
                "transactions_growth_90d",
                "transaction_growth",
            ],
        ),

        _first_available(
            row,
            [
                "network_usage_growth",
            ],
        ),
    ]

    available = [
        value
        for value in values
        if not pd.isna(value)
    ]

    if not available:
        return np.nan

    return float(
        np.median(
            available
        )
    )


def extract_tvl_change(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "tvl_growth_90d",
            "chain_tvl_growth_90d",
        ],
    )


def extract_onchain_change(
    row: pd.Series,
) -> float:

    return _first_available(
        row,
        [
            "onchain_change",
        ],
    )


def extract_price_underreaction(
    row: pd.Series,
) -> float:

    explicit = _first_available(
        row,
        [
            "price_underreaction",
            "price_underreaction_90d",
        ],
    )

    if not pd.isna(explicit):
        return explicit

    price_return = _first_available(
        row,
        [
            "return_90d",
            "price_change_90d",
        ],
    )

    fundamental_change = (
        extract_fundamental_change(
            row
        )
    )

    if (
        pd.isna(price_return)
        or pd.isna(fundamental_change)
    ):
        return np.nan

    return (
        fundamental_change
        - price_return
    )


# ============================================================
# DIVERGENCE NORMALIZATION
# ============================================================

def score_positive_change(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    lower = -0.40
    upper = 0.80

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


def score_price_underreaction(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    lower = -0.50
    upper = 1.00

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
# DIVERGENCE SCORE
# ============================================================

def calculate_fundamental_price_divergence_score(
    row: pd.Series,
) -> float:

    if not config.DIVERGENCE_ENABLED:
        return np.nan

    fundamental_change = (
        extract_fundamental_change(
            row
        )
    )

    revenue_change = (
        extract_revenue_change(
            row
        )
    )

    usage_change = (
        extract_usage_change(
            row
        )
    )

    tvl_change = (
        extract_tvl_change(
            row
        )
    )

    onchain_change = (
        extract_onchain_change(
            row
        )
    )

    price_underreaction = (
        extract_price_underreaction(
            row
        )
    )

    components = {
        "fundamental_change":
            (
                score_positive_change(
                    fundamental_change
                ),
                config.DIVERGENCE_WEIGHTS[
                    "fundamental_change"
                ],
            ),

        "revenue_change":
            (
                score_positive_change(
                    revenue_change
                ),
                config.DIVERGENCE_WEIGHTS[
                    "revenue_change"
                ],
            ),

        "usage_change":
            (
                score_positive_change(
                    usage_change
                ),
                config.DIVERGENCE_WEIGHTS[
                    "usage_change"
                ],
            ),

        "tvl_change":
            (
                score_positive_change(
                    tvl_change
                ),
                config.DIVERGENCE_WEIGHTS[
                    "tvl_change"
                ],
            ),

        "onchain_change":
            (
                score_positive_change(
                    onchain_change
                ),
                config.DIVERGENCE_WEIGHTS[
                    "onchain_change"
                ],
            ),

        "price_underreaction":
            (
                score_price_underreaction(
                    price_underreaction
                ),
                config.DIVERGENCE_WEIGHTS[
                    "price_underreaction"
                ],
            ),
    }

    score = _weighted_average(
        components
    )

    if pd.isna(score):

        fallback = _first_available(
            row,
            [
                "fundamental_price_divergence_score",
                "fundamental_divergence_score",
            ],
        )

        if pd.isna(fallback):
            return np.nan

        return round(
            _clip_score(
                fallback
            ),
            2,
        )

    return round(
        _clip_score(score),
        2,
    )


# ============================================================
# DIVERGENCE CLASSIFICATION
# ============================================================

def classify_divergence(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= config.EXTREME_DIVERGENCE_SCORE:
        return "EXTREME"

    if score >= config.STRONG_DIVERGENCE_SCORE:
        return "STRONG"

    if score >= config.MIN_POSITIVE_DIVERGENCE_SCORE:
        return "POSITIVE"

    if score >= 40:
        return "NEUTRAL"

    return "NEGATIVE"


# ============================================================
# OPPORTUNITY SCORE
# ============================================================

def calculate_opportunity_score(
    row: pd.Series,
) -> float:

    base_score = (
        calculate_base_opportunity_score(
            row
        )
    )

    if pd.isna(base_score):
        return np.nan

    divergence_score = (
        calculate_fundamental_price_divergence_score(
            row
        )
    )

    if (
        not config.DIVERGENCE_ENABLED
        or pd.isna(divergence_score)
    ):
        return round(
            base_score,
            2,
        )

    # Divergência é sinal central de assimetria,
    # mas não substitui a qualidade estrutural.
    final_score = (
        base_score
        * 0.85
        + divergence_score
        * 0.15
    )

    return round(
        _clip_score(
            final_score
        ),
        2,
    )


# ============================================================
# SCORE COMPLETENESS
# ============================================================

def calculate_opportunity_data_completeness(
    row: pd.Series,
) -> float:

    scores = (
        extract_component_scores(
            row
        )
    )

    total_weight = sum(
        config.OPPORTUNITY_WEIGHTS.values()
    )

    available_weight = 0.0

    for component, weight in (
        config.OPPORTUNITY_WEIGHTS.items()
    ):

        value = scores.get(
            component,
            np.nan,
        )

        if not pd.isna(value):
            available_weight += weight

    if total_weight <= 0:
        return 0.0

    return round(
        available_weight
        / total_weight,
        4,
    )


# ============================================================
# OPPORTUNITY EVIDENCE POLICY
# ============================================================

# Opportunity is the engine's central objective. Missing data is NOT converted
# to zero because absence of coverage is not evidence of poor economics.
# However, a high score calculated from only a small fraction of the canonical
# model must not be treated as a fully qualified opportunity.
#
# 65% preserves discovery capacity while requiring that most of the weighted
# opportunity model is actually observed before the Opportunity Gate can pass.
MIN_OPPORTUNITY_DATA_COMPLETENESS = 0.65


def opportunity_evidence_gate(
    row: pd.Series,
) -> bool:

    completeness = (
        calculate_opportunity_data_completeness(
            row
        )
    )

    return (
        completeness
        >= MIN_OPPORTUNITY_DATA_COMPLETENESS
    )


# ============================================================
# QUALITY GATES
# ============================================================

def evaluate_quality_gates(
    row: pd.Series,
    opportunity_score: float,
) -> Dict[str, bool]:

    fundamental_score = (
        _first_available(
            row,
            [
                "fundamental_acceleration_score",
            ],
        )
    )

    revenue_score = (
        _first_available(
            row,
            [
                "revenue_score",
            ],
        )
    )

    holder_value_score = (
        _first_available(
            row,
            [
                "holder_value_score",
            ],
        )
    )

    dilution_score = (
        _first_available(
            row,
            [
                "dilution_score",
            ],
        )
    )

    return {
        "opportunity_gate":
            (
                not pd.isna(
                    opportunity_score
                )
                and opportunity_score
                >= config.MIN_OPPORTUNITY_SCORE
                and opportunity_evidence_gate(
                    row
                )
            ),

        "fundamental_gate":
            (
                not pd.isna(
                    fundamental_score
                )
                and fundamental_score
                >= config.MIN_FUNDAMENTAL_SCORE
            ),

        "revenue_gate":
            (
                not pd.isna(
                    revenue_score
                )
                and revenue_score
                >= config.MIN_REVENUE_SCORE
            ),

        "holder_value_gate":
            (
                not pd.isna(
                    holder_value_score
                )
                and holder_value_score
                >= config.MIN_HOLDER_VALUE_SCORE
            ),

        "dilution_gate":
            (
                not pd.isna(
                    dilution_score
                )
                and dilution_score
                >= config.MIN_DILUTION_SCORE
                and not _safe_bool(
                    row.get(
                        "dilution_hard_block"
                    ),
                    False,
                )
            ),
    }


def calculate_quality_gate(
    gates: Dict[str, bool],
) -> bool:

    if not gates:
        return False

    return all(
        gates.values()
    )


# ============================================================
# BLOCKS
# ============================================================

def detect_hard_block(
    row: pd.Series,
) -> bool:

    dilution_block = _safe_bool(
        row.get(
            "dilution_hard_block"
        ),
        False,
    )

    risk_block = _safe_bool(
        row.get(
            "risk_hard_block"
        ),
        False,
    )

    risk_level = (
        str(
            row.get(
                "risk_level",
                "",
            )
        )
        .upper()
        .strip()
    )

    if (
        config.BLOCK_CRITICAL_RISK
        and risk_level == "CRITICAL"
    ):
        risk_block = True

    return (
        dilution_block
        or risk_block
    )


def hard_block_reasons(
    row: pd.Series,
) -> List[str]:

    reasons = []

    if _safe_bool(
        row.get(
            "dilution_hard_block"
        ),
        False,
    ):
        reasons.append(
            "CRITICAL_DILUTION"
        )

    if _safe_bool(
        row.get(
            "risk_hard_block"
        ),
        False,
    ):
        reasons.append(
            "CRITICAL_RISK"
        )

    risk_level = (
        str(
            row.get(
                "risk_level",
                "",
            )
        )
        .upper()
        .strip()
    )

    if (
        risk_level == "CRITICAL"
        and "CRITICAL_RISK"
        not in reasons
    ):
        reasons.append(
            "CRITICAL_RISK"
        )

    return reasons


# ============================================================
# QUALITY FAILURE REASONS
# ============================================================

def quality_failure_reasons(
    gates: Dict[str, bool],
    row: pd.Series = None,
    opportunity_score: float = np.nan,
) -> List[str]:

    mapping = {
        "opportunity_gate":
            "LOW_OPPORTUNITY_SCORE",

        "fundamental_gate":
            "WEAK_FUNDAMENTALS",

        "revenue_gate":
            "WEAK_REVENUE",

        "holder_value_gate":
            "WEAK_HOLDER_VALUE_CAPTURE",

        "dilution_gate":
            "WEAK_DILUTION_PROFILE",
    }

    reasons = []

    for gate, passed in gates.items():

        if not passed:

            if (
                gate == "opportunity_gate"
                and row is not None
                and not pd.isna(opportunity_score)
                and opportunity_score
                >= config.MIN_OPPORTUNITY_SCORE
                and not opportunity_evidence_gate(row)
            ):
                reasons.append(
                    "INSUFFICIENT_OPPORTUNITY_EVIDENCE"
                )
                continue

            reasons.append(
                mapping.get(
                    gate,
                    gate.upper(),
                )
            )

    return reasons


# ============================================================
# DECISION SIGNAL
# ============================================================

def determine_signal(
    row: pd.Series,
    opportunity_score: float,
    quality_gate: bool,
    hard_block: bool,
) -> str:

    if hard_block:
        return config.SIGNAL_BLOCKED

    if not quality_gate:
        return config.SIGNAL_REJECTED

    timing_score = _first_available(
        row,
        [
            "timing_score",
        ],
    )

    if pd.isna(timing_score):
        return config.SIGNAL_WAIT

    strong_rule = (
        config.DECISION_RULES[
            "STRONG_ENTRY"
        ]
    )

    if (
        opportunity_score
        >= strong_rule[
            "min_opportunity"
        ]
        and timing_score
        >= strong_rule[
            "min_timing"
        ]
    ):
        return config.SIGNAL_STRONG_ENTRY

    entry_rule = (
        config.DECISION_RULES[
            "ENTRY"
        ]
    )

    if (
        opportunity_score
        >= entry_rule[
            "min_opportunity"
        ]
        and timing_score
        >= entry_rule[
            "min_timing"
        ]
    ):
        return config.SIGNAL_ENTRY

    watch_rule = (
        config.DECISION_RULES[
            "WATCH"
        ]
    )

    if (
        opportunity_score
        >= watch_rule[
            "min_opportunity"
        ]
        and timing_score
        >= watch_rule[
            "min_timing"
        ]
    ):
        return config.SIGNAL_WATCH

    return config.SIGNAL_WAIT


# ============================================================
# OPPORTUNITY CLASSIFICATION
# ============================================================

def classify_opportunity(
    score: float,
) -> str:

    if pd.isna(score):
        return "DATA_INSUFFICIENT"

    if score >= config.VERY_STRONG_OPPORTUNITY_SCORE:
        return "VERY_STRONG"

    if score >= config.STRONG_OPPORTUNITY_SCORE:
        return "STRONG"

    if score >= config.MIN_OPPORTUNITY_SCORE:
        return "QUALIFIED"

    if score >= 50:
        return "WEAK"

    return "VERY_WEAK"


# ============================================================
# DECISION CONFIDENCE
# ============================================================

def calculate_decision_confidence(
    row: pd.Series,
) -> float:

    completeness_values = []

    fields = [
        "fundamental_data_completeness",
        "revenue_data_completeness",
        "holder_value_data_completeness",
        "dilution_data_completeness",
        "valuation_data_completeness",
        "onchain_data_completeness",
        "tvl_data_completeness",
        "narrative_data_completeness",
        "risk_data_completeness",
        "timing_data_completeness",
    ]

    for field in fields:

        value = _safe_float(
            row.get(
                field
            )
        )

        if pd.isna(value):
            continue

        if value > 1:
            value /= 100

        completeness_values.append(
            float(
                np.clip(
                    value,
                    0,
                    1,
                )
            )
        )

    opportunity_completeness = (
        calculate_opportunity_data_completeness(
            row
        )
    )

    completeness_values.append(
        opportunity_completeness
    )

    if not completeness_values:
        return 0.0

    return round(
        float(
            np.mean(
                completeness_values
            )
        )
        * 100,
        2,
    )


# ============================================================
# INDIVIDUAL EVALUATION
# ============================================================

def evaluate_ranking(
    row: pd.Series,
) -> Dict:

    base_score = (
        calculate_base_opportunity_score(
            row
        )
    )

    divergence_score = (
        calculate_fundamental_price_divergence_score(
            row
        )
    )

    opportunity_score = (
        calculate_opportunity_score(
            row
        )
    )

    gates = (
        evaluate_quality_gates(
            row,
            opportunity_score,
        )
    )

    quality_gate = (
        calculate_quality_gate(
            gates
        )
    )

    hard_block = (
        detect_hard_block(
            row
        )
    )

    signal = (
        determine_signal(
            row,
            opportunity_score,
            quality_gate,
            hard_block,
        )
    )

    block_reasons = (
        hard_block_reasons(
            row
        )
    )

    failed_gates = (
        quality_failure_reasons(
            gates,
            row=row,
            opportunity_score=opportunity_score,
        )
    )

    return {
        "base_opportunity_score":
            base_score,

        "fundamental_price_divergence":
            divergence_score,

        "fundamental_price_divergence_status":
            classify_divergence(
                divergence_score
            ),

        "opportunity_score":
            opportunity_score,

        "opportunity_status":
            classify_opportunity(
                opportunity_score
            ),

        "opportunity_gate":
            gates[
                "opportunity_gate"
            ],

        "opportunity_evidence_gate":
            opportunity_evidence_gate(
                row
            ),

        "fundamental_gate":
            gates[
                "fundamental_gate"
            ],

        "revenue_gate":
            gates[
                "revenue_gate"
            ],

        "holder_value_gate":
            gates[
                "holder_value_gate"
            ],

        "dilution_gate":
            gates[
                "dilution_gate"
            ],

        "quality_gate":
            quality_gate,

        "hard_block":
            hard_block,

        "hard_block_reasons":
            "|".join(
                block_reasons
            ),

        "quality_failure_reasons":
            "|".join(
                failed_gates
            ),

        "signal":
            signal,

        "opportunity_data_completeness":
            calculate_opportunity_data_completeness(
                row
            ),

        "decision_confidence":
            calculate_decision_confidence(
                row
            ),
    }


# ============================================================
# SIGNAL PRIORITY
# ============================================================

def signal_priority(
    signal: str,
) -> int:

    mapping = {
        config.SIGNAL_STRONG_ENTRY: 0,
        config.SIGNAL_ENTRY: 1,
        config.SIGNAL_WATCH: 2,
        config.SIGNAL_WAIT: 3,
        config.SIGNAL_REJECTED: 4,
        config.SIGNAL_BLOCKED: 5,
    }

    return mapping.get(
        signal,
        99,
    )


# ============================================================
# RANKING
# ============================================================

def build_final_ranking(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    result = dataset.copy()

    result[
        "_signal_priority"
    ] = result[
        "signal"
    ].apply(
        signal_priority
    )

    if "timing_score" not in result.columns:
        result[
            "timing_score"
        ] = np.nan

    if "decision_confidence" not in result.columns:
        result[
            "decision_confidence"
        ] = np.nan

    result = result.sort_values(
        by=[
            "hard_block",
            "quality_gate",
            "_signal_priority",
            "opportunity_score",
            "fundamental_price_divergence",
            "timing_score",
            "decision_confidence",
        ],
        ascending=[
            True,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    result = result.drop(
        columns=[
            "_signal_priority",
        ]
    )

    return result


# ============================================================
# TOP OPPORTUNITIES
# ============================================================

def get_top_opportunities(
    dataset: pd.DataFrame,
    limit: int = config.TOP_OPPORTUNITIES,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    qualified = dataset[
        (
            dataset[
                "quality_gate"
            ]
            == True
        )
        &
        (
            dataset[
                "hard_block"
            ]
            == False
        )
    ].copy()

    if qualified.empty:
        return qualified

    qualified = qualified.sort_values(
        by=[
            "opportunity_score",
            "fundamental_price_divergence",
            "timing_score",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        na_position="last",
    )

    qualified = qualified.head(
        limit
    ).reset_index(
        drop=True
    )

    qualified[
        "rank"
    ] = np.arange(
        1,
        len(qualified) + 1,
    )

    return qualified


# ============================================================
# DATAFRAME ENGINE
# ============================================================

def run_ranking_engine(
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
            "[ranking_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_ranking(
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

    result = build_final_ranking(
        result
    )

    return result


# ============================================================
# REPORT VIEW
# ============================================================

def build_report_view(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    if dataset.empty:
        return dataset.copy()

    fields = [
        field
        for field in config.REPORT_FIELDS
        if field in dataset.columns
    ]

    return dataset[
        fields
    ].copy()


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

    from data.tokenomics_data import (
        enrich_with_tokenomics_data,
    )

    from data.developer_data import (
        enrich_with_developer_data,
    )

    from engines.fundamental_engine import (
        run_fundamental_engine,
    )

    from engines.revenue_engine import (
        run_revenue_engine,
    )

    from engines.holder_value_engine import (
        run_holder_value_engine,
    )

    from engines.dilution_engine import (
        run_dilution_engine,
    )

    from engines.tvl_engine import (
        run_tvl_engine,
    )

    from engines.valuation_engine import (
        run_valuation_engine,
    )

    from engines.onchain_engine import (
        run_onchain_engine,
    )

    from engines.narrative_engine import (
        run_narrative_engine,
    )

    from engines.risk_engine import (
        run_risk_engine,
    )

    from engines.timing_engine import (
        run_timing_engine,
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=50,
    )

    market = market.head(
        50
    )

    market = enrich_with_defi_data(
        market
    )

    market = enrich_with_onchain_data(
        market
    )

    market = enrich_with_tokenomics_data(
        market,
        fetch_history=True,
    )

    market = enrich_with_developer_data(
        market
    )

    market = run_fundamental_engine(
        market
    )

    market = run_revenue_engine(
        market
    )

    market = run_holder_value_engine(
        market
    )

    market = run_dilution_engine(
        market
    )

    market = run_tvl_engine(
        market
    )

    market = run_valuation_engine(
        market
    )

    market = run_onchain_engine(
        market
    )

    market = run_narrative_engine(
        market
    )

    market = run_risk_engine(
        market
    )

    market = run_timing_engine(
        market
    )

    result = run_ranking_engine(
        market
    )

    top = get_top_opportunities(
        result
    )

    report = build_report_view(
        top
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "TOP OPORTUNIDADES CRIPTO"
    )

    print(
        "============================================================"
    )

    if report.empty:

        print(
            "Nenhum ativo passou por todos os Quality Gates."
        )

    else:

        print(
            report.to_string(
                index=False
            )
        )
