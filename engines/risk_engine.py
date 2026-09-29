"""
CRYPTO OPPORTUNITY ENGINE
Risk Engine

OBJETIVO:
Criar uma camada independente de risco.

Princípio:
RISCO NÃO É BÔNUS NEM DESCONTO NO OPPORTUNITY SCORE.

O Opportunity Score mede oportunidade.
O Risk Engine determina se a oportunidade pode ou não ser aceita.

Analisa:
- smart contract;
- bridge;
- centralização;
- governança;
- liquidez;
- concentração de holders;
- histórico de segurança;
- dependência de oracle;
- risco regulatório;
- token unlock/diluição.

REGRA:
Risco CRITICAL pode bloquear o ativo independentemente
do Opportunity Score.

Dados ausentes não são interpretados como risco baixo.
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


def _first_text(
    row: pd.Series,
    fields: Iterable[str],
) -> str:

    for field in fields:

        if field not in row.index:
            continue

        value = row.get(field)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def _normalize_risk_score(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    if 0 <= value <= 1:
        value *= 100

    return _clip_score(
        value
    )


def _risk_text_to_score(
    value: str,
) -> float:

    if not value:
        return np.nan

    text = (
        str(value)
        .upper()
        .strip()
    )

    mapping = {
        "LOW": 15.0,
        "VERY_LOW": 5.0,
        "MODERATE": 45.0,
        "MEDIUM": 45.0,
        "HIGH": 75.0,
        "CRITICAL": 100.0,
    }

    return mapping.get(
        text,
        np.nan,
    )


# ============================================================
# GENERIC RISK EXTRACTION
# ============================================================

def extract_risk_factor(
    row: pd.Series,
    factor: str,
    aliases: Iterable[str] = (),
) -> float:

    fields = [
        f"{factor}_risk_score",
        f"{factor}_risk",
    ]

    fields.extend(
        aliases
    )

    numeric = _first_available(
        row,
        fields,
    )

    if not pd.isna(numeric):
        return _normalize_risk_score(
            numeric
        )

    text = _first_text(
        row,
        fields,
    )

    return _risk_text_to_score(
        text
    )


# ============================================================
# SMART CONTRACT
# ============================================================

def calculate_smart_contract_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "smart_contract",
        [
            "contract_risk_score",
        ],
    )

    if not pd.isna(direct):
        return direct

    audits = _first_available(
        row,
        [
            "security_audits",
            "audit_count",
        ],
    )

    if pd.isna(audits):
        return np.nan

    if audits >= 3:
        return 15.0

    if audits >= 2:
        return 25.0

    if audits >= 1:
        return 40.0

    return 80.0


# ============================================================
# BRIDGE
# ============================================================

def calculate_bridge_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "bridge",
    )

    if not pd.isna(direct):
        return direct

    bridge_dependency = _first_available(
        row,
        [
            "bridge_dependency",
            "bridge_dependency_ratio",
        ],
    )

    if pd.isna(bridge_dependency):
        return np.nan

    if bridge_dependency <= 1:
        bridge_dependency *= 100

    return _clip_score(
        bridge_dependency
    )


# ============================================================
# CENTRALIZATION
# ============================================================

def calculate_centralization_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "centralization",
    )

    if not pd.isna(direct):
        return direct

    insider = _first_available(
        row,
        [
            "insider_concentration",
            "team_vc_allocation",
        ],
    )

    if pd.isna(insider):
        return np.nan

    if insider <= 1:
        insider *= 100

    return _clip_score(
        insider
    )


# ============================================================
# GOVERNANCE
# ============================================================

def calculate_governance_risk(
    row: pd.Series,
) -> float:

    return extract_risk_factor(
        row,
        "governance",
        [
            "governance_concentration_risk",
        ],
    )


# ============================================================
# LIQUIDITY
# ============================================================

def calculate_liquidity_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "liquidity",
    )

    if not pd.isna(direct):
        return direct

    market_cap = _safe_float(
        row.get(
            "market_cap"
        )
    )

    volume = _safe_float(
        row.get(
            "daily_volume"
        )
    )

    if (
        pd.isna(market_cap)
        or pd.isna(volume)
        or market_cap <= 0
    ):
        return np.nan

    ratio = (
        volume
        / market_cap
    )

    if ratio >= 0.20:
        return 10.0

    if ratio >= 0.10:
        return 20.0

    if ratio >= 0.05:
        return 35.0

    if ratio >= 0.02:
        return 55.0

    if ratio >= 0.01:
        return 75.0

    return 95.0


# ============================================================
# HOLDER CONCENTRATION
# ============================================================

def calculate_holder_concentration_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "holder_concentration",
    )

    if not pd.isna(direct):
        return direct

    concentration = _first_available(
        row,
        [
            "top_10_holder_ratio",
            "top10_holder_ratio",
            "insider_concentration",
        ],
    )

    if pd.isna(concentration):
        return np.nan

    if concentration <= 1:
        concentration *= 100

    return _clip_score(
        concentration
    )


# ============================================================
# SECURITY HISTORY
# ============================================================

def calculate_security_history_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "security_history",
    )

    if not pd.isna(direct):
        return direct

    exploits = _first_available(
        row,
        [
            "historical_exploits",
            "exploit_count",
            "security_incidents",
        ],
    )

    if pd.isna(exploits):
        return np.nan

    if exploits <= 0:
        return 10.0

    if exploits == 1:
        return 50.0

    if exploits == 2:
        return 75.0

    return 100.0


# ============================================================
# ORACLE DEPENDENCY
# ============================================================

def calculate_oracle_dependency_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "oracle_dependency",
    )

    if not pd.isna(direct):
        return direct

    dependency = _first_available(
        row,
        [
            "oracle_dependency_ratio",
        ],
    )

    if pd.isna(dependency):
        return np.nan

    if dependency <= 1:
        dependency *= 100

    return _clip_score(
        dependency
    )


# ============================================================
# REGULATORY
# ============================================================

def calculate_regulatory_risk(
    row: pd.Series,
) -> float:

    return extract_risk_factor(
        row,
        "regulatory",
        [
            "regulatory_risk_score",
        ],
    )


# ============================================================
# TOKEN UNLOCK
# ============================================================

def calculate_token_unlock_risk(
    row: pd.Series,
) -> float:

    direct = extract_risk_factor(
        row,
        "token_unlock",
    )

    if not pd.isna(direct):
        return direct

    dilution_risk = _first_text(
        row,
        [
            "dilution_risk",
        ],
    )

    mapped = _risk_text_to_score(
        dilution_risk
    )

    if not pd.isna(mapped):
        return mapped

    dilution_score = _first_available(
        row,
        [
            "dilution_score",
        ],
    )

    if not pd.isna(dilution_score):

        return _clip_score(
            100
            - dilution_score
        )

    unlock_90d = _first_available(
        row,
        [
            "unlock_90d",
        ],
    )

    if pd.isna(unlock_90d):
        return np.nan

    if unlock_90d > 1:
        unlock_90d /= 100

    if unlock_90d <= 0.025:
        return 10.0

    if unlock_90d <= 0.05:
        return 25.0

    if unlock_90d <= 0.10:
        return 45.0

    if unlock_90d <= 0.20:
        return 75.0

    return 100.0


# ============================================================
# RISK COMPONENTS
# ============================================================

def calculate_risk_components(
    row: pd.Series,
) -> Dict[str, float]:

    return {
        "smart_contract":
            calculate_smart_contract_risk(
                row
            ),

        "bridge":
            calculate_bridge_risk(
                row
            ),

        "centralization":
            calculate_centralization_risk(
                row
            ),

        "governance":
            calculate_governance_risk(
                row
            ),

        "liquidity":
            calculate_liquidity_risk(
                row
            ),

        "holder_concentration":
            calculate_holder_concentration_risk(
                row
            ),

        "security_history":
            calculate_security_history_risk(
                row
            ),

        "oracle_dependency":
            calculate_oracle_dependency_risk(
                row
            ),

        "regulatory":
            calculate_regulatory_risk(
                row
            ),

        "token_unlock":
            calculate_token_unlock_risk(
                row
            ),
    }


# ============================================================
# AGGREGATE RISK
# ============================================================

def calculate_risk_score(
    row: pd.Series,
) -> float:

    components = (
        calculate_risk_components(
            row
        )
    )

    available = [
        value
        for value in components.values()
        if not pd.isna(value)
    ]

    if not available:
        return np.nan

    average_risk = float(
        np.mean(
            available
        )
    )

    max_risk = float(
        np.max(
            available
        )
    )

    # O pior risco recebe peso adicional para impedir
    # que riscos críticos sejam diluídos pela média.
    aggregate = (
        average_risk
        * 0.65
        + max_risk
        * 0.35
    )

    return round(
        _clip_score(
            aggregate
        ),
        2,
    )


# ============================================================
# CRITICAL FACTORS
# ============================================================

def detect_critical_risk_factors(
    row: pd.Series,
) -> List[str]:

    components = (
        calculate_risk_components(
            row
        )
    )

    critical = []

    for factor, value in components.items():

        if (
            not pd.isna(value)
            and value >= 90
        ):
            critical.append(
                factor
            )

    dilution_hard_block = row.get(
        "dilution_hard_block",
        False,
    )

    if (
        dilution_hard_block is True
        and "token_unlock"
        not in critical
    ):
        critical.append(
            "token_unlock"
        )

    return critical


# ============================================================
# HIGH RISK FACTORS
# ============================================================

def detect_high_risk_factors(
    row: pd.Series,
) -> List[str]:

    components = (
        calculate_risk_components(
            row
        )
    )

    high = []

    for factor, value in components.items():

        if (
            not pd.isna(value)
            and 70 <= value < 90
        ):
            high.append(
                factor
            )

    return high


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk_level(
    row: pd.Series,
    risk_score: float,
) -> str:

    critical = (
        detect_critical_risk_factors(
            row
        )
    )

    if critical:
        return "CRITICAL"

    if pd.isna(risk_score):
        return "DATA_INSUFFICIENT"

    if risk_score >= 70:
        return "HIGH"

    if risk_score >= 40:
        return "MODERATE"

    return "LOW"


# ============================================================
# HARD BLOCK
# ============================================================

def risk_hard_block(
    risk_level: str,
) -> bool:

    if not config.BLOCK_CRITICAL_RISK:
        return False

    return (
        risk_level
        == "CRITICAL"
    )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def calculate_risk_data_completeness(
    row: pd.Series,
) -> float:

    components = (
        calculate_risk_components(
            row
        )
    )

    expected_factors = list(
        config.RISK_FACTORS
    )

    available = sum(
        1
        for factor in expected_factors
        if (
            factor in components
            and not pd.isna(
                components[
                    factor
                ]
            )
        )
    )

    if not expected_factors:
        return 0.0

    return round(
        available
        / len(
            expected_factors
        ),
        4,
    )


# ============================================================
# RISK DATA QUALITY
# ============================================================

def classify_risk_data_quality(
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

def evaluate_risk(
    row: pd.Series,
) -> Dict:

    components = (
        calculate_risk_components(
            row
        )
    )

    risk_score = (
        calculate_risk_score(
            row
        )
    )

    critical_factors = (
        detect_critical_risk_factors(
            row
        )
    )

    high_factors = (
        detect_high_risk_factors(
            row
        )
    )

    risk_level = (
        classify_risk_level(
            row,
            risk_score,
        )
    )

    hard_block = (
        risk_hard_block(
            risk_level
        )
    )

    completeness = (
        calculate_risk_data_completeness(
            row
        )
    )

    return {
        "smart_contract_risk":
            components[
                "smart_contract"
            ],

        "bridge_risk":
            components[
                "bridge"
            ],

        "centralization_risk":
            components[
                "centralization"
            ],

        "governance_risk":
            components[
                "governance"
            ],

        "liquidity_risk":
            components[
                "liquidity"
            ],

        "holder_concentration_risk":
            components[
                "holder_concentration"
            ],

        "security_history_risk":
            components[
                "security_history"
            ],

        "oracle_dependency_risk":
            components[
                "oracle_dependency"
            ],

        "regulatory_risk":
            components[
                "regulatory"
            ],

        "token_unlock_risk":
            components[
                "token_unlock"
            ],

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "risk_hard_block":
            hard_block,

        "critical_risk_factors":
            "|".join(
                critical_factors
            ),

        "high_risk_factors":
            "|".join(
                high_factors
            ),

        "risk_data_completeness":
            completeness,

        "risk_data_quality":
            classify_risk_data_quality(
                completeness
            ),
    }


# ============================================================
# DATAFRAME
# ============================================================

def run_risk_engine(
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
            "[risk_engine] "
            f"{symbol} "
            f"({position}/{total})"
        )

        evaluations.append(
            evaluate_risk(
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

    from engines.dilution_engine import (
        run_dilution_engine,
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

    market = run_dilution_engine(
        market
    )

    result = run_risk_engine(
        market
    )

    columns = [
        "symbol",
        "name",
        "risk_score",
        "risk_level",
        "risk_hard_block",
        "critical_risk_factors",
        "high_risk_factors",
        "liquidity_risk",
        "centralization_risk",
        "holder_concentration_risk",
        "token_unlock_risk",
        "risk_data_completeness",
        "risk_data_quality",
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
            "risk_score",
            ascending=True,
            na_position="last",
        )
        .head(50)
        .to_string(
            index=False
        )
    )
