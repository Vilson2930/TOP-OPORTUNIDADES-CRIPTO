"""
CRYPTO OPPORTUNITY ENGINE
Universe Engine

FOCO:
Construir o universo investível para busca de OPORTUNIDADES.

Este módulo NÃO escolhe vencedores.
Ele remove ativos que não atendem aos requisitos mínimos
para uma análise séria de oportunidade.

Fluxo:
mercado
→ exclusões
→ liquidez
→ market cap
→ qualidade dos dados
→ histórico mínimo
→ universo elegível
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def _bool(value) -> bool:

    if pd.isna(value):
        return False

    return bool(value)


# ============================================================
# EXCLUSÕES ESTRUTURAIS
# ============================================================

def check_structural_exclusion(
    row: pd.Series,
) -> Tuple[bool, List[str]]:

    reasons: List[str] = []

    symbol = str(
        row.get("symbol", "")
    ).upper().strip()

    name = str(
        row.get("name", "")
    ).lower().strip()

    if symbol in config.EXCLUDED_SYMBOLS:

        reasons.append(
            "EXCLUDED_SYMBOL"
        )

    if config.EXCLUDE_STABLECOINS:

        stablecoin_terms = [
            "stablecoin",
            "stable coin",
            "usd stable",
            "dollar stable",
        ]

        stablecoin_symbols = {
            "USDT",
            "USDC",
            "DAI",
            "FDUSD",
            "USDE",
            "USDS",
            "TUSD",
            "USDP",
            "PYUSD",
            "FRAX",
            "LUSD",
            "GUSD",
            "USD0",
            "USD1",
            "USDD",
            "CRVUSD",
            "SUSD",
            "EURC",
            "EURT",
            "EURS",
            "RLUSD",
            "USDG",
            "GHO",
            "USDAI",
            "MUSD",
            "BUSD",
            "FRXUSD",
            "FXUSD",
        }

        if (
            symbol in stablecoin_symbols
            or any(
                term in name
                for term in stablecoin_terms
            )
        ):
            reasons.append(
                "STABLECOIN"
            )

    if config.EXCLUDE_WRAPPED_ASSETS:

        wrapped_symbols = {
            "WBTC",
            "WETH",
            "WBNB",
            "WMATIC",
            "WAVAX",
            "WSOL",
            "CBBTC",
            "TBTC",
            "WSTETH",
            "STETH",
            "RETH",
            "CBETH",
            "FRXETH",
            "BTC.B",
        }

        if (
            symbol in wrapped_symbols
            or name.startswith("wrapped ")
        ):
            reasons.append(
                "WRAPPED_ASSET"
            )

    return (
        len(reasons) == 0,
        reasons,
    )


# ============================================================
# MARKET CAP
# ============================================================

def check_market_cap(
    row: pd.Series,
) -> Tuple[bool, str]:

    market_cap = _safe_float(
        row.get("market_cap")
    )

    if pd.isna(market_cap):

        return (
            False,
            "MARKET_CAP_MISSING",
        )

    if (
        market_cap
        < config.MIN_MARKET_CAP_USD
    ):

        return (
            False,
            "MARKET_CAP_TOO_LOW",
        )

    return (
        True,
        "",
    )


# ============================================================
# LIQUIDEZ
# ============================================================

def check_liquidity(
    row: pd.Series,
) -> Tuple[bool, str]:

    volume = _safe_float(
        row.get("daily_volume")
    )

    if pd.isna(volume):

        return (
            False,
            "VOLUME_MISSING",
        )

    if (
        volume
        < config.MIN_DAILY_VOLUME_USD
    ):

        return (
            False,
            "LIQUIDITY_TOO_LOW",
        )

    return (
        True,
        "",
    )


# ============================================================
# VOLUME / MARKET CAP
# ============================================================

def calculate_volume_market_cap_ratio(
    row: pd.Series,
) -> float:

    volume = _safe_float(
        row.get("daily_volume")
    )

    market_cap = _safe_float(
        row.get("market_cap")
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


# ============================================================
# QUALIDADE DOS DADOS
# ============================================================

def calculate_data_completeness(
    row: pd.Series,
) -> float:

    important_fields = [
        "price",
        "market_cap",
        "daily_volume",
        "circulating_supply",
        "total_supply",
        "fdv",
    ]

    available = 0
    total = len(
        important_fields
    )

    for field in important_fields:

        value = row.get(
            field
        )

        if (
            value is not None
            and not pd.isna(value)
        ):
            available += 1

    if total == 0:
        return 0.0

    return (
        available
        / total
    )


def check_data_quality(
    row: pd.Series,
) -> Tuple[bool, str, float]:

    existing = _safe_float(
        row.get(
            "market_data_completeness"
        )
    )

    if pd.isna(existing):

        completeness = (
            calculate_data_completeness(
                row
            )
        )

    else:

        completeness = existing

    if (
        completeness
        < config.MIN_DATA_COMPLETENESS
    ):

        return (
            False,
            "INSUFFICIENT_DATA",
            completeness,
        )

    return (
        True,
        "",
        completeness,
    )


# ============================================================
# HISTÓRICO
# ============================================================

def estimate_history_availability(
    row: pd.Series,
) -> Tuple[bool, str]:

    """
    Quando o dataset já possui retornos históricos,
    usamos esses campos como evidência de histórico.

    Caso o pipeline ainda não tenha carregado histórico,
    o ativo não é eliminado aqui. A verificação definitiva
    poderá ocorrer nas camadas posteriores.
    """

    historical_fields = [
        "return_30d",
        "return_90d",
        "return_180d",
        "return_365d",
    ]

    fields_present = [
        field
        for field in historical_fields
        if field in row.index
    ]

    if not fields_present:

        return (
            True,
            "",
        )

    available = sum(
        1
        for field in fields_present
        if not pd.isna(
            row.get(field)
        )
    )

    if available == 0:

        return (
            False,
            "PRICE_HISTORY_INSUFFICIENT",
        )

    return (
        True,
        "",
    )


# ============================================================
# DILUIÇÃO CRÍTICA
# ============================================================

def check_known_dilution_block(
    row: pd.Series,
) -> Tuple[bool, str]:

    if (
        "dilution_hard_block"
        not in row.index
    ):

        return (
            True,
            "",
        )

    value = row.get(
        "dilution_hard_block"
    )

    if (
        value is not None
        and not pd.isna(value)
        and bool(value)
    ):

        return (
            False,
            "CRITICAL_DILUTION",
        )

    return (
        True,
        "",
    )


# ============================================================
# RISCO CRÍTICO
# ============================================================

def check_known_risk_block(
    row: pd.Series,
) -> Tuple[bool, str]:

    if (
        "risk_level"
        not in row.index
    ):

        return (
            True,
            "",
        )

    risk = str(
        row.get(
            "risk_level",
            ""
        )
    ).upper().strip()

    if (
        config.BLOCK_CRITICAL_RISK
        and risk == "CRITICAL"
    ):

        return (
            False,
            "CRITICAL_RISK",
        )

    return (
        True,
        "",
    )


# ============================================================
# ANÁLISE INDIVIDUAL
# ============================================================

def evaluate_asset(
    row: pd.Series,
) -> Dict:

    reasons: List[str] = []

    structural_pass, structural_reasons = (
        check_structural_exclusion(
            row
        )
    )

    reasons.extend(
        structural_reasons
    )

    market_cap_pass, reason = (
        check_market_cap(
            row
        )
    )

    if not market_cap_pass:
        reasons.append(
            reason
        )

    liquidity_pass, reason = (
        check_liquidity(
            row
        )
    )

    if not liquidity_pass:
        reasons.append(
            reason
        )

    (
        data_quality_pass,
        reason,
        completeness,
    ) = check_data_quality(
        row
    )

    if not data_quality_pass:
        reasons.append(
            reason
        )

    history_pass, reason = (
        estimate_history_availability(
            row
        )
    )

    if not history_pass:
        reasons.append(
            reason
        )

    dilution_pass, reason = (
        check_known_dilution_block(
            row
        )
    )

    if not dilution_pass:
        reasons.append(
            reason
        )

    risk_pass, reason = (
        check_known_risk_block(
            row
        )
    )

    if not risk_pass:
        reasons.append(
            reason
        )

    eligible = (
        structural_pass
        and market_cap_pass
        and liquidity_pass
        and data_quality_pass
        and history_pass
        and dilution_pass
        and risk_pass
    )

    volume_market_cap_ratio = (
        calculate_volume_market_cap_ratio(
            row
        )
    )

    return {
        "universe_eligible":
            eligible,

        "universe_rejection_reasons":
            "|".join(
                reasons
            ),

        "universe_data_completeness":
            round(
                completeness,
                4,
            ),

        "volume_market_cap_ratio":
            volume_market_cap_ratio,

        "universe_structural_pass":
            structural_pass,

        "universe_market_cap_pass":
            market_cap_pass,

        "universe_liquidity_pass":
            liquidity_pass,

        "universe_data_quality_pass":
            data_quality_pass,

        "universe_history_pass":
            history_pass,

        "universe_dilution_pass":
            dilution_pass,

        "universe_risk_pass":
            risk_pass,
    }


# ============================================================
# PROCESSAMENTO DO UNIVERSO
# ============================================================

def evaluate_universe(
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    if market_df.empty:
        return market_df.copy()

    result = market_df.copy()

    evaluations = []

    for _, row in result.iterrows():

        evaluations.append(
            evaluate_asset(
                row
            )
        )

    evaluation_df = (
        pd.DataFrame(
            evaluations,
            index=result.index,
        )
    )

    for column in evaluation_df.columns:

        result[
            column
        ] = evaluation_df[
            column
        ]

    return result


# ============================================================
# UNIVERSO ELEGÍVEL
# ============================================================

def build_eligible_universe(
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    evaluated = (
        evaluate_universe(
            market_df
        )
    )

    if evaluated.empty:
        return evaluated

    eligible = evaluated[
        evaluated[
            "universe_eligible"
        ]
    ].copy()

    eligible = eligible.sort_values(
        by=[
            "market_cap",
            "daily_volume",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )

    return eligible.reset_index(
        drop=True
    )


# ============================================================
# REJEITADOS
# ============================================================

def build_rejected_universe(
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    evaluated = (
        evaluate_universe(
            market_df
        )
    )

    if evaluated.empty:
        return evaluated

    rejected = evaluated[
        ~evaluated[
            "universe_eligible"
        ]
    ].copy()

    return rejected.reset_index(
        drop=True
    )


# ============================================================
# ESTATÍSTICAS
# ============================================================

def universe_statistics(
    evaluated_df: pd.DataFrame,
) -> Dict:

    if evaluated_df.empty:

        return {
            "total": 0,
            "eligible": 0,
            "rejected": 0,
            "eligibility_rate": 0.0,
        }

    total = len(
        evaluated_df
    )

    eligible = int(
        evaluated_df[
            "universe_eligible"
        ].sum()
    )

    rejected = (
        total
        - eligible
    )

    rate = (
        eligible / total
        if total > 0
        else 0
    )

    return {
        "total":
            total,

        "eligible":
            eligible,

        "rejected":
            rejected,

        "eligibility_rate":
            round(
                rate,
                4,
            ),
    }


# ============================================================
# MOTIVOS DE REJEIÇÃO
# ============================================================

def rejection_statistics(
    evaluated_df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        evaluated_df.empty
        or "universe_rejection_reasons"
        not in evaluated_df.columns
    ):

        return pd.DataFrame(
            columns=[
                "reason",
                "count",
            ]
        )

    reasons: List[str] = []

    rejected = evaluated_df[
        ~evaluated_df[
            "universe_eligible"
        ]
    ]

    for value in rejected[
        "universe_rejection_reasons"
    ].fillna(""):

        for reason in str(
            value
        ).split("|"):

            reason = reason.strip()

            if reason:
                reasons.append(
                    reason
                )

    if not reasons:

        return pd.DataFrame(
            columns=[
                "reason",
                "count",
            ]
        )

    counts = (
        pd.Series(
            reasons
        )
        .value_counts()
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="count"
        )
    )

    return counts


# ============================================================
# PIPELINE
# ============================================================

def run_universe_engine(
    market_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict,
]:

    print(
        "[universe_engine] "
        "Avaliando universo..."
    )

    evaluated = (
        evaluate_universe(
            market_df
        )
    )

    stats = (
        universe_statistics(
            evaluated
        )
    )

    eligible = evaluated[
        evaluated[
            "universe_eligible"
        ]
    ].copy()

    eligible = (
        eligible
        .sort_values(
            by="market_cap",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "[universe_engine] "
        f"Total: {stats['total']} | "
        f"Elegíveis: {stats['eligible']} | "
        f"Rejeitados: {stats['rejected']}"
    )

    return (
        eligible,
        evaluated,
        stats,
    )


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    from data.market_data import (
        build_market_dataset,
    )

    market = (
        build_market_dataset(
            include_history=False
        )
    )

    (
        eligible,
        evaluated,
        stats,
    ) = run_universe_engine(
        market
    )

    print(
        "\nESTATÍSTICAS"
    )

    print(
        stats
    )

    print(
        "\nMOTIVOS DE REJEIÇÃO"
    )

    print(
        rejection_statistics(
            evaluated
        ).to_string(
            index=False
        )
    )

    columns = [
        "symbol",
        "name",
        "market_cap",
        "daily_volume",
        "volume_market_cap_ratio",
        "universe_data_completeness",
    ]

    available = [
        column
        for column in columns
        if column in eligible.columns
    ]

    print(
        "\nUNIVERSO ELEGÍVEL"
    )

    print(
        eligible[
            available
        ]
        .head(50)
        .to_string(
            index=False
        )
    )
