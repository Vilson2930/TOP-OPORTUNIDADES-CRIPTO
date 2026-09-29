"""
CRYPTO OPPORTUNITY ENGINE
Tokenomics / Dilution Data Layer

FOCO:
Encontrar oportunidades sem confundir preço baixo com token barato.

Analisa:
- circulating supply;
- total supply;
- max supply;
- market cap / FDV;
- supply ainda não circulante;
- diluição potencial;
- pressão estrutural de oferta;
- risco de grande quantidade de tokens ainda por liberar.

IMPORTANTE:
Diluição crítica pode bloquear uma oportunidade.

ARQUITETURA:
Esta camada é SNAPSHOT-ONLY.

Ela NÃO realiza:
- chamadas HTTP;
- consultas ao CoinGecko;
- consultas históricas individuais;
- consultas individuais por ativo.

Todos os dados utilizados nesta etapa devem chegar previamente
pelo market_data.py.

Dados históricos de inflação/unlocks que não estiverem disponíveis
no dataset são tratados como DATA_INSUFFICIENT por meio de NaN.

Isso impede que centenas de ativos gerem HTTP 429 no GitHub Actions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

import config


# ============================================================
# HELPERS
# ============================================================

def _safe_float(
    value,
) -> float:

    try:

        if value is None:
            return np.nan

        result = float(
            value
        )

        if not np.isfinite(
            result
        ):
            return np.nan

        return result

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return np.nan


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    try:

        result = (
            numerator
            / denominator
        )

        if not np.isfinite(
            result
        ):
            return np.nan

        return float(
            result
        )

    except Exception:
        return np.nan


def _clip_score(
    value: float,
) -> float:

    if pd.isna(
        value
    ):
        return np.nan

    return float(
        np.clip(
            value,
            0,
            100,
        )
    )


def _first_valid(
    row: pd.Series,
    columns: list[str],
) -> float:

    for column in columns:

        if column not in row.index:
            continue

        value = _safe_float(
            row.get(
                column
            )
        )

        if not pd.isna(
            value
        ):
            return value

    return np.nan


# ============================================================
# SUPPLY METRICS
# ============================================================

def calculate_supply_metrics(
    circulating_supply: float,
    total_supply: float,
    max_supply: float,
) -> Dict[str, float]:

    circulating_ratio_total = (
        _safe_divide(
            circulating_supply,
            total_supply,
        )
    )

    circulating_ratio_max = (
        _safe_divide(
            circulating_supply,
            max_supply,
        )
    )

    if (
        not pd.isna(total_supply)
        and not pd.isna(circulating_supply)
        and total_supply >= 0
        and circulating_supply >= 0
    ):

        non_circulating_supply = max(
            total_supply
            - circulating_supply,
            0.0,
        )

    else:

        non_circulating_supply = np.nan

    non_circulating_ratio = (
        _safe_divide(
            non_circulating_supply,
            total_supply,
        )
    )

    return {
        "circulating_supply":
            circulating_supply,

        "total_supply":
            total_supply,

        "max_supply":
            max_supply,

        "circulating_supply_ratio":
            circulating_ratio_total,

        "circulating_max_supply_ratio":
            circulating_ratio_max,

        "non_circulating_supply":
            non_circulating_supply,

        "non_circulating_ratio":
            non_circulating_ratio,
    }


# ============================================================
# FDV METRICS
# ============================================================

def calculate_fdv_metrics(
    market_cap: float,
    fdv: float,
) -> Dict[str, float]:

    market_cap_fdv_ratio = (
        _safe_divide(
            market_cap,
            fdv,
        )
    )

    if (
        not pd.isna(fdv)
        and not pd.isna(market_cap)
        and market_cap > 0
    ):

        fdv_premium = (
            fdv
            / market_cap
            - 1
        )

    else:

        fdv_premium = np.nan

    return {
        "market_cap_fdv_ratio":
            market_cap_fdv_ratio,

        "fdv_premium":
            fdv_premium,
    }


# ============================================================
# SUPPLY OVERHANG
# ============================================================

def calculate_supply_overhang(
    circulating_supply: float,
    total_supply: float,
) -> float:

    if (
        pd.isna(circulating_supply)
        or pd.isna(total_supply)
        or total_supply <= 0
    ):
        return np.nan

    future_supply = max(
        total_supply
        - circulating_supply,
        0.0,
    )

    result = (
        future_supply
        / total_supply
    )

    return float(
        np.clip(
            result,
            0,
            1,
        )
    )


# ============================================================
# SCORE — CIRCULATING SUPPLY
# ============================================================

def score_circulating_ratio(
    ratio: float,
) -> float:

    if pd.isna(
        ratio
    ):
        return np.nan

    if ratio >= 0.90:
        return 100.0

    if ratio >= 0.75:
        return 90.0

    if ratio >= 0.60:
        return 75.0

    if ratio >= 0.50:
        return 60.0

    if ratio >= 0.40:
        return 45.0

    if ratio >= 0.25:
        return 25.0

    return 5.0


# ============================================================
# SCORE — MARKET CAP / FDV
# ============================================================

def score_market_cap_fdv(
    ratio: float,
) -> float:

    if pd.isna(
        ratio
    ):
        return np.nan

    if ratio >= 0.90:
        return 100.0

    if ratio >= 0.75:
        return 90.0

    if ratio >= 0.60:
        return 75.0

    if ratio >= 0.50:
        return 60.0

    if ratio >= 0.40:
        return 45.0

    if ratio >= 0.25:
        return 25.0

    return 5.0


# ============================================================
# SCORE — INFLATION
# ============================================================

def score_inflation(
    inflation: float,
) -> float:

    if pd.isna(
        inflation
    ):
        return np.nan

    if inflation <= 0:
        return 100.0

    if inflation <= 0.02:
        return 95.0

    if inflation <= 0.05:
        return 85.0

    if inflation <= 0.10:
        return 70.0

    if inflation <= 0.15:
        return 55.0

    if inflation <= 0.20:
        return 40.0

    if inflation <= 0.30:
        return 20.0

    return 5.0


# ============================================================
# SCORE — SUPPLY OVERHANG
# ============================================================

def score_supply_overhang(
    overhang: float,
) -> float:

    if pd.isna(
        overhang
    ):
        return np.nan

    return _clip_score(
        (
            1
            - overhang
        )
        * 100
    )


# ============================================================
# DILUTION SCORE
# ============================================================

def calculate_dilution_score(
    circulating_ratio: float,
    market_cap_fdv_ratio: float,
    inflation_90d: float,
    inflation_365d: float,
    supply_overhang: float,
) -> float:

    annualized_90d = np.nan

    if not pd.isna(
        inflation_90d
    ):

        try:

            if inflation_90d > -1:

                annualized_90d = (
                    (
                        1
                        + inflation_90d
                    )
                    ** (
                        365
                        / 90
                    )
                    - 1
                )

        except Exception:

            annualized_90d = np.nan

    components = [
        (
            score_circulating_ratio(
                circulating_ratio
            ),
            0.25,
        ),
        (
            score_market_cap_fdv(
                market_cap_fdv_ratio
            ),
            0.20,
        ),
        (
            score_inflation(
                inflation_365d
            ),
            0.25,
        ),
        (
            score_inflation(
                annualized_90d
            ),
            0.15,
        ),
        (
            score_supply_overhang(
                supply_overhang
            ),
            0.15,
        ),
    ]

    weighted_sum = 0.0
    available_weight = 0.0

    for score, weight in components:

        if pd.isna(
            score
        ):
            continue

        weighted_sum += (
            score
            * weight
        )

        available_weight += (
            weight
        )

    if available_weight == 0:
        return np.nan

    return round(
        weighted_sum
        / available_weight,
        2,
    )


# ============================================================
# DILUTION RISK
# ============================================================

def classify_dilution_risk(
    circulating_ratio: float,
    market_cap_fdv_ratio: float,
    inflation_365d: float,
    dilution_score: float,
) -> str:

    critical_conditions = []

    if (
        not pd.isna(
            circulating_ratio
        )
        and circulating_ratio
        < config.MIN_CIRCULATING_SUPPLY_RATIO_CRITICAL
    ):

        critical_conditions.append(
            "LOW_CIRCULATING_SUPPLY"
        )

    if (
        not pd.isna(
            market_cap_fdv_ratio
        )
        and market_cap_fdv_ratio
        < config.MIN_MARKET_CAP_FDV_RATIO_CRITICAL
    ):

        critical_conditions.append(
            "LOW_MC_FDV"
        )

    if (
        not pd.isna(
            inflation_365d
        )
        and inflation_365d
        > config.MAX_ANNUAL_INFLATION_CRITICAL
    ):

        critical_conditions.append(
            "HIGH_INFLATION"
        )

    if (
        not pd.isna(
            dilution_score
        )
        and dilution_score < 25
    ):

        critical_conditions.append(
            "VERY_LOW_DILUTION_SCORE"
        )

    if critical_conditions:
        return "CRITICAL"

    warning_conditions = []

    if (
        not pd.isna(
            circulating_ratio
        )
        and circulating_ratio
        < config.MIN_CIRCULATING_SUPPLY_RATIO_WARNING
    ):

        warning_conditions.append(
            "CIRCULATING_WARNING"
        )

    if (
        not pd.isna(
            market_cap_fdv_ratio
        )
        and market_cap_fdv_ratio
        < config.MIN_MARKET_CAP_FDV_RATIO_WARNING
    ):

        warning_conditions.append(
            "MC_FDV_WARNING"
        )

    if (
        not pd.isna(
            inflation_365d
        )
        and inflation_365d
        > config.MAX_ANNUAL_INFLATION_WARNING
    ):

        warning_conditions.append(
            "INFLATION_WARNING"
        )

    if warning_conditions:
        return "HIGH"

    if (
        not pd.isna(
            dilution_score
        )
        and dilution_score < 60
    ):
        return "MODERATE"

    return "LOW"


# ============================================================
# HARD BLOCK
# ============================================================

def dilution_hard_block(
    dilution_risk: str,
) -> bool:

    return (
        dilution_risk
        == "CRITICAL"
    )


# ============================================================
# OPPORTUNITY MODIFIER
# ============================================================

def calculate_dilution_opportunity_modifier(
    dilution_score: float,
) -> float:

    """
    Retorna multiplicador de 0 a 1.

    Não cria oportunidade.
    Apenas reduz oportunidade quando
    a estrutura de oferta é ruim.
    """

    if pd.isna(
        dilution_score
    ):
        return np.nan

    if dilution_score >= 80:
        return 1.00

    if dilution_score >= 70:
        return 0.95

    if dilution_score >= 60:
        return 0.90

    if dilution_score >= 50:
        return 0.80

    if dilution_score >= 40:
        return 0.65

    if dilution_score >= 25:
        return 0.40

    return 0.0


# ============================================================
# EXTRAÇÃO DE SUPPLY DO MARKET DATA
# ============================================================

def _extract_market_supply(
    row: pd.Series,
) -> Dict[str, float]:

    circulating_supply = (
        _first_valid(
            row,
            [
                "circulating_supply",
                "circulatingSupply",
            ],
        )
    )

    total_supply = (
        _first_valid(
            row,
            [
                "total_supply",
                "totalSupply",
            ],
        )
    )

    max_supply = (
        _first_valid(
            row,
            [
                "max_supply",
                "maxSupply",
            ],
        )
    )

    market_cap = (
        _first_valid(
            row,
            [
                "market_cap",
                "marketCap",
            ],
        )
    )

    price = (
        _first_valid(
            row,
            [
                "price",
                "current_price",
            ],
        )
    )

    fdv = (
        _first_valid(
            row,
            [
                "fdv",
                "fully_diluted_valuation",
                "fully_diluted_market_cap",
            ],
        )
    )

    # --------------------------------------------------------
    # CIRCULATING SUPPLY FALLBACK
    # --------------------------------------------------------

    if (
        pd.isna(
            circulating_supply
        )
        and not pd.isna(
            market_cap
        )
        and not pd.isna(
            price
        )
        and price > 0
    ):

        circulating_supply = (
            market_cap
            / price
        )

    # --------------------------------------------------------
    # TOTAL SUPPLY FALLBACK
    # --------------------------------------------------------

    if (
        pd.isna(
            total_supply
        )
        and not pd.isna(
            max_supply
        )
        and max_supply > 0
    ):

        total_supply = (
            max_supply
        )

    # --------------------------------------------------------
    # FDV FALLBACK POR PRICE × TOTAL SUPPLY
    # --------------------------------------------------------

    if (
        pd.isna(
            fdv
        )
        and not pd.isna(
            price
        )
        and not pd.isna(
            total_supply
        )
        and price > 0
        and total_supply > 0
    ):

        fdv = (
            price
            * total_supply
        )

    # --------------------------------------------------------
    # FDV FALLBACK POR MARKET CAP E SUPPLY
    # --------------------------------------------------------

    if (
        pd.isna(
            fdv
        )
        and not pd.isna(
            market_cap
        )
        and not pd.isna(
            circulating_supply
        )
        and not pd.isna(
            total_supply
        )
        and circulating_supply > 0
        and total_supply > 0
    ):

        fdv = (
            market_cap
            * total_supply
            / circulating_supply
        )

    return {
        "circulating_supply":
            circulating_supply,

        "total_supply":
            total_supply,

        "max_supply":
            max_supply,

        "market_cap":
            market_cap,

        "price":
            price,

        "fdv":
            fdv,
    }


# ============================================================
# HISTORICAL VALUES ALREADY PRESENT IN DATASET
# ============================================================

def _extract_existing_inflation(
    row: pd.Series,
    days: int,
) -> float:

    candidates = [
        f"estimated_supply_inflation_{days}d",
        f"supply_inflation_{days}d",
        f"inflation_{days}d",
    ]

    return _first_valid(
        row,
        candidates,
    )


# ============================================================
# DATA QUALITY
# ============================================================

def _calculate_tokenomics_data_completeness(
    circulating_supply: float,
    total_supply: float,
    market_cap: float,
    fdv: float,
) -> float:

    values = [
        circulating_supply,
        total_supply,
        market_cap,
        fdv,
    ]

    available = sum(
        1
        for value in values
        if not pd.isna(
            value
        )
    )

    return round(
        available
        / len(values),
        4,
    )


def _classify_tokenomics_data_quality(
    completeness: float,
) -> str:

    if completeness >= 1.00:
        return "HIGH"

    if completeness >= 0.75:
        return "GOOD"

    if completeness >= 0.50:
        return "PARTIAL"

    return "DATA_INSUFFICIENT"


# ============================================================
# ENRIQUECIMENTO
# ============================================================

def enrich_with_tokenomics_data(
    market_df: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:

    """
    Enriquece o dataset utilizando SOMENTE dados já existentes
    no DataFrame recebido.

    Esta função deliberadamente aceita *args e **kwargs para
    compatibilidade com versões antigas do main.py.

    Portanto, mesmo que uma versão antiga chame:

        fetch_history=True

    ou:

        fetch_missing_details=True

    nenhuma chamada HTTP será realizada.

    A camada tokenomics permanece SNAPSHOT-ONLY.
    """

    if (
        market_df is None
        or market_df.empty
    ):

        return (
            market_df.copy()
            if isinstance(
                market_df,
                pd.DataFrame,
            )
            else pd.DataFrame()
        )

    records = []

    total = len(
        market_df
    )

    print(
        "[tokenomics_data] "
        f"Processando {total} ativos."
    )

    print(
        "[tokenomics_data] "
        "Modo SNAPSHOT-ONLY ativo."
    )

    print(
        "[tokenomics_data] "
        "Chamadas HTTP individuais: DESATIVADAS."
    )

    print(
        "[tokenomics_data] "
        "Histórico individual: DESATIVADO."
    )

    print(
        "[tokenomics_data] "
        "Detalhes individuais CoinGecko: DESATIVADOS."
    )

    for position, (_, row) in enumerate(
        market_df.iterrows(),
        start=1,
    ):

        coin_id = (
            row.get(
                "coin_id"
            )
        )

        symbol = (
            row.get(
                "symbol"
            )
        )

        if (
            position == 1
            or position == total
            or position % 25 == 0
        ):

            print(
                "[tokenomics_data] "
                f"{symbol} "
                f"({position}/{total})"
            )

        # ----------------------------------------------------
        # SUPPLY SNAPSHOT
        # ----------------------------------------------------

        market_values = (
            _extract_market_supply(
                row
            )
        )

        circulating_supply = (
            market_values[
                "circulating_supply"
            ]
        )

        total_supply = (
            market_values[
                "total_supply"
            ]
        )

        max_supply = (
            market_values[
                "max_supply"
            ]
        )

        market_cap = (
            market_values[
                "market_cap"
            ]
        )

        fdv = (
            market_values[
                "fdv"
            ]
        )

        # ----------------------------------------------------
        # SUPPLY METRICS
        # ----------------------------------------------------

        supply_metrics = (
            calculate_supply_metrics(
                circulating_supply,
                total_supply,
                max_supply,
            )
        )

        # ----------------------------------------------------
        # FDV METRICS
        # ----------------------------------------------------

        fdv_metrics = (
            calculate_fdv_metrics(
                market_cap,
                fdv,
            )
        )

        # ----------------------------------------------------
        # SUPPLY OVERHANG
        # ----------------------------------------------------

        supply_overhang = (
            calculate_supply_overhang(
                circulating_supply,
                total_supply,
            )
        )

        # ----------------------------------------------------
        # INFLATION
        # ----------------------------------------------------
        # Apenas reutiliza valores caso já existam no dataset.
        # Nenhuma consulta histórica é feita aqui.
        # ----------------------------------------------------

        inflation_30d = (
            _extract_existing_inflation(
                row,
                30,
            )
        )

        inflation_90d = (
            _extract_existing_inflation(
                row,
                90,
            )
        )

        inflation_180d = (
            _extract_existing_inflation(
                row,
                180,
            )
        )

        inflation_365d = (
            _extract_existing_inflation(
                row,
                365,
            )
        )

        # ----------------------------------------------------
        # DILUTION SCORE
        # ----------------------------------------------------

        dilution_score = (
            calculate_dilution_score(
                circulating_ratio=
                    supply_metrics[
                        "circulating_supply_ratio"
                    ],

                market_cap_fdv_ratio=
                    fdv_metrics[
                        "market_cap_fdv_ratio"
                    ],

                inflation_90d=
                    inflation_90d,

                inflation_365d=
                    inflation_365d,

                supply_overhang=
                    supply_overhang,
            )
        )

        # ----------------------------------------------------
        # DILUTION RISK
        # ----------------------------------------------------

        dilution_risk = (
            classify_dilution_risk(
                circulating_ratio=
                    supply_metrics[
                        "circulating_supply_ratio"
                    ],

                market_cap_fdv_ratio=
                    fdv_metrics[
                        "market_cap_fdv_ratio"
                    ],

                inflation_365d=
                    inflation_365d,

                dilution_score=
                    dilution_score,
            )
        )

        # ----------------------------------------------------
        # HARD BLOCK
        # ----------------------------------------------------

        hard_block = (
            dilution_hard_block(
                dilution_risk
            )
        )

        # ----------------------------------------------------
        # OPPORTUNITY MODIFIER
        # ----------------------------------------------------

        modifier = (
            calculate_dilution_opportunity_modifier(
                dilution_score
            )
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        completeness = (
            _calculate_tokenomics_data_completeness(
                circulating_supply=
                    circulating_supply,

                total_supply=
                    total_supply,

                market_cap=
                    market_cap,

                fdv=
                    fdv,
            )
        )

        data_quality = (
            _classify_tokenomics_data_quality(
                completeness
            )
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        records.append(
            {
                "coin_id":
                    coin_id,

                **supply_metrics,

                **fdv_metrics,

                "supply_overhang":
                    supply_overhang,

                "estimated_supply_inflation_30d":
                    inflation_30d,

                "estimated_supply_inflation_90d":
                    inflation_90d,

                "estimated_supply_inflation_180d":
                    inflation_180d,

                "estimated_supply_inflation_365d":
                    inflation_365d,

                "dilution_score":
                    dilution_score,

                "dilution_risk":
                    dilution_risk,

                "dilution_hard_block":
                    hard_block,

                "dilution_opportunity_modifier":
                    modifier,

                "tokenomics_data_completeness":
                    completeness,

                "tokenomics_data_quality":
                    data_quality,

                "tokenomics_data_source":
                    "MARKET_SNAPSHOT",

                "tokenomics_history_status":
                    (
                        "AVAILABLE_IN_INPUT"
                        if any(
                            not pd.isna(
                                value
                            )
                            for value in [
                                inflation_30d,
                                inflation_90d,
                                inflation_180d,
                                inflation_365d,
                            ]
                        )
                        else "DATA_INSUFFICIENT"
                    ),
            }
        )

    # ========================================================
    # TOKENOMICS DATAFRAME
    # ========================================================

    tokenomics_df = (
        pd.DataFrame(
            records
        )
    )

    if tokenomics_df.empty:

        print(
            "[tokenomics_data] "
            "Nenhum registro tokenomics gerado."
        )

        return (
            market_df.copy()
        )

    # ========================================================
    # REMOVE DUPLICATE COLUMNS BEFORE MERGE
    # ========================================================

    duplicate_columns = [
        column
        for column in [
            "circulating_supply",
            "total_supply",
            "max_supply",
            "market_cap_fdv_ratio",
            "circulating_supply_ratio",
            "circulating_max_supply_ratio",
            "non_circulating_supply",
            "non_circulating_ratio",
            "fdv_premium",
            "supply_overhang",
            "estimated_supply_inflation_30d",
            "estimated_supply_inflation_90d",
            "estimated_supply_inflation_180d",
            "estimated_supply_inflation_365d",
            "dilution_score",
            "dilution_risk",
            "dilution_hard_block",
            "dilution_opportunity_modifier",
            "tokenomics_data_completeness",
            "tokenomics_data_quality",
            "tokenomics_data_source",
            "tokenomics_history_status",
        ]
        if column
        in market_df.columns
    ]

    base_df = (
        market_df.drop(
            columns=
                duplicate_columns,
            errors="ignore",
        )
    )

    # ========================================================
    # MERGE
    # ========================================================

    result = (
        base_df.merge(
            tokenomics_df,
            on="coin_id",
            how="left",
        )
    )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    numeric_columns = [
        "circulating_supply",
        "total_supply",
        "max_supply",
        "market_cap_fdv_ratio",
        "circulating_supply_ratio",
        "circulating_max_supply_ratio",
        "non_circulating_supply",
        "non_circulating_ratio",
        "fdv_premium",
        "supply_overhang",
        "estimated_supply_inflation_30d",
        "estimated_supply_inflation_90d",
        "estimated_supply_inflation_180d",
        "estimated_supply_inflation_365d",
        "dilution_score",
        "dilution_opportunity_modifier",
        "tokenomics_data_completeness",
    ]

    for column in numeric_columns:

        if column in result.columns:

            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="coerce",
                )
            )

    # ========================================================
    # LOG
    # ========================================================

    print(
        "[tokenomics_data] "
        f"Concluído: {len(result)} ativos."
    )

    print(
        "[tokenomics_data] "
        "Chamadas HTTP realizadas nesta camada: 0."
    )

    print(
        "[tokenomics_data] "
        "Consultas históricas individuais realizadas: 0."
    )

    print(
        "[tokenomics_data] "
        "CoinGecko individual: NÃO UTILIZADO."
    )

    print(
        "[tokenomics_data] "
        "Tokenomics processado exclusivamente "
        "a partir do snapshot disponível."
    )

    return result


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

    market = (
        market.head(
            50
        )
    )

    result = (
        enrich_with_tokenomics_data(
            market
        )
    )

    columns = [
        "symbol",
        "name",
        "market_cap",
        "fdv",
        "circulating_supply",
        "total_supply",
        "max_supply",
        "circulating_supply_ratio",
        "market_cap_fdv_ratio",
        "supply_overhang",
        "estimated_supply_inflation_90d",
        "estimated_supply_inflation_365d",
        "dilution_score",
        "dilution_risk",
        "dilution_hard_block",
        "tokenomics_data_completeness",
        "tokenomics_data_quality",
        "tokenomics_history_status",
    ]

    available = [
        column
        for column in columns
        if column
        in result.columns
    ]

    if (
        "dilution_score"
        in result.columns
    ):

        output = (
            result[
                available
            ]
            .sort_values(
                "dilution_score",
                ascending=False,
                na_position="last",
            )
            .head(
                50
            )
        )

    else:

        output = (
            result[
                available
            ]
            .head(
                50
            )
        )

    print(
        output.to_string(
            index=False
        )
    )
