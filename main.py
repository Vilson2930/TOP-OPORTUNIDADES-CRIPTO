"""
CRYPTO OPPORTUNITY ENGINE
Main Orchestrator

OBJETIVO:
Executar o pipeline completo do motor de oportunidades cripto.

PIPELINE:

UNIVERSE
→ HARD FILTERS
→ DATA ENRICHMENT
→ FUNDAMENTALS
→ REVENUE
→ HOLDER VALUE
→ DILUTION
→ TVL
→ VALUATION
→ ON-CHAIN
→ NARRATIVE
→ RISK
→ TIMING
→ RANKING
→ DECISION

Princípio:
Encontrar projetos de qualidade onde a melhora econômica e fundamental
esteja avançando mais rapidamente que a precificação do mercado.

BTC é excluído.

A IA NÃO participa da decisão determinística.
A auditoria por IA será executada posteriormente como camada independente.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

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

from data.holder_value_data import (
    enrich_with_holder_value_data,
)

from data.developer_data import (
    enrich_with_developer_data,
)

from engines.universe_engine import (
    run_universe_engine,
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

from engines.ranking_engine import (
    run_ranking_engine,
    get_top_opportunities,
    build_report_view,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
)

FULL_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "crypto_opportunity_full.csv",
)

TOP_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "crypto_opportunities_top.csv",
)

REJECTED_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "crypto_rejected.csv",
)

ENGINE_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "engine_summary.txt",
)


# ============================================================
# HELPERS
# ============================================================

def ensure_output_directory() -> None:

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


def print_header(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )


def print_step(
    step: int,
    total: int,
    title: str,
) -> None:

    print(
        "\n"
        f"[{step:02d}/{total:02d}] "
        f"{title}"
    )


def dataframe_memory_mb(
    dataset: pd.DataFrame,
) -> float:

    if dataset.empty:
        return 0.0

    memory = (
        dataset
        .memory_usage(
            deep=True
        )
        .sum()
    )

    return round(
        memory
        / 1024
        / 1024,
        2,
    )


def safe_count(
    dataset: pd.DataFrame,
    column: str,
    value,
) -> int:

    if (
        dataset.empty
        or column not in dataset.columns
    ):
        return 0

    return int(
        (
            dataset[
                column
            ]
            == value
        )
        .sum()
    )


def safe_numeric_mean(
    dataset: pd.DataFrame,
    column: str,
) -> float:

    if (
        dataset.empty
        or column not in dataset.columns
    ):
        return np.nan

    values = pd.to_numeric(
        dataset[
            column
        ],
        errors="coerce",
    )

    if values.dropna().empty:
        return np.nan

    return float(
        values.mean()
    )


def safe_numeric_median(
    dataset: pd.DataFrame,
    column: str,
) -> float:

    if (
        dataset.empty
        or column not in dataset.columns
    ):
        return np.nan

    values = pd.to_numeric(
        dataset[
            column
        ],
        errors="coerce",
    )

    if values.dropna().empty:
        return np.nan

    return float(
        values.median()
    )


def format_number(
    value,
    decimals: int = 2,
) -> str:

    try:

        value = float(
            value
        )

        if np.isnan(
            value
        ):
            return "N/A"

        return (
            f"{value:.{decimals}f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_market_dataset(
    dataset: pd.DataFrame,
) -> None:

    required = {
        "coin_id",
        "symbol",
        "name",
        "price",
        "market_cap",
        "daily_volume",
    }

    missing = (
        required
        - set(
            dataset.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Market dataset missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def validate_final_dataset(
    dataset: pd.DataFrame,
) -> None:

    required = {
        "symbol",
        "name",
        "opportunity_score",
        "quality_gate",
        "hard_block",
        "signal",
    }

    missing = (
        required
        - set(
            dataset.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Final dataset missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


# ============================================================
# UNIVERSE HANDLING
# ============================================================

def split_universe_result(
    universe_result,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if isinstance(
        universe_result,
        pd.DataFrame,
    ):

        dataset = universe_result.copy()

        if (
            "universe_eligible"
            in dataset.columns
        ):

            eligible = dataset[
                dataset[
                    "universe_eligible"
                ]
                == True
            ].copy()

            rejected = dataset[
                dataset[
                    "universe_eligible"
                ]
                != True
            ].copy()

            return (
                eligible,
                rejected,
            )

        if (
            "eligible"
            in dataset.columns
        ):

            eligible = dataset[
                dataset[
                    "eligible"
                ]
                == True
            ].copy()

            rejected = dataset[
                dataset[
                    "eligible"
                ]
                != True
            ].copy()

            return (
                eligible,
                rejected,
            )

        return (
            dataset,
            pd.DataFrame(),
        )

    if isinstance(
        universe_result,
        dict,
    ):

        eligible = (
            universe_result.get(
                "eligible"
            )
        )

        if eligible is None:

            eligible = (
                universe_result.get(
                    "eligible_universe"
                )
            )

        rejected = (
            universe_result.get(
                "rejected"
            )
        )

        if rejected is None:

            rejected = (
                universe_result.get(
                    "rejected_universe"
                )
            )

        if not isinstance(
            eligible,
            pd.DataFrame,
        ):

            eligible = pd.DataFrame()

        if not isinstance(
            rejected,
            pd.DataFrame,
        ):

            rejected = pd.DataFrame()

        return (
            eligible.copy(),
            rejected.copy(),
        )

    if isinstance(
        universe_result,
        tuple,
    ):

        frames = [
            item
            for item in universe_result
            if isinstance(
                item,
                pd.DataFrame,
            )
        ]

        if len(frames) >= 2:

            first = frames[0].copy()
            second = frames[1].copy()

            # universe_engine may return (full_dataset, eligible_dataset).
            # Detect that shape and derive rejected rows from the full dataset.
            if len(first) >= len(second):

                if (
                    "universe_eligible" in first.columns
                ):

                    eligible = first[
                        first["universe_eligible"] == True
                    ].copy()

                    rejected = first[
                        first["universe_eligible"] != True
                    ].copy()

                    return eligible, rejected

                if (
                    "eligible" in first.columns
                ):

                    eligible = first[
                        first["eligible"] == True
                    ].copy()

                    rejected = first[
                        first["eligible"] != True
                    ].copy()

                    return eligible, rejected

                # If second is a strict subset of first, treat second as eligible
                # and derive rejected rows by coin_id/symbol.
                if len(second) < len(first):

                    key = None

                    for candidate in (
                        "coin_id",
                        "symbol",
                    ):
                        if (
                            candidate in first.columns
                            and candidate in second.columns
                        ):
                            key = candidate
                            break

                    if key is not None:

                        eligible_keys = set(
                            second[key]
                            .dropna()
                            .astype(str)
                        )

                        rejected = first[
                            ~first[key]
                            .astype(str)
                            .isin(eligible_keys)
                        ].copy()

                        return second, rejected

            return first, second

        if len(frames) == 1:

            return (
                frames[0].copy(),
                pd.DataFrame(),
            )

    raise RuntimeError(
        "Unsupported return type from universe_engine."
    )


# ============================================================
# ENGINE STEP WRAPPER
# ============================================================

def run_engine_step(
    name: str,
    function,
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    start = time.time()

    before_rows = len(
        dataset
    )

    result = function(
        dataset
    )

    if not isinstance(
        result,
        pd.DataFrame,
    ):

        raise RuntimeError(
            f"{name} did not return a DataFrame."
        )

    elapsed = (
        time.time()
        - start
    )

    after_rows = len(
        result
    )

    print(
        f"[{name}] "
        f"rows={after_rows} "
        f"before={before_rows} "
        f"memory={dataframe_memory_mb(result)}MB "
        f"time={elapsed:.2f}s"
    )

    return result


# ============================================================
# DATA ENRICHMENT
# ============================================================

def enrich_dataset(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    result = dataset.copy()

    print_header(
        "DATA ENRICHMENT"
    )

    print(
        "\n[defi_data] starting"
    )

    result = enrich_with_defi_data(
        result
    )

    print(
        "[defi_data] completed"
    )

    print(
        "\n[onchain_data] starting"
    )

    result = enrich_with_onchain_data(
        result
    )

    print(
        "[onchain_data] completed"
    )

    print(
        "\n[tokenomics_data] starting"
    )

    result = enrich_with_tokenomics_data(
        result
    )

    print(
        "[tokenomics_data] completed"
    )

    print(
        "\n[holder_value_data] starting"
    )

    result = enrich_with_holder_value_data(
        result
    )

    print(
        "[holder_value_data] completed"
    )

    print(
        "\n[developer_data] starting"
    )

    result = enrich_with_developer_data(
        result
    )

    print(
        "[developer_data] completed"
    )

    return result


# ============================================================
# ENGINE PIPELINE
# ============================================================

def run_analysis_engines(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    result = dataset.copy()

    engines = [
        (
            "fundamental_engine",
            run_fundamental_engine,
        ),

        (
            "revenue_engine",
            run_revenue_engine,
        ),

        (
            "holder_value_engine",
            run_holder_value_engine,
        ),

        (
            "dilution_engine",
            run_dilution_engine,
        ),

        (
            "tvl_engine",
            run_tvl_engine,
        ),

        (
            "valuation_engine",
            run_valuation_engine,
        ),

        (
            "onchain_engine",
            run_onchain_engine,
        ),

        (
            "narrative_engine",
            run_narrative_engine,
        ),

        (
            "risk_engine",
            run_risk_engine,
        ),

        (
            "timing_engine",
            run_timing_engine,
        ),

        (
            "ranking_engine",
            run_ranking_engine,
        ),
    ]

    total = len(
        engines
    )

    print_header(
        "ANALYSIS ENGINES"
    )

    for index, (
        name,
        function,
    ) in enumerate(
        engines,
        start=1,
    ):

        print_step(
            index,
            total,
            name,
        )

        result = run_engine_step(
            name,
            function,
            result,
        )

    return result


# ============================================================
# OUTPUT
# ============================================================

def save_outputs(
    full_result: pd.DataFrame,
    top_result: pd.DataFrame,
    rejected_result: pd.DataFrame,
) -> None:

    ensure_output_directory()

    full_result.to_csv(
        FULL_RESULTS_FILE,
        index=False,
    )

    top_result.to_csv(
        TOP_RESULTS_FILE,
        index=False,
    )

    if not rejected_result.empty:

        rejected_result.to_csv(
            REJECTED_RESULTS_FILE,
            index=False,
        )

    print_header(
        "OUTPUT FILES"
    )

    print(
        FULL_RESULTS_FILE
    )

    print(
        TOP_RESULTS_FILE
    )

    if not rejected_result.empty:

        print(
            REJECTED_RESULTS_FILE
        )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    market_count: int,
    eligible_count: int,
    rejected_count: int,
    final_result: pd.DataFrame,
    top_result: pd.DataFrame,
    started_at: datetime,
    finished_at: datetime,
) -> str:

    duration = (
        finished_at
        - started_at
    ).total_seconds()

    strong_entry = safe_count(
        final_result,
        "signal",
        config.SIGNAL_STRONG_ENTRY,
    )

    entry = safe_count(
        final_result,
        "signal",
        config.SIGNAL_ENTRY,
    )

    watch = safe_count(
        final_result,
        "signal",
        config.SIGNAL_WATCH,
    )

    wait = safe_count(
        final_result,
        "signal",
        config.SIGNAL_WAIT,
    )

    rejected = safe_count(
        final_result,
        "signal",
        config.SIGNAL_REJECTED,
    )

    blocked = safe_count(
        final_result,
        "signal",
        config.SIGNAL_BLOCKED,
    )

    quality_pass = safe_count(
        final_result,
        "quality_gate",
        True,
    )

    avg_opportunity = safe_numeric_mean(
        final_result,
        "opportunity_score",
    )

    median_opportunity = safe_numeric_median(
        final_result,
        "opportunity_score",
    )

    avg_divergence = safe_numeric_mean(
        final_result,
        "fundamental_price_divergence",
    )

    avg_timing = safe_numeric_mean(
        final_result,
        "timing_score",
    )

    avg_confidence = safe_numeric_mean(
        final_result,
        "decision_confidence",
    )

    lines = [
        config.ENGINE_NAME,
        f"Version: {config.ENGINE_VERSION}",
        "",
        f"Started UTC: {started_at.isoformat()}",
        f"Finished UTC: {finished_at.isoformat()}",
        f"Duration seconds: {duration:.2f}",
        "",
        "UNIVERSE",
        f"Market assets: {market_count}",
        f"Eligible assets: {eligible_count}",
        f"Universe rejected: {rejected_count}",
        "",
        "QUALITY",
        f"Quality Gate passed: {quality_pass}",
        "",
        "SIGNALS",
        f"STRONG_ENTRY: {strong_entry}",
        f"ENTRY: {entry}",
        f"WATCH: {watch}",
        f"WAIT: {wait}",
        f"REJECTED: {rejected}",
        f"BLOCKED: {blocked}",
        "",
        "SCORES",
        (
            "Average Opportunity Score: "
            + format_number(
                avg_opportunity
            )
        ),
        (
            "Median Opportunity Score: "
            + format_number(
                median_opportunity
            )
        ),
        (
            "Average Fundamental/Price Divergence: "
            + format_number(
                avg_divergence
            )
        ),
        (
            "Average Timing Score: "
            + format_number(
                avg_timing
            )
        ),
        (
            "Average Decision Confidence: "
            + format_number(
                avg_confidence
            )
        ),
        "",
        (
            "Top opportunities returned: "
            f"{len(top_result)}"
        ),
    ]

    return "\n".join(
        lines
    )


def save_summary(
    summary: str,
) -> None:

    ensure_output_directory()

    with open(
        ENGINE_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            summary
        )


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_top_opportunities(
    top_result: pd.DataFrame,
) -> None:

    print_header(
        "TOP OPORTUNIDADES CRIPTO"
    )

    if top_result.empty:

        print(
            "Nenhum ativo passou por todos os Quality Gates."
        )

        return

    report = build_report_view(
        top_result
    )

    if report.empty:

        print(
            "Nenhum campo de relatório disponível."
        )

        return

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    pd.set_option(
        "display.max_colwidth",
        40,
    )

    print(
        report.to_string(
            index=False
        )
    )


# ============================================================
# EXECUTION
# ============================================================

def run_crypto_opportunity_engine() -> Dict:

    started_at = datetime.now(
        timezone.utc
    )

    ensure_output_directory()

    print_header(
        config.ENGINE_NAME
    )

    print(
        f"Version: {config.ENGINE_VERSION}"
    )

    print(
        "Objective:"
    )

    print(
        config.ENGINE_RULES[
            "primary_objective"
        ]
    )

    print(
        "\nBTC excluded: "
        f"{config.ENGINE_RULES['btc_excluded']}"
    )

    # ========================================================
    # 1. MARKET DATA
    # ========================================================

    print_header(
        "1. MARKET UNIVERSE"
    )

    market = build_market_dataset(
        include_history=True,
        history_top_n=config.UNIVERSE_MAX_ASSETS,
    )

    validate_market_dataset(
        market
    )

    market_count = len(
        market
    )

    print(
        f"Market assets collected: {market_count}"
    )

    print(
        f"Memory: {dataframe_memory_mb(market)} MB"
    )

    if market.empty:

        raise RuntimeError(
            "Market dataset is empty."
        )

    # ========================================================
    # 2. UNIVERSE ENGINE
    # ========================================================

    print_header(
        "2. UNIVERSE ENGINE"
    )

    universe_result = (
        run_universe_engine(
            market
        )
    )

    eligible, rejected = (
        split_universe_result(
            universe_result
        )
    )

    eligible_count = len(
        eligible
    )

    rejected_count = len(
        rejected
    )

    print(
        f"Eligible: {eligible_count}"
    )

    print(
        f"Rejected: {rejected_count}"
    )

    if eligible.empty:

        raise RuntimeError(
            "No assets passed the Universe Engine."
        )

    # ========================================================
    # 3. DATA ENRICHMENT
    # ========================================================

    enriched = enrich_dataset(
        eligible
    )

    # ========================================================
    # 4. ANALYSIS ENGINES
    # ========================================================

    final_result = (
        run_analysis_engines(
            enriched
        )
    )

    validate_final_dataset(
        final_result
    )

    # ========================================================
    # 5. TOP OPPORTUNITIES
    # ========================================================

    top_result = (
        get_top_opportunities(
            final_result,
            limit=config.TOP_OPPORTUNITIES,
        )
    )

    # ========================================================
    # 6. OUTPUT
    # ========================================================

    save_outputs(
        full_result=final_result,
        top_result=top_result,
        rejected_result=rejected,
    )

    # ========================================================
    # 7. SUMMARY
    # ========================================================

    finished_at = datetime.now(
        timezone.utc
    )

    summary = build_summary(
        market_count=market_count,
        eligible_count=eligible_count,
        rejected_count=rejected_count,
        final_result=final_result,
        top_result=top_result,
        started_at=started_at,
        finished_at=finished_at,
    )

    save_summary(
        summary
    )

    print_header(
        "ENGINE SUMMARY"
    )

    print(
        summary
    )

    # ========================================================
    # 8. CONSOLE REPORT
    # ========================================================

    print_top_opportunities(
        top_result
    )

    print_header(
        "ENGINE COMPLETED"
    )

    print(
        "Deterministic engine completed successfully."
    )

    print(
        "AI audit is not part of the deterministic decision layer."
    )

    return {
        "market":
            market,

        "eligible":
            eligible,

        "rejected":
            rejected,

        "final":
            final_result,

        "top":
            top_result,

        "summary":
            summary,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    try:

        run_crypto_opportunity_engine()

        return 0

    except KeyboardInterrupt:

        print(
            "\nExecution interrupted by user."
        )

        return 130

    except Exception as error:

        print_header(
            "ENGINE FAILURE"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        print(
            "\nTRACEBACK"
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
