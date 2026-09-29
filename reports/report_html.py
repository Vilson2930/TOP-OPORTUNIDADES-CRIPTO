"""
CRYPTO OPPORTUNITY ENGINE
HTML Report

OBJETIVO:
Gerar relatório HTML profissional com as principais oportunidades
identificadas pelo motor determinístico.

O relatório apresenta:
- resumo executivo;
- estatísticas do universo;
- distribuição dos sinais;
- Top Oportunidades;
- scores fundamentais;
- divergência fundamentos/preço;
- risco;
- timing;
- confiança da decisão.

A camada de relatório NÃO altera nenhuma decisão do engine.
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(
            __file__
        )
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(
        0,
        BASE_DIR,
    )

import config


# ============================================================
# PATH
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
)

DEFAULT_HTML_FILE = os.path.join(
    OUTPUT_DIR,
    "crypto_opportunity_report.html",
)


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


def _escape(value) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ) and np.isnan(value):
        return ""

    return html.escape(
        str(value)
    )


def _format_number(
    value,
    decimals: int = 2,
) -> str:

    number = _safe_float(
        value
    )

    if pd.isna(number):
        return "N/A"

    return f"{number:,.{decimals}f}"


def _format_usd(
    value,
) -> str:

    number = _safe_float(
        value
    )

    if pd.isna(number):
        return "N/A"

    abs_value = abs(
        number
    )

    if abs_value >= 1_000_000_000:

        return (
            f"${number / 1_000_000_000:,.2f}B"
        )

    if abs_value >= 1_000_000:

        return (
            f"${number / 1_000_000:,.2f}M"
        )

    if abs_value >= 1_000:

        return (
            f"${number / 1_000:,.2f}K"
        )

    return (
        f"${number:,.4f}"
    )


def _format_percent(
    value,
) -> str:

    number = _safe_float(
        value
    )

    if pd.isna(number):
        return "N/A"

    if abs(number) <= 2:

        number *= 100

    return (
        f"{number:.2f}%"
    )


def _count(
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
        ).sum()
    )


def _mean(
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


def _value(
    row: pd.Series,
    field: str,
    default="N/A",
):

    if field not in row.index:
        return default

    value = row.get(
        field
    )

    if value is None:
        return default

    try:

        if pd.isna(value):
            return default

    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


# ============================================================
# CSS
# ============================================================

def _build_css() -> str:

    return """
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            background: #f4f6f8;
            color: #17202a;
            font-family:
                Arial,
                Helvetica,
                sans-serif;
        }

        .container {
            width: 96%;
            max-width: 1500px;
            margin: 24px auto;
        }

        .header {
            background: #111827;
            color: white;
            padding: 28px 32px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .header h1 {
            margin: 0 0 8px 0;
            font-size: 28px;
        }

        .header p {
            margin: 5px 0;
            color: #d1d5db;
            line-height: 1.5;
        }

        .section {
            background: white;
            padding: 22px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow:
                0 1px 4px
                rgba(0, 0, 0, 0.08);
        }

        .section h2 {
            margin-top: 0;
            font-size: 20px;
            border-bottom:
                1px solid #e5e7eb;
            padding-bottom: 10px;
        }

        .cards {
            display: grid;
            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(180px, 1fr)
                );
            gap: 12px;
        }

        .card {
            border:
                1px solid #e5e7eb;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }

        .card-title {
            font-size: 12px;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 8px;
        }

        .card-value {
            font-size: 24px;
            font-weight: bold;
            color: #111827;
        }

        .table-wrapper {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        th {
            background: #111827;
            color: white;
            padding: 9px 7px;
            text-align: center;
            white-space: nowrap;
        }

        td {
            border-bottom:
                1px solid #e5e7eb;
            padding: 8px 7px;
            text-align: center;
            white-space: nowrap;
        }

        tr:nth-child(even) {
            background: #f9fafb;
        }

        .left {
            text-align: left;
        }

        .signal-strong {
            font-weight: bold;
        }

        .signal-entry {
            font-weight: bold;
        }

        .blocked {
            font-weight: bold;
        }

        .small {
            font-size: 11px;
            color: #6b7280;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 11px;
            margin: 25px 0;
        }

        .methodology {
            line-height: 1.6;
            font-size: 13px;
        }

        .methodology strong {
            color: #111827;
        }
    </style>
    """


# ============================================================
# SUMMARY CARDS
# ============================================================

def _summary_cards(
    full_result: pd.DataFrame,
    top_result: pd.DataFrame,
) -> str:

    cards = [
        (
            "Ativos analisados",
            len(
                full_result
            ),
        ),

        (
            "Quality Gate",
            _count(
                full_result,
                "quality_gate",
                True,
            ),
        ),

        (
            "Strong Entry",
            _count(
                full_result,
                "signal",
                config.SIGNAL_STRONG_ENTRY,
            ),
        ),

        (
            "Entry",
            _count(
                full_result,
                "signal",
                config.SIGNAL_ENTRY,
            ),
        ),

        (
            "Watch",
            _count(
                full_result,
                "signal",
                config.SIGNAL_WATCH,
            ),
        ),

        (
            "Blocked",
            _count(
                full_result,
                "signal",
                config.SIGNAL_BLOCKED,
            ),
        ),

        (
            "Top oportunidades",
            len(
                top_result
            ),
        ),

        (
            "Opportunity médio",
            _format_number(
                _mean(
                    full_result,
                    "opportunity_score",
                )
            ),
        ),

        (
            "Divergência média",
            _format_number(
                _mean(
                    full_result,
                    "fundamental_price_divergence",
                )
            ),
        ),

        (
            "Confiança média",
            (
                _format_number(
                    _mean(
                        full_result,
                        "decision_confidence",
                    )
                )
                + "%"
            ),
        ),
    ]

    output = [
        '<div class="cards">'
    ]

    for title, value in cards:

        output.append(
            """
            <div class="card">
                <div class="card-title">
                    {title}
                </div>
                <div class="card-value">
                    {value}
                </div>
            </div>
            """.format(
                title=_escape(
                    title
                ),
                value=_escape(
                    value
                ),
            )
        )

    output.append(
        "</div>"
    )

    return "\n".join(
        output
    )


# ============================================================
# TOP TABLE
# ============================================================

def _build_top_table(
    dataset: pd.DataFrame,
) -> str:

    if dataset.empty:

        return (
            "<p>"
            "Nenhum ativo passou por todos os Quality Gates."
            "</p>"
        )

    columns = [
        (
            "rank",
            "Rank",
        ),

        (
            "symbol",
            "Ativo",
        ),

        (
            "name",
            "Nome",
        ),

        (
            "category",
            "Categoria",
        ),

        (
            "market_cap",
            "Market Cap",
        ),

        (
            "price",
            "Preço",
        ),

        (
            "opportunity_score",
            "Opportunity",
        ),

        (
            "fundamental_acceleration_score",
            "Fundamental",
        ),

        (
            "revenue_score",
            "Revenue",
        ),

        (
            "holder_value_score",
            "Holder Value",
        ),

        (
            "dilution_score",
            "Dilution",
        ),

        (
            "valuation_score",
            "Valuation",
        ),

        (
            "onchain_score",
            "On-chain",
        ),

        (
            "tvl_score",
            "TVL",
        ),

        (
            "narrative_score",
            "Narrative",
        ),

        (
            "fundamental_price_divergence",
            "Divergência",
        ),

        (
            "risk_level",
            "Risco",
        ),

        (
            "timing_score",
            "Timing",
        ),

        (
            "signal",
            "Sinal",
        ),

        (
            "decision_confidence",
            "Confiança",
        ),
    ]

    available = [
        item
        for item in columns
        if item[0] in dataset.columns
    ]

    output = [
        '<div class="table-wrapper">',
        "<table>",
        "<thead>",
        "<tr>",
    ]

    for _, title in available:

        output.append(
            f"<th>{_escape(title)}</th>"
        )

    output.extend(
        [
            "</tr>",
            "</thead>",
            "<tbody>",
        ]
    )

    for _, row in dataset.iterrows():

        output.append(
            "<tr>"
        )

        for field, _ in available:

            value = _value(
                row,
                field,
            )

            css_class = ""

            if field in {
                "symbol",
                "name",
                "category",
            }:
                css_class = "left"

            if field == "signal":

                if (
                    value
                    == config.SIGNAL_STRONG_ENTRY
                ):
                    css_class = (
                        "signal-strong"
                    )

                elif (
                    value
                    == config.SIGNAL_ENTRY
                ):
                    css_class = (
                        "signal-entry"
                    )

                elif (
                    value
                    == config.SIGNAL_BLOCKED
                ):
                    css_class = (
                        "blocked"
                    )

            if field in {
                "market_cap",
            }:

                display = _format_usd(
                    value
                )

            elif field == "price":

                display = _format_usd(
                    value
                )

            elif field in {
                "opportunity_score",
                "fundamental_acceleration_score",
                "revenue_score",
                "holder_value_score",
                "dilution_score",
                "valuation_score",
                "onchain_score",
                "tvl_score",
                "narrative_score",
                "fundamental_price_divergence",
                "timing_score",
                "decision_confidence",
            }:

                display = _format_number(
                    value
                )

            else:

                display = _escape(
                    value
                )

            output.append(
                (
                    f'<td class="{css_class}">'
                    f"{display}"
                    "</td>"
                )
            )

        output.append(
            "</tr>"
        )

    output.extend(
        [
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )

    return "\n".join(
        output
    )
