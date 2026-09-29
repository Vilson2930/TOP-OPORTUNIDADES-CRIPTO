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
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(
            __file__
        )
    )
)

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


# ============================================================
# SIGNAL TABLE
# ============================================================

def _build_signal_table(
    dataset: pd.DataFrame,
) -> str:

    signals = [
        config.SIGNAL_STRONG_ENTRY,
        config.SIGNAL_ENTRY,
        config.SIGNAL_WATCH,
        config.SIGNAL_WAIT,
        config.SIGNAL_REJECTED,
        config.SIGNAL_BLOCKED,
    ]

    rows = []

    for signal in signals:

        rows.append(
            (
                signal,
                _count(
                    dataset,
                    "signal",
                    signal,
                ),
            )
        )

    output = [
        "<table>",
        "<thead>",
        "<tr>",
        "<th>Sinal</th>",
        "<th>Quantidade</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for signal, count in rows:

        output.append(
            "<tr>"
            f"<td>{_escape(signal)}</td>"
            f"<td>{count}</td>"
            "</tr>"
        )

    output.extend(
        [
            "</tbody>",
            "</table>",
        ]
    )

    return "\n".join(
        output
    )


# ============================================================
# METHODOLOGY
# ============================================================

def _build_methodology() -> str:

    weights = (
        config.OPPORTUNITY_WEIGHTS
    )

    return f"""
    <div class="methodology">

        <p>
            <strong>Objetivo:</strong>
            encontrar criptoativos de qualidade em que a melhora
            econômica e fundamental esteja avançando mais rapidamente
            que a precificação do mercado.
        </p>

        <p>
            <strong>Opportunity Score:</strong>
            Fundamental Acceleration
            {weights["fundamental_acceleration"] * 100:.0f}% |
            Revenue/Economics
            {weights["revenue_economics"] * 100:.0f}% |
            Holder Value Capture
            {weights["holder_value_capture"] * 100:.0f}% |
            Valuation
            {weights["valuation"] * 100:.0f}% |
            Tokenomics/Dilution
            {weights["tokenomics_dilution"] * 100:.0f}% |
            On-chain Growth
            {weights["onchain_growth"] * 100:.0f}% |
            TVL/Utilization
            {weights["tvl_utilization"] * 100:.0f}% |
            Narrative
            {weights["narrative"] * 100:.0f}%.
        </p>

        <p>
            <strong>Quality Gates:</strong>
            Opportunity ≥ {config.MIN_OPPORTUNITY_SCORE},
            Fundamental ≥ {config.MIN_FUNDAMENTAL_SCORE},
            Revenue ≥ {config.MIN_REVENUE_SCORE},
            Holder Value ≥ {config.MIN_HOLDER_VALUE_SCORE},
            Dilution ≥ {config.MIN_DILUTION_SCORE}.
        </p>

        <p>
            <strong>Divergência Fundamentos/Preço:</strong>
            é um sinal central de assimetria.
            O motor procura melhora em fundamentos, receita,
            utilização, TVL e on-chain que ainda não tenha sido
            totalmente refletida no preço.
        </p>

        <p>
            <strong>Timing:</strong>
            é analisado separadamente.
            Um timing forte não pode resgatar um projeto que falhou
            nos Quality Gates.
        </p>

        <p>
            <strong>Risco:</strong>
            é independente do Opportunity Score.
            Risco crítico ou diluição crítica podem bloquear o ativo.
        </p>

        <p>
            <strong>Narrativa:</strong>
            possui peso limitado a
            {weights["narrative"] * 100:.0f}% e não pode substituir
            fundamentos reais.
        </p>

    </div>
    """


# ============================================================
# HTML DOCUMENT
# ============================================================

def build_html_report(
    full_result: pd.DataFrame,
    top_result: Optional[pd.DataFrame] = None,
    generated_at: Optional[datetime] = None,
) -> str:

    if generated_at is None:

        generated_at = datetime.now(
            timezone.utc
        )

    if top_result is None:

        if (
            "quality_gate"
            in full_result.columns
            and "hard_block"
            in full_result.columns
        ):

            top_result = full_result[
                (
                    full_result[
                        "quality_gate"
                    ]
                    == True
                )
                &
                (
                    full_result[
                        "hard_block"
                    ]
                    == False
                )
            ].copy()

            if (
                "opportunity_score"
                in top_result.columns
            ):

                top_result = (
                    top_result
                    .sort_values(
                        "opportunity_score",
                        ascending=False,
                        na_position="last",
                    )
                    .head(
                        config.TOP_OPPORTUNITIES
                    )
                )

        else:

            top_result = pd.DataFrame()

    generated_text = (
        generated_at
        .astimezone(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    html_document = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">

    <head>

        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <title>
            {html.escape(config.ENGINE_NAME)}
        </title>

        {_build_css()}

    </head>

    <body>

        <div class="container">

            <div class="header">

                <h1>
                    {html.escape(config.ENGINE_NAME)}
                </h1>

                <p>
                    Versão:
                    {html.escape(config.ENGINE_VERSION)}
                </p>

                <p>
                    Gerado em:
                    {html.escape(generated_text)}
                </p>

                <p>
                    {html.escape(
                        config.ENGINE_RULES[
                            "primary_objective"
                        ]
                    )}
                </p>

            </div>

            <div class="section">

                <h2>
                    Resumo Executivo
                </h2>

                {_summary_cards(
                    full_result,
                    top_result,
                )}

            </div>

            <div class="section">

                <h2>
                    Distribuição dos Sinais
                </h2>

                {_build_signal_table(
                    full_result
                )}

            </div>

            <div class="section">

                <h2>
                    Top Oportunidades Cripto
                </h2>

                <p class="small">
                    Ranking determinístico.
                    BTC excluído.
                    Ativos bloqueados ou que falharam nos Quality Gates
                    não aparecem como oportunidades aprovadas.
                </p>

                {_build_top_table(
                    top_result
                )}

            </div>

            <div class="section">

                <h2>
                    Metodologia
                </h2>

                {_build_methodology()}

            </div>

            <div class="footer">

                {html.escape(config.ENGINE_NAME)}
                —
                relatório gerado automaticamente pelo motor determinístico.

                <br>

                A camada de relatório não altera scores,
                sinais ou decisões do engine.

            </div>

        </div>

    </body>

    </html>
    """

    return html_document


# ============================================================
# SAVE
# ============================================================

def save_html_report(
    full_result: pd.DataFrame,
    top_result: Optional[pd.DataFrame] = None,
    output_file: str = DEFAULT_HTML_FILE,
) -> str:

    os.makedirs(
        os.path.dirname(
            output_file
        ),
        exist_ok=True,
    )

    html_content = (
        build_html_report(
            full_result=full_result,
            top_result=top_result,
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            html_content
        )

    print(
        "[report_html] "
        f"saved: {output_file}"
    )

    return output_file


# ============================================================
# EXECUÇÃO ISOLADA
# ============================================================

if __name__ == "__main__":

    full_file = os.path.join(
        OUTPUT_DIR,
        "crypto_opportunity_full.csv",
    )

    top_file = os.path.join(
        OUTPUT_DIR,
        "crypto_opportunities_top.csv",
    )

    if not os.path.exists(
        full_file
    ):

        raise FileNotFoundError(
            "Execute main.py primeiro. "
            "Arquivo não encontrado: "
            f"{full_file}"
        )

    full_result = pd.read_csv(
        full_file
    )

    if os.path.exists(
        top_file
    ):

        top_result = pd.read_csv(
            top_file
        )

    else:

        top_result = None

    save_html_report(
        full_result=full_result,
        top_result=top_result,
    )
