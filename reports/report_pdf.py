"""
CRYPTO OPPORTUNITY ENGINE
PDF Report

OBJETIVO:
Gerar relatório PDF profissional com o resultado determinístico
do Crypto Opportunity Engine.

O relatório apresenta:
- identificação do engine;
- resumo executivo;
- distribuição dos sinais;
- Top Oportunidades Cripto;
- scores fundamentais;
- divergência fundamentos/preço;
- risco;
- timing;
- metodologia.

A camada de relatório NÃO altera nenhuma decisão do engine.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

import config


# ============================================================
# PATHS
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

DEFAULT_PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "crypto_opportunity_report.pdf",
)


# ============================================================
# PAGE
# ============================================================

PAGE_SIZE = landscape(
    A4
)

PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE

LEFT_MARGIN = 10 * mm
RIGHT_MARGIN = 10 * mm
TOP_MARGIN = 14 * mm
BOTTOM_MARGIN = 12 * mm


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


def _format_number(
    value,
    decimals: int = 2,
) -> str:

    number = _safe_float(
        value
    )

    if pd.isna(number):
        return "N/A"

    return f"{number:.{decimals}f}"


def _format_usd(
    value,
) -> str:

    number = _safe_float(
        value
    )

    if pd.isna(number):
        return "N/A"

    absolute = abs(
        number
    )

    if absolute >= 1_000_000_000:

        return (
            f"${number / 1_000_000_000:.2f}B"
        )

    if absolute >= 1_000_000:

        return (
            f"${number / 1_000_000:.2f}M"
        )

    if absolute >= 1_000:

        return (
            f"${number / 1_000:.2f}K"
        )

    if absolute >= 1:

        return (
            f"${number:.2f}"
        )

    return (
        f"${number:.6f}"
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
# STYLES
# ============================================================

def _build_styles():

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "EngineTitle",
        parent=styles[
            "Title"
        ],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    subtitle = ParagraphStyle(
        "EngineSubtitle",
        parent=styles[
            "Normal"
        ],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor(
            "#374151"
        ),
        spaceAfter=5,
    )

    heading = ParagraphStyle(
        "SectionHeading",
        parent=styles[
            "Heading2"
        ],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor(
            "#111827"
        ),
        spaceBefore=5,
        spaceAfter=8,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles[
            "BodyText"
        ],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor(
            "#1f2937"
        ),
        spaceAfter=5,
    )

    small = ParagraphStyle(
        "Small",
        parent=styles[
            "BodyText"
        ],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor(
            "#6b7280"
        ),
    )

    card_title = ParagraphStyle(
        "CardTitle",
        parent=styles[
            "Normal"
        ],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor(
            "#6b7280"
        ),
        alignment=TA_CENTER,
    )

    card_value = ParagraphStyle(
        "CardValue",
        parent=styles[
            "Normal"
        ],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor(
            "#111827"
        ),
        alignment=TA_CENTER,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles[
            "Normal"
        ],
        fontName="Helvetica-Bold",
        fontSize=5.8,
        leading=7,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles[
            "Normal"
        ],
        fontName="Helvetica",
        fontSize=5.8,
        leading=7,
        textColor=colors.HexColor(
            "#111827"
        ),
        alignment=TA_CENTER,
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=table_cell,
        alignment=TA_LEFT,
    )

    return {
        "title": title,
        "subtitle": subtitle,
        "heading": heading,
        "body": body,
        "small": small,
        "card_title": card_title,
        "card_value": card_value,
        "table_header": table_header,
        "table_cell": table_cell,
        "table_cell_left": table_cell_left,
    }


# ============================================================
# HEADER / FOOTER
# ============================================================

def _page_header_footer(
    canvas,
    doc,
):

    canvas.saveState()

    canvas.setStrokeColor(
        colors.HexColor(
            "#d1d5db"
        )
    )

    canvas.setLineWidth(
        0.4
    )

    canvas.line(
        LEFT_MARGIN,
        PAGE_HEIGHT - 9 * mm,
        PAGE_WIDTH - RIGHT_MARGIN,
        PAGE_HEIGHT - 9 * mm,
    )

    canvas.setFont(
        "Helvetica-Bold",
        7,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#374151"
        )
    )

    canvas.drawString(
        LEFT_MARGIN,
        PAGE_HEIGHT - 7 * mm,
        config.ENGINE_NAME,
    )

    canvas.setFont(
        "Helvetica",
        6.5,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#6b7280"
        )
    )

    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        6 * mm,
        f"Página {doc.page}",
    )

    canvas.drawString(
        LEFT_MARGIN,
        6 * mm,
        (
            "Relatório determinístico — "
            "a camada de relatório não altera decisões."
        ),
    )

    canvas.restoreState()


# ============================================================
# SUMMARY CARDS
# ============================================================

def _summary_table(
    full_result: pd.DataFrame,
    top_result: pd.DataFrame,
    styles,
):

    cards = [
        (
            "ATIVOS ANALISADOS",
            str(
                len(
                    full_result
                )
            ),
        ),

        (
            "QUALITY GATE",
            str(
                _count(
                    full_result,
                    "quality_gate",
                    True,
                )
            ),
        ),

        (
            "STRONG ENTRY",
            str(
                _count(
                    full_result,
                    "signal",
                    config.SIGNAL_STRONG_ENTRY,
                )
            ),
        ),

        (
            "ENTRY",
            str(
                _count(
                    full_result,
                    "signal",
                    config.SIGNAL_ENTRY,
                )
            ),
        ),

        (
            "WATCH",
            str(
                _count(
                    full_result,
                    "signal",
                    config.SIGNAL_WATCH,
                )
            ),
        ),

        (
            "BLOCKED",
            str(
                _count(
                    full_result,
                    "signal",
                    config.SIGNAL_BLOCKED,
                )
            ),
        ),

        (
            "TOP",
            str(
                len(
                    top_result
                )
            ),
        ),

        (
            "OPP. MÉDIO",
            _format_number(
                _mean(
                    full_result,
                    "opportunity_score",
                )
            ),
        ),
    ]

    first_row = []
    second_row = []

    for title, value in cards:

        cell = [
            Paragraph(
                title,
                styles[
                    "card_title"
                ],
            ),
            Spacer(
                1,
                2,
            ),
            Paragraph(
                value,
                styles[
                    "card_value"
                ],
            ),
        ]

        if len(
            first_row
        ) < 4:

            first_row.append(
                cell
            )

        else:

            second_row.append(
                cell
            )

    data = [
        first_row,
        second_row,
    ]

    table = Table(
        data,
        colWidths=[
            65 * mm,
            65 * mm,
            65 * mm,
            65 * mm,
        ],
        rowHeights=[
            20 * mm,
            20 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#f9fafb"
                    ),
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#d1d5db"
                    ),
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#e5e7eb"
                    ),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


# ============================================================
# SIGNAL TABLE
# ============================================================

def _signal_table(
    dataset: pd.DataFrame,
    styles,
):

    signals = [
        config.SIGNAL_STRONG_ENTRY,
        config.SIGNAL_ENTRY,
        config.SIGNAL_WATCH,
        config.SIGNAL_WAIT,
        config.SIGNAL_REJECTED,
        config.SIGNAL_BLOCKED,
    ]

    data = [
        [
            Paragraph(
                "SINAL",
                styles[
                    "table_header"
                ],
            ),
            Paragraph(
                "QUANTIDADE",
                styles[
                    "table_header"
                ],
            ),
        ]
    ]

    for signal in signals:

        data.append(
            [
                Paragraph(
                    signal,
                    styles[
                        "table_cell_left"
                    ],
                ),

                Paragraph(
                    str(
                        _count(
                            dataset,
                            "signal",
                            signal,
                        )
                    ),
                    styles[
                        "table_cell"
                    ],
                ),
            ]
        )

    table = Table(
        data,
        colWidths=[
            70 * mm,
            35 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#111827"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor(
                        "#d1d5db"
                    ),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#f9fafb"
                        ),
                    ],
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


# ============================================================
# TOP OPPORTUNITIES TABLE
# ============================================================

def _top_opportunities_table(
    dataset: pd.DataFrame,
    styles,
):

    if dataset.empty:

        return Paragraph(
            (
                "Nenhum ativo passou por todos "
                "os Quality Gates."
            ),
            styles[
                "body"
            ],
        )

    columns = [
        (
            "rank",
            "#",
            7,
        ),

        (
            "symbol",
            "ATIVO",
            13,
        ),

        (
            "category",
            "CATEG.",
            18,
        ),

        (
            "market_cap",
            "M.CAP",
            20,
        ),

        (
            "price",
            "PREÇO",
            18,
        ),

        (
            "opportunity_score",
            "OPP.",
            13,
        ),

        (
            "fundamental_acceleration_score",
            "FUND.",
            13,
        ),

        (
            "revenue_score",
            "REV.",
            13,
        ),

        (
            "holder_value_score",
            "HOLDER",
            14,
        ),

        (
            "dilution_score",
            "DIL.",
            13,
        ),

        (
            "valuation_score",
            "VAL.",
            13,
        ),

        (
            "onchain_score",
            "ONCH.",
            13,
        ),

        (
            "tvl_score",
            "TVL",
            12,
        ),

        (
            "narrative_score",
            "NARR.",
            13,
        ),

        (
            "fundamental_price_divergence",
            "DIV.",
            13,
        ),

        (
            "risk_level",
            "RISCO",
            17,
        ),

        (
            "timing_score",
            "TIMING",
            14,
        ),

        (
            "signal",
            "SINAL",
            24,
        ),

        (
            "decision_confidence",
            "CONF.",
            14,
        ),
    ]

    available = [
        column
        for column in columns
        if column[0] in dataset.columns
    ]

    header = []

    for _, label, _ in available:

        header.append(
            Paragraph(
                label,
                styles[
                    "table_header"
                ],
            )
        )

    data = [
        header
    ]

    numeric_score_fields = {
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
    }

    left_fields = {
        "symbol",
        "category",
        "signal",
    }

    for _, row in dataset.iterrows():

        table_row = []

        for field, _, _ in available:

            value = _value(
                row,
                field,
            )

            if field == "market_cap":

                display = _format_usd(
                    value
                )

            elif field == "price":

                display = _format_usd(
                    value
                )

            elif field in numeric_score_fields:

                display = _format_number(
                    value
                )

            else:

                display = str(
                    value
                )

            style = (
                styles[
                    "table_cell_left"
                ]
                if field in left_fields
                else styles[
                    "table_cell"
                ]
            )

            table_row.append(
                Paragraph(
                    display,
                    style,
                )
            )

        data.append(
            table_row
        )

    total_units = sum(
        column[2]
        for column in available
    )

    available_width = (
        PAGE_WIDTH
        - LEFT_MARGIN
        - RIGHT_MARGIN
    )

    col_widths = [
        available_width
        * column[2]
        / total_units
        for column in available
    ]

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#111827"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.HexColor(
                        "#d1d5db"
                    ),
                ),

                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#f9fafb"
                        ),
                    ],
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    return table


# ============================================================
# METHODOLOGY
# ============================================================

def _methodology_story(
    styles,
):

    weights = (
        config.OPPORTUNITY_WEIGHTS
    )

    paragraphs = [
        (
            "<b>Objetivo.</b> "
            "Encontrar criptoativos de qualidade em que a melhora "
            "econômica e fundamental esteja avançando mais rapidamente "
            "que a precificação do mercado."
        ),

        (
            "<b>Opportunity Score.</b> "
            f"Fundamental Acceleration "
            f"{weights['fundamental_acceleration'] * 100:.0f}%; "
            f"Revenue/Economics "
            f"{weights['revenue_economics'] * 100:.0f}%; "
            f"Holder Value Capture "
            f"{weights['holder_value_capture'] * 100:.0f}%; "
            f"Valuation "
            f"{weights['valuation'] * 100:.0f}%; "
            f"Tokenomics/Dilution "
            f"{weights['tokenomics_dilution'] * 100:.0f}%; "
            f"On-chain Growth "
            f"{weights['onchain_growth'] * 100:.0f}%; "
            f"TVL/Utilization "
            f"{weights['tvl_utilization'] * 100:.0f}%; "
            f"Narrative "
            f"{weights['narrative'] * 100:.0f}%."
        ),

        (
            "<b>Quality Gates.</b> "
            f"Opportunity >= {config.MIN_OPPORTUNITY_SCORE}; "
            f"Fundamental >= {config.MIN_FUNDAMENTAL_SCORE}; "
            f"Revenue >= {config.MIN_REVENUE_SCORE}; "
            f"Holder Value >= {config.MIN_HOLDER_VALUE_SCORE}; "
            f"Dilution >= {config.MIN_DILUTION_SCORE}."
        ),

        (
            "<b>Divergência Fundamentos/Preço.</b> "
            "O motor procura melhora em fundamentos, receita, "
            "utilização, TVL e atividade on-chain que ainda não tenha "
            "sido totalmente refletida no preço."
        ),

        (
            "<b>Holder Value.</b> "
            "Receita do protocolo não é suficiente por si só. "
            "O motor procura mecanismos de captura econômica pelo token, "
            "incluindo distribuição de taxas, real yield, buyback, burn, "
            "uso obrigatório do token e staking econômico."
        ),

        (
            "<b>Diluição.</b> "
            "Inflação, baixa oferta circulante, diferença entre Market Cap "
            "e FDV e grandes unlocks podem reduzir ou bloquear a oportunidade."
        ),

        (
            "<b>TVL.</b> "
            "TVL é analisado conforme a categoria. "
            "TVL alto isoladamente não é considerado oportunidade."
        ),

        (
            "<b>Narrativa.</b> "
            f"Possui apenas {weights['narrative'] * 100:.0f}% do score. "
            "Narrativa não pode substituir fundamentos."
        ),

        (
            "<b>Timing.</b> "
            "É analisado separadamente e não pode resgatar projeto "
            "que falhou nos Quality Gates."
        ),

        (
            "<b>Risco.</b> "
            "É independente do Opportunity Score. "
            "Risco crítico ou diluição crítica podem bloquear o ativo."
        ),

        (
            "<b>BTC.</b> "
            "Bitcoin e representações wrapped configuradas pelo engine "
            "são excluídos do universo de oportunidades."
        ),

        (
            "<b>Auditoria IA.</b> "
            "A auditoria por inteligência artificial é uma camada "
            "independente e posterior. Ela não participa da decisão "
            "determinística do engine."
        ),
    ]

    story = []

    for text in paragraphs:

        story.append(
            Paragraph(
                text,
                styles[
                    "body"
                ],
            )
        )

        story.append(
            Spacer(
                1,
                2,
            )
        )

    return story


# ============================================================
# BUILD PDF
# ============================================================

def build_pdf_report(
    full_result: pd.DataFrame,
    top_result: Optional[pd.DataFrame] = None,
    output_file: str = DEFAULT_PDF_FILE,
) -> str:

    os.makedirs(
        os.path.dirname(
            output_file
        ),
        exist_ok=True,
    )

    styles = _build_styles()

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
                    .reset_index(
                        drop=True
                    )
                )

                if (
                    "rank"
                    not in top_result.columns
                ):

                    top_result[
                        "rank"
                    ] = np.arange(
                        1,
                        len(
                            top_result
                        ) + 1,
                    )

        else:

            top_result = pd.DataFrame()

    generated_at = datetime.now(
        timezone.utc
    )

    generated_text = (
        generated_at.strftime(
            "%d/%m/%Y %H:%M:%S UTC"
        )
    )

    doc = BaseDocTemplate(
        output_file,
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=config.ENGINE_NAME,
        author="CRYPTO OPPORTUNITY ENGINE",
        subject=(
            "Relatório de oportunidades cripto"
        ),
    )

    frame = Frame(
        LEFT_MARGIN,
        BOTTOM_MARGIN,
        PAGE_WIDTH
        - LEFT_MARGIN
        - RIGHT_MARGIN,
        PAGE_HEIGHT
        - TOP_MARGIN
        - BOTTOM_MARGIN,
        id="main_frame",
    )

    template = PageTemplate(
        id="main_template",
        frames=[
            frame
        ],
        onPage=_page_header_footer,
    )

    doc.addPageTemplates(
        [
            template
        ]
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            config.ENGINE_NAME,
            styles[
                "title"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                f"Versão {config.ENGINE_VERSION} "
                f"| Gerado em {generated_text}"
            ),
            styles[
                "subtitle"
            ],
        )
    )

    story.append(
        Paragraph(
            config.ENGINE_RULES[
                "primary_objective"
            ],
            styles[
                "subtitle"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Resumo Executivo",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        _summary_table(
            full_result,
            top_result,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    opportunity_mean = _mean(
        full_result,
        "opportunity_score",
    )

    divergence_mean = _mean(
        full_result,
        "fundamental_price_divergence",
    )

    timing_mean = _mean(
        full_result,
        "timing_score",
    )

    confidence_mean = _mean(
        full_result,
        "decision_confidence",
    )

    executive_text = (
        f"Opportunity Score médio: "
        f"<b>{_format_number(opportunity_mean)}</b> | "
        f"Divergência média: "
        f"<b>{_format_number(divergence_mean)}</b> | "
        f"Timing médio: "
        f"<b>{_format_number(timing_mean)}</b> | "
        f"Confiança média: "
        f"<b>{_format_number(confidence_mean)}%</b>."
    )

    story.append(
        Paragraph(
            executive_text,
            styles[
                "body"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    # ========================================================
    # SIGNALS
    # ========================================================

    story.append(
        Paragraph(
            "Distribuição dos Sinais",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        _signal_table(
            full_result,
            styles,
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # TOP OPPORTUNITIES
    # ========================================================

    story.append(
        Paragraph(
            "Top Oportunidades Cripto",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Ranking determinístico. "
                "Ativos bloqueados ou que falharam nos Quality Gates "
                "não aparecem como oportunidades aprovadas."
            ),
            styles[
                "small"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        _top_opportunities_table(
            top_result,
            styles,
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    story.append(
        Paragraph(
            "Metodologia do Engine",
            styles[
                "heading"
            ],
        )
    )

    methodology = (
        _methodology_story(
            styles
        )
    )

    story.extend(
        methodology
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    # ========================================================
    # ENGINE RULES
    # ========================================================

    story.append(
        Paragraph(
            "Regras Estruturais",
            styles[
                "heading"
            ],
        )
    )

    rules = [
        (
            "Narrativa não pode substituir fundamentos.",
            config.ENGINE_RULES.get(
                "narrative_cannot_override_fundamentals",
                True,
            ),
        ),

        (
            "Timing não pode resgatar projeto ruim.",
            config.ENGINE_RULES.get(
                "timing_cannot_rescue_bad_project",
                True,
            ),
        ),

        (
            "TVL alto isoladamente não representa oportunidade.",
            config.ENGINE_RULES.get(
                "high_tvl_alone_is_not_opportunity",
                True,
            ),
        ),

        (
            "Receita sem captura de valor pelo holder não é suficiente.",
            config.ENGINE_RULES.get(
                "high_revenue_without_holder_capture_is_not_enough",
                True,
            ),
        ),

        (
            "Yield originado apenas por emissão não é real yield.",
            config.ENGINE_RULES.get(
                "high_yield_from_token_emission_is_not_real_yield",
                True,
            ),
        ),

        (
            "Diluição crítica pode bloquear o ativo.",
            config.ENGINE_RULES.get(
                "critical_dilution_can_block_asset",
                True,
            ),
        ),

        (
            "Risco crítico pode bloquear o ativo.",
            config.ENGINE_RULES.get(
                "critical_risk_can_block_asset",
                True,
            ),
        ),

        (
            "Divergência fundamentos/preço é sinal central.",
            config.ENGINE_RULES.get(
                "fundamental_price_divergence_is_core_signal",
                True,
            ),
        ),

        (
            "Métricas específicas por categoria são obrigatórias.",
            config.ENGINE_RULES.get(
                "category_specific_metrics_required",
                True,
            ),
        ),
    ]

    rule_data = [
        [
            Paragraph(
                "REGRA",
                styles[
                    "table_header"
                ],
            ),
            Paragraph(
                "ATIVA",
                styles[
                    "table_header"
                ],
            ),
        ]
    ]

    for description, enabled in rules:

        rule_data.append(
            [
                Paragraph(
                    description,
                    styles[
                        "table_cell_left"
                    ],
                ),

                Paragraph(
                    (
                        "SIM"
                        if enabled
                        else "NÃO"
                    ),
                    styles[
                        "table_cell"
                    ],
                ),
            ]
        )

    rules_table = Table(
        rule_data,
        colWidths=[
            220 * mm,
            30 * mm,
        ],
        repeatRows=1,
    )

    rules_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#111827"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor(
                        "#d1d5db"
                    ),
                ),

                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#f9fafb"
                        ),
                    ],
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                rules_table
            ]
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Importante:</b> "
                "este relatório representa a saída do motor quantitativo "
                "determinístico. A futura auditoria por IA deverá atuar "
                "como camada independente de validação e não deverá "
                "reescrever scores, alterar Quality Gates ou substituir "
                "a decisão produzida pelo engine."
            ),
            styles[
                "body"
            ],
        )
    )

    doc.build(
        story
    )

    print(
        "[report_pdf] "
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

    build_pdf_report(
        full_result=full_result,
        top_result=top_result,
    )
