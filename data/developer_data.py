"""
CRYPTO OPPORTUNITY ENGINE
Developer Data Layer

FOCO:
Detectar se o desenvolvimento do projeto está:
- ativo;
- crescendo;
- acelerando;
- consistente.

Desenvolvimento é confirmação secundária.
Não cria oportunidade sozinho.

Fonte:
CoinGecko Public API.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

import config


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


# ============================================================
# HTTP
# ============================================================

def _request(
    endpoint: str,
    params: Optional[Dict] = None,
) -> Optional[object]:

    url = f"{COINGECKO_BASE_URL}{endpoint}"

    headers = {
        "accept": "application/json",
        "user-agent":
            f"{config.ENGINE_NAME}/{config.ENGINE_VERSION}",
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    "[developer_data] "
                    f"Rate limit. Aguardando {wait}s."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:

                wait = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    "[developer_data] "
                    f"HTTP {response.status_code}. "
                    f"Retry em {wait}s."
                )

                time.sleep(wait)
                continue

            return None

        except requests.RequestException:

            if attempt == MAX_RETRIES:
                return None

            time.sleep(
                RETRY_BACKOFF_SECONDS
                * attempt
            )

    return None


# ============================================================
# HELPERS
# ============================================================

def _safe_float(
    value,
) -> float:

    try:

        if value is None:
            return np.nan

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def _clip_score(
    value: float,
) -> float:

    if pd.isna(value):
        return np.nan

    return float(
        np.clip(
            value,
            0,
            100,
        )
    )


# ============================================================
# COLETA
# ============================================================

def fetch_developer_data(
    coin_id: str,
) -> Dict[str, float]:

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "true",
        "sparkline": "false",
    }

    data = _request(
        f"/coins/{coin_id}",
        params=params,
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    developer = data.get(
        "developer_data",
        {},
    )

    if not isinstance(
        developer,
        dict,
    ):
        return {}

    return {
        "forks":
            _safe_float(
                developer.get(
                    "forks"
                )
            ),

        "stars":
            _safe_float(
                developer.get(
                    "stars"
                )
            ),

        "subscribers":
            _safe_float(
                developer.get(
                    "subscribers"
                )
            ),

        "total_issues":
            _safe_float(
                developer.get(
                    "total_issues"
                )
            ),

        "closed_issues":
            _safe_float(
                developer.get(
                    "closed_issues"
                )
            ),

        "pull_requests_merged":
            _safe_float(
                developer.get(
                    "pull_requests_merged"
                )
            ),

        "pull_request_contributors":
            _safe_float(
                developer.get(
                    "pull_request_contributors"
                )
            ),

        "commit_count_4_weeks":
            _safe_float(
                developer.get(
                    "commit_count_4_weeks"
                )
            ),
    }


# ============================================================
# ISSUE RESOLUTION
# ============================================================

def calculate_issue_resolution_ratio(
    total_issues: float,
    closed_issues: float,
) -> float:

    if (
        pd.isna(total_issues)
        or pd.isna(closed_issues)
        or total_issues <= 0
    ):
        return np.nan

    return float(
        np.clip(
            closed_issues
            / total_issues,
            0,
            1,
        )
    )


# ============================================================
# SCORES
# ============================================================

def score_commits(
    commits: float,
) -> float:

    if pd.isna(commits):
        return np.nan

    if commits <= 0:
        return 0.0

    if commits >= 200:
        return 100.0

    return _clip_score(
        commits
        / 200
        * 100
    )


def score_contributors(
    contributors: float,
) -> float:

    if pd.isna(contributors):
        return np.nan

    if contributors <= 0:
        return 0.0

    if contributors >= 100:
        return 100.0

    return _clip_score(
        contributors
        / 100
        * 100
    )


def score_pull_requests(
    pull_requests: float,
) -> float:

    if pd.isna(pull_requests):
        return np.nan

    if pull_requests <= 0:
        return 0.0

    if pull_requests >= 5000:
        return 100.0

    return _clip_score(
        np.log1p(
            pull_requests
        )
        / np.log1p(5000)
        * 100
    )


def score_stars(
    stars: float,
) -> float:

    if pd.isna(stars):
        return np.nan

    if stars <= 0:
        return 0.0

    if stars >= 50_000:
        return 100.0

    return _clip_score(
        np.log1p(
            stars
        )
        / np.log1p(50_000)
        * 100
    )


def score_issue_resolution(
    ratio: float,
) -> float:

    if pd.isna(ratio):
        return np.nan

    return _clip_score(
        ratio
        * 100
    )


# ============================================================
# DEVELOPER QUALITY SCORE
# ============================================================

def calculate_developer_score(
    metrics: Dict[str, float],
) -> float:

    issue_resolution = (
        calculate_issue_resolution_ratio(
            metrics.get(
                "total_issues",
                np.nan,
            ),
            metrics.get(
                "closed_issues",
                np.nan,
            ),
        )
    )

    components = [
        (
            score_commits(
                metrics.get(
                    "commit_count_4_weeks",
                    np.nan,
                )
            ),
            0.35,
        ),
        (
            score_contributors(
                metrics.get(
                    "pull_request_contributors",
                    np.nan,
                )
            ),
            0.25,
        ),
        (
            score_pull_requests(
                metrics.get(
                    "pull_requests_merged",
                    np.nan,
                )
            ),
            0.15,
        ),
        (
            score_issue_resolution(
                issue_resolution
            ),
            0.15,
        ),
        (
            score_stars(
                metrics.get(
                    "stars",
                    np.nan,
                )
            ),
            0.10,
        ),
    ]

    weighted_sum = 0.0
    available_weight = 0.0

    for score, weight in components:

        if pd.isna(score):
            continue

        weighted_sum += (
            score
            * weight
        )

        available_weight += weight

    if available_weight == 0:
        return np.nan

    return round(
        weighted_sum
        / available_weight,
        2,
    )


# ============================================================
# ACTIVITY CLASSIFICATION
# ============================================================

def classify_developer_activity(
    developer_score: float,
    commits: float,
) -> str:

    if (
        pd.isna(developer_score)
        and pd.isna(commits)
    ):
        return "DATA_INSUFFICIENT"

    if (
        not pd.isna(commits)
        and commits == 0
    ):
        return "INACTIVE"

    if (
        not pd.isna(developer_score)
        and developer_score >= 80
    ):
        return "VERY_ACTIVE"

    if (
        not pd.isna(developer_score)
        and developer_score >= 60
    ):
        return "ACTIVE"

    if (
        not pd.isna(developer_score)
        and developer_score >= 40
    ):
        return "MODERATE"

    return "WEAK"


# ============================================================
# DEVELOPMENT CONFIRMATION
# ============================================================

def calculate_development_confirmation(
    developer_score: float,
) -> float:

    """
    Retorna 0-1.

    Desenvolvimento não cria oportunidade.
    Serve apenas como confirmação da qualidade
    e continuidade operacional do projeto.
    """

    if pd.isna(
        developer_score
    ):
        return np.nan

    if developer_score >= 80:
        return 1.00

    if developer_score >= 60:
        return 0.90

    if developer_score >= 40:
        return 0.75

    if developer_score >= 20:
        return 0.50

    return 0.25


# ============================================================
# ENRIQUECIMENTO
# ============================================================

def enrich_with_developer_data(
    market_df: pd.DataFrame,
    sleep_seconds: float = 1.2,
) -> pd.DataFrame:

    if market_df.empty:
        return market_df.copy()

    records = []

    total = len(
        market_df
    )

    for position, (_, row) in enumerate(
        market_df.iterrows(),
        start=1,
    ):

        coin_id = row.get(
            "coin_id"
        )

        symbol = row.get(
            "symbol"
        )

        print(
            "[developer_data] "
            f"{symbol} "
            f"({position}/{total})"
        )

        metrics = (
            fetch_developer_data(
                coin_id
            )
        )

        if not metrics:

            records.append(
                {
                    "coin_id":
                        coin_id,

                    "developer_data_available":
                        False,

                    "developer_score":
                        np.nan,

                    "developer_activity":
                        "DATA_INSUFFICIENT",

                    "development_confirmation":
                        np.nan,
                }
            )

            time.sleep(
                sleep_seconds
            )

            continue

        issue_resolution = (
            calculate_issue_resolution_ratio(
                metrics.get(
                    "total_issues",
                    np.nan,
                ),
                metrics.get(
                    "closed_issues",
                    np.nan,
                ),
            )
        )

        developer_score = (
            calculate_developer_score(
                metrics
            )
        )

        activity = (
            classify_developer_activity(
                developer_score,
                metrics.get(
                    "commit_count_4_weeks",
                    np.nan,
                ),
            )
        )

        confirmation = (
            calculate_development_confirmation(
                developer_score
            )
        )

        records.append(
            {
                "coin_id":
                    coin_id,

                "developer_data_available":
                    True,

                **metrics,

                "issue_resolution_ratio":
                    issue_resolution,

                "developer_score":
                    developer_score,

                "developer_activity":
                    activity,

                "development_confirmation":
                    confirmation,
            }
        )

        time.sleep(
            sleep_seconds
        )

    developer_df = pd.DataFrame(
        records
    )

    return market_df.merge(
        developer_df,
        on="coin_id",
        how="left",
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

    market = market.head(
        50
    )

    result = (
        enrich_with_developer_data(
            market
        )
    )

    columns = [
        "symbol",
        "name",
        "commit_count_4_weeks",
        "pull_request_contributors",
        "pull_requests_merged",
        "issue_resolution_ratio",
        "developer_score",
        "developer_activity",
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
            "developer_score",
            ascending=False,
            na_position="last",
        )
        .head(50)
        .to_string(
            index=False
        )
    )
