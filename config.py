"""
CRYPTO OPPORTUNITY ENGINE
Configuração central.

OBJETIVO:
Encontrar oportunidades em criptoativos (excluindo BTC) onde a melhora
econômica/fundamental esteja avançando mais rapidamente que a precificação
do mercado.

Princípio:
QUALIDADE + ACELERAÇÃO + CAPTURA DE VALOR + BAIXA DILUIÇÃO
+ VALUATION + DIVERGÊNCIA FUNDAMENTOS/PREÇO + TIMING.

Narrativa é complementar e nunca deve aprovar um projeto ruim.
"""

# ============================================================
# ENGINE
# ============================================================

ENGINE_NAME = "CRYPTO OPPORTUNITY ENGINE"
ENGINE_VERSION = "1.0.0"

BASE_CURRENCY = "USD"

EXCLUDED_SYMBOLS = {
    "BTC",
    "WBTC",
    "CBBTC",
    "TBTC",
}

EXCLUDE_STABLECOINS = True
EXCLUDE_WRAPPED_ASSETS = True
EXCLUDE_LEVERAGED_TOKENS = True


# ============================================================
# UNIVERSO
# ============================================================

UNIVERSE_MAX_ASSETS = 500

MIN_MARKET_CAP_USD = 50_000_000
MIN_DAILY_VOLUME_USD = 2_000_000

MIN_PROJECT_AGE_DAYS = 180

MIN_DATA_COMPLETENESS = 0.70


# ============================================================
# HARD FILTERS
# Não recebem pontos.
# Podem eliminar o ativo.
# ============================================================

HARD_FILTERS = {
    "minimum_liquidity": True,
    "minimum_market_cap": True,
    "minimum_project_age": True,
    "minimum_data_quality": True,
    "critical_security_risk": True,
    "critical_dilution_risk": True,
    "critical_concentration_risk": True,
}


# ============================================================
# OPPORTUNITY SCORE
# Total = 100
# ============================================================

OPPORTUNITY_WEIGHTS = {

    # Projeto está melhorando?
    "fundamental_acceleration": 0.20,

    # Está gerando atividade econômica/receita?
    "revenue_economics": 0.15,

    # Holder captura parte do valor?
    "holder_value_capture": 0.15,

    # Está barato frente aos fundamentos?
    "valuation": 0.15,

    # Oferta futura é saudável?
    "tokenomics_dilution": 0.15,

    # Blockchain confirma crescimento?
    "onchain_growth": 0.10,

    # Capital/utilização do protocolo
    "tvl_utilization": 0.05,

    # Tendência/narrativa apenas complementar
    "narrative": 0.05,
}

assert round(sum(OPPORTUNITY_WEIGHTS.values()), 10) == 1.0


# ============================================================
# FUNDAMENTAL ACCELERATION
# ============================================================

FUNDAMENTAL_ACCELERATION_WEIGHTS = {
    "users_growth": 0.20,
    "transactions_growth": 0.15,
    "fees_growth": 0.20,
    "revenue_growth": 0.20,
    "tvl_growth": 0.15,
    "developer_growth": 0.10,
}


GROWTH_WINDOWS_DAYS = [
    30,
    90,
    180,
    365,
]


# ============================================================
# RECEITA / ECONOMIA
# ============================================================

REVENUE_WEIGHTS = {
    "revenue_growth": 0.30,
    "fees_growth": 0.20,
    "revenue_quality": 0.20,
    "revenue_consistency": 0.15,
    "economic_activity_growth": 0.15,
}


# ============================================================
# HOLDER VALUE CAPTURE
# ============================================================

HOLDER_VALUE_WEIGHTS = {
    "real_yield": 0.25,
    "fee_distribution": 0.20,
    "buyback": 0.15,
    "burn": 0.10,
    "token_required_for_usage": 0.15,
    "economic_staking": 0.15,
}

MIN_HOLDER_VALUE_SCORE = 40


# ============================================================
# TOKENOMICS / DILUIÇÃO
# ============================================================

DILUTION_WEIGHTS = {
    "circulating_supply_ratio": 0.20,
    "market_cap_fdv_ratio": 0.15,
    "annual_inflation": 0.20,
    "unlock_90d": 0.15,
    "unlock_365d": 0.15,
    "insider_concentration": 0.10,
    "unlock_vs_liquidity": 0.05,
}


# Limites de alerta

MAX_ANNUAL_INFLATION_WARNING = 0.10
MAX_ANNUAL_INFLATION_CRITICAL = 0.20

MAX_UNLOCK_90D_WARNING = 0.10
MAX_UNLOCK_90D_CRITICAL = 0.20

MAX_UNLOCK_365D_WARNING = 0.20
MAX_UNLOCK_365D_CRITICAL = 0.40

MIN_CIRCULATING_SUPPLY_RATIO_WARNING = 0.50
MIN_CIRCULATING_SUPPLY_RATIO_CRITICAL = 0.25

MIN_MARKET_CAP_FDV_RATIO_WARNING = 0.50
MIN_MARKET_CAP_FDV_RATIO_CRITICAL = 0.25

MIN_DILUTION_SCORE = 40


# ============================================================
# VALUATION
# ============================================================

VALUATION_WEIGHTS = {
    "market_cap_revenue": 0.25,
    "fdv_revenue": 0.20,
    "market_cap_fees": 0.15,
    "market_cap_tvl": 0.15,
    "historical_valuation": 0.15,
    "peer_relative_valuation": 0.10,
}


# ============================================================
# TVL
# TVL tem peso variável conforme categoria.
# ============================================================

TVL_WEIGHTS = {
    "tvl_growth_30d": 0.15,
    "tvl_growth_90d": 0.25,
    "tvl_growth_180d": 0.20,
    "revenue_per_tvl": 0.15,
    "fees_per_tvl": 0.10,
    "market_cap_tvl": 0.10,
    "tvl_quality": 0.05,
}


TVL_CATEGORY_RELEVANCE = {

    "LENDING": 1.00,
    "DEX": 1.00,
    "LIQUID_STAKING": 1.00,
    "YIELD": 1.00,

    "L1": 0.70,
    "L2": 0.80,

    "RWA": 0.80,
    "TOKENIZATION": 0.80,

    "DEFI_INFRASTRUCTURE": 0.60,

    "ORACLE": 0.20,
    "INTEROPERABILITY": 0.30,
    "DEPIN": 0.30,
    "AI": 0.20,

    "OTHER": 0.30,
}


# ============================================================
# ON-CHAIN
# ============================================================

ONCHAIN_WEIGHTS = {
    "active_addresses_growth": 0.20,
    "transaction_growth": 0.15,
    "new_holders_growth": 0.15,
    "stablecoin_inflows": 0.15,
    "exchange_netflow": 0.10,
    "whale_accumulation": 0.10,
    "network_usage_growth": 0.15,
}


# ============================================================
# NARRATIVA
# Peso total limitado a 5%.
# ============================================================

NARRATIVE_WEIGHTS = {
    "real_adoption": 0.40,
    "capital_growth": 0.25,
    "institutional_adoption": 0.20,
    "sector_growth": 0.15,
}


TRACKED_NARRATIVES = {
    "RWA",
    "TOKENIZATION",
    "TOKENIZED_STOCKS",
    "TOKENIZED_TREASURIES",
    "DEFI",
    "L1",
    "L2",
    "DEPIN",
    "AI",
    "ORACLE",
    "INTEROPERABILITY",
}


# ============================================================
# RWA / TOKENIZAÇÃO
# ============================================================

RWA_METRICS = {
    "tokenized_assets_growth",
    "tokenized_equities_growth",
    "tokenized_treasuries_growth",
    "rwa_tvl_growth",
    "rwa_volume_growth",
    "rwa_users_growth",
    "institutional_integrations",
    "protocol_revenue_from_rwa",
}


# ============================================================
# FUNDAMENTAL × PRICE DIVERGENCE
# Núcleo da busca por oportunidade.
# ============================================================

DIVERGENCE_ENABLED = True

DIVERGENCE_WINDOWS_DAYS = [
    30,
    90,
    180,
]

DIVERGENCE_WEIGHTS = {
    "fundamental_change": 0.35,
    "revenue_change": 0.20,
    "usage_change": 0.15,
    "tvl_change": 0.10,
    "onchain_change": 0.10,
    "price_underreaction": 0.10,
}


# Quanto maior, mais os fundamentos melhoraram
# sem acompanhamento proporcional do preço.

MIN_POSITIVE_DIVERGENCE_SCORE = 60

STRONG_DIVERGENCE_SCORE = 75
EXTREME_DIVERGENCE_SCORE = 90


# ============================================================
# QUALITY GATES
# ============================================================

MIN_OPPORTUNITY_SCORE = 65

MIN_FUNDAMENTAL_SCORE = 55
MIN_REVENUE_SCORE = 45
MIN_HOLDER_VALUE_SCORE = 40
MIN_DILUTION_SCORE = 40

STRONG_OPPORTUNITY_SCORE = 75
VERY_STRONG_OPPORTUNITY_SCORE = 85


# ============================================================
# RISK ENGINE
# Risco não aumenta score.
# Pode bloquear ativo.
# ============================================================

RISK_LEVELS = [
    "LOW",
    "MODERATE",
    "HIGH",
    "CRITICAL",
]

BLOCK_CRITICAL_RISK = True

RISK_FACTORS = {
    "smart_contract",
    "bridge",
    "centralization",
    "governance",
    "liquidity",
    "holder_concentration",
    "security_history",
    "oracle_dependency",
    "regulatory",
    "token_unlock",
}


# ============================================================
# TIMING ENGINE
# Timing NÃO faz parte do Opportunity Score.
# ============================================================

TIMING_WEIGHTS = {
    "momentum_3m": 0.15,
    "momentum_6m": 0.15,
    "relative_strength": 0.15,
    "volume_confirmation": 0.10,
    "trend_structure": 0.15,
    "drawdown_position": 0.10,
    "funding": 0.10,
    "open_interest": 0.10,
}


TIMING_STRONG_ENTRY = 80
TIMING_ENTRY = 65
TIMING_WATCH = 50


# ============================================================
# CLASSIFICAÇÃO OPERACIONAL
# ============================================================

SIGNAL_STRONG_ENTRY = "STRONG_ENTRY"
SIGNAL_ENTRY = "ENTRY"
SIGNAL_WATCH = "WATCH"
SIGNAL_WAIT = "WAIT"
SIGNAL_REJECTED = "REJECTED"
SIGNAL_BLOCKED = "BLOCKED"


# ============================================================
# REGRA DE DECISÃO
# ============================================================

DECISION_RULES = {

    "STRONG_ENTRY": {
        "min_opportunity": 85,
        "min_timing": 80,
        "quality_gate": True,
    },

    "ENTRY": {
        "min_opportunity": 75,
        "min_timing": 65,
        "quality_gate": True,
    },

    "WATCH": {
        "min_opportunity": 65,
        "min_timing": 50,
        "quality_gate": True,
    },
}


# ============================================================
# RELATÓRIO
# ============================================================

TOP_OPPORTUNITIES = 20

REPORT_FIELDS = [
    "rank",
    "symbol",
    "name",
    "category",
    "market_cap",
    "price",
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
    "risk_level",
    "timing_score",
    "signal",
]


# ============================================================
# PRINCÍPIOS FIXOS DO MOTOR
# ============================================================

ENGINE_RULES = {

    "focus": "OPPORTUNITY",

    "primary_objective":
        "Find quality crypto assets where economic and fundamental "
        "improvement is occurring faster than market repricing.",

    "btc_excluded": True,

    "narrative_cannot_override_fundamentals": True,

    "timing_cannot_rescue_bad_project": True,

    "high_tvl_alone_is_not_opportunity": True,

    "high_revenue_without_holder_capture_is_not_enough": True,

    "high_yield_from_token_emission_is_not_real_yield": True,

    "critical_dilution_can_block_asset": True,

    "critical_risk_can_block_asset": True,

    "fundamental_price_divergence_is_core_signal": True,

    "category_specific_metrics_required": True,
}
