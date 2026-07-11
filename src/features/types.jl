# src/features/types.jl

"""
    AbstractFeatureConfig

Base abstract type for all feature configurations.
"""
abstract type AbstractFeatureConfig end

# --- Core Features ---
struct TeamIDsFeature <: AbstractFeatureConfig end
struct GoalsFeature <: AbstractFeatureConfig end
# Per-match league/tournament index for pooled multi-division segments (e.g. ScottishLower
# [56, 57]). Indices are keyed off the FULL DataStore (stable across splits) and a
# match_id -> league_idx lookup is stashed for prediction-time reconstruction.
struct LeagueFeature <: AbstractFeatureConfig end

# --- Stats Features ---
struct XGFeature <: AbstractFeatureConfig end
struct ShotsFeature <: AbstractFeatureConfig end
struct BigChanceFeature <: AbstractFeatureConfig end          # bigChanceCreated
struct ShotsInsideBoxFeature <: AbstractFeatureConfig end     # totalShotsInsideBox
struct FinalThirdEntriesFeature <: AbstractFeatureConfig end  # finalThirdEntries
struct TouchesInOppBoxFeature <: AbstractFeatureConfig end    # touchesInOppBox

# --- Market Features ---
abstract type AbstractMarketFeatureConfig <: AbstractFeatureConfig end

LINES::Tuple{Vararg{Symbol}} = (:result_1x2, :btts, :over_05, :under_05, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35, :over_45, :under_45)
# 1. Double Poisson (Independent, ρ = 0)
Base.@kwdef struct DoublePoissonMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = LINES
end

# 2. Dixon-Coles (with ρ)
Base.@kwdef struct DixonColesMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = LINES
end

# 3. Regularized Frank Copula Negative Binomial
Base.@kwdef struct RegularizedFrankCopulaMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = LINES
    prior_r::Float64 = 15.0
    penalty_weight::Float64 = 0.05
end

# 4. Double Negative Binomial
Base.@kwdef struct RegularizedDoubleNegativeBinomialMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = LINES
    prior_r::Float64 = 15.0
    penalty_weight::Float64 = 0.05
end

# 5. Dixon-Coles Negative Binomial
Base.@kwdef struct RegularizedDixonColesNegativeBinomialMarketFeature <: AbstractMarketFeatureConfig
    lines::Tuple{Vararg{Symbol}} = LINES
    prior_r::Float64 = 15.0
    penalty_weight::Float64 = 0.05
end

# 6. Local-Intensity SMILE target (per-strike market-implied Poisson rate Λ^mkt(K)).
# NB: a plain AbstractFeatureConfig (NOT AbstractMarketFeatureConfig) — it ships its own
# dedicated `add_feature!` (Poisson-CDF inversion of de-vigged O/U) in market_extractors.jl,
# and must NOT be caught by the generic market-fit extractor. Kmax=4 is the keeper default
# (strikes 5,6 are thin/selection-biased). See docs: local-intensity / smile pillar.
Base.@kwdef struct MarketSmileFeature <: AbstractFeatureConfig
    Kmax::Int = 4
end

# --- Time Features ---
struct TimeIndicesFeature <: AbstractFeatureConfig end
struct DatesFeature <: AbstractFeatureConfig end
struct MonthFeature <: AbstractFeatureConfig end
struct MidweekFeature <: AbstractFeatureConfig end
struct PlasticPitchFeature <: AbstractFeatureConfig end

# --- Player Tracking Features ---
"""
    AbstractRatingTracker

Abstract type for player rating tracking algorithms (e.g., EWMA, Bayesian).
"""
abstract type AbstractRatingTracker end

struct PlayerRatingsFeature{T <: AbstractRatingTracker} <: AbstractFeatureConfig
    tracker::T
end
