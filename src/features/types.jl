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

# --- BBC Features ---
# Per-side total shots from `ds.bbc` (BBC match pages), for the two-layer thinned-Poisson funnel
# Shots ~ Poisson(λ_s), Goals ~ Poisson(λ_s·p₂). Emits Int counts with a 0 dummy plus a Float64
# usability mask; segments with no BBC coverage get an all-zero mask, never an error.
struct ShotsFunnelFeature <: AbstractFeatureConfig end

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

# --- Plus-Minus (RAPM) Player Rating Features ---
#
# ARCHITECTURE NOTE — why this is a SIBLING of PlayerRatingsFeature, not another tracker.
# `AbstractRatingTracker` is a per-player RECURSIVE FILTER over one player's own rating history
# (see trackers/bayesian.jl). RAPM is a GLOBAL CROSS-MATCH RIDGE REGRESSION solved for every player
# at once, so it cannot be an `AbstractRatingTracker`. It becomes its own feature family instead —
# mirroring `AbstractMarketFeatureConfig` — with one concrete struct per plus-minus target and a
# single shared `add_feature!` (extractors/plus_minus_extractors.jl) dispatching all of them.
# Swapping the APM from shots to xG is therefore just swapping the struct.
#
# The estimator knobs travel WITH the struct, so a variant is fully described by its config.
# Defaults are the research's per-target tuned cells; `w_sim = 0` everywhere on purpose — see
# src/features/plus_minus/ridge.jl for why the Brier-optimal `w_sim = 0.9` is NOT the default.
abstract type AbstractPlusMinusFeature <: AbstractFeatureConfig end

# The GREEN-LIT cell: split-half reliability 0.669 vs the SofaScore rating's 0.660, and better
# match-outcome retrodiction than the SofaScore-fed model on both held-out seasons.
Base.@kwdef struct ShotsPlusMinusFeature <: AbstractPlusMinusFeature
    w_sim::Float64          = 0.0
    λ::Float64              = 1000.0
    half_life_days::Float64 = 730.0
end

Base.@kwdef struct ShotsOnTargetPlusMinusFeature <: AbstractPlusMinusFeature
    w_sim::Float64          = 0.0
    λ::Float64              = 1000.0
    half_life_days::Float64 = 730.0
end

Base.@kwdef struct GoalsPlusMinusFeature <: AbstractPlusMinusFeature
    w_sim::Float64          = 0.0
    λ::Float64              = 1000.0
    half_life_days::Float64 = 730.0
end

# The LEAST TEAM-LOADED cell (club R² 0.212 vs 0.389 for shots), at the cost of a lower split-half
# reliability (0.407). λ optimised lower than the other targets.
Base.@kwdef struct XGPlusMinusFeature <: AbstractPlusMinusFeature
    w_sim::Float64          = 0.0
    λ::Float64              = 200.0
    half_life_days::Float64 = 730.0
end

"""
    pm_target(config) -> Symbol

Which segment response column this variant regresses on (see features/plus_minus/targets.jl).
"""
pm_target(::ShotsPlusMinusFeature)         = :y_shots
pm_target(::ShotsOnTargetPlusMinusFeature) = :y_sot
pm_target(::GoalsPlusMinusFeature)         = :y_goals
pm_target(::XGPlusMinusFeature)            = :y_xg

# --- Rating scale accessor ---
"""
    rating_base(config) -> Float64

The value a "neutral" player rating sits at, which engines subtract before weighting. The nine
`outfield_*` xG engines hard-code `config.player_ratings_feature.tracker.prior_mean` for this; new
engines should call `rating_base` so any feature family can supply the rating.

RAPM is zero-centred by construction (it is a ridge coefficient shrunk toward 0), so the plus-minus
family returns 0.0 and the centring is a no-op.
"""
rating_base(::AbstractFeatureConfig)      = 0.0
rating_base(c::PlayerRatingsFeature)      = c.tracker.prior_mean
rating_base(::AbstractPlusMinusFeature)   = 0.0
