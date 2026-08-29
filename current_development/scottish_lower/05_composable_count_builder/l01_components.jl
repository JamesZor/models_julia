# ==============================================================================
# 05 — Composable Count Model Builder : THE COMPONENT LIBRARY
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# WHAT THIS FILE IS. Two new component families, both of which the existing
# `src/models/pregame/components/` library is missing, and whose absence is the
# actual cause of the engine explosion documented in
# docs/architecture/composable_model_builder_specification.md §1:
#
#   1. COVARIATES  (AbstractCovariateConfig)
#      A scalar weight `w` times a per-match design vector `x`, entering the two
#      log-intensities. Wealth and travel distance are both instances. Today each
#      one costs a struct, a `@model`, a `required_features`, and an
#      `extract_parameters` — and each PAIR costs another set (`_engw`, `_engd`,
#      `_engj` in 02_poisson_wealth/l00_feature_poisson.jl are the same eleven
#      lines three times).
#
#   2. OBSERVATIONS  (AbstractObservationConfig)
#      The count likelihood the two log-intensities are fed into. Poisson,
#      Negative Binomial, Dixon-Coles, Frank copula. Every one of these shares the
#      IDENTICAL linear predictor; only the observation layer differs. Today that
#      difference is an entire engine file.
#
# The existing `src` component families (interception, dynamics, home advantage,
# dispersion) are NOT re-implemented here. They are reused verbatim, which is what
# makes the parity claim in r01_demo.jl meaningful.
#
# ==============================================================================

using BayesianFootball
using Turing
using Distributions
using DataFrames
using Dates
using Statistics
using MCMCChains

const CB_PG       = BayesianFootball.Models.PreGame
const CB_Features = BayesianFootball.Features
const CB_TI       = BayesianFootball.TypesInterfaces

# The wealth and distance FEATURE extractors are reused, not rewritten. Parity
# against arms 02/03/04 is only evidence of anything if the design vectors are
# produced by literally the same code.
if !isdefined(@__MODULE__, :SLFPLogSumWealthFeature)
    include(joinpath(@__DIR__, "..", "02_poisson_wealth", "l00_feature_poisson.jl"))
end
if !isdefined(@__MODULE__, :ScottishDistanceFeature)
    include(joinpath(@__DIR__, "..", "03_poisson_distance", "l00_distance_feature.jl"))
end


# ==============================================================================
# 1. COVARIATE ROLES — how a covariate reaches the two sides
# ==============================================================================
#
# A covariate contributes `q = w .* x` to the log-intensities. There are exactly
# two ways it can land, and which one it is, is a property of the covariate, not
# of the engine:
#
#   SUPREMACY   η_h += q,  η_a -= q      shifts WHO scores; total goals unchanged
#   LEVEL       η_h += q,  η_a += q      shifts HOW MANY goals; supremacy unchanged
#
# Wealth and travel distance are both supremacy covariates (a richer or a
# less-travelled side is better, symmetrically). A pitch or referee effect would be
# a level covariate. This is dispatch, not a branch: `covariate_sides` is resolved
# from the type at compile time, so it costs nothing on the ReverseDiff tape.

abstract type AbstractCovariateRole end

"η_h += q, η_a -= q — moves the result, holds the total."
struct SupremacyRole <: AbstractCovariateRole end

"η_h += q, η_a += q — moves the total, holds the result."
struct LevelRole <: AbstractCovariateRole end

covariate_sides(::SupremacyRole, q) = (q, -q)
covariate_sides(::LevelRole,     q) = (q,  q)


# ==============================================================================
# 2. THE COVARIATE CONTRACT
# ==============================================================================
#
# Six methods. Implement them and the builder derives everything else: the site
# name in the chain, the entry in `required_features`, the design vector at fit
# time, the design vector at prediction time, and the extraction of `w`.
#
# There are deliberately NO fallbacks. A covariate that forgets a method fails at
# `build()` with the method's name, not at hour three of an MCMC run.

abstract type AbstractCovariateConfig end

"""
    covariate_name(c) -> Symbol

The chain site prefix. `:wealth` produces the sampling site `wealth.w`. Must be
unique within a model and must not collide with a structural prefix
(`inter`, `ha`, `dyn`, `disp`, `dc`, `cop`) — `build()` enforces both.
"""
covariate_name(c::AbstractCovariateConfig) = _cov_missing(c, :covariate_name)

"""
    covariate_role(c) -> AbstractCovariateRole

`SupremacyRole()` or `LevelRole()`. See §1.
"""
covariate_role(c::AbstractCovariateConfig) = _cov_missing(c, :covariate_role)

"""
    covariate_prior(c) -> UnivariateDistribution

The prior on the scalar weight `w`.
"""
covariate_prior(c::AbstractCovariateConfig) = _cov_missing(c, :covariate_prior)

"""
    covariate_features(c) -> Vector{<:AbstractFeatureConfig}

What `Features.create_features` must extract for this covariate. The builder
concatenates these onto the structural features to derive `required_features`.
"""
covariate_features(c::AbstractCovariateConfig) = _cov_missing(c, :covariate_features)

"""
    covariate_column(c, feature_set) -> Vector{Float64}

The fit-time design vector, one entry per fitted match, in `ordered_match_ids`
order. Must be finite everywhere.

MISSINGNESS IS A ZERO, NOT A MASK. For a linear term `w * x`, imputing an absent
covariate to `0.0` is *exactly* a binary mask (`w * 0 == 0`), so the AD guide's
masking rule is satisfied without carrying a second vector. This only holds because
the term is linear in `x`; a non-linear covariate would need a real mask.
"""
covariate_column(c::AbstractCovariateConfig, fs) = _cov_missing(c, :covariate_column)

"""
    covariate_oos(c, feature_set, df) -> Vector{Float64}

The prediction-time design vector, one entry per row of `df`. Separate from
`covariate_column` because extraction has no `DataStore` argument and must reach
out-of-sample fixtures through a point-in-time bridge stashed at feature-build time.
"""
covariate_oos(c::AbstractCovariateConfig, fs, df) = _cov_missing(c, :covariate_oos)

_cov_missing(c, hook::Symbol) = error("$(typeof(c)) must implement $(hook)")


# ==============================================================================
# 3. CONCRETE COVARIATES
# ==============================================================================

"""
    WealthCovariate

Point-in-time starting-XI squad market valuation differential,
`x = (log Σ value_home − log Σ value_away) / log_scale`.

Wraps `SLFPLogSumWealthFeature` from 02_poisson_wealth/l00_feature_poisson.jl
unchanged: same lookup, same kickoff-stamped causal filter, same fallback rule.
The prior default is arm 02's.
"""
Base.@kwdef struct WealthCovariate{F, D<:Distribution, R<:AbstractCovariateRole} <: AbstractCovariateConfig
    feature::F = SLFPLogSumWealthFeature()
    prior::D   = truncated(Normal(0.10, 0.05), lower = 0.0)
    role::R    = SupremacyRole()
end

covariate_name(::WealthCovariate)      = :wealth
covariate_role(c::WealthCovariate)     = c.role
covariate_prior(c::WealthCovariate)    = c.prior
covariate_features(c::WealthCovariate) = CB_Features.AbstractFeatureConfig[c.feature]

covariate_column(::WealthCovariate, fs) = Vector{Float64}(fs.data[:flat_delta_wealth_logsum])

function covariate_oos(::WealthCovariate, fs, df)
    # A materialised column wins, so a caller can price a hypothetical lineup.
    hasproperty(df, :delta_wealth_logsum) && return Float64.(df.delta_wealth_logsum)
    # Otherwise the causal bridge built at feature time (no valuation stamped after
    # its own kickoff is ever used). An unknown fixture prices at the league mean, 0.
    bridge = fs.data[:wealth_oos_bridge_by_match_id]
    return Float64[get(bridge, Int(r.match_id), 0.0) for r in eachrow(df)]
end

"""
    DistanceCovariate

Static away-travel burden between the two grounds. Wraps `ScottishDistanceFeature`
from 03_poisson_distance/l00_distance_feature.jl unchanged: catalog-fixed
standardisation, Haversine miles, deterministic 45-mile fallback for unmapped grounds.
The prior default is arm 03's.
"""
Base.@kwdef struct DistanceCovariate{F, D<:Distribution, R<:AbstractCovariateRole} <: AbstractCovariateConfig
    feature::F = ScottishDistanceFeature(metric = :log_dist_z)
    prior::D   = truncated(Normal(0.04, 0.03), lower = 0.0)
    role::R    = SupremacyRole()
end

covariate_name(::DistanceCovariate)      = :distance
covariate_role(c::DistanceCovariate)     = c.role
covariate_prior(c::DistanceCovariate)    = c.prior
covariate_features(c::DistanceCovariate) = CB_Features.AbstractFeatureConfig[c.feature]

covariate_column(::DistanceCovariate, fs) = Vector{Float64}(fs.data[:flat_distance])

# Column of the distance table that each metric selects. Distance is a static
# function of the two grounds, so an OOS fixture can simply be recomputed —
# there is no point-in-time bridge to maintain.
const _CB_DIST_COLUMN = Dict(
    :log_dist_z    => :log_dist_z,
    :dist_z        => :dist_z,
    :hav_miles     => :hav_miles,
    :road_miles    => :road_miles,
    :drive_minutes => :drive_minutes,
)

function covariate_oos(c::DistanceCovariate, fs, df)
    hasproperty(df, :distance_z) && return Float64.(df.distance_z)
    table = build_match_distance_table(
        DataFrame(df);
        geocodes_df = load_scottish_stadium_catalog(c.feature.geocodes_csv),
    )
    col = get(_CB_DIST_COLUMN, c.feature.metric, :log_dist_z)
    return Float64.(getproperty(table, col))
end


# ==============================================================================
# ==============================================================================
# 3b. THE NUMERICAL GUARD
# ==============================================================================
#
# Whether the log-intensities are bounded before they reach `exp` is a MODELLING
# DECISION, not a house style, and the Scottish arms disagree about it: arm 00's
# engine has no guard, arms 02/03/04 clamp to [-10, 10]. Hard-coding either one
# into a shared engine silently overrides one of them.
#
# It is also not free. On fold 1 the clamp is the ENTIRE measured difference
# between this engine and arm 00's (r01_demo.jl §8). Making it a component means
# that cost is a choice with a name rather than a line nobody reads.

abstract type AbstractRateGuard end

"""
    ClampGuard(lo = -10.0, hi = 10.0)

Bound η before `exp`, so an excursion during warm-up cannot produce `Inf`.
Arms 02/03/04's behaviour, and the default: a guard that is never needed costs a
few microseconds, and one that is needed and absent costs the run.
"""
Base.@kwdef struct ClampGuard <: AbstractRateGuard
    lo::Float64 = -10.0
    hi::Float64 = 10.0
end

"""
    NoGuard()

No bound. Arm 00's behaviour. Identical results wherever the clamp would not
bind — which `cb_clamp_headroom` measures rather than assumes.
"""
struct NoGuard <: AbstractRateGuard end

apply_guard(g::ClampGuard, η) = clamp.(η, g.lo, g.hi)
apply_guard(::NoGuard, η)     = η

guard_describe(g::ClampGuard) = "clamp to [$(g.lo), $(g.hi)]"
guard_describe(::NoGuard)     = "none — η is unbounded before exp"



# 4. THE OBSERVATION LAYER
# ==============================================================================
#
# THE ARCHITECTURAL POINT OF THIS FILE. Poisson, Negative Binomial, Dixon-Coles and
# the Frank copula are not four models. They are one linear predictor
#
#     η_h = μ_s + δ_m + γ_h + α_h + β_a + Σ_k q_k,h
#     η_a = μ_s + δ_m       + α_a + β_h + Σ_k q_k,a
#
# read by four different observation densities. Splitting them into four engine
# files means every covariate added later must be added four times.
#
# Each observation config also declares which `TypesInterfaces` supertype the built
# model must carry, because that supertype is what routes score-matrix computation
# in `src/predictions/score_computation/`. Getting it wrong does not error — it
# silently prices with the wrong grid. `build()` therefore takes the supertype FROM
# the observation config rather than letting the caller pick it.

abstract type AbstractObservationConfig end

"""
    PoissonObservation

`y ~ Poisson(exp(η))`, evaluated in log-intensity space as
`y·η − exp(η) − log Γ(y+1)`. No dispersion parameter. The default.
"""
struct PoissonObservation <: AbstractObservationConfig end

"""
    NegativeBinomialObservation

`y ~ RobustNegativeBinomial(r, exp(η))`. Wraps any `src` dispersion config
(`GlobalDispersion`, `HomeAwayDispersion`, `AdvancedVolatilityDispersion`)
verbatim, so the `disp.*` chain sites are identical to the `src` NegBin engines.
"""
Base.@kwdef struct NegativeBinomialObservation{D<:CB_PG.AbstractDispersionConfig} <: AbstractObservationConfig
    dispersion::D = CB_PG.GlobalDispersion()
end

"""
    DixonColesCorrelation

Low-score correction τ(y_h, y_a; λ_h, λ_a, ρ) on top of the double-Poisson
likelihood.

DECLARED, NOT WIRED. The builder accepts it, validation reports it, and `build()`
refuses it with the reason. Wiring it is five lines in `l03_engine.jl` (the τ
masks are pre-computable in the builder exactly as the AD guide §4 describes) plus
a `DixonColesRates` extraction that carries ρ — but there is no hand-written
Scottish arm to check the result against, and an unverified likelihood is worth
less than an honest gap. See the specification §7.2.
"""
Base.@kwdef struct DixonColesCorrelation{C<:CB_PG.AbstractDixonColesConfig} <: AbstractObservationConfig
    correlation::C = CB_PG.GlobalDixonColesConfig()
end

"""
    FrankCopulaCorrelation

Frank copula joint likelihood over the two negative-binomial marginals.

DECLARED, NOT WIRED — same reasoning as `DixonColesCorrelation`. See the
specification §7.2.
"""
Base.@kwdef struct FrankCopulaCorrelation{C<:CB_PG.AbstractCopulaConfig} <: AbstractObservationConfig
    correlation::C = CB_PG.HierarchicalFrankCopulaConfig()
end

# --- observation traits -------------------------------------------------------

"The `TypesInterfaces` supertype the built model must carry, so that score-matrix
dispatch in `src/predictions/` reaches the matching grid."
observation_family(::PoissonObservation)           = :poisson
observation_family(::NegativeBinomialObservation)  = :negbin
observation_family(::DixonColesCorrelation)        = :dixon_coles
observation_family(::FrankCopulaCorrelation)       = :frank_copula

"Is the observation density actually implemented in `l03_engine.jl`?"
observation_wired(::AbstractObservationConfig)     = false
observation_wired(::PoissonObservation)            = true
# The two scalar dispersion variants return a plain `(h, a)` pair, which the engine can
# broadcast without a branch. `AdvancedVolatilityDispersion` returns per-team and per-month
# volatility components that have to be re-assembled per match, and the `src` NegBin engine
# does that with a `hasproperty` branch inside `@model` — a construct the AD guide forbids
# and which this engine will not inherit. Wiring it means a `_reconstruct_dispersion`
# submodel that dispatches on the dispersion type, not a branch. See specification §7.2.
observation_wired(o::NegativeBinomialObservation)  =
    o.dispersion isa Union{CB_PG.GlobalDispersion, CB_PG.HomeAwayDispersion}

"Chain-site prefixes the observation layer owns, for the site-collision check."
observation_prefixes(::PoissonObservation)          = Symbol[]
observation_prefixes(::NegativeBinomialObservation) = [:disp]
observation_prefixes(::DixonColesCorrelation)       = [:dc]
observation_prefixes(::FrankCopulaCorrelation)      = [:cop]

"What still has to be built before this observation can be used."
observation_gap(o::AbstractObservationConfig) = "no engine method in l03_engine.jl"
observation_gap(o::NegativeBinomialObservation) =
    "$(nameof(typeof(o.dispersion))) needs a per-match dispersion reconstruction submodel; " *
    "GlobalDispersion and HomeAwayDispersion are wired — specification §7.2"
observation_gap(::DixonColesCorrelation) =
    "τ low-score correction and DixonColesRates (ρ) extraction — specification §7.2"
observation_gap(::FrankCopulaCorrelation) =
    "Frank copula joint density over NegBin marginals and κ extraction — specification §7.2"
