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

# Dependencies and aliases are loaded once by builder-module.jl. Production
# builder code never includes files from current_development/.


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
# There are deliberately NO working fallbacks. A covariate that forgets a method
# fails in `build_count_model()` with the method's name, not during sampling.

abstract type AbstractCovariateConfig <: CB_PG.AbstractModelComponent end

"""
    covariate_name(c) -> Symbol

The chain site prefix. `:wealth` produces the sampling site `wealth.w`. Must be
unique within a model and must not collide with a structural prefix
(`inter`, `ha`, `dyn`, `disp`, `dc`, `cop`) — `build_count_model()` enforces both.
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

# ------------------------------------------------------------------------------
# 3a. Age-weighting curves for production wealth
# ------------------------------------------------------------------------------

"Abstract contract for a scalar, allocation-free player-age productivity curve."
abstract type AbstractAgeWeightingCurve end

"""
    RichardsSigmoid(x0 = 23.0, k = 0.80, nu = 2.0)

Asymmetric maturation curve
`ϕ(age) = (1 + exp(-k * (age - x0)))^(-1 / nu)`.
"""
Base.@kwdef struct RichardsSigmoid <: AbstractAgeWeightingCurve
    x0::Float64 = 23.0
    k::Float64 = 0.80
    nu::Float64 = 2.0
end

"""
    ShiftedGamma(a0 = 16.0, peak = 27.5, alpha = 3.5)

Shifted gamma curve divided by its value at the mode, so `ϕ(peak) == 1`.
Ages at or below `a0` have zero production weight.
"""
Base.@kwdef struct ShiftedGamma <: AbstractAgeWeightingCurve
    a0::Float64 = 16.0
    peak::Float64 = 27.5
    alpha::Float64 = 3.5
end

"""
    GaussianPrime(mu = 26.5, sigma = 4.5)

Symmetric prime-age benchmark, normalized to one at `mu`.
"""
Base.@kwdef struct GaussianPrime <: AbstractAgeWeightingCurve
    mu::Float64 = 26.5
    sigma::Float64 = 4.5
end

"""
    age_weight(curve, age) -> Float64

Evaluate a candidate player-age productivity curve. These scalar kernels allocate
nothing and are suitable for the inner player-valuation loop.
"""
@inline function age_weight(curve::RichardsSigmoid, age::Real)
    return (1.0 + exp(-curve.k * (age - curve.x0)))^(-1.0 / curve.nu)
end

@inline function age_weight(curve::ShiftedGamma, age::Real)
    age <= curve.a0 && return 0.0
    x = age - curve.a0
    mode = curve.peak - curve.a0
    exponent = curve.alpha - 1.0
    return (x / mode)^exponent * exp(-(exponent / mode) * (x - mode))
end

@inline function age_weight(curve::GaussianPrime, age::Real)
    z = (age - curve.mu) / curve.sigma
    return exp(-0.5 * z * z)
end

@inline (curve::AbstractAgeWeightingCurve)(age::Real) = age_weight(curve, age)

function _cb_validate_age_curve(curve::RichardsSigmoid)
    isfinite(curve.x0) || error("RichardsSigmoid.x0 must be finite")
    isfinite(curve.k) && curve.k > 0.0 ||
        error("RichardsSigmoid.k must be finite and > 0")
    isfinite(curve.nu) && curve.nu > 0.0 ||
        error("RichardsSigmoid.nu must be finite and > 0")
    return nothing
end

function _cb_validate_age_curve(curve::ShiftedGamma)
    isfinite(curve.a0) || error("ShiftedGamma.a0 must be finite")
    isfinite(curve.peak) && curve.peak > curve.a0 ||
        error("ShiftedGamma.peak must be finite and > a0")
    isfinite(curve.alpha) && curve.alpha > 1.0 ||
        error("ShiftedGamma.alpha must be finite and > 1")
    return nothing
end

function _cb_validate_age_curve(curve::GaussianPrime)
    isfinite(curve.mu) || error("GaussianPrime.mu must be finite")
    isfinite(curve.sigma) && curve.sigma > 0.0 ||
        error("GaussianPrime.sigma must be finite and > 0")
    return nothing
end

"""
    LogSumWealthFeature

Point-in-time starting-XI squad wealth used by the verified Scottish builder:
`(log(sum(value_home)) - log(sum(value_away))) / log_scale`. Invalid player
values use `fallback_default`; a fixture is neutral unless both sides contain at
least one valid, kickoff-safe valuation.
"""
Base.@kwdef struct LogSumWealthFeature <: CB_Features.AbstractFeatureConfig
    fallback_default::Float64 = 100_000.0
    log_scale::Float64 = 1.0
end

# Name retained for scripts written against the verified prototype.
const SLFPLogSumWealthFeature = LogSumWealthFeature

function _cb_wealth_datetime(value)
    ismissing(value) && return nothing
    value isa DateTime && return value
    value isa Date && return DateTime(value)
    if hasproperty(value, :zone) # TimeZones.ZonedDateTime
        return DateTime(value, Dates.UTC)
    elseif value isa AbstractString
        return tryparse(DateTime, String(value))
    else
        return tryparse(DateTime, string(value))
    end
end

function _cb_match_kickoffs(matches)
    out = Dict{Int,DateTime}()
    columns = propertynames(matches)
    for row in eachrow(matches)
        match_id = Int(row.match_id)
        if :start_timestamp in columns
            stamp = _cb_wealth_datetime(row.start_timestamp)
            if stamp !== nothing
                out[match_id] = stamp
                continue
            end
        end
        if :match_date in columns && !ismissing(row.match_date)
            hour = :match_hour in columns ? Int(coalesce(row.match_hour, 0)) : 0
            out[match_id] = DateTime(row.match_date) + Hour(hour)
        end
    end
    return out
end

function _cb_logsum_wealth_side(row, columns)
    if :team_side in columns
        ismissing(row.team_side) && return nothing
        side = row.team_side
        # The production fetcher emits lowercase `home`/`away`; compare those
        # directly so the per-player hot path does not allocate a String.
        side == "home" && return true
        side == "away" && return false
        normalized = lowercase(String(side))
        normalized == "home" && return true
        normalized == "away" && return false
        return nothing
    elseif :is_home_team in columns
        ismissing(row.is_home_team) && return nothing
        return Bool(row.is_home_team)
    elseif :is_home in columns
        ismissing(row.is_home) && return nothing
        return Bool(row.is_home)
    end
    return nothing
end

function _cb_logsum_wealth_lookup(lineups, matches, ids, config::LogSumWealthFeature)
    config.fallback_default > 0.0 || error("fallback_default must be > 0")
    config.log_scale > 0.0 || error("log_scale must be > 0")

    wanted = Set(Int.(ids))
    values = Dict{Tuple{Int,Bool},Vector{Float64}}()
    valid_counts = Dict{Tuple{Int,Bool},Int}()
    kickoffs = _cb_match_kickoffs(matches)
    columns = propertynames(lineups)

    for row in eachrow(lineups)
        match_id = Int(row.match_id)
        match_id in wanted || continue
        if :is_substitute in columns && coalesce(row.is_substitute, false)
            continue
        end

        side = _cb_logsum_wealth_side(row, columns)
        side === nothing && continue
        key = (match_id, side)
        side_values = get!(values, key, Float64[])

        raw = :proposed_market_value in columns ? row.proposed_market_value :
              :market_value in columns ? row.market_value : missing
        valuation_stamp = :valuation_timestamp in columns ?
                          _cb_wealth_datetime(row.valuation_timestamp) : nothing
        kickoff = get(kickoffs, match_id, nothing)
        stamp_ok = (valuation_stamp === nothing) || (kickoff === nothing) ||
                   (valuation_stamp < kickoff)

        parsed = if ismissing(raw)
            nothing
        else
            try
                Float64(raw)
            catch
                nothing
            end
        end
        if parsed !== nothing && stamp_ok && isfinite(parsed) && parsed > 0.0
            push!(side_values, parsed)
            valid_counts[key] = get(valid_counts, key, 0) + 1
        else
            push!(side_values, config.fallback_default)
        end
    end

    out = Dict{Int,Float64}()
    for match_id in wanted
        home = get(values, (match_id, true), Float64[])
        away = get(values, (match_id, false), Float64[])
        if isempty(home) || isempty(away) ||
           get(valid_counts, (match_id, true), 0) == 0 ||
           get(valid_counts, (match_id, false), 0) == 0
            continue
        end
        out[match_id] = (log(sum(home)) - log(sum(away))) / config.log_scale
    end
    return out
end

function CB_Features.add_feature!(F_data::Dict, config::LogSumWealthFeature,
                                  ordered_ids, team_map::Dict,
                                  ds::CB_Features.Data.DataStore)
    selected = _cb_logsum_wealth_lookup(ds.lineups, ds.matches, ordered_ids, config)
    F_data[:flat_delta_wealth_logsum] = Float64[
        get(selected, Int(match_id), 0.0) for match_id in ordered_ids]
    F_data[:flat_wealth_fallback] = Int[
        haskey(selected, Int(match_id)) ? 0 : 1 for match_id in ordered_ids]
    F_data[:wealth_logsum_by_match_id] = selected

    # Extraction bridge is point-in-time per fixture: the PIT guard above accepts
    # no valuation timestamp at or after that fixture's own kickoff.
    all_ids = Int.(ds.matches.match_id)
    F_data[:wealth_oos_bridge_by_match_id] =
        _cb_logsum_wealth_lookup(ds.lineups, ds.matches, all_ids, config)
    return nothing
end

const _CB_SECONDS_PER_YEAR = 365.25 * 86_400.0

"""
    ProductionWealthFeature(
        curve = RichardsSigmoid(),
        fallback_default = 100_000.0,
        fallback_age = 26.5,
        log_scale = 1.0,
    )

Point-in-time age-adjusted starting-XI wealth:

`(log(Σ value_home * ϕ(age)) - log(Σ value_away * ϕ(age))) / log_scale`.

Player age is measured at that fixture's kickoff from the SofaScore Unix DOB
stamp. A missing or malformed DOB uses the prime-neutral `fallback_age`; its raw
value never reaches the curve. A valuation stamped at or after kickoff is not
used (the safe `fallback_default` is substituted), and a fixture is neutral
unless both sides have at least one positive kickoff-safe valuation.
"""
Base.@kwdef struct ProductionWealthFeature{
    C<:AbstractAgeWeightingCurve,
} <: CB_Features.AbstractFeatureConfig
    curve::C = RichardsSigmoid()
    fallback_default::Float64 = 100_000.0
    fallback_age::Float64 = 26.5
    log_scale::Float64 = 1.0
end

function _cb_unix_seconds(value)
    ismissing(value) && return nothing
    value isa DateTime && return datetime2unix(value)
    value isa Date && return datetime2unix(DateTime(value))
    if value isa Real
        parsed = Float64(value)
        return isfinite(parsed) ? parsed : nothing
    end
    parsed = tryparse(Float64, String(value))
    return parsed !== nothing && isfinite(parsed) ? parsed : nothing
end

@inline function _cb_player_age(kickoff::Union{Nothing,DateTime}, dob_value,
                                fallback_age::Float64)
    kickoff === nothing && return fallback_age
    dob = _cb_unix_seconds(dob_value)
    dob === nothing && return fallback_age
    kickoff_seconds = datetime2unix(kickoff)
    # A DOB at or after kickoff is invalid. Ignore it rather than allowing a
    # negative age to alter the valuation; the documented prime-age fallback is
    # the neutral replacement for an unmapped or invalid player.
    0.0 < dob < kickoff_seconds || return fallback_age
    age = (kickoff_seconds - dob) / _CB_SECONDS_PER_YEAR
    return isfinite(age) && age > 0.0 ? age : fallback_age
end

function _cb_validate_production_feature(config::ProductionWealthFeature)
    isfinite(config.fallback_default) && config.fallback_default > 0.0 ||
        error("fallback_default must be finite and > 0")
    isfinite(config.fallback_age) && config.fallback_age > 0.0 ||
        error("fallback_age must be finite and > 0")
    isfinite(config.log_scale) && config.log_scale > 0.0 ||
        error("log_scale must be finite and > 0")
    _cb_validate_age_curve(config.curve)
    fallback_weight = age_weight(config.curve, config.fallback_age)
    isfinite(fallback_weight) && fallback_weight > 0.0 ||
        error("the configured curve must have a finite positive weight at fallback_age")
    return nothing
end

function _cb_production_wealth_lookup(lineups, matches, ids,
                                      config::ProductionWealthFeature)
    _cb_validate_production_feature(config)

    wanted = Set(Int.(ids))
    totals = Dict{Tuple{Int,Bool},Float64}()
    valid_counts = Dict{Tuple{Int,Bool},Int}()
    sizehint!(totals, 2 * length(wanted))
    sizehint!(valid_counts, 2 * length(wanted))
    kickoffs = _cb_match_kickoffs(matches)
    columns = propertynames(lineups)

    for row in eachrow(lineups)
        match_id = Int(row.match_id)
        match_id in wanted || continue
        if :is_substitute in columns && coalesce(row.is_substitute, false)
            continue
        end

        side = _cb_logsum_wealth_side(row, columns)
        side === nothing && continue
        key = (match_id, side)

        raw_value = :proposed_market_value in columns ? row.proposed_market_value :
                    :market_value in columns ? row.market_value : missing
        parsed_value = if ismissing(raw_value)
            nothing
        else
            try
                Float64(raw_value)
            catch
                nothing
            end
        end

        valuation_stamp = :valuation_timestamp in columns ?
                          _cb_wealth_datetime(row.valuation_timestamp) : nothing
        kickoff = get(kickoffs, match_id, nothing)
        # Match-row values do not always carry a separate observation timestamp.
        # In that case (or when kickoff metadata is unavailable), accept them; if
        # both stamps exist, enforce strict pre-kickoff availability.
        stamp_ok = (valuation_stamp === nothing) || (kickoff === nothing) ||
                   (valuation_stamp < kickoff)
        value = config.fallback_default
        valid_value = false
        if parsed_value !== nothing && stamp_ok &&
           isfinite(parsed_value) && parsed_value > 0.0
            value = parsed_value
            valid_value = true
        end

        dob_value = :date_of_birth_timestamp in columns ?
                    row.date_of_birth_timestamp : missing
        age = _cb_player_age(kickoff, dob_value, config.fallback_age)
        production_value = value * age_weight(config.curve, age)
        isfinite(production_value) && production_value > 0.0 || continue

        totals[key] = get(totals, key, 0.0) + production_value
        if valid_value
            valid_counts[key] = get(valid_counts, key, 0) + 1
        end
    end

    out = Dict{Int,Float64}()
    for match_id in wanted
        home_key = (match_id, true)
        away_key = (match_id, false)
        home = get(totals, home_key, 0.0)
        away = get(totals, away_key, 0.0)
        if home <= 0.0 || away <= 0.0 ||
           get(valid_counts, home_key, 0) == 0 ||
           get(valid_counts, away_key, 0) == 0
            continue
        end
        delta = (log(home) - log(away)) / config.log_scale
        isfinite(delta) && (out[match_id] = delta)
    end
    return out
end

function CB_Features.add_feature!(F_data::Dict, config::ProductionWealthFeature,
                                  ordered_ids, team_map::Dict,
                                  ds::CB_Features.Data.DataStore)
    selected = _cb_production_wealth_lookup(
        ds.lineups, ds.matches, ordered_ids, config)
    F_data[:flat_delta_production_wealth] = Float64[
        get(selected, Int(match_id), 0.0) for match_id in ordered_ids]
    F_data[:flat_production_wealth_fallback] = Int[
        haskey(selected, Int(match_id)) ? 0 : 1 for match_id in ordered_ids]
    F_data[:production_wealth_by_match_id] = selected

    # Keep an all-fixture causal bridge for prediction-time extraction. Every
    # fixture is evaluated against its own kickoff; no future-stamped valuation
    # can enter an earlier match through this bridge.
    all_ids = Int.(ds.matches.match_id)
    F_data[:production_wealth_oos_bridge_by_match_id] =
        _cb_production_wealth_lookup(ds.lineups, ds.matches, all_ids, config)
    return nothing
end

"""
    WealthCovariate

Point-in-time starting-XI squad market valuation differential,
`x = (log Σ value_home − log Σ value_away) / log_scale`.

Uses `LogSumWealthFeature`, preserving the verified prototype's feature equation,
kickoff filtration, and neutral fallback. The prior default is arm 02's.
"""
Base.@kwdef struct WealthCovariate{
    F<:Union{LogSumWealthFeature,CB_Features.SquadWealthFeature},
    D<:UnivariateDistribution,
    R<:AbstractCovariateRole,
} <: AbstractCovariateConfig
    feature::F = LogSumWealthFeature()
    prior::D   = truncated(Normal(0.10, 0.05), lower = 0.0)
    role::R    = SupremacyRole()
end

covariate_name(::WealthCovariate)      = :wealth
covariate_role(c::WealthCovariate)     = c.role
covariate_prior(c::WealthCovariate)    = c.prior
covariate_features(c::WealthCovariate) = CB_Features.AbstractFeatureConfig[c.feature]

function covariate_column(c::WealthCovariate, fs)
    key = c.feature isa LogSumWealthFeature ? :flat_delta_wealth_logsum :
          :flat_delta_wealth
    haskey(fs.data, key) || error(
        "WealthCovariate with $(nameof(typeof(c.feature))) requires :$key")
    return Vector{Float64}(fs.data[key])
end

function covariate_oos(c::WealthCovariate, fs, df)
    # A materialised column wins, so a caller can price a hypothetical lineup.
    if c.feature isa LogSumWealthFeature
        hasproperty(df, :delta_wealth_logsum) && return Float64.(df.delta_wealth_logsum)
        bridge = get(fs.data, :wealth_oos_bridge_by_match_id, Dict{Int,Float64}())
    else
        hasproperty(df, :delta_wealth) && return Float64.(df.delta_wealth)
        bridge = get(fs.data, :wealth_by_match_id, Dict{Int32,Float64}())
    end
    return Float64[get(bridge, Int(r.match_id), 0.0) for r in eachrow(df)]
end

"""
    ProductionWealthCovariate

Age-adjusted point-in-time starting-XI production wealth. The design column is
built by `ProductionWealthFeature` and enters the home/away log-rates as a
supremacy term. Its default prior matches the raw wealth covariate, making the
curve transform—not a prior change—the distinction between the two components.
"""
Base.@kwdef struct ProductionWealthCovariate{
    F<:ProductionWealthFeature,
    D<:UnivariateDistribution,
    R<:AbstractCovariateRole,
} <: AbstractCovariateConfig
    feature::F = ProductionWealthFeature()
    prior::D = truncated(Normal(0.10, 0.05), lower = 0.0)
    role::R = SupremacyRole()
end

covariate_name(::ProductionWealthCovariate) = :production_wealth
covariate_role(c::ProductionWealthCovariate) = c.role
covariate_prior(c::ProductionWealthCovariate) = c.prior
covariate_features(c::ProductionWealthCovariate) =
    CB_Features.AbstractFeatureConfig[c.feature]

function covariate_column(::ProductionWealthCovariate, fs)
    haskey(fs.data, :flat_delta_production_wealth) || error(
        "ProductionWealthCovariate requires :flat_delta_production_wealth")
    return Vector{Float64}(fs.data[:flat_delta_production_wealth])
end

function covariate_oos(::ProductionWealthCovariate, fs, df)
    # A caller may override the causal bridge with a materialised hypothetical
    # lineup, following the same extraction convention as WealthCovariate.
    hasproperty(df, :delta_production_wealth) &&
        return Float64.(df.delta_production_wealth)
    bridge = get(fs.data, :production_wealth_oos_bridge_by_match_id,
                 Dict{Int,Float64}())
    return Float64[get(bridge, Int(row.match_id), 0.0) for row in eachrow(df)]
end

"""
    DistanceCovariate

Static away-travel burden between the two grounds. Uses the production
`Features.DistanceFeature`: catalog-fixed standardisation, Haversine miles, and
a deterministic 45-mile fallback for unmapped grounds. The prior default is arm 03's.
"""
Base.@kwdef struct DistanceCovariate{
    F<:CB_Features.DistanceFeature,
    D<:UnivariateDistribution,
    R<:AbstractCovariateRole,
} <: AbstractCovariateConfig
    feature::F = CB_Features.DistanceFeature(metric = :log_dist_z)
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
    hasproperty(df, c.feature.metric) &&
        return Float64.(getproperty(df, c.feature.metric))
    # Legacy Scottish arm materialised its default log-distance under this name.
    c.feature.metric === :log_dist_z && hasproperty(df, :distance_z) &&
        return Float64.(df.distance_z)

    table = CB_Features.build_match_distance_table(
        DataFrame(df);
        geocodes_df = CB_Features.load_stadium_catalog(c.feature.geocodes_csv),
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

abstract type AbstractRateGuard <: CB_PG.AbstractModelComponent end

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
# silently prices with the wrong grid. `build_count_model()` therefore takes the supertype FROM
# the observation config rather than letting the caller pick it.

abstract type AbstractObservationConfig <: CB_PG.AbstractModelComponent end

"""
    PoissonObservation

`y ~ Poisson(exp(η))`, evaluated in log-intensity space as
`y·η − exp(η) − log Γ(y+1)`. No dispersion parameter. The default.
"""
struct PoissonObservation <: AbstractObservationConfig end

"""
    NegativeBinomialObservation

`y ~ RobustNegativeBinomial(r, exp(η))`. Wraps any `src` dispersion config
(`GlobalDispersion`, `HomeAwayDispersion`) with their existing priors and
`disp.*` chain schema. The builder applies its own smooth AD-safe log-dispersion
bound; advanced volatility remains an explicit validation refusal until its
per-match reconstruction is AD-safe.
"""
Base.@kwdef struct NegativeBinomialObservation{D<:CB_PG.AbstractDispersionConfig} <: AbstractObservationConfig
    dispersion::D = CB_PG.GlobalDispersion()
end

"""
    DixonColesCorrelation

Low-score correction τ(y_h, y_a; λ_h, λ_a, ρ) on top of the double-Poisson
likelihood.

DECLARED, NOT WIRED. The builder accepts it, validation reports it, and
`build_count_model()` refuses it with the reason. Wiring it belongs in `engine.jl` (the τ
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

"Is the observation density implemented in the production builder engine?"
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
observation_gap(o::AbstractObservationConfig) = "no observation method in builder/engine.jl"
observation_gap(o::NegativeBinomialObservation) =
    "$(nameof(typeof(o.dispersion))) needs a per-match dispersion reconstruction submodel; " *
    "GlobalDispersion and HomeAwayDispersion are wired — specification §7.2"
observation_gap(::DixonColesCorrelation) =
    "τ low-score correction and DixonColesRates (ρ) extraction — specification §7.2"
observation_gap(::FrankCopulaCorrelation) =
    "Frank copula joint density over NegBin marginals and κ extraction — specification §7.2"
