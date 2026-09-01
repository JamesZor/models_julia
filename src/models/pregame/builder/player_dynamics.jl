# ==============================================================================
# Pure player-level lineup dynamics for the composable count-model builder
# ==============================================================================

abstract type AbstractPlayerAggregation end

struct OutfieldPlayerAggregation <: AbstractPlayerAggregation end

Base.@kwdef struct BenchWeightedPlayerAggregation <: AbstractPlayerAggregation
    w_bench::Float64 = 0.25
end

struct PositionalPlayerAggregation <: AbstractPlayerAggregation end
struct MinuteWeightedPlayerAggregation <: AbstractPlayerAggregation end

"""
    PlayerLineupDynamics(; feature=XGPlusMinusFeature(), aggregation=OutfieldPlayerAggregation(), …)

Replace latent team attack/defence with global weights on pre-match lineup ratings. The
rating feature is fit in the feature layer; the Turing model receives only contiguous
`Vector{Float64}` aggregates. `w_bench_prior=nothing` uses the aggregation's fixed bench
weight (0.25 for `BenchWeightedPlayerAggregation` and `PositionalPlayerAggregation`).
"""
Base.@kwdef struct PlayerLineupDynamics{
    F<:CB_Features.AbstractFeatureConfig,
    A<:AbstractPlayerAggregation,
    WA<:Distribution,
    WD<:Distribution,
    WB<:Union{Nothing,Distribution},
} <: CB_PG.AbstractDynamicsConfig
    feature::F = CB_Features.XGPlusMinusFeature()
    aggregation::A = OutfieldPlayerAggregation()
    w_att_prior::WA = Normal(0.0, 0.3)
    w_def_prior::WD = Normal(0.0, 0.3)
    w_bench_prior::WB = nothing
end

abstract type AbstractPlayerLineupDesign end

struct OutfieldPlayerDesign <: AbstractPlayerLineupDesign
    home::Vector{Float64}
    away::Vector{Float64}
end

struct BenchWeightedPlayerDesign <: AbstractPlayerLineupDesign
    home::Vector{Float64}
    away::Vector{Float64}
    bench_home::Vector{Float64}
    bench_away::Vector{Float64}
end

struct PositionalPlayerDesign <: AbstractPlayerLineupDesign
    home_D::Vector{Float64}
    home_M::Vector{Float64}
    home_F::Vector{Float64}
    away_D::Vector{Float64}
    away_M::Vector{Float64}
    away_F::Vector{Float64}
    bench_home_D::Vector{Float64}
    bench_home_M::Vector{Float64}
    bench_home_F::Vector{Float64}
    bench_away_D::Vector{Float64}
    bench_away_M::Vector{Float64}
    bench_away_F::Vector{Float64}
end

struct MinuteWeightedPlayerDesign <: AbstractPlayerLineupDesign
    home::Vector{Float64}
    away::Vector{Float64}
end

"Extra features owned by a structural dynamics component."
dynamics_features(::CB_PG.AbstractDynamicsConfig) = CB_Features.AbstractFeatureConfig[]
dynamics_features(config::PlayerLineupDynamics) =
    CB_Features.AbstractFeatureConfig[config.feature]

"Likelihood weights are time-decayed for legacy dynamics and uniform for pure lineup dynamics."
dynamics_match_weights(config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics},
                       dates::Vector{Float64}) =
    0.5 .^ (dates ./ Float64(config.days_half_life))
dynamics_match_weights(::PlayerLineupDynamics, dates::Vector{Float64}) = ones(Float64, length(dates))

_dynamics_weighting_valid(::CB_PG.AbstractDynamicsConfig) = false
_dynamics_weighting_detail(config::CB_PG.AbstractDynamicsConfig) =
    "$(nameof(typeof(config))) has no composable likelihood-weighting adapter"
_dynamics_weighting_valid(config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics}) =
    isfinite(config.days_half_life) && config.days_half_life > 0
_dynamics_weighting_valid(::PlayerLineupDynamics) = true
_dynamics_weighting_detail(config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics}) =
    "time decay half-life = $(config.days_half_life) days"
_dynamics_weighting_detail(::PlayerLineupDynamics) =
    "uniform likelihood weights; lineup ratings are point-in-time"

function _player_column(feature_set, key::Symbol, n_matches::Int)
    haskey(feature_set.data, key) || error(
        "PlayerLineupDynamics requires :$key. Use an AbstractPlusMinusFeature extractor " *
        "or another feature that implements the flat lineup-vector contract.")
    values = Vector{Float64}(feature_set.data[key])
    length(values) == n_matches || error(
        "PlayerLineupDynamics column :$key has length $(length(values)); expected $n_matches")
    all(isfinite, values) || error("PlayerLineupDynamics column :$key has non-finite entries")
    return values
end

function player_lineup_design(::OutfieldPlayerAggregation, feature_set, n_matches::Int)
    return OutfieldPlayerDesign(
        _player_column(feature_set, :flat_home_outfield_rating, n_matches),
        _player_column(feature_set, :flat_away_outfield_rating, n_matches),
    )
end

function player_lineup_design(::BenchWeightedPlayerAggregation, feature_set, n_matches::Int)
    return BenchWeightedPlayerDesign(
        _player_column(feature_set, :flat_home_outfield_rating, n_matches),
        _player_column(feature_set, :flat_away_outfield_rating, n_matches),
        _player_column(feature_set, :flat_home_bench_rating, n_matches),
        _player_column(feature_set, :flat_away_bench_rating, n_matches),
    )
end

function player_lineup_design(::PositionalPlayerAggregation, feature_set, n_matches::Int)
    keys = (
        :flat_home_D_rating, :flat_home_M_rating, :flat_home_F_rating,
        :flat_away_D_rating, :flat_away_M_rating, :flat_away_F_rating,
        :flat_home_bench_D_rating, :flat_home_bench_M_rating, :flat_home_bench_F_rating,
        :flat_away_bench_D_rating, :flat_away_bench_M_rating, :flat_away_bench_F_rating,
    )
    values = map(key -> _player_column(feature_set, key, n_matches), keys)
    return PositionalPlayerDesign(values...)
end

function player_lineup_design(::MinuteWeightedPlayerAggregation, feature_set, n_matches::Int)
    return MinuteWeightedPlayerDesign(
        _player_column(feature_set, :flat_home_minute_weighted_rating, n_matches),
        _player_column(feature_set, :flat_away_minute_weighted_rating, n_matches),
    )
end

player_lineup_design(config::PlayerLineupDynamics, feature_set, n_matches::Int) =
    player_lineup_design(config.aggregation, feature_set, n_matches)

dynamics_design(::CB_PG.AbstractDynamicsConfig, feature_set, n_matches::Int) = nothing
dynamics_design(config::PlayerLineupDynamics, feature_set, n_matches::Int) =
    player_lineup_design(config, feature_set, n_matches)

@model function _player_bench_weight(::Nothing, fixed_weight::Float64)
    return fixed_weight
end

@model function _player_bench_weight(prior::Distribution, fixed_weight::Float64)
    w_bench ~ prior
    return w_bench
end

@model function _player_lineup_effects(config::PlayerLineupDynamics{<:Any,<:OutfieldPlayerAggregation},
                                       design::OutfieldPlayerDesign)
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    return (;
        att_h = w_att .* design.home,
        def_a = .-w_def .* design.away,
        att_a = w_att .* design.away,
        def_h = .-w_def .* design.home,
    )
end

@model function _player_lineup_effects(config::PlayerLineupDynamics{<:Any,<:BenchWeightedPlayerAggregation},
                                       design::BenchWeightedPlayerDesign)
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    bench ~ to_submodel(
        _player_bench_weight(config.w_bench_prior, config.aggregation.w_bench), false)
    home = design.home .+ bench .* design.bench_home
    away = design.away .+ bench .* design.bench_away
    return (;
        att_h = w_att .* home,
        def_a = .-w_def .* away,
        att_a = w_att .* away,
        def_h = .-w_def .* home,
    )
end

@model function _player_lineup_effects(config::PlayerLineupDynamics{<:Any,<:PositionalPlayerAggregation},
                                       design::PositionalPlayerDesign)
    w_att_F ~ config.w_att_prior
    w_att_M ~ config.w_att_prior
    w_def_D ~ config.w_def_prior
    w_def_M ~ config.w_def_prior
    bench ~ to_submodel(_player_bench_weight(config.w_bench_prior, 0.25), false)

    home_F = design.home_F .+ bench .* design.bench_home_F
    home_M = design.home_M .+ bench .* design.bench_home_M
    home_D = design.home_D .+ bench .* design.bench_home_D
    away_F = design.away_F .+ bench .* design.bench_away_F
    away_M = design.away_M .+ bench .* design.bench_away_M
    away_D = design.away_D .+ bench .* design.bench_away_D
    return (;
        att_h = w_att_F .* home_F .+ w_att_M .* home_M,
        def_a = .-w_def_D .* away_D .- w_def_M .* away_M,
        att_a = w_att_F .* away_F .+ w_att_M .* away_M,
        def_h = .-w_def_D .* home_D .- w_def_M .* home_M,
    )
end

@model function _player_lineup_effects(config::PlayerLineupDynamics{<:Any,<:MinuteWeightedPlayerAggregation},
                                       design::MinuteWeightedPlayerDesign)
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    return (;
        att_h = w_att .* design.home,
        def_a = .-w_def .* design.away,
        att_a = w_att .* design.away,
        def_h = .-w_def .* design.home,
    )
end

function _player_chain_vector(chain::Chains, name::Symbol)
    return vec(Array(chain[name]))
end

function extract_player_dynamics(chain::Chains,
                                 config::PlayerLineupDynamics{<:Any,<:OutfieldPlayerAggregation},
                                 prefix::String)
    return (; w_att = _player_chain_vector(chain, Symbol("$prefix.w_att")),
              w_def = _player_chain_vector(chain, Symbol("$prefix.w_def")))
end

function extract_player_dynamics(chain::Chains,
                                 config::PlayerLineupDynamics{<:Any,<:BenchWeightedPlayerAggregation},
                                 prefix::String)
    fixed = config.aggregation.w_bench
    n_samples = size(chain, 1) * size(chain, 3)
    bench = config.w_bench_prior === nothing ? fill(fixed, n_samples) :
            _player_chain_vector(chain, Symbol("$prefix.w_bench"))
    return (; w_att = _player_chain_vector(chain, Symbol("$prefix.w_att")),
              w_def = _player_chain_vector(chain, Symbol("$prefix.w_def")),
              w_bench = bench)
end

function extract_player_dynamics(chain::Chains,
                                 config::PlayerLineupDynamics{<:Any,<:PositionalPlayerAggregation},
                                 prefix::String)
    n_samples = size(chain, 1) * size(chain, 3)
    bench = config.w_bench_prior === nothing ? fill(0.25, n_samples) :
            _player_chain_vector(chain, Symbol("$prefix.w_bench"))
    return (;
        w_att_F = _player_chain_vector(chain, Symbol("$prefix.w_att_F")),
        w_att_M = _player_chain_vector(chain, Symbol("$prefix.w_att_M")),
        w_def_D = _player_chain_vector(chain, Symbol("$prefix.w_def_D")),
        w_def_M = _player_chain_vector(chain, Symbol("$prefix.w_def_M")),
        w_bench = bench,
    )
end

function extract_player_dynamics(chain::Chains,
                                 config::PlayerLineupDynamics{<:Any,<:MinuteWeightedPlayerAggregation},
                                 prefix::String)
    return (; w_att = _player_chain_vector(chain, Symbol("$prefix.w_att")),
              w_def = _player_chain_vector(chain, Symbol("$prefix.w_def")))
end

_player_neutral_aggregate() = CB_Features._pm_empty_lineup_aggregate()

function player_oos_effects(::OutfieldPlayerAggregation, draw, value)
    return (; att_h = draw.w_att .* value.home_outfield,
              def_a = .-draw.w_def .* value.away_outfield,
              att_a = draw.w_att .* value.away_outfield,
              def_h = .-draw.w_def .* value.home_outfield)
end

function player_oos_effects(::BenchWeightedPlayerAggregation, draw, value)
    home = value.home_outfield .+ draw.w_bench .* value.home_bench
    away = value.away_outfield .+ draw.w_bench .* value.away_bench
    return (; att_h = draw.w_att .* home, def_a = .-draw.w_def .* away,
              att_a = draw.w_att .* away, def_h = .-draw.w_def .* home)
end

function player_oos_effects(::PositionalPlayerAggregation, draw, value)
    home_F = value.home_F .+ draw.w_bench .* value.home_bench_F
    home_M = value.home_M .+ draw.w_bench .* value.home_bench_M
    home_D = value.home_D .+ draw.w_bench .* value.home_bench_D
    away_F = value.away_F .+ draw.w_bench .* value.away_bench_F
    away_M = value.away_M .+ draw.w_bench .* value.away_bench_M
    away_D = value.away_D .+ draw.w_bench .* value.away_bench_D
    return (;
        att_h = draw.w_att_F .* home_F .+ draw.w_att_M .* home_M,
        def_a = .-draw.w_def_D .* away_D .- draw.w_def_M .* away_M,
        att_a = draw.w_att_F .* away_F .+ draw.w_att_M .* away_M,
        def_h = .-draw.w_def_D .* home_D .- draw.w_def_M .* home_M,
    )
end

function player_oos_effects(::MinuteWeightedPlayerAggregation, draw, value)
    return (; att_h = draw.w_att .* value.home_minute,
              def_a = .-draw.w_def .* value.away_minute,
              att_a = draw.w_att .* value.away_minute,
              def_h = .-draw.w_def .* value.home_minute)
end
