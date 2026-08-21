# current_development/scottish_lower/open_play/l02_open_play_engines.jl
#
# LOADER 2/2 — High-Performance Open-Play Noise-Reduction Bayesian Negative Binomial Engines
#
# Models:
#   1. TeamGoalsNegBinOpenPlayModel: Baseline Open-Play NegBin Goals (y_np_nog)
#   2. TeamGoalsNegBinWealthOpenPlayModel: Open-Play NegBin Goals + Starting-XI Squad Wealth (ΔW)
#   3. TeamPxGGoalsAPMNegBinWealthOpenPlayModel: Clean Open-Play pxG + RAPM + Wealth + Open-Play Goals
#
# Uses vectorized precomputed loggamma NegBin likelihoods with 0-allocation ReverseDiff tape evaluation.

using Turing
using DynamicPPL: to_submodel
using Distributions
using DataFrames
using Dates
using Statistics
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma
using StatsFuns: logit
using LinearAlgebra

using BayesianFootball
const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

using BayesianFootball.MyDistributions: RobustNegativeBinomial

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l01_proxy_xg_feature.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l02_pxg_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l01_negbin_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/wealth/l01_wealth_data.jl"))

# ==============================================================================
# 0. OPEN-PLAY FEATURE REGISTRATION & EXTRACTORS
# ==============================================================================

"""
    OpenPlayGoalsFeature <: Features.AbstractFeatureConfig

Extracts clean Non-Penalty, Non-Own-Goal (NP-NOG) match scores for Turing training.
"""
struct OpenPlayGoalsFeature <: Features.AbstractFeatureConfig end

function Features.add_feature!(F_data::Dict, ::OpenPlayGoalsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    op_df = extract_open_play_match_data(ds; include_referees=false)
    score_map = Dict(row.match_id => (row.y_np_nog_h, row.y_np_nog_a) for row in eachrow(op_df))
    
    F_data[:flat_home_goals] = [Int(score_map[id][1]) for id in ordered_ids]
    F_data[:flat_away_goals] = [Int(score_map[id][2]) for id in ordered_ids]
end

function Features.add_feature!(F_data::Dict, config::CleanProxyXGFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    clean_pxg_df = aggregate_clean_pxg_by_match(ds; k = config.k)
    pxg_map = Dict(row.match_id => (row.clean_pxg_h, row.clean_pxg_a) for row in eachrow(clean_pxg_df))
    
    F_data[:flat_home_pxg] = [coalesce(pxg_map[id][1], NaN) for id in ordered_ids]
    F_data[:flat_away_pxg] = [coalesce(pxg_map[id][2], NaN) for id in ordered_ids]
end

function _pxg_core_open_play(data, config)
    date_deltas = Vector{Int}(data[:dates])
    wealth_vec = haskey(data, :flat_wealth_diff) ? Vector{Float64}(data[:flat_wealth_diff]) : zeros(Float64, length(data[:flat_home_ids]))
    return (;
        home_ids    = Vector{Int}(data[:flat_home_ids]),
        away_ids    = Vector{Int}(data[:flat_away_ids]),
        season_idx  = Vector{Int}(data[:season_indices]),
        month_idx   = Vector{Int}(data[:flat_months]),
        league_idx  = Vector{Int}(data[:flat_league_ids]),
        home_goals  = Vector{Int}(data[:flat_home_goals]),
        away_goals  = Vector{Int}(data[:flat_away_goals]),
        wealth_diff = wealth_vec,
        w           = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life),
        n_teams     = Int(data[:n_teams]),
        n_seasons   = Int(data[:n_seasons]),
        n_months    = 12,
        n_leagues   = Int(data[:n_leagues]),
    )
end

function _pxg_ratings_open_play(data, config, n::Int)
    config.apm_on || return (zeros(Float64, n), zeros(Float64, n))
    base = Features.rating_base(config.player_ratings_feature)
    h = _pxg_outfield(Vector{Float64}(data[:flat_home_D_rating]),
                      Vector{Float64}(data[:flat_home_M_rating]),
                      Vector{Float64}(data[:flat_home_F_rating]), base)
    a = _pxg_outfield(Vector{Float64}(data[:flat_away_D_rating]),
                      Vector{Float64}(data[:flat_away_M_rating]),
                      Vector{Float64}(data[:flat_away_F_rating]), base)
    return (h, a)
end

# ==============================================================================
# 1. MODEL 1: TeamGoalsNegBinOpenPlayModel (Baseline Open-Play Goals)
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinOpenPlayModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_goals_negbin_open_play"
end

function Features.required_features(model::TeamGoalsNegBinOpenPlayModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), OpenPlayGoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_goals_negbin_open_play_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    nb_h::NamedTuple, nb_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64,
    config
)
    # 1. Submodels
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))
    disp  ~ to_submodel(PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # 2. Linear Predictor
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    bad_h  = isnan.(log_λ_h)
    bad_a  = isnan.(log_λ_a)
    is_bad = any(bad_h) || any(bad_a)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    r_h = disp.h
    r_a = disp.a

    # 3. Negative Binomial Likelihood
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamGoalsNegBinOpenPlayModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_open_play(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_open_play(data, config, n)
    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)
    return build_goals_negbin_open_play_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 2. MODEL 2: TeamGoalsNegBinWealthOpenPlayModel (Open-Play Goals + Wealth)
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinWealthOpenPlayModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    wealth_feature::W         = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution = truncated(Normal(0.15, 0.08), lower = 0.0)
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_goals_negbin_wealth_open_play"
end

function Features.required_features(model::TeamGoalsNegBinWealthOpenPlayModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), OpenPlayGoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_goals_negbin_wealth_open_play_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    wealth_diff::Vector{Float64},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    nb_h::NamedTuple, nb_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64,
    config
)
    # 1. Submodels
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))
    disp  ~ to_submodel(PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    w_wealth ~ config.w_wealth_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # 2. Linear Predictor
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    w_shift = w_wealth .* wealth_diff

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ w_shift, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- w_shift, -10.0, 10.0)

    bad_h  = isnan.(log_λ_h)
    bad_a  = isnan.(log_λ_a)
    is_bad = any(bad_h) || any(bad_a)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    r_h = disp.h
    r_a = disp.a

    # 3. Negative Binomial Likelihood
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamGoalsNegBinWealthOpenPlayModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_open_play(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_open_play(data, config, n)
    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)
    return build_goals_negbin_wealth_open_play_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 3. MODEL 3: TeamPxGGoalsAPMNegBinWealthOpenPlayModel (Clean pxG + RAPM + Wealth + Open-Play Goals)
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMNegBinWealthOpenPlayModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature,
    X<:Features.AbstractFeatureConfig,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    clean_pxg_feature::X      = CleanProxyXGFeature()
    wealth_feature::W         = ScottishTeamWealthFeature()
    ν_prior::Distribution     = PXG_NU_PRIOR
    log_κ_prior::Distribution = PXG_LOGK_PRIOR
    w_wealth_prior::Distribution = truncated(Normal(0.15, 0.08), lower = 0.0)
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_pxg_goals_apm_negbin_wealth_open_play"
end

function Features.required_features(model::TeamPxGGoalsAPMNegBinWealthOpenPlayModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), OpenPlayGoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.clean_pxg_feature, model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_pxg_goals_apm_negbin_wealth_open_play_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    wealth_diff::Vector{Float64},
    sx_h::NamedTuple, sx_a::NamedTuple,
    home_goals::Vector{Int}, away_goals::Vector{Int},
    nb_h::NamedTuple, nb_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64,
    config
)
    # 1. Submodels
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))
    disp  ~ to_submodel(PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    w_wealth ~ config.w_wealth_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    ν_xg  ~ config.ν_prior
    log_κ ~ config.log_κ_prior

    # 2. Expected Scoring Intensity
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    w_shift = w_wealth .* wealth_diff

    log_μ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ w_shift, -10.0, 10.0)
    log_μ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- w_shift, -10.0, 10.0)

    bad_h  = isnan.(log_μ_h)
    bad_a  = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # 3. Clean Proxy xG Gamma Likelihood
    ll_x_h = _pxg_gamma_loglik(log_μ_h, ν_xg, sx_h)
    ll_x_a = _pxg_gamma_loglik(log_μ_a, ν_xg, sx_a)

    # 4. Open-Play Goals NegBin Likelihood (mean = kappa * mu)
    log_λ_h = log_μ_h .+ log_κ
    log_λ_a = log_μ_a .+ log_κ

    r_h = disp.h
    r_a = disp.a

    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_x_h + ll_x_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinWealthOpenPlayModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_open_play(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_open_play(data, config, n)

    pxg_h = Vector{Float64}(data[:flat_home_pxg])
    pxg_a = Vector{Float64}(data[:flat_away_pxg])

    sx_h = _pxg_precompute(pxg_h, d.w)
    sx_a = _pxg_precompute(pxg_a, d.w)
    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_pxg_goals_apm_negbin_wealth_open_play_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        sx_h, sx_a,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 4. PARAMETER EXTRACTOR & PREDICTION OVERLOADS
# ==============================================================================

# Union for dispatch
const OpenPlayNegBinModels = Union{
    TeamGoalsNegBinOpenPlayModel,
    TeamGoalsNegBinWealthOpenPlayModel,
    TeamPxGGoalsAPMNegBinWealthOpenPlayModel
}

function Pred.extract_parameters(model::OpenPlayNegBinModels, chain::MCMCChains.Chains, ds::Data.DataStore; kwargs...)
    return _negbin_extract_parameters(model, chain, ds; kwargs...)
end

function Pred.compute_score_matrix(model::OpenPlayNegBinModels, params::NamedTuple, max_goals::Int = 10)
    return _negbin_compute_score_matrix(model, params, max_goals)
end
