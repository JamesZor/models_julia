# current_development/scottish_lower/neg_bin/l02_negbin_wealth_engines.jl
#
# LOADER: Robust Negative Binomial (NB2) + Scottish Team Wealth Bayesian Engines
#
# 1. TeamGoalsNegBinWealthModel            (Goals-Only NegBin Control + Starting-XI Wealth Delta)
# 2. TeamPxGGoalsAPMNegBinWealthModel      (Arm A: Proxy xG Gamma + RAPM + Team Wealth + NegBin Goals)
# 3. TeamFunnelPxGGoalsAPMNegBinWealthModel(Arm B: 3-Layer Shots Funnel + Proxy xG Quality + RAPM + Team Wealth + NegBin Goals)
#
# Uses HomeAwayDispersion (r_a = exp(log_r), r_h = exp(log_r + δ_r_home))
# with 0-allocation integer recurrence log-PMF evaluation.

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
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l01_proxy_xg_feature.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l02_pxg_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/wealth/l01_wealth_data.jl"))
include("l01_negbin_engines.jl")

# ==============================================================================
# 0. CORE WEALTH HELPERS
# ==============================================================================

function _pxg_core_wealth(data, config)
    date_deltas = Vector{Int}(data[:dates])
    return (;
        home_ids    = Vector{Int}(data[:flat_home_ids]),
        away_ids    = Vector{Int}(data[:flat_away_ids]),
        season_idx  = Vector{Int}(data[:season_indices]),
        month_idx   = Vector{Int}(data[:flat_months]),
        league_idx  = Vector{Int}(data[:flat_league_ids]),
        home_goals  = Vector{Int}(data[:flat_home_goals]),
        away_goals  = Vector{Int}(data[:flat_away_goals]),
        wealth_diff = Vector{Float64}(data[:flat_wealth_diff]),
        w           = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life),
        n_teams     = Int(data[:n_teams]),
        n_seasons   = Int(data[:n_seasons]),
        n_months    = 12,
        n_leagues   = Int(data[:n_leagues]),
    )
end

function _pxg_ratings_wealth(data, config, n::Int)
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
# 1. MODEL 1: TeamGoalsNegBinWealthModel (Goals NegBin + Wealth)
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinWealthModel{
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
    name::String              = "team_goals_negbin_wealth"
end

function Features.required_features(model::TeamGoalsNegBinWealthModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_goals_negbin_wealth_engine(
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

    # 3. Vectorized SIMD Robust Negative Binomial Likelihood
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamGoalsNegBinWealthModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)
    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)
    return build_goals_negbin_wealth_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 2. MODEL 2: TeamPxGGoalsAPMNegBinWealthModel (Arm A + Wealth)
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMNegBinWealthModel{
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
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
    wealth_feature::W         = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution = truncated(Normal(0.15, 0.08), lower = 0.0)
    ν_prior::Distribution     = PXG_NU_PRIOR
    log_κ_prior::Distribution = PXG_LOGK_PRIOR
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "pxg_apm_negbin_wealth"
end

function Features.required_features(model::TeamPxGGoalsAPMNegBinWealthModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.proxy_feature, model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_pxg_goals_apm_negbin_wealth_engine(
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

    # 2. Linear Predictor for True xG
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

    bad_h = isnan.(log_μ_h)
    bad_a = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(ν_xg) || isnan(log_κ)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    μ_h = exp.(log_μ_h)
    μ_a = exp.(log_μ_a)
    κ   = exp(log_κ)

    # 3. Layer 1: Proxy xG Gamma Likelihood
    ll_x_h = (ν_xg - 1.0) * sx_h.S_logx - ν_xg * sum(sx_h.c_x ./ μ_h) - ν_xg * sum(sx_h.c_m .* log_μ_h) + sx_h.S_m * (ν_xg * log(ν_xg) - loggamma(ν_xg))
    ll_x_a = (ν_xg - 1.0) * sx_a.S_logx - ν_xg * sum(sx_a.c_x ./ μ_a) - ν_xg * sum(sx_a.c_m .* log_μ_a) + sx_a.S_m * (ν_xg * log(ν_xg) - loggamma(ν_xg))

    # 4. Layer 2: Goals Negative Binomial Likelihood with Conversion Rate κ
    log_λ_h = log_μ_h .+ log_κ
    log_λ_a = log_μ_a .+ log_κ

    r_h = disp.h
    r_a = disp.a

    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_x_h + ll_x_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinWealthModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)
    xg_h   = Vector{Float64}(data[:flat_home_xg_proxy])
    xg_a   = Vector{Float64}(data[:flat_away_xg_proxy])
    mask_h = Vector{Float64}(data[:flat_pxg_mask_h])
    mask_a = Vector{Float64}(data[:flat_pxg_mask_a])
    sx_h   = _pxg_suff(xg_h, mask_h, d.home_goals, d.w)
    sx_a   = _pxg_suff(xg_a, mask_a, d.away_goals, d.w)
    nb_h   = _negbin_precompute(d.home_goals, d.w)
    nb_a   = _negbin_precompute(d.away_goals, d.w)
    return build_pxg_goals_apm_negbin_wealth_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        sx_h, sx_a,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 3. MODEL 3: TeamFunnelPxGGoalsAPMNegBinWealthModel (Arm B + Wealth)
# ==============================================================================

Base.@kwdef struct TeamFunnelPxGGoalsAPMNegBinWealthModel{
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
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
    wealth_feature::W         = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution = truncated(Normal(0.12, 0.06), lower = 0.0)
    shot_scale::Float64       = 2.29
    ν_prior::Distribution     = PXG_NU_PRIOR
    log_κ_prior::Distribution = PXG_LOGK_PRIOR
    q_prior::Distribution     = PXG_Q_PRIOR
    σ_q_prior::Distribution   = PXG_SIGQ_PRIOR
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    team_quality_on::Bool     = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "funnel_pxg_apm_negbin_wealth"
end

function Features.required_features(model::TeamFunnelPxGGoalsAPMNegBinWealthModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.ShotsFunnelFeature(), model.proxy_feature, model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_funnel_pxg_apm_negbin_wealth_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    wealth_diff::Vector{Float64},
    home_shots::Vector{Int}, away_shots::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    sf_h::NamedTuple, sf_a::NamedTuple,
    nb_h::NamedTuple, nb_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64, team_quality_active::Float64,
    shot_scale::Float64,
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

    # Quality & Conversion Hyperparameters
    logit_q_base ~ config.q_prior
    σ_q_raw      ~ config.σ_q_prior
    σ_q = team_quality_active * σ_q_raw
    η_att_raw ~ filldist(Normal(0.0, 1.0), n_teams)
    η_def_raw ~ filldist(Normal(0.0, 1.0), n_teams)
    η_att = team_quality_active .* (η_att_raw .- mean(η_att_raw))
    η_def = team_quality_active .* (η_def_raw .- mean(η_def_raw))

    ν_q   ~ config.ν_prior
    log_κ ~ config.log_κ_prior

    # 2. Shot Volume Linear Predictor
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    w_shift = w_wealth .* wealth_diff

    log_λ_s_h = clamp.(log(shot_scale) .+ int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                       view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ w_shift, -10.0, 10.0)
    log_λ_s_a = clamp.(log(shot_scale) .+ int_m .+ lg .+
                       view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- w_shift, -10.0, 10.0)

    # 3. Shot Quality Linear Predictor
    logit_q_h = clamp.(logit_q_base .+ σ_q .* (view(η_att, home_ids) .- view(η_def, away_ids)), -10.0, 10.0)
    logit_q_a = clamp.(logit_q_base .+ σ_q .* (view(η_att, away_ids) .- view(η_def, home_ids)), -10.0, 10.0)

    # AD-safe check
    bad = any(isnan.(log_λ_s_h)) || any(isnan.(log_λ_s_a)) || any(isnan.(logit_q_h)) || any(isnan.(logit_q_a)) || isnan(ν_q) || isnan(log_κ)
    Turing.@addlogprob! ifelse(bad, -Inf, 0.0)

    # 4. Layer 1: Shot Volume Poisson Likelihood
    ll_s_h = sum(sf_h.c_s_lin .* log_λ_s_h) - sum(sf_h.c_s_rate .* exp.(log_λ_s_h))
    ll_s_a = sum(sf_a.c_s_lin .* log_λ_s_a) - sum(sf_a.c_s_rate .* exp.(log_λ_s_a))

    # 5. Layer 2: Shot Quality Gamma Likelihood (Optimized SIMD)
    inv_q_h = 1.0 .+ exp.(-logit_q_h)
    log_q_h = -log1pexp.(-logit_q_h)
    inv_q_a = 1.0 .+ exp.(-logit_q_a)
    log_q_a = -log1pexp.(-logit_q_a)

    ll_q_h = (ν_q - 1.0) * sf_h.S_logx - ν_q * sum(sf_h.c_x .* inv_q_h) + ν_q * sum(sf_h.c_s_ev .* log_q_h) +
             sf_h.S_w_ev * (ν_q * log(ν_q) - loggamma(ν_q)) - sum(sf_h.cq_m .* loggamma.(ν_q .* sf_h.cq_n))
    ll_q_a = (ν_q - 1.0) * sf_a.S_logx - ν_q * sum(sf_a.c_x .* inv_q_a) + ν_q * sum(sf_a.c_s_ev .* log_q_a) +
             sf_a.S_w_ev * (ν_q * log(ν_q) - loggamma(ν_q)) - sum(sf_a.cq_m .* loggamma.(ν_q .* sf_a.cq_n))

    # 6. Layer 3: Goals Negative Binomial Likelihood
    log_q_h_full = -log1pexp.(-logit_q_h)
    log_q_a_full = -log1pexp.(-logit_q_a)
    log_μ_h = log_λ_s_h .+ log_q_h_full
    log_μ_a = log_λ_s_a .+ log_q_a_full
    log_λ_g_h = log_μ_h .+ log_κ
    log_λ_g_a = log_μ_a .+ log_κ

    r_h = disp.h
    r_a = disp.a

    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_g_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_g_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMNegBinWealthModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)
    shots_h = Vector{Int}(data[:flat_home_shots])
    shots_a = Vector{Int}(data[:flat_away_shots])
    xg_h    = Vector{Float64}(data[:flat_home_xg_proxy])
    xg_a    = Vector{Float64}(data[:flat_away_xg_proxy])
    mask_h  = Vector{Float64}(data[:flat_pxg_mask_h])
    mask_a  = Vector{Float64}(data[:flat_pxg_mask_a])
    sf_h    = _funnel_suff_opt(shots_h, xg_h, mask_h, d.w)
    sf_a    = _funnel_suff_opt(shots_a, xg_a, mask_a, d.w)
    nb_h    = _negbin_precompute(d.home_goals, d.w)
    nb_a    = _negbin_precompute(d.away_goals, d.w)
    return build_funnel_pxg_apm_negbin_wealth_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        shots_h, shots_a,
        d.home_goals, d.away_goals,
        sf_h, sf_a,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        _pxg_active(config.team_quality_on),
        config.shot_scale,
        config
    )
end

# ==============================================================================
# 4. EXTRACTION & PREDICTION INTERFACES
# ==============================================================================

const ScottishNegBinWealthModelUnion = Union{
    TeamGoalsNegBinWealthModel,
    TeamPxGGoalsAPMNegBinWealthModel,
    TeamFunnelPxGGoalsAPMNegBinWealthModel
}

function PreGame.extract_parameters(
    model::ScottishNegBinWealthModelUnion,
    df::DataFrame,
    feature_set::Features.FeatureSet,
    chain::MCMCChains.Chains
)
    n_samples = length(chain)
    c_names = names(chain)

    μ_base_cols = [col for col in c_names if startswith(string(col), "inter.μ_base[")]
    μ_base = isempty(μ_base_cols) ? zeros(n_samples) : chain[μ_base_cols[end]].data[:, 1]

    # Home Advantage
    ha_samples = if any(startswith(string(col), "ha.home_advantage[") for col in c_names)
        ha_cols = [col for col in c_names if startswith(string(col), "ha.home_advantage[")]
        chain[ha_cols].value.data[:, :, 1]
    elseif in(Symbol("ha.home_advantage"), c_names)
        reshape(chain[Symbol("ha.home_advantage")].data[:, 1], n_samples, 1)
    else
        zeros(n_samples, 1)
    end

    # Team Dynamics (Attacking & Defending)
    att_cols = [col for col in c_names if startswith(string(col), "dyn.α[")]
    def_cols = [col for col in c_names if startswith(string(col), "dyn.β[")]
    att = isempty(att_cols) ? zeros(n_samples, 1) : chain[att_cols].value.data[:, :, 1]
    def = isempty(def_cols) ? zeros(n_samples, 1) : chain[def_cols].value.data[:, :, 1]

    # Dispersion Parameters
    r_h_samples = in(Symbol("disp.h"), c_names) ? chain[Symbol("disp.h")].data[:, 1] :
                  (in(Symbol("disp.log_r"), c_names) ? exp.(chain[Symbol("disp.log_r")].data[:, 1] .+ (in(Symbol("disp.δ_r_home"), c_names) ? chain[Symbol("disp.δ_r_home")].data[:, 1] : 0.0)) : fill(23.66, n_samples))
    r_a_samples = in(Symbol("disp.a"), c_names) ? chain[Symbol("disp.a")].data[:, 1] :
                  (in(Symbol("disp.log_r"), c_names) ? exp.(chain[Symbol("disp.log_r")].data[:, 1]) : fill(9.25, n_samples))

    # APM Outfield Ratings
    w_att = in(:w_att, c_names) ? chain[:w_att].data[:, 1] : zeros(n_samples)
    w_def = in(:w_def, c_names) ? chain[:w_def].data[:, 1] : zeros(n_samples)

    # Wealth Weight
    w_wealth = in(:w_wealth, c_names) ? chain[:w_wealth].data[:, 1] : zeros(n_samples)

    # Conversion Rate κ
    κ = in(:log_κ, c_names) ? exp.(chain[:log_κ].data[:, 1]) : ones(n_samples)

    # Team Quality (for Funnel model)
    q_base = in(:logit_q_base, c_names) ? chain[:logit_q_base].data[:, 1] : fill(logit(0.133), n_samples)
    σ_q    = in(:σ_q_raw, c_names) ? chain[:σ_q_raw].data[:, 1] : zeros(n_samples)
    q_att_cols = [col for col in c_names if startswith(string(col), "η_att_raw[")]
    q_def_cols = [col for col in c_names if startswith(string(col), "η_def_raw[")]
    aq = isempty(q_att_cols) ? zeros(n_samples, 1) : chain[q_att_cols].value.data[:, :, 1]
    dq = isempty(q_def_cols) ? zeros(n_samples, 1) : chain[q_def_cols].value.data[:, :, 1]

    contexts = Features.extract_match_contexts(feature_set, df)
    wealth_map = get(feature_set.data, :wealth_map, Dict{Int, Float64}())
    results = Dict{Int, NamedTuple}()

    for c in contexts
        h_idx = max(c.h_idx, 1)
        a_idx = max(c.a_idx, 1)
        ha_idx = min(h_idx, size(ha_samples, 2))

        # APM pillar
        base_r = Features.rating_base(model.player_ratings_feature)
        h_out = model.apm_on ? ((c.h_D + c.h_M + c.h_F) - 10.0 * base_r) : 0.0
        a_out = model.apm_on ? ((c.a_D + c.a_M + c.a_F) - 10.0 * base_r) : 0.0
        pillar_h = model.apm_on ? (w_att .* h_out .- w_def .* a_out) : zeros(n_samples)
        pillar_a = model.apm_on ? (w_att .* a_out .- w_def .* h_out) : zeros(n_samples)

        # Wealth differential
        delta_w = get(wealth_map, c.match_id, 0.0)
        w_shift = w_wealth .* delta_w

        if model isa TeamFunnelPxGGoalsAPMNegBinWealthModel
            # 3-Layer Funnel computation
            q_h_idx = min(h_idx, size(aq, 2))
            q_a_idx = min(a_idx, size(dq, 2))
            η_a_h = c.h_idx > 0 ? aq[:, q_h_idx] : zeros(n_samples)
            η_d_a = c.a_idx > 0 ? dq[:, q_a_idx] : zeros(n_samples)
            η_a_a = c.a_idx > 0 ? aq[:, q_a_idx] : zeros(n_samples)
            η_d_h = c.h_idx > 0 ? dq[:, q_h_idx] : zeros(n_samples)

            logit_q_h = clamp.(q_base .+ σ_q .* (η_a_h .- η_d_a), -10.0, 10.0)
            logit_q_a = clamp.(q_base .+ σ_q .* (η_a_a .- η_d_h), -10.0, 10.0)
            log_q_h = -log1pexp.(-logit_q_h)
            log_q_a = -log1pexp.(-logit_q_a)

            log_λ_s_h = clamp.(log(model.shot_scale) .+ μ_base .+ ha_samples[:, ha_idx] .+
                               att[:, h_idx] .+ def[:, a_idx] .+ pillar_h .+ w_shift, -10.0, 10.0)
            log_λ_s_a = clamp.(log(model.shot_scale) .+ μ_base .+
                               att[:, a_idx] .+ def[:, h_idx] .+ pillar_a .- w_shift, -10.0, 10.0)

            log_μ_h = log_λ_s_h .+ log_q_h
            log_μ_a = log_λ_s_a .+ log_q_a
            log_λ_h = log_μ_h .+ log.(κ)
            log_λ_a = log_μ_a .+ log.(κ)

            results[c.match_id] = (
                λ_h = exp.(log_λ_h),
                λ_a = exp.(log_λ_a),
                r_h = r_h_samples,
                r_a = r_a_samples,
                true_xg_h = exp.(log_μ_h),
                true_xg_a = exp.(log_μ_a),
                κ   = κ,
                w_wealth = w_wealth
            )
        else
            log_μ_h = clamp.(μ_base .+ ha_samples[:, ha_idx] .+
                             att[:, h_idx] .+ def[:, a_idx] .+ pillar_h .+ w_shift, -10.0, 10.0)
            log_μ_a = clamp.(μ_base .+
                             att[:, a_idx] .+ def[:, h_idx] .+ pillar_a .- w_shift, -10.0, 10.0)

            log_λ_h = model isa TeamPxGGoalsAPMNegBinWealthModel ? (log_μ_h .+ log.(κ)) : log_μ_h
            log_λ_a = model isa TeamPxGGoalsAPMNegBinWealthModel ? (log_μ_a .+ log.(κ)) : log_μ_a

            results[c.match_id] = (
                λ_h = exp.(log_λ_h),
                λ_a = exp.(log_λ_a),
                r_h = r_h_samples,
                r_a = r_a_samples,
                true_xg_h = exp.(log_μ_h),
                true_xg_a = exp.(log_μ_a),
                κ   = κ,
                w_wealth = w_wealth
            )
        end
    end

    return results
end

PreGame.extract_parameters(model::ScottishNegBinWealthModelUnion, df::DataFrame, feature_tuple::Tuple, chain::MCMCChains.Chains) =
    PreGame.extract_parameters(model, df, feature_tuple[1], chain)

Pred.extract_params(::ScottishNegBinWealthModelUnion, row) = (
    λ_h = row.λ_h,
    λ_a = row.λ_a,
    r_h = hasproperty(row, :r_h) ? row.r_h : fill(23.66, length(row.λ_h)),
    r_a = hasproperty(row, :r_a) ? row.r_a : fill(9.25, length(row.λ_a))
)

function Pred.compute_score_matrix(
    model::ScottishNegBinWealthModelUnion,
    params;
    max_goals::Int = 12
)
    λ_h, λ_a = params.λ_h, params.λ_a
    r_h, r_a = params.r_h, params.r_a
    n_samples = length(λ_h)

    S = zeros(Float64, max_goals, max_goals, n_samples)

    @inbounds for k in 1:n_samples
        dist_h = RobustNegativeBinomial(max(Float64(r_h[k]), 1e-4), max(Float64(λ_h[k]), 1e-4))
        dist_a = RobustNegativeBinomial(max(Float64(r_a[k]), 1e-4), max(Float64(λ_a[k]), 1e-4))
        p_h = [pdf(dist_h, i) for i in 0:max_goals-1]
        p_a = [pdf(dist_a, j) for j in 0:max_goals-1]
        S[:, :, k] = p_h .* transpose(p_a)
    end

    return Pred.ScoreMatrix(S)
end
