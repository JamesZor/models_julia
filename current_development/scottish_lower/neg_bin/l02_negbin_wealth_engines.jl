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
    sf_h::NamedTuple, sf_a::NamedTuple,
    home_goals::Vector{Int}, away_goals::Vector{Int},
    nb_h::NamedTuple, nb_a::NamedTuple,
    shot_scale::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64, team_quality_active::Float64,
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

    ν_q   ~ config.ν_prior
    log_κ ~ config.log_κ_prior
    q_raw ~ config.q_prior
    σ_q   ~ config.σ_q_prior

    # Team shot quality offsets
    aq_raw ~ filldist(Normal(0.0, σ_q), n_teams)
    dq_raw ~ filldist(Normal(0.0, σ_q), n_teams)
    aq     = team_quality_active .* (aq_raw .- mean(aq_raw))
    dq     = team_quality_active .* (dq_raw .- mean(dq_raw))

    # 2. Linear Predictor (Volume)
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

    # 3. Quality Layer (q = 1 / (1 + exp(-z)), 1/q = 1 + exp(-z), log q = -log(1 + exp(-z)))
    logit_q_h = clamp.(q_raw .+ view(aq, home_ids) .- view(dq, away_ids), -10.0, 10.0)
    logit_q_a = clamp.(q_raw .+ view(aq, away_ids) .- view(dq, home_ids), -10.0, 10.0)

    bad_h  = isnan.(log_λ_h)
    bad_a  = isnan.(log_λ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ) || isnan(q_raw) || isnan(σ_q)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    λ_h     = exp.(log_λ_h)
    λ_a     = exp.(log_λ_a)

    inv_q_h = 1.0 .+ exp.(.-logit_q_h)
    inv_q_a = 1.0 .+ exp.(.-logit_q_a)
    log_q_h = .-log.(inv_q_h)
    log_q_a = .-log.(inv_q_a)

    lνq     = log(ν_q)

    r_h = disp.h
    r_a = disp.a

    # 4. Volume Likelihood
    ll_s_h = sum(sf_h.c_s_lin .* log_λ_h) - sum(sf_h.c_s_rate .* λ_h)
    ll_s_a = sum(sf_a.c_s_lin .* log_λ_a) - sum(sf_a.c_s_rate .* λ_a)

    # 5. Quality Likelihood (Collapsed loggamma across ~20 unique shot counts)
    ll_gamma_q_h = isempty(sf_h.u_shots_f64) ? 0.0 : sum(sf_h.shot_weights .* loggamma.(ν_q .* sf_h.u_shots_f64))
    ll_gamma_q_a = isempty(sf_a.u_shots_f64) ? 0.0 : sum(sf_a.shot_weights .* loggamma.(ν_q .* sf_a.u_shots_f64))

    ll_q_h = ν_q * sf_h.S_Slogx - sf_h.S_logx -
             ν_q * sum(sf_h.cq_x .* inv_q_h) -
             ν_q * sum(sf_h.cq_S .* log_q_h) +
             ν_q * lνq * sf_h.S_cq_S -
             ll_gamma_q_h
    ll_q_a = ν_q * sf_a.S_Slogx - sf_a.S_logx -
             ν_q * sum(sf_a.cq_x .* inv_q_a) -
             ν_q * sum(sf_a.cq_S .* log_q_a) +
             ν_q * lνq * sf_a.S_cq_S -
             ll_gamma_q_a

    # 6. Goals Robust Negative Binomial Likelihood (SIMD vectorized)
    log_λ_gh = log_κ .+ log_λ_h .+ log_q_h
    log_λ_ga = log_κ .+ log_λ_a .+ log_q_a
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_gh, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_ga, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(ll_s_h) || isnan(ll_s_a) || isnan(ll_q_h) || isnan(ll_q_a) || isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMNegBinWealthModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)

    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_funnel_pxg_apm_negbin_wealth_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        _funnel_suff_opt(Vector{Int}(data[:flat_home_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_h]),
                         Vector{Float64}(data[:flat_home_xg_proxy]),
                         Vector{Int}(data[:flat_home_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_h]),
                         d.home_goals, d.w),
        _funnel_suff_opt(Vector{Int}(data[:flat_away_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_a]),
                         Vector{Float64}(data[:flat_away_xg_proxy]),
                         Vector{Int}(data[:flat_away_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_a]),
                         d.away_goals, d.w),
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        config.shot_scale,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        _pxg_active(config.team_quality_on),
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

function _pxg_league_offsets_wealth(chain, n_leagues::Int, sym::String)
    n_samples = size(chain, 1) * size(chain, 3)
    offsets = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        col_sym = Symbol("$(sym)[$i]")
        if col_sym in keys(chain)
            offsets[:, i] = vec(Array(chain[col_sym]))
        end
    end
    return offsets .- mean(offsets, dims = 2)
end

function _get_or_build_wealth_map(data, df)
    if haskey(data, :wealth_map)
        return data[:wealth_map]
    end
    cache_path = joinpath(ROOT, "current_development/scottish_lower/wealth/cache/scottish_val_catalog.jls")
    local val_cat
    if isfile(cache_path)
        val_cat = deserialize(cache_path)
    else
        conn = wealth_db_connect()
        val_cat = fetch_scottish_player_valuations(conn, tournament_ids=[56, 57])
        close(conn)
        mkpath(dirname(cache_path))
        serialize(cache_path, val_cat)
    end
    ds_temp = Data.DataStore(
        matches = df,
        odds = DataFrame(),
        betfair_odds = DataFrame(),
        statistics = DataFrame(),
        lineups = hasproperty(df, :lineups) ? df.lineups : DataFrame(),
        incidents = DataFrame()
    )
    lineup_vals = fetch_match_lineup_values(ds_temp, val_cat; fallback_default=100_000.0)
    wealth_df   = build_match_wealth_table(lineup_vals)
    return Dict(r.match_id => Float64(r.delta_w) for r in eachrow(wealth_df))
end

function _pxg_extract_core_wealth(model, df, feature_set, chain)
    data      = _pxg_get_data(feature_set)
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_leagues = Int(data[:n_leagues])
    team_map  = data[:team_map]
    league_lookup = data[:league_lookup]
    ratings_map   = get(data, :player_ratings_map, Dict{Int, Dict{Tuple{String, String}, Float64}}())
    wealth_map    = _get_or_build_wealth_map(data, df)

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = PreGame.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    δ_mat    = _pxg_league_offsets_wealth(chain, n_leagues, "δ_league_raw")

    n_samples = size(chain, 1) * size(chain, 3)
    γ_mat = model.league_ha_on ? _pxg_league_offsets_wealth(chain, n_leagues, "γ_league_raw") :
                                 zeros(n_samples, n_leagues)

    w_att = vec(Array(chain[:w_att]))
    w_def = vec(Array(chain[:w_def]))
    w_wealth = vec(Array(chain[:w_wealth]))
    apm_a = _pxg_active(model.apm_on)
    base  = Features.rating_base(model.player_ratings_feature)

    out = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid   = Int(row.match_id)
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)
        l_idx = get(league_lookup, mid, 0)

        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)
        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        lg  = l_idx > 0 ? δ_mat[:, l_idx] : zeros(n_samples)
        γlg = l_idx > 0 ? γ_mat[:, l_idx] : zeros(n_samples)

        m_r = get(ratings_map, mid, Dict{Tuple{String, String}, Float64}())
        r_h = (get(m_r, ("home", "D"), 0.0) + get(m_r, ("home", "M"), 0.0) +
               get(m_r, ("home", "F"), 0.0)) - 10.0 * base
        r_a = (get(m_r, ("away", "D"), 0.0) + get(m_r, ("away", "M"), 0.0) +
               get(m_r, ("away", "F"), 0.0)) - 10.0 * base

        pillar_h = apm_a .* (w_att .* r_h .- w_def .* r_a)
        pillar_a = apm_a .* (w_att .* r_a .- w_def .* r_h)

        # Starting-XI wealth differential shift
        w_diff = get(wealth_map, mid, 0.0)
        w_shift = w_wealth .* w_diff

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        int_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        out[mid] = (;
            h_idx, a_idx, n_samples,
            lin_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a .+ pillar_h .+ w_shift, -10.0, 10.0),
            lin_a = clamp.(int_v .+ lg .+               α_a .+ β_h .+ pillar_a .- w_shift, -10.0, 10.0),
        )
    end
    return out, n_samples, n_teams
end

function PreGame.extract_parameters(
    model::ScottishNegBinWealthModelUnion,
    df::DataFrame,
    feature_set::Features.FeatureSet,
    chain::MCMCChains.Chains
)
    core, n_samples, n_teams = _pxg_extract_core_wealth(model, df, feature_set, chain)

    disp = PreGame.extract_dispersion(chain, model.dispersion_config)
    r_h_samples = disp.h
    r_a_samples = disp.a

    has_kappa = Symbol("log_κ") in keys(chain)
    κ = has_kappa ? exp.(vec(Array(chain[Symbol("log_κ")]))) : ones(Float64, n_samples)

    is_funnel = model isa TeamFunnelPxGGoalsAPMNegBinWealthModel

    results = Dict{Int, NamedTuple}()
    if is_funnel
        q_raw_samples = vec(Array(chain[Symbol("q_raw")]))
        σ_q           = vec(Array(chain[Symbol("σ_q")]))
        qa            = _pxg_active(model.team_quality_on)

        aq = zeros(n_samples, n_teams); dq = zeros(n_samples, n_teams)
        for i in 1:n_teams
            col_aq = Symbol("aq_raw[$i]")
            col_dq = Symbol("dq_raw[$i]")
            if col_aq in keys(chain)
                aq[:, i] = qa .* vec(Array(chain[col_aq]))
            end
            if col_dq in keys(chain)
                dq[:, i] = qa .* vec(Array(chain[col_dq]))
            end
        end
        aq .-= mean(aq, dims = 2); dq .-= mean(dq, dims = 2)

        for (mid, c) in core
            a_h = c.h_idx > 0 ? aq[:, c.h_idx] : zeros(n_samples)
            d_h = c.h_idx > 0 ? dq[:, c.h_idx] : zeros(n_samples)
            a_a = c.a_idx > 0 ? aq[:, c.a_idx] : zeros(n_samples)
            d_a = c.a_idx > 0 ? dq[:, c.a_idx] : zeros(n_samples)

            log_λ_s_h = model.shot_scale .+ c.lin_h
            log_λ_s_a = model.shot_scale .+ c.lin_a

            logit_q_h = clamp.(q_raw_samples .+ a_h .- d_a, -10.0, 10.0)
            logit_q_a = clamp.(q_raw_samples .+ a_a .- d_h, -10.0, 10.0)
            log_q_h   = .-log.(1.0 .+ exp.(.-logit_q_h))
            log_q_a   = .-log.(1.0 .+ exp.(.-logit_q_a))

            μ_h = exp.(log_λ_s_h .+ log_q_h)
            μ_a = exp.(log_λ_s_a .+ log_q_a)

            results[mid] = (;
                λ_h = κ .* μ_h,
                λ_a = κ .* μ_a,
                r_h = r_h_samples,
                r_a = r_a_samples,
                true_xg_h = μ_h,
                true_xg_a = μ_a,
                κ   = κ
            )
        end
    else
        for (mid, c) in core
            μ_h = exp.(c.lin_h)
            μ_a = exp.(c.lin_a)

            results[mid] = (;
                λ_h = κ .* μ_h,
                λ_a = κ .* μ_a,
                r_h = r_h_samples,
                r_a = r_a_samples,
                true_xg_h = μ_h,
                true_xg_a = μ_a,
                κ   = κ
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
