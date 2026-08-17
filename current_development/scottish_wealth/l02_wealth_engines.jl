# current_development/scottish_wealth/l02_wealth_engines.jl
#
# LOADER: High-Performance Scottish Lower Wealth-Augmented Bayesian Engines
#
# Embeds `ScottishTeamWealthFeature` (Starting-XI wealth delta ΔW) into the Scottish models:
# 1. Arm A: TeamPxGGoalsAPMWealthModel
# 2. Arm B: TeamFunnelPxGGoalsAPMWealthModel (Champion 3-Layer)

using Turing
using DynamicPPL: to_submodel
using Distributions
using DataFrames
using Dates
using Statistics
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma
using StatsFuns: logit

using BayesianFootball
const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l01_proxy_xg_feature.jl"))
include("l01_wealth_data.jl")

const PXG_NU_PRIOR    = truncated(Normal(4.0, 1.5), lower = 0.5)
const PXG_LOGK_PRIOR  = Normal(0.0, 0.2)
const PXG_Q_PRIOR     = Normal(logit(0.133), 0.5)
const PXG_SIGQ_PRIOR  = truncated(Normal(0.0, 0.15), lower = 0.0)

_pxg_outfield(D, M, F, base) = (D .+ M .+ F) .- 10.0 * base
_pxg_active(b::Bool) = b ? 1.0 : 0.0

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
# 1. ARM A: TeamPxGGoalsAPMWealthModel
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMWealthModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    P<:Features.AbstractPlusMinusFeature,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
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
end

function Features.required_features(model::TeamPxGGoalsAPMWealthModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.proxy_feature, model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function _pxg_suff(xg::Vector{Float64}, mask::Vector{Float64}, goals::Vector{Int}, w::Vector{Float64})
    wm      = w .* mask
    c_x     = wm .* xg
    c_mlogx = wm .* log.(xg)
    c_g_lin = w .* goals
    return (
        c_x      = c_x,
        c_m      = wm,
        S_m      = sum(wm),
        S_logx   = sum(c_mlogx),
        c_g_lin  = c_g_lin,
        S_g      = sum(c_g_lin),
        c_g_rate = w
    )
end

@model function build_pxg_goals_apm_wealth_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    wealth_diff::Vector{Float64},
    sx_h::NamedTuple, sx_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64,
    config
)
    # Components
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    ν_xg  ~ config.ν_prior
    log_κ ~ config.log_κ_prior

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

    # AD-safe rejection
    bad_h = isnan.(log_μ_h)
    bad_a = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(ν_xg) || isnan(log_κ)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    μ_h = exp.(log_μ_h)
    μ_a = exp.(log_μ_a)
    κ   = exp(log_κ)

    # 1. Proxy xG Likelihood
    ll_x_h = (ν_xg - 1.0) * sx_h.S_logx - ν_xg * sum(sx_h.c_x ./ μ_h) - ν_xg * sum(sx_h.c_m .* log_μ_h) + sx_h.S_m * (ν_xg * log(ν_xg) - loggamma(ν_xg))
    ll_x_a = (ν_xg - 1.0) * sx_a.S_logx - ν_xg * sum(sx_a.c_x ./ μ_a) - ν_xg * sum(sx_a.c_m .* log_μ_a) + sx_a.S_m * (ν_xg * log(ν_xg) - loggamma(ν_xg))

    # 2. Goals Likelihood
    ll_g_h = sx_h.S_g * log_κ + sum(sx_h.c_g_lin .* log_μ_h) - κ * sum(sx_h.c_g_rate .* μ_h)
    ll_g_a = sx_a.S_g * log_κ + sum(sx_a.c_g_lin .* log_μ_a) - κ * sum(sx_a.c_g_rate .* μ_a)

    Turing.@addlogprob! ll_x_h + ll_x_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMWealthModel, feature_set)
    data = feature_set.data
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)
    sx_h = _pxg_suff(Vector{Float64}(data[:flat_home_pxg]),
                     Vector{Float64}(data[:flat_home_pxg_mask]),
                     d.home_goals, d.w)
    sx_a = _pxg_suff(Vector{Float64}(data[:flat_away_pxg]),
                     Vector{Float64}(data[:flat_away_pxg_mask]),
                     d.away_goals, d.w)
    return build_pxg_goals_apm_wealth_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        sx_h, sx_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 2. ARM B: TeamFunnelPxGGoalsAPMWealthModel (Champion 3-Layer)
# ==============================================================================

Base.@kwdef struct TeamFunnelPxGGoalsAPMWealthModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    P<:Features.AbstractPlusMinusFeature,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
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
end

function Features.required_features(model::TeamFunnelPxGGoalsAPMWealthModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.ShotsFunnelFeature(), model.proxy_feature, model.wealth_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function _pxg_funnel_suff(shots_bbc::Vector{Int}, mask_s::Vector{Float64},
                          xg::Vector{Float64}, n_ev::Vector{Int}, mask_x::Vector{Float64},
                          goals::Vector{Int}, w::Vector{Float64})
    ws      = w .* mask_s
    wx      = w .* mask_x
    logx    = log.(xg)
    n_safe  = [mask_x[i] > 0 && n_ev[i] > 0 ? Float64(n_ev[i]) : 1.0 for i in eachindex(n_ev)]
    cq_S    = wx .* n_safe
    c_g_lin = w .* goals
    return (
        c_s_lin  = ws .* shots_bbc,
        c_s_rate = ws,
        cq_m     = wx,
        cq_S     = cq_S,
        cq_x     = wx .* xg,
        n_ev     = n_safe,
        S_Slogx  = sum(cq_S .* logx),
        S_logx   = sum(wx .* logx),
        S_cq_S   = sum(cq_S),
        c_g_lin  = c_g_lin,
        S_g      = sum(c_g_lin),
        c_g_rate = w
    )
end

@model function build_funnel_pxg_goals_apm_wealth_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    wealth_diff::Vector{Float64},
    sf_h::NamedTuple, sf_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    shot_scale::Float64,
    league_ha_active::Float64, apm_active::Float64, quality_active::Float64,
    config
)
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    ν_q   ~ config.ν_prior
    log_κ ~ config.log_κ_prior
    q_raw ~ config.q_prior
    σ_q   ~ config.σ_q_prior

    raw_aq ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_dq ~ filldist(Normal(0.0, 1.0), n_teams)
    aq = quality_active .* (raw_aq .* σ_q); aq = aq .- mean(aq)
    dq = quality_active .* (raw_dq .* σ_q); dq = dq .- mean(dq)

    int_m = shot_scale .+ view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    w_shift = w_wealth .* wealth_diff

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ w_shift, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- w_shift, -10.0, 10.0)

    # Quality Pillar
    qlin_h  = q_raw .+ view(aq, home_ids) .- view(dq, away_ids)
    qlin_a  = q_raw .+ view(aq, away_ids) .- view(dq, home_ids)
    log_q_h = .-log1pexp.(.-qlin_h)
    log_q_a = .-log1pexp.(.-qlin_a)

    bad_h  = isnan.(log_λ_h) .| isnan.(log_q_h)
    bad_a  = isnan.(log_λ_a) .| isnan.(log_q_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(ν_q) || isnan(log_κ)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    log_q_h = ifelse.(bad_h, zero.(log_q_h), log_q_h)
    log_q_a = ifelse.(bad_a, zero.(log_q_a), log_q_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    λ_h     = exp.(log_λ_h);   λ_a     = exp.(log_λ_a)
    q_h     = exp.(log_q_h);   q_a     = exp.(log_q_a)
    inv_q_h = exp.(.-log_q_h); inv_q_a = exp.(.-log_q_a)

    κ   = exp(log_κ)
    lνq = log(ν_q)

    # 1. Volume Likelihood
    ll_s_h = sum(sf_h.c_s_lin .* log_λ_h) - sum(sf_h.c_s_rate .* λ_h)
    ll_s_a = sum(sf_a.c_s_lin .* log_λ_a) - sum(sf_a.c_s_rate .* λ_a)

    # 2. Quality Likelihood
    ll_q_h = ν_q * sf_h.S_Slogx - sf_h.S_logx -
             ν_q * sum(sf_h.cq_x .* inv_q_h) -
             ν_q * sum(sf_h.cq_S .* log_q_h) +
             ν_q * lνq * sf_h.S_cq_S -
             sum(sf_h.cq_m .* loggamma.(ν_q .* sf_h.n_ev))
    ll_q_a = ν_q * sf_a.S_Slogx - sf_a.S_logx -
             ν_q * sum(sf_a.cq_x .* inv_q_a) -
             ν_q * sum(sf_a.cq_S .* log_q_a) +
             ν_q * lνq * sf_a.S_cq_S -
             sum(sf_a.cq_m .* loggamma.(ν_q .* sf_a.n_ev))

    # 3. Goals Likelihood
    ll_g_h = sf_h.S_g * log_κ + sum(sf_h.c_g_lin .* (log_λ_h .+ log_q_h)) -
             κ * sum(sf_h.c_g_rate .* λ_h .* q_h)
    ll_g_a = sf_a.S_g * log_κ + sum(sf_a.c_g_lin .* (log_λ_a .+ log_q_a)) -
             κ * sum(sf_a.c_g_rate .* λ_a .* q_a)

    Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMWealthModel, feature_set)
    data = feature_set.data
    d    = _pxg_core_wealth(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_wealth(data, config, n)
    sf_h = _pxg_funnel_suff(Vector{Int}(data[:flat_home_shots]),
                            Vector{Float64}(data[:flat_home_shots_mask]),
                            Vector{Float64}(data[:flat_home_pxg]),
                            Vector{Int}(data[:flat_home_pxg_events]),
                            Vector{Float64}(data[:flat_home_pxg_mask]),
                            d.home_goals, d.w)
    sf_a = _pxg_funnel_suff(Vector{Int}(data[:flat_away_shots]),
                            Vector{Float64}(data[:flat_away_shots_mask]),
                            Vector{Float64}(data[:flat_away_pxg]),
                            Vector{Int}(data[:flat_away_pxg_events]),
                            Vector{Float64}(data[:flat_away_pxg_mask]),
                            d.away_goals, d.w)
    return build_funnel_pxg_goals_apm_wealth_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.wealth_diff,
        sf_h, sf_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.shot_scale,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        _pxg_active(config.team_quality_on),
        config
    )
end

# ==============================================================================
# 3. PREDICTION DISPATCH (Poisson score matrix override)
# ==============================================================================

const ScottishWealthModelUnion = Union{
    TeamPxGGoalsAPMWealthModel,
    TeamFunnelPxGGoalsAPMWealthModel
}

function Pred.extract_params(model::ScottishWealthModelUnion, chain::MCMCChains.Chains, split)
    df = DataFrame(chain)
    return (
        chain_df = df,
        w_att = mean(df[!, "w_att"]),
        w_def = mean(df[!, "w_def"]),
        w_wealth = hasproperty(df, :w_wealth) ? mean(df[!, :w_wealth]) : 0.0,
        log_kappa = hasproperty(df, :log_kappa) ? mean(df[!, :log_kappa]) : 0.0
    )
end

@info "Scottish Lower Wealth-Augmented Models defined successfully"
