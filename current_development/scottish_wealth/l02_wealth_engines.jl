# current_development/scottish_wealth/l02_wealth_engines.jl
#
# LOADER: High-Performance Scottish Lower Wealth-Augmented Bayesian Engines
#
# Embeds `ScottishTeamWealthFeature` (Starting-XI wealth delta ΔW) into the 3 Scottish models:
# 1. Baseline: DynamicFunnelPlusMinusWealthModel
# 2. Arm A:    TeamPxGGoalsAPMWealthModel
# 3. Arm B:    TeamFunnelPxGGoalsAPMWealthModel (Champion 3-Layer)

using Turing
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
    base = Features.required_features(PreGame.DynamicGoalsPlusMinusLeagueTimeDecayModel(
        interception_config    = model.interception_config,
        dynamics_config        = model.dynamics_config,
        homeadvantage_config   = model.homeadvantage_config,
        player_ratings_feature = model.player_ratings_feature,
    ))
    return vcat(base, [model.proxy_feature, model.wealth_feature])
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
        S_mlogx  = sum(c_mlogx),
        c_g_lin  = c_g_lin,
        S_g      = sum(c_g_lin),
        c_g_rate = w
    )
end

@model function build_pxg_goals_apm_wealth_engine(
    c, ratings, suff_h, suff_a,
    config::TeamPxGGoalsAPMWealthModel,
    n_obs::Int
)
    # Interception, HA, Dynamics, League components
    inter   = Turing.@submodel PreGame.build_interception_component(config.interception_config, c.n_seasons, c.n_months)
    ha      = Turing.@submodel PreGame.build_home_advantage_component(config.homeadvantage_config, c.n_teams)
    tdyn_a  = Turing.@submodel PreGame.build_dynamics(config.dynamics_config, c.n_teams)
    tdyn_d  = Turing.@submodel PreGame.build_dynamics(config.dynamics_config, c.n_teams)
    
    raw_δ_lg ~ filldist(Normal(0.0, 1.0), c.n_leagues)
    δ_lg = (raw_δ_lg .- mean(raw_δ_lg)) .* config.league_offset_sd
    
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    
    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    log_kappa ~ config.log_κ_prior
    kappa = exp(log_kappa)
    nu ~ config.ν_prior

    # Linear Predictor
    int_m = view(inter.μ_base, c.season_idx) .+ view(inter.δ_month, c.month_idx)
    lg_m  = view(δ_lg, c.league_idx)
    ha_m  = view(ha, c.home_ids)
    
    (R_h, R_a) = ratings
    att_h = w_att .* R_h
    def_a = w_def .* R_a
    att_a = w_att .* R_a
    def_h = w_def .* R_h
    
    w_shift = w_wealth .* c.wealth_diff
    
    log_μ_h = clamp.(int_m .+ lg_m .+ ha_m .+ view(tdyn_a, c.home_ids) .- view(tdyn_d, c.away_ids) .+ att_h .- def_a .+ w_shift, -15.0, 15.0)
    log_μ_a = clamp.(int_m .+ lg_m        .+ view(tdyn_a, c.away_ids) .- view(tdyn_d, c.home_ids) .+ att_a .- def_h .- w_shift, -15.0, 15.0)
    
    μ_h = exp.(log_μ_h)
    μ_a = exp.(log_μ_a)
    
    # Pillar A: Proxy xG (Gamma)
    ll_xg_h = (nu - 1.0)*suff_h.S_mlogx - nu*sum(suff_h.c_x ./ μ_h) - nu*sum(suff_h.c_m .* log_μ_h) + suff_h.S_m*(nu*log(nu) - loggamma(nu))
    ll_xg_a = (nu - 1.0)*suff_a.S_mlogx - nu*sum(suff_a.c_x ./ μ_a) - nu*sum(suff_a.c_m .* log_μ_a) + suff_a.S_m*(nu*log(nu) - loggamma(nu))
    
    # Pillar B: Goals (Poisson)
    ll_g_h = suff_h.S_g*log_kappa + sum(suff_h.c_g_lin .* log_μ_h) - kappa*sum(suff_h.c_g_rate .* μ_h)
    ll_g_a = suff_a.S_g*log_kappa + sum(suff_a.c_g_lin .* log_μ_a) - kappa*sum(suff_a.c_g_rate .* μ_a)
    
    Turing.@addlogprob! ll_xg_h + ll_xg_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMWealthModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    c = _pxg_core_wealth(d, config)
    n = length(c.home_goals)
    ratings = _pxg_ratings_wealth(d, config, n)
    
    xg_h = Vector{Float64}(d[:flat_home_pxg])
    xg_a = Vector{Float64}(d[:flat_away_pxg])
    m_h  = Vector{Float64}(d[:flat_home_pxg_mask])
    m_a  = Vector{Float64}(d[:flat_away_pxg_mask])
    
    suff_h = _pxg_suff(xg_h, m_h, c.home_goals, c.w)
    suff_a = _pxg_suff(xg_a, m_a, c.away_goals, c.w)
    
    return build_pxg_goals_apm_wealth_engine(c, ratings, suff_h, suff_a, config, n)
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
    base = Features.required_features(PreGame.DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel(
        interception_config    = model.interception_config,
        dynamics_config        = model.dynamics_config,
        homeadvantage_config   = model.homeadvantage_config,
        player_ratings_feature = model.player_ratings_feature,
    ))
    return vcat(base, [model.proxy_feature, model.wealth_feature])
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
        S_m      = sum(wx),
        c_g_lin  = c_g_lin,
        S_g      = sum(c_g_lin),
        c_g_rate = w
    )
end

@model function build_funnel_pxg_goals_apm_wealth_engine(
    c, ratings, suff_h, suff_a,
    config::TeamFunnelPxGGoalsAPMWealthModel,
    n_obs::Int
)
    inter   = Turing.@submodel PreGame.build_interception_component(config.interception_config, c.n_seasons, c.n_months)
    ha      = Turing.@submodel PreGame.build_home_advantage_component(config.homeadvantage_config, c.n_teams)
    tdyn_a  = Turing.@submodel PreGame.build_dynamics(config.dynamics_config, c.n_teams)
    tdyn_d  = Turing.@submodel PreGame.build_dynamics(config.dynamics_config, c.n_teams)
    
    raw_δ_lg ~ filldist(Normal(0.0, 1.0), c.n_leagues)
    δ_lg = (raw_δ_lg .- mean(raw_δ_lg)) .* config.league_offset_sd
    
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    
    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    log_kappa ~ config.log_κ_prior
    kappa = exp(log_kappa)
    
    q_raw ~ config.q_prior
    σ_q   ~ config.σ_q_prior
    ν_q   ~ config.ν_prior
    
    raw_aq ~ filldist(Normal(0.0, 1.0), c.n_teams)
    raw_dq ~ filldist(Normal(0.0, 1.0), c.n_teams)
    aq = (raw_aq .- mean(raw_aq)) .* σ_q
    dq = (raw_dq .- mean(raw_dq)) .* σ_q

    int_m = view(inter.μ_base, c.season_idx) .+ view(inter.δ_month, c.month_idx)
    lg_m  = view(δ_lg, c.league_idx)
    ha_m  = view(ha, c.home_ids)
    
    (R_h, R_a) = ratings
    att_h = w_att .* R_h
    def_a = w_def .* R_a
    att_a = w_att .* R_a
    def_h = w_def .* R_h
    
    w_shift = w_wealth .* c.wealth_diff
    
    log_λ_s_h = config.shot_scale .+ int_m .+ lg_m .+ ha_m .+ view(tdyn_a, c.home_ids) .- view(tdyn_d, c.away_ids) .+ att_h .- def_a .+ w_shift
    log_λ_s_a = config.shot_scale .+ int_m .+ lg_m        .+ view(tdyn_a, c.away_ids) .- view(tdyn_d, c.home_ids) .+ att_a .- def_h .- w_shift
    
    λ_s_h = exp.(clamp.(log_λ_s_h, -15.0, 15.0))
    λ_s_a = exp.(clamp.(log_λ_s_a, -15.0, 15.0))

    # Quality Pillar
    logit_q_h = q_raw .+ view(aq, c.home_ids) .- view(dq, c.away_ids)
    logit_q_a = q_raw .+ view(aq, c.away_ids) .- view(dq, c.home_ids)
    
    log_q_h = -log1pexp.(-logit_q_h)
    log_q_a = -log1pexp.(-logit_q_a)
    inv_q_h = 1.0 .+ exp.(-logit_q_h)
    inv_q_a = 1.0 .+ exp.(-logit_q_a)
    
    # 1. Volume Likelihood
    ll_vol_h = sum(suff_h.c_s_lin .* log_λ_s_h) - sum(suff_h.c_s_rate .* λ_s_h)
    ll_vol_a = sum(suff_a.c_s_lin .* log_λ_s_a) - sum(suff_a.c_s_rate .* λ_s_a)

    # 2. Quality Likelihood
    ll_q_h = ν_q * suff_h.S_Slogx - suff_h.S_logx - ν_q*sum(suff_h.cq_x .* inv_q_h) - ν_q*sum(suff_h.cq_S .* log_q_h) +
             ν_q*log(ν_q)*sum(suff_h.cq_S) - sum(suff_h.cq_m .* loggamma.(ν_q .* suff_h.n_ev))
    ll_q_a = ν_q * suff_a.S_Slogx - suff_a.S_logx - ν_q*sum(suff_a.cq_x .* inv_q_a) - ν_q*sum(suff_a.cq_S .* log_q_a) +
             ν_q*log(ν_q)*sum(suff_a.cq_S) - sum(suff_a.cq_m .* loggamma.(ν_q .* suff_a.n_ev))

    # 3. Goals Likelihood
    log_mu_h = log_λ_s_h .+ log_q_h
    log_mu_a = log_λ_s_a .+ log_q_a
    mu_h     = λ_s_h .* exp.(log_q_h)
    mu_a     = λ_s_a .* exp.(log_q_a)
    
    ll_g_h = suff_h.S_g*log_kappa + sum(suff_h.c_g_lin .* log_mu_h) - kappa*sum(suff_h.c_g_rate .* mu_h)
    ll_g_a = suff_a.S_g*log_kappa + sum(suff_a.c_g_lin .* log_mu_a) - kappa*sum(suff_a.c_g_rate .* mu_a)

    Turing.@addlogprob! ll_vol_h + ll_vol_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMWealthModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    c = _pxg_core_wealth(d, config)
    n = length(c.home_goals)
    ratings = _pxg_ratings_wealth(d, config, n)
    
    shots_h = Vector{Int}(d[:flat_home_shots])
    shots_a = Vector{Int}(d[:flat_away_shots])
    m_s_h   = Vector{Float64}(d[:flat_home_shots_mask])
    m_s_a   = Vector{Float64}(d[:flat_away_shots_mask])
    
    xg_h = Vector{Float64}(d[:flat_home_pxg])
    xg_a = Vector{Float64}(d[:flat_away_pxg])
    nev_h = Vector{Int}(d[:flat_home_pxg_events])
    nev_a = Vector{Int}(d[:flat_away_pxg_events])
    m_x_h = Vector{Float64}(d[:flat_home_pxg_mask])
    m_x_a = Vector{Float64}(d[:flat_away_pxg_mask])
    
    suff_h = _pxg_funnel_suff(shots_h, m_s_h, xg_h, nev_h, m_x_h, c.home_goals, c.w)
    suff_a = _pxg_funnel_suff(shots_a, m_s_a, xg_a, nev_a, m_x_a, c.away_goals, c.w)
    
    return build_funnel_pxg_goals_apm_wealth_engine(c, ratings, suff_h, suff_a, config, n)
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
