# current_development/scottish_lower/neg_bin/l01_negbin_engines.jl
#
# LOADER: Robust Negative Binomial (NB2) Goals Likelihood Engines for Scottish Lower (56/57)
#
# 1. TeamGoalsNegBinModel               (Baseline Goals-Only NegBin Control)
# 2. TeamPxGGoalsAPMNegBinModel         (Arm A: Proxy xG Gamma + RAPM + NegBin Goals)
# 3. TeamFunnelPxGGoalsAPMNegBinModel   (Arm B: Shots Volume Poisson + Proxy xG Quality Gamma + RAPM + NegBin Goals)
#
# Dispersion: Uses HomeAwayDispersion (r_a = exp(log_r), r_h = exp(log_r + δ_r_home))
# capturing the empirical Scottish Lower asymmetry (r_away ≈ 9.25 vs r_home ≈ 23.66).

using Turing
using Distributions
using DataFrames
using Dates
using Statistics
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma
using StatsFuns: logit

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

using BayesianFootball.MyDistributions: RobustNegativeBinomial

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l01_proxy_xg_feature.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l02_pxg_engines.jl"))

# ==============================================================================
# 0. SHARED DISPERSION DEFAULT & PRIORS
# ==============================================================================

const SCOTTISH_HOMEAWAY_DISPERSION = PreGame.HomeAwayDispersion(
    log_r     = Normal(2.6, 0.5),
    δ_r_home  = Normal(0.6, 0.5)
)

_pxg_get_data(fs::Dict) = fs
_pxg_get_data(fs::Features.FeatureSet) = fs.data
_pxg_get_data(fs::Tuple) = _pxg_get_data(first(fs))
_pxg_get_data(fs) = hasproperty(fs, :data) ? fs.data : fs

# ==============================================================================
# 1. MODEL 1: BASELINE GOALS-ONLY NEGATIVE BINOMIAL MODEL
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinModel{
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
    name::String              = "team_goals_negbin"
end

@model function build_goals_negbin_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    home_goals::Vector{Int}, away_goals::Vector{Int}, weights::Vector{Float64},
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

    λ_h = exp.(log_λ_h)
    λ_a = exp.(log_λ_a)

    r_h = disp.h
    r_a = disp.a

    # 3. Robust Negative Binomial Likelihood
    ll_g_h = logpdf.(RobustNegativeBinomial.(r_h, λ_h), home_goals)
    ll_g_a = logpdf.(RobustNegativeBinomial.(r_a, λ_a), away_goals)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a), -Inf, 0.0)
    Turing.@addlogprob! sum(weights .* ll_g_h)
    Turing.@addlogprob! sum(weights .* ll_g_a)
end

function Features.required_features(model::TeamGoalsNegBinModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function PreGame.build_turing_model(config::TeamGoalsNegBinModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings(data, config, n)

    return build_goals_negbin_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.home_goals, d.away_goals, d.w,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 2. MODEL 2: ARM A (PROXY xG + RAPM + ROBUST NEGATIVE BINOMIAL GOALS)
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMNegBinModel{
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
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
    ν_prior::Distribution     = PXG_NU_PRIOR
    log_κ_prior::Distribution = PXG_LOGK_PRIOR
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_pxg_goals_apm_negbin"
end

@model function build_pxg_goals_apm_negbin_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sx_h::NamedTuple, sx_a::NamedTuple,
    home_goals::Vector{Int}, away_goals::Vector{Int}, weights::Vector{Float64},
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

    ν_xg  ~ config.ν_prior
    log_κ ~ config.log_κ_prior

    # 2. Expected Scoring Intensity
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_μ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_μ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    bad_h  = isnan.(log_μ_h)
    bad_a  = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    μ_h     = exp.(log_μ_h);   μ_a     = exp.(log_μ_a)
    inv_μ_h = exp.(.-log_μ_h); inv_μ_a = exp.(.-log_μ_a)
    κ       = exp(log_κ)

    r_h = disp.h
    r_a = disp.a

    # 3. Pillar A: Proxy xG (Gamma)
    cν = ν_xg * log(ν_xg) - loggamma(ν_xg)
    ll_xg_h = (ν_xg - 1.0) * sx_h.S_logx - ν_xg * sum(sx_h.c_x .* inv_μ_h) -
              ν_xg * sum(sx_h.c_m .* log_μ_h) + cν * sx_h.S_m
    ll_xg_a = (ν_xg - 1.0) * sx_a.S_logx - ν_xg * sum(sx_a.c_x .* inv_μ_a) -
              ν_xg * sum(sx_a.c_m .* log_μ_a) + cν * sx_a.S_m

    # 4. Pillar B: Goals (Robust Negative Binomial)
    ll_g_h = logpdf.(RobustNegativeBinomial.(r_h, κ .* μ_h), home_goals)
    ll_g_a = logpdf.(RobustNegativeBinomial.(r_a, κ .* μ_a), away_goals)

    Turing.@addlogprob! ifelse(isnan(ll_xg_h) || isnan(ll_xg_a) || isnan(r_h) || isnan(r_a), -Inf, 0.0)
    Turing.@addlogprob! ll_xg_h
    Turing.@addlogprob! ll_xg_a
    Turing.@addlogprob! sum(weights .* ll_g_h)
    Turing.@addlogprob! sum(weights .* ll_g_a)
end

function Features.required_features(model::TeamPxGGoalsAPMNegBinModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(), model.proxy_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings(data, config, n)

    xg_h   = Vector{Float64}(data[:flat_home_xg_proxy])
    xg_a   = Vector{Float64}(data[:flat_away_xg_proxy])
    mask_h = Vector{Float64}(data[:flat_pxg_mask_h])
    mask_a = Vector{Float64}(data[:flat_pxg_mask_a])

    return build_pxg_goals_apm_negbin_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        _pxg_suff(xg_h, mask_h, d.home_goals, d.w),
        _pxg_suff(xg_a, mask_a, d.away_goals, d.w),
        d.home_goals, d.away_goals, d.w,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 3. MODEL 3: ARM B (3-LAYER FUNNEL: SHOTS POISSON -> QUALITY GAMMA -> GOALS NEGBIN)
# ==============================================================================

Base.@kwdef struct TeamFunnelPxGGoalsAPMNegBinModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature
} <: PreGame.AbstractTimeDecayPlayerModel
    shot_scale::Float64       = log(10.0)
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
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
    name::String              = "team_funnel_pxg_goals_apm_negbin"
end

@model function build_funnel_pxg_goals_apm_negbin_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sf_h::NamedTuple, sf_a::NamedTuple,
    home_goals::Vector{Int}, away_goals::Vector{Int}, weights::Vector{Float64},
    shot_scale::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64, quality_active::Float64,
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

    ν_q   ~ config.ν_prior
    log_κ ~ config.log_κ_prior
    q_raw ~ config.q_prior
    σ_q   ~ config.σ_q_prior

    raw_aq ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_dq ~ filldist(Normal(0.0, 1.0), n_teams)
    aq = quality_active .* (raw_aq .* σ_q); aq = aq .- mean(aq)
    dq = quality_active .* (raw_dq .* σ_q); dq = dq .- mean(dq)

    # 2. Volume Layer (λ_s)
    int_m = shot_scale .+ view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    # 3. Quality Layer (q)
    logit_q_h = clamp.(q_raw .+ view(aq, home_ids) .- view(dq, away_ids), -10.0, 10.0)
    logit_q_a = clamp.(q_raw .+ view(aq, away_ids) .- view(dq, home_ids), -10.0, 10.0)

    bad_h  = isnan.(log_λ_h)
    bad_a  = isnan.(log_λ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ) || isnan(q_raw) || isnan(σ_q)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    λ_h     = exp.(log_λ_h);   λ_a     = exp.(log_λ_a)
    log_q_h = .-log1pexp.(.-logit_q_h)
    log_q_a = .-log1pexp.(.-logit_q_a)
    q_h     = exp.(log_q_h);   q_a     = exp.(log_q_a)
    inv_q_h = exp.(.-log_q_h); inv_q_a = exp.(.-log_q_a)

    κ       = exp(log_κ)
    lνq     = log(ν_q)

    r_h = disp.h
    r_a = disp.a

    # 4. Volume Likelihood
    ll_s_h = sum(sf_h.c_s_lin .* log_λ_h) - sum(sf_h.c_s_rate .* λ_h)
    ll_s_a = sum(sf_a.c_s_lin .* log_λ_a) - sum(sf_a.c_s_rate .* λ_a)

    # 5. Quality Likelihood
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

    # 6. Goals Robust Negative Binomial Likelihood
    ll_g_h = logpdf.(RobustNegativeBinomial.(r_h, κ .* λ_h .* q_h), home_goals)
    ll_g_a = logpdf.(RobustNegativeBinomial.(r_a, κ .* λ_a .* q_a), away_goals)

    Turing.@addlogprob! ifelse(isnan(ll_s_h) || isnan(ll_s_a) || isnan(ll_q_h) || isnan(ll_q_a) || isnan(r_h) || isnan(r_a), -Inf, 0.0)
    Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a
    Turing.@addlogprob! sum(weights .* ll_g_h)
    Turing.@addlogprob! sum(weights .* ll_g_a)
end

function Features.required_features(model::TeamFunnelPxGGoalsAPMNegBinModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.ShotsFunnelFeature(), model.proxy_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMNegBinModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings(data, config, n)

    return build_funnel_pxg_goals_apm_negbin_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        _pxg_funnel_suff(Vector{Int}(data[:flat_home_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_h]),
                         Vector{Float64}(data[:flat_home_xg_proxy]),
                         Vector{Int}(data[:flat_home_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_h]),
                         d.home_goals, d.w),
        _pxg_funnel_suff(Vector{Int}(data[:flat_away_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_a]),
                         Vector{Float64}(data[:flat_away_xg_proxy]),
                         Vector{Int}(data[:flat_away_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_a]),
                         d.away_goals, d.w),
        d.home_goals, d.away_goals, d.w,
        config.shot_scale,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        _pxg_active(config.team_quality_on),
        config
    )
end

# ==============================================================================
# 5. PARAMETER EXTRACTORS & PREDICTIVE SCORE MATRICES
# ==============================================================================

const ScottishNegBinModelUnion = Union{
    TeamGoalsNegBinModel,
    TeamPxGGoalsAPMNegBinModel,
    TeamFunnelPxGGoalsAPMNegBinModel
}

function PreGame.extract_parameters(
    model::ScottishNegBinModelUnion,
    df::DataFrame,
    feature_set,
    chain::MCMCChains.Chains
)
    n_matches = nrow(df)
    n_samples = size(chain, 1) * size(chain, 3)

    # Extract common linear components via _pxg_extract_core
    c = _pxg_extract_core(model, df, feature_set, chain)

    # Extract Dispersion (r_h, r_a)
    disp = PreGame.extract_dispersion(chain, model.dispersion_config)
    r_h_samples = disp.h
    r_a_samples = disp.a

    has_kappa = Symbol("log_κ") in keys(chain)
    κ_samples = has_kappa ? exp.(vec(Array(chain[Symbol("log_κ")]))) : ones(Float64, n_samples)

    is_funnel = model isa TeamFunnelPxGGoalsAPMNegBinModel
    
    λ_h_mat = zeros(Float64, n_matches, n_samples)
    λ_a_mat = zeros(Float64, n_matches, n_samples)
    r_h_mat = zeros(Float64, n_matches, n_samples)
    r_a_mat = zeros(Float64, n_matches, n_samples)

    if is_funnel
        q_raw_samples = vec(Array(chain[Symbol("q_raw")]))
        aq = model.team_quality_on ? _pxg_league_offsets(chain, c.n_teams, "raw_aq") .* vec(Array(chain[Symbol("σ_q")])) : zeros(n_samples, c.n_teams)
        dq = model.team_quality_on ? _pxg_league_offsets(chain, c.n_teams, "raw_dq") .* vec(Array(chain[Symbol("σ_q")])) : zeros(n_samples, c.n_teams)

        for i in 1:n_matches
            hid = c.home_ids[i]; aid = c.away_ids[i]
            log_λ_h = model.shot_scale .+ c.core_h[i, :]
            log_λ_a = model.shot_scale .+ c.core_a[i, :]

            logit_q_h = clamp.(q_raw_samples .+ aq[:, hid] .- dq[:, aid], -10.0, 10.0)
            logit_q_a = clamp.(q_raw_samples .+ aq[:, aid] .- dq[:, hid], -10.0, 10.0)
            log_q_h   = .-log1pexp.(.-logit_q_h)
            log_q_a   = .-log1pexp.(.-logit_q_a)

            μ_h = exp.(log_λ_h .+ log_q_h)
            μ_a = exp.(log_λ_a .+ log_q_a)

            λ_h_mat[i, :] = κ_samples .* μ_h
            λ_a_mat[i, :] = κ_samples .* μ_a
            r_h_mat[i, :] = r_h_samples
            r_a_mat[i, :] = r_a_samples
        end
    else
        for i in 1:n_matches
            μ_h = exp.(c.core_h[i, :])
            μ_a = exp.(c.core_a[i, :])

            λ_h_mat[i, :] = κ_samples .* μ_h
            λ_a_mat[i, :] = κ_samples .* μ_a
            r_h_mat[i, :] = r_h_samples
            r_a_mat[i, :] = r_a_samples
        end
    end

    return Dict{String, Any}(
        "λ_h" => [λ_h_mat[i, :] for i in 1:n_matches],
        "λ_a" => [λ_a_mat[i, :] for i in 1:n_matches],
        "r_h" => [r_h_mat[i, :] for i in 1:n_matches],
        "r_a" => [r_a_mat[i, :] for i in 1:n_matches],
        "κ"   => fill(κ_samples, n_matches)
    )
end

Pred.extract_params(::ScottishNegBinModelUnion, row) = (
    λ_h = row.λ_h,
    λ_a = row.λ_a,
    r_h = hasproperty(row, :r_h) ? row.r_h : 23.66,
    r_a = hasproperty(row, :r_a) ? row.r_a : 9.25
)

function _negbin_score_matrix(λ_h::Real, λ_a::Real, r_h::Real, r_a::Real; max_goals::Int = 12)
    dist_h = RobustNegativeBinomial(max(r_h, 1e-4), max(λ_h, 1e-4))
    dist_a = RobustNegativeBinomial(max(r_a, 1e-4), max(λ_a, 1e-4))
    p_h = [pdf(dist_h, i) for i in 0:max_goals]
    p_a = [pdf(dist_a, j) for j in 0:max_goals]
    return p_h * p_a'
end

function Pred.compute_score_matrix(::ScottishNegBinModelUnion, params; max_goals::Int = 12)
    return _negbin_score_matrix(params.λ_h, params.λ_a, params.r_h, params.r_a; max_goals = max_goals)
end
