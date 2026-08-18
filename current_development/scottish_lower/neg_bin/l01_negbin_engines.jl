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
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
    shot_scale::Float64       = 2.2
    q_prior::Distribution     = PXG_Q_PRIOR
    σ_q_prior::Distribution   = PXG_SIGQ_PRIOR
    ν_q_prior::Distribution   = PXG_NU_PRIOR
    log_κ_prior::Distribution = PXG_LOGK_PRIOR
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_funnel_pxg_goals_apm_negbin"
end

@model function build_funnel_pxg_goals_apm_negbin_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sx_h::NamedTuple, sx_a::NamedTuple,
    sq_h::NamedTuple, sq_a::NamedTuple,
    shots_h::Vector{Float64}, shots_a::Vector{Float64}, shots_mask::Vector{Float64},
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

    q_base ~ config.q_prior
    σ_q    ~ config.σ_q_prior
    q_att_raw ~ filldist(Normal(0.0, 1.0), n_teams)
    q_def_raw ~ filldist(Normal(0.0, 1.0), n_teams)
    q_att = q_att_raw .* σ_q
    q_def = q_def_raw .* σ_q

    ν_q   ~ config.ν_q_prior
    log_κ ~ config.log_κ_prior

    # 2. Volume Layer (λ_s)
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_λ_s_h = clamp.(config.shot_scale .+ int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                       view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_λ_s_a = clamp.(config.shot_scale .+ int_m .+ lg .+
                       view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    # 3. Quality Layer (q)
    logit_q_h = clamp.(q_base .+ view(q_att, home_ids) .- view(q_def, away_ids), -10.0, 10.0)
    logit_q_a = clamp.(q_base .+ view(q_att, away_ids) .- view(q_def, home_ids), -10.0, 10.0)
    q_h = 1.0 ./ (1.0 .+ exp.(.-logit_q_h))
    q_a = 1.0 ./ (1.0 .+ exp.(.-logit_q_a))

    log_q_h = .-log1pexp.(.-logit_q_h)
    log_q_a = .-log1pexp.(.-logit_q_a)

    log_μ_h = log_λ_s_h .+ log_q_h
    log_μ_a = log_λ_s_a .+ log_q_a

    bad_h  = isnan.(log_μ_h)
    bad_a  = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ) || isnan(q_base) || isnan(σ_q)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    λ_s_h = exp.(log_λ_s_h); λ_s_a = exp.(log_λ_s_a)
    μ_h   = exp.(log_μ_h);   μ_a   = exp.(log_μ_a)
    κ     = exp(log_κ)

    r_h = disp.h
    r_a = disp.a

    # 4. Layer 1: Shots Volume (Poisson)
    suff_s_h = loggamma.(shots_h .+ 1.0)
    suff_s_a = loggamma.(shots_a .+ 1.0)
    ll_s_h = sum(shots_mask .* ((shots_h .* log_λ_s_h) .- λ_s_h .- suff_s_h))
    ll_s_a = sum(shots_mask .* ((shots_a .* log_λ_s_a) .- λ_s_a .- suff_s_a))

    # 5. Layer 2: Quality Gamma (Conditional on Event Shots)
    α_q_h = ν_q .* sq_h.c_S
    α_q_a = ν_q .* sq_a.c_S
    lν    = log(ν_q)
    ll_q_h = sum(α_q_h .* sq_h.c_mlogx) - sq_h.S_logx - ν_q * sum(sq_h.c_x ./ q_h) -
             sum(α_q_h .* (log_q_h .- lν)) - sum(sq_h.mask .* loggamma.(α_q_h))
    ll_q_a = sum(α_q_a .* sq_a.c_mlogx) - sq_a.S_logx - ν_q * sum(sq_a.c_x ./ q_a) -
             sum(α_q_a .* (log_q_a .- lν)) - sum(sq_a.mask .* loggamma.(α_q_a))

    # 6. Layer 3: Goals Robust Negative Binomial
    ll_g_h = logpdf.(RobustNegativeBinomial.(r_h, κ .* μ_h), home_goals)
    ll_g_a = logpdf.(RobustNegativeBinomial.(r_a, κ .* μ_a), away_goals)

    Turing.@addlogprob! ifelse(isnan(ll_s_h) || isnan(ll_s_a) || isnan(ll_q_h) || isnan(ll_q_a) || isnan(r_h) || isnan(r_a), -Inf, 0.0)
    Turing.@addlogprob! ll_s_h
    Turing.@addlogprob! ll_s_a
    Turing.@addlogprob! ll_q_h
    Turing.@addlogprob! ll_q_a
    Turing.@addlogprob! sum(weights .* ll_g_h)
    Turing.@addlogprob! sum(weights .* ll_g_a)
end

# ==============================================================================
# 4. BUILDERS (Data Packaging)
# ==============================================================================

function PreGame.build_turing_model(model::TeamGoalsNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    n = length(core.home_ids)
    rat_h, rat_a = _pxg_ratings(data, model, n)

    return build_goals_negbin_engine(
        core.home_ids, core.away_ids,
        core.season_idx, core.month_idx, core.league_idx,
        rat_h, rat_a,
        core.home_goals, core.away_goals, core.w,
        core.n_teams, core.n_seasons, core.n_months, core.n_leagues,
        _pxg_active(model.league_ha_on), _pxg_active(model.apm_on),
        model
    )
end

function PreGame.build_turing_model(model::TeamPxGGoalsAPMNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    n = length(core.home_ids)
    rat_h, rat_a = _pxg_ratings(data, model, n)

    xg_h   = Vector{Float64}(data[:proxy_xg_home])
    xg_a   = Vector{Float64}(data[:proxy_xg_away])
    mask_h = Vector{Float64}(data[:proxy_xg_mask_home])
    mask_a = Vector{Float64}(data[:proxy_xg_mask_away])

    sx_h = _pxg_suff(xg_h, mask_h, core.home_goals, core.w)
    sx_a = _pxg_suff(xg_a, mask_a, core.away_goals, core.w)

    return build_pxg_goals_apm_negbin_engine(
        core.home_ids, core.away_ids,
        core.season_idx, core.month_idx, core.league_idx,
        rat_h, rat_a,
        sx_h, sx_a,
        core.home_goals, core.away_goals, core.w,
        core.n_teams, core.n_seasons, core.n_months, core.n_leagues,
        _pxg_active(model.league_ha_on), _pxg_active(model.apm_on),
        model
    )
end

function PreGame.build_turing_model(model::TeamFunnelPxGGoalsAPMNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    n = length(core.home_ids)
    rat_h, rat_a = _pxg_ratings(data, model, n)

    xg_h   = Vector{Float64}(data[:proxy_xg_home])
    xg_a   = Vector{Float64}(data[:proxy_xg_away])
    mask_h = Vector{Float64}(data[:proxy_xg_mask_home])
    mask_a = Vector{Float64}(data[:proxy_xg_mask_away])
    S_h    = Vector{Float64}(data[:proxy_xg_shots_home])
    S_a    = Vector{Float64}(data[:proxy_xg_shots_away])

    shots_h = Vector{Float64}(data[:bbc_shots_home])
    shots_a = Vector{Float64}(data[:bbc_shots_away])
    shots_mask = Vector{Float64}(data[:bbc_shots_mask])

    sx_h = _pxg_suff(xg_h, mask_h, core.home_goals, core.w)
    sx_a = _pxg_suff(xg_a, mask_a, core.away_goals, core.w)
    sq_h = _funnel_quality_suff(xg_h, mask_h, S_h)
    sq_a = _funnel_quality_suff(xg_a, mask_a, S_a)

    return build_funnel_pxg_goals_apm_negbin_engine(
        core.home_ids, core.away_ids,
        core.season_idx, core.month_idx, core.league_idx,
        rat_h, rat_a,
        sx_h, sx_a,
        sq_h, sq_a,
        shots_h, shots_a, shots_mask,
        core.home_goals, core.away_goals, core.w,
        core.n_teams, core.n_seasons, core.n_months, core.n_leagues,
        _pxg_active(model.league_ha_on), _pxg_active(model.apm_on),
        model
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
    feature_set::Features.FeatureSet,
    chain::MCMCChains.Chains
)
    n_matches = nrow(df)
    n_samples = size(chain, 1) * size(chain, 3)
    data = feature_set.features

    μ_base = PreGame.extract_interception(chain, model.interception_config, 1).μ_base
    n_seasons = size(μ_base, 2)
    inter_ext = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    μ_base_samples = inter_ext.μ_base
    δ_month_samples = inter_ext.δ_month

    ha_samples = PreGame.extract_home_advantage(chain, model.homeadvantage_config)
    dyn_samples = PreGame.extract_dynamics(chain, model.dynamics_config)

    n_leagues = Int(data[:n_leagues])
    δ_league = _pxg_league_offsets(chain, n_leagues, "δ_league_raw")
    γ_league = model.league_ha_on ? _pxg_league_offsets(chain, n_leagues, "γ_league_raw") : zeros(n_samples, n_leagues)

    w_att = model.apm_on ? vec(Array(chain[Symbol("w_att")])) : zeros(n_samples)
    w_def = model.apm_on ? vec(Array(chain[Symbol("w_def")])) : zeros(n_samples)

    disp = PreGame.extract_dispersion(chain, model.dispersion_config)
    r_h_samples = disp.h
    r_a_samples = disp.a

    has_kappa = hasproperty(model, :log_κ_prior) && Symbol("log_κ") in names(chain)
    κ_samples = has_kappa ? exp.(vec(Array(chain[Symbol("log_κ")]))) : ones(Float64, n_samples)

    is_funnel = model isa TeamFunnelPxGGoalsAPMNegBinModel
    q_base = is_funnel ? vec(Array(chain[Symbol("q_base")])) : zeros(n_samples)
    σ_q = is_funnel ? vec(Array(chain[Symbol("σ_q")])) : zeros(n_samples)
    q_att_arr = is_funnel ? Array(chain[Symbol("q_att_raw")]) : zeros(n_samples, 1)
    q_def_arr = is_funnel ? Array(chain[Symbol("q_def_raw")]) : zeros(n_samples, 1)

    home_ids = Vector{Int}(data[:flat_home_ids])
    away_ids = Vector{Int}(data[:flat_away_ids])
    season_idx = Vector{Int}(data[:season_indices])
    month_idx = Vector{Int}(data[:flat_months])
    league_idx = Vector{Int}(data[:flat_league_ids])

    rat_h, rat_a = _pxg_ratings(data, model, n_matches)

    λ_h_mat = zeros(Float64, n_matches, n_samples)
    λ_a_mat = zeros(Float64, n_matches, n_samples)
    r_h_mat = zeros(Float64, n_matches, n_samples)
    r_a_mat = zeros(Float64, n_matches, n_samples)

    for i in 1:n_matches
        h_id = home_ids[i]; a_id = away_ids[i]
        s_id = season_idx[i]; m_id = month_idx[i]; l_id = league_idx[i]

        int_m = μ_base_samples[:, s_id] .+ δ_month_samples[:, m_id]
        lg    = δ_league[:, l_id]
        γ_lg  = γ_league[:, l_id]

        ha_h  = ha_samples[:, h_id]
        att_h = dyn_samples.α[:, h_id]; def_a = dyn_samples.β[:, a_id]
        att_a = dyn_samples.α[:, a_id]; def_h = dyn_samples.β[:, h_id]

        pm_h = w_att .* rat_h[i] .- w_def .* rat_a[i]
        pm_a = w_att .* rat_a[i] .- w_def .* rat_h[i]

        if is_funnel
            log_λ_s_h = model.shot_scale .+ int_m .+ lg .+ ha_h .+ γ_lg .+ att_h .+ def_a .+ pm_h
            log_λ_s_a = model.shot_scale .+ int_m .+ lg .+ att_a .+ def_h .+ pm_a
            λ_s_h = exp.(clamp.(log_λ_s_h, -10.0, 10.0))
            λ_s_a = exp.(clamp.(log_λ_s_a, -10.0, 10.0))

            logit_q_h = q_base .+ q_att_arr[:, h_id] .* σ_q .- q_def_arr[:, a_id] .* σ_q
            logit_q_a = q_base .+ q_att_arr[:, a_id] .* σ_q .- q_def_arr[:, h_id] .* σ_q
            q_h = 1.0 ./ (1.0 .+ exp.(.-clamp.(logit_q_h, -10.0, 10.0)))
            q_a = 1.0 ./ (1.0 .+ exp.(.-clamp.(logit_q_a, -10.0, 10.0)))

            μ_h = λ_s_h .* q_h
            μ_a = λ_s_a .* q_a
        else
            log_μ_h = int_m .+ lg .+ ha_h .+ γ_lg .+ att_h .+ def_a .+ pm_h
            log_μ_a = int_m .+ lg .+ att_a .+ def_h .+ pm_a
            μ_h = exp.(clamp.(log_μ_h, -10.0, 10.0))
            μ_a = exp.(clamp.(log_μ_a, -10.0, 10.0))
        end

        λ_h_mat[i, :] = κ_samples .* μ_h
        λ_a_mat[i, :] = κ_samples .* μ_a
        r_h_mat[i, :] = r_h_samples
        r_a_mat[i, :] = r_a_samples
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
