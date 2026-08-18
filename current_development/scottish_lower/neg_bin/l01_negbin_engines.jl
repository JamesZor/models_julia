# current_development/scottish_lower/neg_bin/l01_negbin_engines.jl
#
# LOADER: Robust Negative Binomial (NB2) Goals Likelihood Engines for Scottish Lower (56/57)
#
# Models implemented:
# 1. TeamGoalsNegBinModel               (Baseline Goals-Only NegBin Control)
# 2. TeamPxGGoalsAPMNegBinModel         (Arm A: Proxy xG Gamma + RAPM + NegBin Goals)
# 3. TeamFunnelPxGGoalsAPMNegBinModel   (Arm B: Shots Volume Poisson + Proxy xG Quality Gamma + RAPM + NegBin Goals)
#
# Mathematical Foundation:
# - Decouples goal outcome variance from mean intensity: Var(G) = μ + μ²/r.
# - Uses HomeAwayDispersion (r_a = exp(log_r), r_h = exp(log_r + δ_r_home)) to capture
#   the empirical asymmetry discovered in Stage-A EDA (r_away ≈ 9.25 vs r_home ≈ 23.66).
# - Retains exact ReverseDiff SIMD broadcasting (logpdf.(RobustNegativeBinomial.(r, λ), g))
#   with 1-node GradientTape execution adhering to docs/turing_ad_performance_guide.md.

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

# Scottish Lower empirical EDA: r_away ≈ 9.25 (log r ≈ 2.22), r_home ≈ 23.66 (log r ≈ 3.16)
# Default prior centred on log_r ≈ 2.5 (r ≈ 12.2) and δ_r_home ≈ 0.6 (r_h/r_a ≈ 1.82)
const SCOTTISH_HOMEAWAY_DISPERSION = PreGame.HomeAwayDispersion(
    log_r     = Normal(2.6, 0.5),
    δ_r_home  = Normal(0.6, 0.5)
)

# ==============================================================================
# 1. MODEL 1: BASELINE GOALS-ONLY NEGATIVE BINOMIAL MODEL
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinModel{
    I<:PreGame.AbstractInterceptionConfig,
    D<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    DISP<:PreGame.AbstractDispersionConfig,
    R<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    dynamics_config::D
    homeadvantage_config::H
    dispersion_config::DISP              = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::R            = Features.XGPlusMinusFeature()
    w_att_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    w_def_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    league_split::Bool                   = true
    time_decay::Bool                     = true
    name::String                         = "team_goals_negbin"
end

@model function _team_goals_negbin_turing(
    core,
    ratings_h,
    ratings_a,
    config::TeamGoalsNegBinModel
)
    # 1. Global Interception & Season/Month/League Dynamics
    μ_base ~ config.interception_config.μ_base
    σ_month ~ config.interception_config.σ_month
    d_month ~ filldist(Normal(0, σ_month), core.n_seasons * 12)

    d_league = zeros(typeof(μ_base), core.n_teams)
    if config.league_split
        σ_league ~ Normal(0.0, 0.3)
        league_raw ~ filldist(Normal(0, 1), 2)
        league_offset = (league_raw .- mean(league_raw)) .* σ_league
        d_league = league_offset[core.league_idx]
    end

    # 2. Team Home Advantage & Attack/Defense Strengths
    ha_global ~ config.homeadvantage_config.ha_global
    σ_ha ~ config.homeadvantage_config.σ_ha
    ha_raw ~ filldist(Normal(0, 1), core.n_teams)
    ha_team = ha_global .+ ha_raw .* σ_ha

    σ_att ~ Normal(0.0, 0.4)
    σ_def ~ Normal(0.0, 0.4)
    att_raw ~ filldist(Normal(0, 1), core.n_teams)
    def_raw ~ filldist(Normal(0, 1), core.n_teams)
    att_team = att_raw .* σ_att
    def_team = def_raw .* σ_def

    # 3. Player RAPM Pillar
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    pm_h = w_att .* ratings_h.att .- w_def .* ratings_a.def
    pm_a = w_att .* ratings_a.att .- w_def .* ratings_h.def

    # 4. Expected Goal Intensities
    log_λ_h = μ_base .+ d_month[core.month_idx] .+ d_league .+ ha_team[core.home_ids] .+
              att_team[core.home_ids] .- def_team[core.away_ids] .+ pm_h
    log_λ_a = μ_base .+ d_month[core.month_idx] .+ d_league .+
              att_team[core.away_ids] .- def_team[core.home_ids] .+ pm_a

    λ_h = exp.(clamp.(log_λ_h, -5.0, 4.0))
    λ_a = exp.(clamp.(log_λ_a, -5.0, 4.0))

    # 5. Robust Negative Binomial Dispersion
    disp = PreGame.build_dispersion(config.dispersion_config)
    r_h = disp.h
    r_a = disp.a

    # 6. Goals Likelihood
    log_lik_g_h = logpdf.(RobustNegativeBinomial.(r_h, λ_h), core.home_goals)
    log_lik_g_a = logpdf.(RobustNegativeBinomial.(r_a, λ_a), core.away_goals)

    if config.time_decay
        Turing.@addlogprob! sum(core.w .* log_lik_g_h)
        Turing.@addlogprob! sum(core.w .* log_lik_g_a)
    else
        Turing.@addlogprob! sum(log_lik_g_h)
        Turing.@addlogprob! sum(log_lik_g_a)
    end
end

# ==============================================================================
# 2. MODEL 2: ARM A (PROXY xG + RAPM + ROBUST NEGATIVE BINOMIAL GOALS)
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMNegBinModel{
    I<:PreGame.AbstractInterceptionConfig,
    D<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    DISP<:PreGame.AbstractDispersionConfig,
    R<:Features.AbstractFeatureConfig,
    PXG<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    dynamics_config::D
    homeadvantage_config::H
    kappa_config::K                      = PreGame.GlobalKappa(log_κ = PXG_LOGK_PRIOR)
    dispersion_config::DISP              = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::R            = Features.XGPlusMinusFeature()
    pxg_feature::PXG                     = Features.ScottishProxyXGFeature()
    w_att_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    w_def_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    ν_xg_prior::ContinuousUnivariateDistribution  = PXG_NU_PRIOR
    league_split::Bool                   = true
    time_decay::Bool                     = true
    name::String                         = "team_pxg_goals_apm_negbin"
end

@model function _team_pxg_goals_apm_negbin_turing(
    core,
    ratings_h,
    ratings_a,
    pxg,
    config::TeamPxGGoalsAPMNegBinModel
)
    # 1. Global Interception & Season/Month/League Dynamics
    μ_base ~ config.interception_config.μ_base
    σ_month ~ config.interception_config.σ_month
    d_month ~ filldist(Normal(0, σ_month), core.n_seasons * 12)

    d_league = zeros(typeof(μ_base), core.n_teams)
    if config.league_split
        σ_league ~ Normal(0.0, 0.3)
        league_raw ~ filldist(Normal(0, 1), 2)
        league_offset = (league_raw .- mean(league_raw)) .* σ_league
        d_league = league_offset[core.league_idx]
    end

    # 2. Team Home Advantage & Strengths
    ha_global ~ config.homeadvantage_config.ha_global
    σ_ha ~ config.homeadvantage_config.σ_ha
    ha_raw ~ filldist(Normal(0, 1), core.n_teams)
    ha_team = ha_global .+ ha_raw .* σ_ha

    σ_att ~ Normal(0.0, 0.4)
    σ_def ~ Normal(0.0, 0.4)
    att_raw ~ filldist(Normal(0, 1), core.n_teams)
    def_raw ~ filldist(Normal(0, 1), core.n_teams)
    att_team = att_raw .* σ_att
    def_team = def_raw .* σ_def

    # 3. Player RAPM Pillar
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    pm_h = w_att .* ratings_h.att .- w_def .* ratings_a.def
    pm_a = w_att .* ratings_a.att .- w_def .* ratings_h.def

    # 4. Underlying True Scoring Intensity (μ)
    log_μ_h = μ_base .+ d_month[core.month_idx] .+ d_league .+ ha_team[core.home_ids] .+
              att_team[core.home_ids] .- def_team[core.away_ids] .+ pm_h
    log_μ_a = μ_base .+ d_month[core.month_idx] .+ d_league .+
              att_team[core.away_ids] .- def_team[core.home_ids] .+ pm_a

    μ_h = exp.(clamp.(log_μ_h, -5.0, 4.0))
    μ_a = exp.(clamp.(log_μ_a, -5.0, 4.0))

    # 5. Finishing Conversion (κ) & Dispersion (r_h, r_a)
    log_κ ~ config.kappa_config.log_κ
    κ = exp(clamp(log_κ, -1.0, 1.0))
    λ_goals_h = κ .* μ_h
    λ_goals_a = κ .* μ_a

    disp = PreGame.build_dispersion(config.dispersion_config)
    r_h = disp.h
    r_a = disp.a

    # 6. Proxy xG Gamma Pillar (Co-training)
    ν ~ config.ν_xg_prior
    shape_h = ν
    scale_h = μ_h ./ ν
    shape_a = ν
    scale_a = μ_a ./ ν

    log_lik_xg_h = pxg.mask_h .* (
        (shape_h - 1.0) .* pxg.log_xg_h .- pxg.xg_h ./ scale_h .-
        shape_h .* log.(scale_h) .- loggamma(shape_h)
    )
    log_lik_xg_a = pxg.mask_a .* (
        (shape_a - 1.0) .* pxg.log_xg_a .- pxg.xg_a ./ scale_a .-
        shape_a .* log.(scale_a) .- loggamma(shape_a)
    )

    # 7. Goals Robust Negative Binomial Pillar
    log_lik_g_h = logpdf.(RobustNegativeBinomial.(r_h, λ_goals_h), core.home_goals)
    log_lik_g_a = logpdf.(RobustNegativeBinomial.(r_a, λ_goals_a), core.away_goals)

    if config.time_decay
        Turing.@addlogprob! sum(core.w .* log_lik_xg_h)
        Turing.@addlogprob! sum(core.w .* log_lik_xg_a)
        Turing.@addlogprob! sum(core.w .* log_lik_g_h)
        Turing.@addlogprob! sum(core.w .* log_lik_g_a)
    else
        Turing.@addlogprob! sum(log_lik_xg_h)
        Turing.@addlogprob! sum(log_lik_xg_a)
        Turing.@addlogprob! sum(log_lik_g_h)
        Turing.@addlogprob! sum(log_lik_g_a)
    end
end

# ==============================================================================
# 3. MODEL 3: ARM B (3-LAYER FUNNEL: SHOTS POISSON -> QUALITY GAMMA -> GOALS NEGBIN)
# ==============================================================================

Base.@kwdef struct TeamFunnelPxGGoalsAPMNegBinModel{
    I<:PreGame.AbstractInterceptionConfig,
    D<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    DISP<:PreGame.AbstractDispersionConfig,
    R<:Features.AbstractFeatureConfig,
    PXG<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    dynamics_config::D
    homeadvantage_config::H
    kappa_config::K                      = PreGame.GlobalKappa(log_κ = PXG_LOGK_PRIOR)
    dispersion_config::DISP              = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::R            = Features.XGPlusMinusFeature()
    pxg_feature::PXG                     = Features.ScottishProxyXGFeature()
    shot_scale::Float64                  = 2.2
    q_prior::ContinuousUnivariateDistribution     = PXG_Q_PRIOR
    σ_q_prior::ContinuousUnivariateDistribution   = PXG_SIGQ_PRIOR
    ν_q_prior::ContinuousUnivariateDistribution   = PXG_NU_PRIOR
    w_att_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    w_def_prior::ContinuousUnivariateDistribution = truncated(Normal(0.05, 0.05), lower = 0.0)
    league_split::Bool                   = true
    time_decay::Bool                     = true
    name::String                         = "team_funnel_pxg_goals_apm_negbin"
end

@model function _team_funnel_pxg_goals_apm_negbin_turing(
    core,
    ratings_h,
    ratings_a,
    pxg,
    shots_h,
    shots_a,
    shots_mask,
    config::TeamFunnelPxGGoalsAPMNegBinModel
)
    # 1. Global Interception & Season/Month/League Dynamics
    μ_base ~ config.interception_config.μ_base
    σ_month ~ config.interception_config.σ_month
    d_month ~ filldist(Normal(0, σ_month), core.n_seasons * 12)

    d_league = zeros(typeof(μ_base), core.n_teams)
    if config.league_split
        σ_league ~ Normal(0.0, 0.3)
        league_raw ~ filldist(Normal(0, 1), 2)
        league_offset = (league_raw .- mean(league_raw)) .* σ_league
        d_league = league_offset[core.league_idx]
    end

    # 2. Team Home Advantage & Volume Ratings
    ha_global ~ config.homeadvantage_config.ha_global
    σ_ha ~ config.homeadvantage_config.σ_ha
    ha_raw ~ filldist(Normal(0, 1), core.n_teams)
    ha_team = ha_global .+ ha_raw .* σ_ha

    σ_att ~ Normal(0.0, 0.4)
    σ_def ~ Normal(0.0, 0.4)
    att_raw ~ filldist(Normal(0, 1), core.n_teams)
    def_raw ~ filldist(Normal(0, 1), core.n_teams)
    att_team = att_raw .* σ_att
    def_team = def_raw .* σ_def

    # 3. Player RAPM Pillar
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    pm_h = w_att .* ratings_h.att .- w_def .* ratings_a.def
    pm_a = w_att .* ratings_a.att .- w_def .* ratings_h.def

    # 4. Volume Layer (Expected Shots λ_s)
    log_λ_s_h = config.shot_scale .+ μ_base .+ d_month[core.month_idx] .+ d_league .+
                ha_team[core.home_ids] .+ att_team[core.home_ids] .- def_team[core.away_ids] .+ pm_h
    log_λ_s_a = config.shot_scale .+ μ_base .+ d_month[core.month_idx] .+ d_league .+
                att_team[core.away_ids] .- def_team[core.home_ids] .+ pm_a

    λ_s_h = exp.(clamp.(log_λ_s_h, -2.0, 4.5))
    λ_s_a = exp.(clamp.(log_λ_s_a, -2.0, 4.5))

    # 5. Quality Layer (Shot Quality q = xG / Shot)
    q_base ~ config.q_prior
    σ_q ~ config.σ_q_prior
    q_att_raw ~ filldist(Normal(0, 1), core.n_teams)
    q_def_raw ~ filldist(Normal(0, 1), core.n_teams)
    q_att = q_att_raw .* σ_q
    q_def = q_def_raw .* σ_q

    logit_q_h = q_base .+ q_att[core.home_ids] .- q_def[core.away_ids]
    logit_q_a = q_base .+ q_att[core.away_ids] .- q_def[core.home_ids]
    q_h = 1.0 ./ (1.0 .+ exp.(-clamp.(logit_q_h, -4.0, 1.0)))
    q_a = 1.0 ./ (1.0 .+ exp.(-clamp.(logit_q_a, -4.0, 1.0)))

    # 6. Compound Expected Goals (μ = λ_s · q)
    μ_h = λ_s_h .* q_h
    μ_a = λ_s_a .* q_a

    # 7. Finishing Conversion (κ) & Dispersion (r_h, r_a)
    log_κ ~ config.kappa_config.log_κ
    κ = exp(clamp(log_κ, -1.0, 1.0))
    λ_goals_h = κ .* μ_h
    λ_goals_a = κ .* μ_a

    disp = PreGame.build_dispersion(config.dispersion_config)
    r_h = disp.h
    r_a = disp.a

    # 8. Likelihood Layer 1: Shots Volume (Poisson)
    suff_s_h = loggamma.(shots_h .+ 1.0)
    suff_s_a = loggamma.(shots_a .+ 1.0)
    log_lik_s_h = shots_mask .* ((shots_h .* log.(λ_s_h)) .- λ_s_h .- suff_s_h)
    log_lik_s_a = shots_mask .* ((shots_a .* log.(λ_s_a)) .- λ_s_a .- suff_s_a)

    # 9. Likelihood Layer 2: Quality Gamma (Conditional on Commentary Shots)
    ν_q ~ config.ν_q_prior
    S_cond_h = pxg.shots_comm_h
    S_cond_a = pxg.shots_comm_a
    shape_q_h = ν_q .* S_cond_h
    scale_q_h = q_h ./ ν_q
    shape_q_a = ν_q .* S_cond_a
    scale_q_a = q_a ./ ν_q

    log_lik_q_h = pxg.mask_h .* (
        (shape_q_h .- 1.0) .* pxg.log_xg_h .- pxg.xg_h ./ scale_q_h .-
        shape_q_h .* log.(scale_q_h) .- loggamma.(shape_q_h)
    )
    log_lik_q_a = pxg.mask_a .* (
        (shape_q_a .- 1.0) .* pxg.log_xg_a .- pxg.xg_a ./ scale_q_a .-
        shape_q_a .* log.(scale_q_a) .- loggamma.(shape_q_a)
    )

    # 10. Likelihood Layer 3: Goals Robust Negative Binomial
    log_lik_g_h = logpdf.(RobustNegativeBinomial.(r_h, λ_goals_h), core.home_goals)
    log_lik_g_a = logpdf.(RobustNegativeBinomial.(r_a, λ_goals_a), core.away_goals)

    if config.time_decay
        Turing.@addlogprob! sum(core.w .* log_lik_s_h)
        Turing.@addlogprob! sum(core.w .* log_lik_s_a)
        Turing.@addlogprob! sum(core.w .* log_lik_q_h)
        Turing.@addlogprob! sum(core.w .* log_lik_q_a)
        Turing.@addlogprob! sum(core.w .* log_lik_g_h)
        Turing.@addlogprob! sum(core.w .* log_lik_g_a)
    else
        Turing.@addlogprob! sum(log_lik_s_h)
        Turing.@addlogprob! sum(log_lik_s_a)
        Turing.@addlogprob! sum(log_lik_q_h)
        Turing.@addlogprob! sum(log_lik_q_a)
        Turing.@addlogprob! sum(log_lik_g_h)
        Turing.@addlogprob! sum(log_lik_g_a)
    end
end

# ==============================================================================
# 4. BUILDERS (Data Packaging)
# ==============================================================================

function PreGame.build_turing_model(model::TeamGoalsNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    ratings_h, ratings_a = _pxg_ratings(data)
    return _team_goals_negbin_turing(core, ratings_h, ratings_a, model)
end

function PreGame.build_turing_model(model::TeamPxGGoalsAPMNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    ratings_h, ratings_a = _pxg_ratings(data)
    pxg = _pxg_tensors(data)
    return _team_pxg_goals_apm_negbin_turing(core, ratings_h, ratings_a, pxg, model)
end

function PreGame.build_turing_model(model::TeamFunnelPxGGoalsAPMNegBinModel, feature_set::Features.FeatureSet)
    data = feature_set.features
    core = _pxg_core(data, model)
    ratings_h, ratings_a = _pxg_ratings(data)
    pxg = _pxg_tensors(data)
    shots_h = Vector{Float64}(data[:bbc_shots_home])
    shots_a = Vector{Float64}(data[:bbc_shots_away])
    shots_mask = Vector{Float64}(data[:bbc_shots_mask])
    return _team_funnel_pxg_goals_apm_negbin_turing(core, ratings_h, ratings_a, pxg, shots_h, shots_a, shots_mask, model)
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
    n_samples = length(chain)
    data = feature_set.features

    μ_base = vec(Array(chain[Symbol("μ_base")]))
    d_month_arr = Array(chain[Symbol("d_month")])
    σ_month = vec(Array(chain[Symbol("σ_month")]))

    ha_global = vec(Array(chain[Symbol("ha_global")]))
    ha_raw_arr = Array(chain[Symbol("ha_raw")])
    σ_ha = vec(Array(chain[Symbol("σ_ha")]))

    att_raw_arr = Array(chain[Symbol("att_raw")])
    def_raw_arr = Array(chain[Symbol("def_raw")])
    σ_att = vec(Array(chain[Symbol("σ_att")]))
    σ_def = vec(Array(chain[Symbol("σ_def")]))

    w_att = vec(Array(chain[Symbol("w_att")]))
    w_def = vec(Array(chain[Symbol("w_def")]))

    # Extract Dispersion (r_h, r_a)
    disp = PreGame.extract_dispersion(chain, model.dispersion_config)
    r_h_samples = disp.h
    r_a_samples = disp.a

    has_kappa = hasproperty(model, :kappa_config) && Symbol("log_κ") in names(chain)
    κ_samples = has_kappa ? exp.(vec(Array(chain[Symbol("log_κ")]))) : ones(Float64, n_samples)

    is_funnel = model isa TeamFunnelPxGGoalsAPMNegBinModel
    q_base = is_funnel ? vec(Array(chain[Symbol("q_base")])) : zeros(Float64, n_samples)
    σ_q = is_funnel ? vec(Array(chain[Symbol("σ_q")])) : zeros(Float64, n_samples)
    q_att_arr = is_funnel ? Array(chain[Symbol("q_att_raw")]) : zeros(Float64, n_samples, 1)
    q_def_arr = is_funnel ? Array(chain[Symbol("q_def_raw")]) : zeros(Float64, n_samples, 1)

    ratings_h, ratings_a = _pxg_ratings(data)
    flat_home = Vector{Int}(data[:flat_home_ids])
    flat_away = Vector{Int}(data[:flat_away_ids])
    flat_months = Vector{Int}(data[:flat_months])
    flat_leagues = Vector{Int}(data[:flat_league_ids])

    has_league = model.league_split && Symbol("σ_league") in names(chain)
    σ_league = has_league ? vec(Array(chain[Symbol("σ_league")])) : zeros(Float64, n_samples)
    league_raw_arr = has_league ? Array(chain[Symbol("league_raw")]) : zeros(Float64, n_samples, 2)

    λ_h_mat = zeros(Float64, n_matches, n_samples)
    λ_a_mat = zeros(Float64, n_matches, n_samples)
    r_h_mat = zeros(Float64, n_matches, n_samples)
    r_a_mat = zeros(Float64, n_matches, n_samples)

    for i in 1:n_matches
        h_id = flat_home[i]
        a_id = flat_away[i]
        m_idx = flat_months[i]
        l_idx = flat_leagues[i]

        rh_att = ratings_h.att[i]; rh_def = ratings_h.def[i]
        ra_att = ratings_a.att[i]; ra_def = ratings_a.def[i]

        d_m = m_idx <= size(d_month_arr, 2) ? d_month_arr[:, m_idx] : randn(n_samples) .* σ_month
        ha_h = ha_global .+ ha_raw_arr[:, h_id] .* σ_ha
        att_h = att_raw_arr[:, h_id] .* σ_att
        def_a = def_raw_arr[:, a_id] .* σ_def
        att_a = att_raw_arr[:, a_id] .* σ_att
        def_h = def_raw_arr[:, h_id] .* σ_def

        d_l = zeros(Float64, n_samples)
        if has_league
            l_mean = (league_raw_arr[:, 1] .+ league_raw_arr[:, 2]) ./ 2.0
            d_l = (league_raw_arr[:, l_idx] .- l_mean) .* σ_league
        end

        pm_h = w_att .* rh_att .- w_def .* ra_def
        pm_a = w_att .* ra_att .- w_def .* rh_def

        if is_funnel
            log_λ_s_h = model.shot_scale .+ μ_base .+ d_m .+ d_l .+ ha_h .+ att_h .- def_a .+ pm_h
            log_λ_s_a = model.shot_scale .+ μ_base .+ d_m .+ d_l .+ att_a .- def_h .+ pm_a
            λ_s_h = exp.(clamp.(log_λ_s_h, -2.0, 4.5))
            λ_s_a = exp.(clamp.(log_λ_s_a, -2.0, 4.5))

            logit_q_h = q_base .+ q_att_arr[:, h_id] .* σ_q .- q_def_arr[:, a_id] .* σ_q
            logit_q_a = q_base .+ q_att_arr[:, a_id] .* σ_q .- q_def_arr[:, h_id] .* σ_q
            q_h = 1.0 ./ (1.0 .+ exp.(-clamp.(logit_q_h, -4.0, 1.0)))
            q_a = 1.0 ./ (1.0 .+ exp.(-clamp.(logit_q_a, -4.0, 1.0)))

            μ_h = λ_s_h .* q_h
            μ_a = λ_s_a .* q_a
        else
            log_μ_h = μ_base .+ d_m .+ d_l .+ ha_h .+ att_h .- def_a .+ pm_h
            log_μ_a = μ_base .+ d_m .+ d_l .+ att_a .- def_h .+ pm_a
            μ_h = exp.(clamp.(log_μ_h, -5.0, 4.0))
            μ_a = exp.(clamp.(log_μ_a, -5.0, 4.0))
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

# Dispatch overrides for Prediction Layer
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
