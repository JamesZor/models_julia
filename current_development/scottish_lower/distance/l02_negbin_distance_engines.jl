# current_development/scottish_lower/distance/l02_negbin_distance_engines.jl
#
# LOADER: Robust Negative Binomial (NB2) + Travel Distance Bayesian Engines
#
# 1. TeamGoalsNegBinDistanceModel            (Goals-Only NegBin Control + Travel Distance)
# 2. TeamPxGGoalsAPMNegBinDistanceModel      (Arm A: Proxy xG Gamma + RAPM + Travel Distance + NegBin Goals)
# 3. TeamFunnelPxGGoalsAPMNegBinDistanceModel(Arm B: 3-Layer Shots Funnel + Proxy xG Quality + RAPM + Travel Distance + NegBin Goals)
#
# Uses HomeAwayDispersion (r_a = exp(log_r), r_h = exp(log_r + δ_r_home))
# with 0-allocation integer recurrence log-PMF evaluation and travel fatigue shift.

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
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l01_negbin_engines.jl"))
include("l01_distance_features.jl")

# ==============================================================================
# 0. CORE DISTANCE HELPERS
# ==============================================================================

function _pxg_core_distance(data, config)
    date_deltas = Vector{Int}(data[:dates])
    dist_vec = haskey(data, :flat_distance) ? Vector{Float64}(data[:flat_distance]) : (haskey(data, :flat_log_distance_z) ? Vector{Float64}(data[:flat_log_distance_z]) : zeros(Float64, length(data[:flat_home_ids])))
    return (;
        home_ids    = Vector{Int}(data[:flat_home_ids]),
        away_ids    = Vector{Int}(data[:flat_away_ids]),
        season_idx  = Vector{Int}(data[:season_indices]),
        month_idx   = Vector{Int}(data[:flat_months]),
        league_idx  = Vector{Int}(data[:flat_league_ids]),
        home_goals  = Vector{Int}(data[:flat_home_goals]),
        away_goals  = Vector{Int}(data[:flat_away_goals]),
        distance_z  = dist_vec,
        w           = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life),
        n_teams     = Int(data[:n_teams]),
        n_seasons   = Int(data[:n_seasons]),
        n_months    = 12,
        n_leagues   = Int(data[:n_leagues]),
    )
end

function _pxg_ratings_distance(data, config, n::Int)
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
# 1. MODEL 1: TeamGoalsNegBinDistanceModel (Goals NegBin + Distance)
# ==============================================================================

Base.@kwdef struct TeamGoalsNegBinDistanceModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature,
    Dist<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    distance_feature::Dist    = ScottishDistanceFeature()
    w_dist_prior::Distribution= truncated(Normal(0.04, 0.03), lower = 0.0)
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_goals_negbin_distance"
end

function Features.required_features(model::TeamGoalsNegBinDistanceModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.distance_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_goals_negbin_distance_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    distance_z::Vector{Float64},
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
    w_dist ~ config.w_dist_prior

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

    dist_shift = w_dist .* distance_z

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ dist_shift, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- dist_shift, -10.0, 10.0)

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

function PreGame.build_turing_model(config::TeamGoalsNegBinDistanceModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_distance(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_distance(data, config, n)
    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)
    return build_goals_negbin_distance_engine(
        d.home_ids, d.away_ids,
        d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.distance_z,
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on),
        _pxg_active(config.apm_on),
        config
    )
end

# ==============================================================================
# 2. MODEL 2: ARM A (PROXY xG + RAPM + DISTANCE + NEGATIVE BINOMIAL GOALS)
# ==============================================================================

Base.@kwdef struct TeamPxGGoalsAPMNegBinDistanceModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    D<:PreGame.AbstractDispersionConfig,
    P<:Features.AbstractPlusMinusFeature,
    Dist<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
    dynamics_config::T        = PreGame.TimeDecayDynamics()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    dispersion_config::D      = SCOTTISH_HOMEAWAY_DISPERSION
    player_ratings_feature::P = Features.XGPlusMinusFeature()
    distance_feature::Dist    = ScottishDistanceFeature()
    proxy_feature::ProxyXGFeature = ProxyXGFeature()
    w_dist_prior::Distribution= truncated(Normal(0.04, 0.03), lower = 0.0)
    w_att_prior::Distribution = Normal(0.0, 0.3)
    w_def_prior::Distribution = Normal(0.0, 0.3)
    apm_on::Bool              = true
    kappa_prior::Distribution = Normal(0.0, 0.2)
    nu_prior::Distribution    = Exponential(1.0)
    league_offset_sd::Float64 = 0.1
    league_ha_sd::Float64     = 0.1
    league_ha_on::Bool        = false
    name::String              = "team_pxg_goals_apm_negbin_distance"
end

function Features.required_features(model::TeamPxGGoalsAPMNegBinDistanceModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        model.proxy_feature, model.distance_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

@model function build_pxg_goals_apm_negbin_distance_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    distance_z::Vector{Float64},
    pxg_h::NamedTuple, pxg_a::NamedTuple,
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

    log_κ ~ config.kappa_prior
    ν_raw ~ config.nu_prior
    ν     = ν_raw + 1.0

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior
    w_dist ~ config.w_dist_prior

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

    dist_shift = w_dist .* distance_z

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h .+ dist_shift, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a .- dist_shift, -10.0, 10.0)

    # 3. Proxy xG Gamma Likelihood
    lν = log(ν)
    ll_gamma = pxg_h.S_w_pxg * loggamma(ν)

    ll_pxg_h = ν * pxg_h.S_w_log_pxg - pxg_h.S_log_pxg -
               ν * sum(pxg_h.w_pxg_val .* exp.(-log_λ_h)) -
               ν * sum(pxg_h.w_pxg .* log_λ_h) +
               ν * lν * pxg_h.S_w_pxg - ll_gamma

    ll_pxg_a = ν * pxg_a.S_w_log_pxg - pxg_a.S_log_pxg -
               ν * sum(pxg_a.w_pxg_val .* exp.(-log_λ_a)) -
               ν * sum(pxg_a.w_pxg .* log_λ_a) +
               ν * lν * pxg_a.S_w_pxg - ll_gamma

    # 4. Goals Robust Negative Binomial Likelihood
    log_λ_gh = log_κ .+ log_λ_h
    log_λ_ga = log_κ .+ log_λ_a
    r_h = disp.h
    r_a = disp.a

    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_gh, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_ga, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(ll_pxg_h) || isnan(ll_pxg_a) || isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_pxg_h + ll_pxg_a + ll_g_h + ll_g_a
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinDistanceModel, feature_set)
    data = _pxg_get_data(feature_set)
    d    = _pxg_core_distance(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings_distance(data, config, n)

    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_pxg_goals_apm_negbin_distance_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        d.distance_z,
        _pxg_suff_opt(Vector{Float64}(data[:flat_home_xg_proxy]), Vector{Float64}(data[:flat_pxg_mask_h]), d.w),
        _pxg_suff_opt(Vector{Float64}(data[:flat_away_xg_proxy]), Vector{Float64}(data[:flat_pxg_mask_a]), d.w),
        d.home_goals, d.away_goals,
        nb_h, nb_a,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config
    )
end

# FeatureSet overrides
PreGame.build_turing_model(config::TeamGoalsNegBinDistanceModel, fs::Features.FeatureSet) = PreGame.build_turing_model(config, fs.data)
PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinDistanceModel, fs::Features.FeatureSet) = PreGame.build_turing_model(config, fs.data)

# ==============================================================================
# 3. PARAMETER EXTRACTOR FOR DISTANCE MODELS
# ==============================================================================

const ScottishNegBinDistanceUnion = Union{
    TeamGoalsNegBinDistanceModel,
    TeamPxGGoalsAPMNegBinDistanceModel
}

function PreGame.extract_parameters(
    model::ScottishNegBinDistanceUnion,
    df::DataFrame,
    feature_set::Features.FeatureSet,
    chain::MCMCChains.Chains
)
    core, n_samples, n_teams = _pxg_extract_core(model, df, feature_set, chain)

    disp = PreGame.extract_dispersion(chain, model.dispersion_config)
    r_h_samples = disp.h
    r_a_samples = disp.a

    has_kappa = Symbol("log_κ") in keys(chain)
    κ = has_kappa ? exp.(vec(Array(chain[Symbol("log_κ")]))) : ones(Float64, n_samples)

    w_dist_samples = Symbol("w_dist") in keys(chain) ? vec(Array(chain[Symbol("w_dist")])) : zeros(Float64, n_samples)

    results = Dict{Int, NamedTuple}()
    for (i, row) in enumerate(eachrow(df))
        mid = row.match_id
        h_idx, a_idx, s_idx, m_idx, lg_idx, w_att, w_def = _pxg_row_covars(core, row, i)
        
        # Pull match distance
        dist_z_val = 0.0
        if haskey(feature_set.data, :flat_distance)
            dist_z_val = Float64(feature_set.data[:flat_distance][i])
        elseif haskey(feature_set.data, :distance_df)
            dist_match_row = filter(r -> r.match_id == mid, feature_set.data[:distance_df])
            if nrow(dist_match_row) > 0
                dist_z_val = Float64(dist_match_row[1, :log_dist_z])
            end
        end

        mu_base = core.μ_base[:, s_idx]
        delta_m = core.δ_month[:, m_idx]
        ha_val  = core.ha[:, h_idx]

        h_rat = (core.rat_h[i] .- core.base)
        a_rat = (core.rat_a[i] .- core.base)
        pillar_h = core.apm_on ? (w_att .* h_rat .- w_def .* a_rat) : zeros(Float64, n_samples)
        pillar_a = core.apm_on ? (w_att .* a_rat .- w_def .* h_rat) : zeros(Float64, n_samples)

        lg_h = core.δ_league[:, lg_idx]
        lg_ha = core.league_ha_on ? core.γ_league[:, lg_idx] : zeros(Float64, n_samples)

        dist_shift = w_dist_samples .* dist_z_val

        log_λ_h = mu_base .+ delta_m .+ ha_val .+ lg_h .+ lg_ha .+
                  core.α[:, h_idx] .+ core.β[:, a_idx] .+ pillar_h .+ dist_shift
        log_λ_a = mu_base .+ delta_m .+ lg_h .+
                  core.α[:, a_idx] .+ core.β[:, h_idx] .+ pillar_a .- dist_shift

        λ_h = exp.(log_λ_h) .* κ
        λ_a = exp.(log_λ_a) .* κ

        results[mid] = (
            λ_home = λ_h,
            λ_away = λ_a,
            r_home = r_h_samples,
            r_away = r_a_samples
        )
    end
    return results
end

# 4. PREDICTION MATRICES
function Pred.predict_match_rates(model::ScottishNegBinDistanceUnion, params::NamedTuple)
    return (params.λ_home, params.λ_away)
end

function Pred.predict_match_score_grid(model::ScottishNegBinDistanceUnion, params::NamedTuple, max_goals::Int = 10)
    λ_h, λ_a = params.λ_home, params.λ_away
    r_h, r_a = params.r_home, params.r_away
    n_samples = length(λ_h)

    P = zeros(Float64, max_goals + 1, max_goals + 1)
    for s in 1:n_samples
        dist_h = RobustNegativeBinomial(r_h[s], λ_h[s])
        dist_a = RobustNegativeBinomial(r_a[s], λ_a[s])

        p_h = [pdf(dist_h, g) for g in 0:max_goals]
        p_a = [pdf(dist_a, g) for g in 0:max_goals]
        p_h ./= sum(p_h)
        p_a ./= sum(p_a)

        P .+= p_h * p_a'
    end
    P ./= n_samples
    return P
end
