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

"""
Precomputes sufficient statistics and integer goal count weights for 0-allocation Negative Binomial AD.
Uses the Gamma function recurrence identity:
logΓ(k + r) - logΓ(r) = sum_{j=0}^{k-1} log(r + j)
to eliminate all loggamma calls and Array{TrackedReal} allocations on the tape.
"""
function _negbin_precompute(goals::Vector{Int}, w::Vector{Float64})
    max_k = isempty(goals) ? 0 : maximum(goals)
    if max_k > 0
        N_j = Float64[sum(w[goals .> j]) for j in 0:(max_k - 1)]
        j_offsets = Float64.(0:(max_k - 1))
    else
        N_j = Float64[]
        j_offsets = Float64[]
    end
    return (
        w = w,
        c_g_lin = w .* goals,
        S_w = sum(w),
        N_j = N_j,
        j_offsets = j_offsets
    )
end

@inline function _negbin_vector_loglik(
    goals::Vector{Int},
    log_λ::AbstractVector,
    r::Real,
    nb::NamedTuple
)
    lr = log(max(r, 1e-6))
    λ  = exp.(log_λ)

    # 1. Scalar r*log(r) block
    ll_r = nb.S_w * r * lr

    # 2. Integer recurrence log(r + j) terms (O(max_goals), zero loggamma, zero allocations)
    ll_gamma = isempty(nb.N_j) ? 0.0 : sum(nb.N_j .* log.(r .+ nb.j_offsets))

    # 3. Vector (k + r) * log(r + λ)
    ll_denom = sum(nb.w .* (goals .+ r) .* log.(r .+ λ))

    # 4. Vector k * log(λ)
    ll_numer = sum(nb.c_g_lin .* log_λ)

    return ll_r + ll_gamma + ll_numer - ll_denom
end

@model function build_goals_negbin_engine(
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

    # 3. Vectorized SIMD Robust Negative Binomial Likelihood
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_h, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_a, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_g_h + ll_g_a
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

    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_goals_negbin_engine(
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

    r_h = disp.h
    r_a = disp.a

    # 3. Pillar A: Proxy xG (Gamma)
    cν = ν_xg * log(ν_xg) - loggamma(ν_xg)
    ll_xg_h = (ν_xg - 1.0) * sx_h.S_logx - ν_xg * sum(sx_h.c_x .* inv_μ_h) -
              ν_xg * sum(sx_h.c_m .* log_μ_h) + cν * sx_h.S_m
    ll_xg_a = (ν_xg - 1.0) * sx_a.S_logx - ν_xg * sum(sx_a.c_x .* inv_μ_a) -
              ν_xg * sum(sx_a.c_m .* log_μ_a) + cν * sx_a.S_m

    # 4. Pillar B: Goals (Robust Negative Binomial, SIMD vectorized)
    log_λ_gh = log_κ .+ log_μ_h
    log_λ_ga = log_κ .+ log_μ_a
    ll_g_h = _negbin_vector_loglik(home_goals, log_λ_gh, r_h, nb_h)
    ll_g_a = _negbin_vector_loglik(away_goals, log_λ_ga, r_a, nb_a)

    Turing.@addlogprob! ifelse(isnan(ll_xg_h) || isnan(ll_xg_a) || isnan(r_h) || isnan(r_a) || isnan(ll_g_h) || isnan(ll_g_a), -Inf, 0.0)
    Turing.@addlogprob! ll_xg_h + ll_xg_a + ll_g_h + ll_g_a
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

    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_pxg_goals_apm_negbin_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        _pxg_suff(xg_h, mask_h, d.home_goals, d.w),
        _pxg_suff(xg_a, mask_a, d.away_goals, d.w),
        d.home_goals, d.away_goals,
        nb_h, nb_a,
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

function _funnel_suff_opt(shots_bbc::Vector{Int}, mask_s::Vector{Float64},
                          xg::Vector{Float64}, n_ev::Vector{Int}, mask_x::Vector{Float64},
                          goals::Vector{Int}, w::Vector{Float64})
    ws      = w .* mask_s
    wx      = w .* mask_x
    logx    = log.(xg)
    n_safe  = [mask_x[i] > 0 && n_ev[i] > 0 ? Float64(n_ev[i]) : 1.0 for i in eachindex(n_ev)]
    cq_S    = wx .* n_safe
    c_g_lin = w .* goals

    # Collapse distinct non-zero shot counts for loggamma evaluation
    # sum_i wx_i * logΓ(ν_q * S_i) == sum_k W_k * logΓ(ν_q * s_k)
    active_idx = findall(i -> mask_x[i] > 0 && n_ev[i] > 0, eachindex(n_ev))
    if !isempty(active_idx)
        unique_shots = sort(unique(n_ev[active_idx]))
        shot_weights = [sum(wx[i] for i in active_idx if n_ev[i] == s) for s in unique_shots]
        u_shots_f64  = Float64.(unique_shots)
    else
        unique_shots = Int[]
        shot_weights = Float64[]
        u_shots_f64  = Float64[]
    end

    return (
        # volume
        c_s_lin  = ws .* shots_bbc,
        c_s_rate = ws,
        # quality (conditional Gamma)
        cq_S     = cq_S,                     # multiplies log q
        cq_x     = wx .* xg,                 # multiplies 1/q
        S_Slogx  = sum(cq_S .* logx),        # multiplies nu_q
        S_logx   = sum(wx .* logx),          # multiplies -1
        S_cq_S   = sum(cq_S),                # multiplies nu_q*log nu_q
        # collapsed loggamma terms:
        shot_weights = shot_weights,
        u_shots_f64  = u_shots_f64,
        # goals
        c_g_lin  = c_g_lin,
        c_g_rate = w,
        S_g      = sum(c_g_lin),
    )
end

@model function build_funnel_pxg_goals_apm_negbin_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sf_h::NamedTuple, sf_a::NamedTuple,
    home_goals::Vector{Int}, away_goals::Vector{Int},
    nb_h::NamedTuple, nb_a::NamedTuple,
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

    nb_h = _negbin_precompute(d.home_goals, d.w)
    nb_a = _negbin_precompute(d.away_goals, d.w)

    return build_funnel_pxg_goals_apm_negbin_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
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

# Explicit FeatureSet overrides to cleanly overwrite old REPL method tables
PreGame.build_turing_model(config::TeamGoalsNegBinModel, fs::Features.FeatureSet) = PreGame.build_turing_model(config, fs.data)
PreGame.build_turing_model(config::TeamPxGGoalsAPMNegBinModel, fs::Features.FeatureSet) = PreGame.build_turing_model(config, fs.data)
PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMNegBinModel, fs::Features.FeatureSet) = PreGame.build_turing_model(config, fs.data)

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

PreGame.extract_parameters(model::ScottishNegBinModelUnion, df::DataFrame, feature_tuple::Tuple, chain::MCMCChains.Chains) =
    PreGame.extract_parameters(model, df, feature_tuple[1], chain)

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
