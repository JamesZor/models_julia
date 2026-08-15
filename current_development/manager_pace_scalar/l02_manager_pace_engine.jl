# current_development/manager_pace_scalar/l02_manager_pace_engine.jl
#
# ==============================================================================
# LOADER: Scalar Manager Tactical Pace & Team Wealth Turing Engine
# ==============================================================================
#
# PURPOSE:
#   Implements a clean, 100% ReverseDiff AD-safe model estimating a single global
#   scalar slope w_pace alongside w_wealth, avoiding categorical parameter bloat.
#
# ==============================================================================

using Turing
using DynamicPPL
using Distributions
using ReverseDiff
using DataFrames
using LinearAlgebra
using Statistics

using BayesianFootball
const Models   = BayesianFootball.Models
const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features

import BayesianFootball.Models.PreGame: build_turing_model, extract_parameters
import BayesianFootball.Features: required_features

include(joinpath(@__DIR__, "l01_manager_pace_data.jl"))

# ==============================================================================
# 1. MASTER MODEL STRUCT
# ==============================================================================

Base.@kwdef struct DynamicSmileDoublePoissonXGWealthManagerPaceModel{
    I<:PreGame.AbstractInterceptionConfig,
    P<:PreGame.OutfieldPlayerDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    K<:PreGame.AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    W<:Features.AbstractFeatureConfig,
    MP<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I           = PreGame.HierarchicalMonthlyInterception()
    player_dynamics_config::P        = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0)
    dispersion_config                = PreGame.HomeAwayDispersion()
    homeadvantage_config::H          = PreGame.HierarchicalTeamHomeAdvantage()
    kappa_config::K                  = PreGame.HierarchicalTeamKappa()
    player_ratings_feature::R        = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
    wealth_feature::W                = TeamWealthFeature()
    manager_pace_feature::MP         = ManagerPaceFeature(pseudo_matches = 15.0)
    market_feature_config::M         = Features.DoublePoissonMarketFeature()
    smile_feature::Features.MarketSmileFeature = Features.MarketSmileFeature(Kmax=4)
    w_wealth_prior::Distribution     = truncated(Normal(0.105, 0.05), lower=0.0)
    w_pace_prior::Distribution       = truncated(Normal(0.05, 0.03), lower=0.0)
    ν_xg::Distribution               = Gamma(2.0, 0.5)
    σ_supremacy_prior::Distribution  = truncated(Normal(0.0, 0.20), lower=0.0)
    σ_smile_prior::Distribution      = truncated(Normal(0.0, 0.20), lower=0.0)
    market_on::Bool                  = false
    supremacy_weight::Float64        = 0.4
    smile_weight::Float64            = 0.4
end

const DynamicSmileDoublePoissonXGWealthManagerPacePlayerTimeDecayModel = DynamicSmileDoublePoissonXGWealthManagerPaceModel

function Features.required_features(model::DynamicSmileDoublePoissonXGWealthManagerPaceModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.market_feature_config,
        model.smile_feature,
        model.player_ratings_feature,
        model.wealth_feature,
        model.manager_pace_feature,
        Features.TimeIndicesFeature()
    ]
end

# ==============================================================================
# 2. TURING PROBABILISTIC ENGINE
# ==============================================================================

@model function build_double_poisson_smile_xg_wealth_manager_pace_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64}, home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64}, home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64}, away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64}, away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64},     # [n_matches × nK]
    smile_mask::Matrix{Float64},     # [n_matches × nK]
    wealth_diff::Vector{Float64},    # [n_matches]
    pace_sum::Vector{Float64},       # [n_matches]
    n_strikes::Int,
    market_active::Float64,
    supremacy_weight::Float64,
    smile_weight::Float64,
    smile_shape_sd::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicSmileDoublePoissonXGWealthManagerPaceModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg     ~ config.ν_xg
    w_wealth ~ config.w_wealth_prior
    w_pace   ~ config.w_pace_prior

    if market_active > 0.5
        σ_sup   ~ config.σ_supremacy_prior
        σ_smile ~ config.σ_smile_prior
    end
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)

    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(PreGame.build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(PreGame.build_dynamics(config.player_dynamics_config, n_teams))

    # --- 2. VECTORIZED INDEXING & MATH ---
    base_rating = hasproperty(config.player_ratings_feature.tracker, :prior_mean) ? config.player_ratings_feature.tracker.prior_mean : 6.0

    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    w_shift = w_wealth .* wealth_diff
    p_shift = w_pace .* pace_sum

    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a .+ w_shift .+ p_shift, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h .- w_shift .+ p_shift, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)
    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C1: Market Supremacy ---
    if market_active > 0.5
        market_rate_h = log_λ_h .+ log.(kap_h)
        market_rate_a = log_λ_a .+ log.(kap_a)
        model_sup = market_rate_h .- market_rate_a
        m_sup     = market_log_λ_h .- market_log_λ_a

        ll_sup = logpdf.(Normal.(m_sup, σ_sup), model_sup)
        Turing.@addlogprob! sum(ll_sup .* match_weights .* market_mask) * supremacy_weight
    end

    # --- Pillar C2: Market Smile ---
    if market_active > 0.5
        φ_strikes = exp.(log_φ)
        ll_smile_h = logpdf.(Poisson.(λ_h .* transpose(φ_strikes)), home_goals)
        ll_smile_a = logpdf.(Poisson.(λ_a .* transpose(φ_strikes)), away_goals)
        ll_smile_tot = (ll_smile_h .+ ll_smile_a) .* match_weights

        Turing.@addlogprob! sum(ll_smile_tot .* smile_mask) * smile_weight
    end
end

# ==============================================================================
# 3. BUILDER & EXTRACTOR INTERFACES
# ==============================================================================

function PreGame.build_turing_model(
    config::DynamicSmileDoublePoissonXGWealthManagerPaceModel,
    feature_set::Features.FeatureSet
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = Int(data[:n_months])

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    match_weights = ones(Float64, length(home_goals))

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]

    _mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]

    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])
    n_strikes  = size(smile_logΛ, 2)

    wealth_diff = Vector{Float64}(get(data, :flat_wealth_diff, zeros(Float64, length(home_goals))))
    pace_sum    = Vector{Float64}(get(data, :pace_sum, zeros(Float64, length(home_goals))))
    market_active = config.market_on ? 1.0 : 0.0

    return build_double_poisson_smile_xg_wealth_manager_pace_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        market_log_h, market_log_a, market_mask,
        smile_logΛ, smile_mask, wealth_diff, pace_sum, n_strikes,
        market_active, config.supremacy_weight, config.smile_weight,
        0.10, n_teams, n_seasons, n_months, config
    )
end

function PreGame.extract_parameters(
    model::DynamicSmileDoublePoissonXGWealthManagerPaceModel,
    target_matches::DataFrame,
    feature_set::Features.FeatureSet,
    chain
)
    F = feature_set.data
    n_samples = size(chain, 1) * size(chain, 3)
    n_teams   = Int(F[:n_teams])
    n_seasons = Int(F[:n_seasons])
    team_map  = F[:team_map]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = PreGame.extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    # Global slopes
    w_wealth_samples = :w_wealth in keys(chain) ? vec(Array(chain[:w_wealth])) : fill(0.105, n_samples)
    w_pace_samples   = :w_pace in keys(chain) ? vec(Array(chain[:w_pace])) : fill(0.05, n_samples)

    nK = haskey(F, :smile_Kmax) ? Int(F[:smile_Kmax]) + 1 : 4
    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        sym = Symbol("log_φ[$k]")
        φ_mat[:, k] = (sym in keys(chain)) ? exp.(vec(Array(chain[sym]))) : ones(n_samples)
    end

    ratings_map    = get(F, :player_ratings_map, Dict())
    wealth_map     = get(F, :wealth_lookup_map, Dict{Int, Float64}())
    manager_z_map  = get(F, :manager_z_map, Dict{String, Float64}())
    match_managers = get(F, :match_managers, Dict{Int, Tuple{String, String}}())
    fallback_mgr   = get(F, :fallback_manager, "Unknown Manager")
    base_r         = hasproperty(model.player_ratings_feature.tracker, :prior_mean) ? model.player_ratings_feature.tracker.prior_mean : 6.0

    n_seasons = size(inter_nt.μ_base, 2)
    results = Dict{Int, Any}()

    for row in eachrow(target_matches)
        m_id = Int(row.match_id)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, m_id, Dict())
        h_G = get(m_ratings, ("home","G"), base_r)
        h_D = get(m_ratings, ("home","D"), 4.0 * base_r)
        h_M = get(m_ratings, ("home","M"), 3.0 * base_r)
        h_F = get(m_ratings, ("home","F"), 3.0 * base_r)
        a_G = get(m_ratings, ("away","G"), base_r)
        a_D = get(m_ratings, ("away","D"), 4.0 * base_r)
        a_M = get(m_ratings, ("away","M"), 3.0 * base_r)
        a_F = get(m_ratings, ("away","F"), 3.0 * base_r)

        h_G_c = h_G - base_r; h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        a_G_c = a_G - base_r; a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att .* h_G_c) .+ (p_dyn_nt.w_Outfield_att .* h_O_c)
        def_h = (p_dyn_nt.w_G_def .* h_G_c) .+ (p_dyn_nt.w_Outfield_def .* h_O_c)
        att_a = (p_dyn_nt.w_G_att .* a_G_c) .+ (p_dyn_nt.w_Outfield_att .* a_O_c)
        def_a = (p_dyn_nt.w_G_def .* a_G_c) .+ (p_dyn_nt.w_Outfield_def .* a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        
        μ_s = (s_idx <= size(inter_nt.μ_base, 2)) ? inter_nt.μ_base[:, s_idx] : inter_nt.μ_base[:, end]
        δ_m = (hasproperty(inter_nt, :δ_month) && m_idx <= size(inter_nt.δ_month, 2)) ? inter_nt.δ_month[:, m_idx] : zeros(n_samples)
        μ_v = μ_s .+ δ_m

        dw = get(wealth_map, m_id, 0.0)
        w_shift = w_wealth_samples .* dw

        h_mgr, a_mgr = get(match_managers, m_id, (fallback_mgr, fallback_mgr))
        zh = get(manager_z_map, h_mgr, 0.0)
        za = get(manager_z_map, a_mgr, 0.0)
        p_shift = w_pace_samples .* (zh + za)

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a .+ w_shift .+ p_shift, -20.0, 20.0)
        log_λ_a = clamp.(μ_v        .+ att_a .+ def_h .- w_shift .+ p_shift, -20.0, 20.0)

        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[m_id] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
            w_wealth = w_wealth_samples,
            w_pace   = w_pace_samples
        )
    end

    return results
end
