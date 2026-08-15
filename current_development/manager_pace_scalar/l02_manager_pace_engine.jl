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
# SECTION 1: MASTER MODEL STRUCT
# ==============================================================================

Base.@kwdef struct DynamicSmileDoublePoissonXGWealthManagerPaceModel{
    I, P, D, H, K, PR, W, MP, MF, SM, WP
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I    = PreGame.HierarchicalMonthlyInterception()
    player_dynamics_config::P = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0)
    dispersion_config::D      = PreGame.HomeAwayDispersion()
    homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
    kappa_config::K           = PreGame.HierarchicalTeamKappa()
    player_ratings_feature::PR= Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
    wealth_feature::W         = TeamWealthFeature()
    manager_pace_feature::MP  = ManagerPaceFeature(pseudo_matches = 15.0)
    w_wealth_prior            = truncated(Normal(0.105, 0.05), lower=0.0)
    w_pace_prior::WP          = truncated(Normal(0.05, 0.03), lower=0.0)
    market_feature_config::MF = Features.DoublePoissonMarketFeature()
    smile_feature::SM         = Features.MarketSmileFeature(Kmax = 4)
    market_on::Bool           = false
    supremacy_weight::Float64 = 0.4
    smile_weight::Float64     = 0.4
end

# Alias for full configuration compatibility
const DynamicSmileDoublePoissonXGWealthManagerPacePlayerTimeDecayModel = DynamicSmileDoublePoissonXGWealthManagerPaceModel

function Features.required_features(model::DynamicSmileDoublePoissonXGWealthManagerPaceModel)
    return [
        model.interception_config,
        model.dispersion_config,
        model.homeadvantage_config,
        model.kappa_config,
        model.player_ratings_feature,
        model.wealth_feature,
        model.manager_pace_feature,
        model.market_feature_config,
        model.smile_feature
    ]
end

# ==============================================================================
# SECTION 2: TURING PROBABILISTIC ENGINE
# ==============================================================================

@model function build_double_poisson_smile_xg_wealth_manager_pace_engine(
    h_goals, a_goals,
    h_xg, a_xg,
    market_sup,
    market_smile_h, market_smile_a,
    match_team_indices,
    n_teams,
    n_time_steps,
    time_indices,
    seasons, months,
    h_G_c, h_O_c, a_G_c, a_O_c,
    wealth_diff,
    pace_sum,
    w_wealth_prior,
    w_pace_prior,
    market_on::Bool,
    supremacy_weight::Float64,
    smile_weight::Float64
)
    n_matches = length(h_goals)

    # 1. Global Slopes
    w_wealth ~ w_wealth_prior
    w_pace   ~ w_pace_prior

    # 2. Interceptions
    n_seasons = maximum(seasons)
    μ_base ~ filldist(Normal(0.15, 0.20), n_seasons)
    δ_month_raw ~ filldist(Normal(0.0, 1.0), 12)
    σ_month ~ truncated(Normal(0.0, 0.10), lower=0.0)
    δ_month = δ_month_raw .* σ_month

    # 3. Home Advantage (Hierarchical Non-Centered)
    γ_base ~ Normal(0.20, 0.10)
    σ_γ ~ truncated(Normal(0.0, 0.15), lower=0.0)
    γ_team_raw ~ filldist(Normal(0.0, 1.0), n_teams)
    γ_team = γ_base .+ (γ_team_raw .* σ_γ)

    # 4. Conversion / Kappa (Hierarchical Non-Centered)
    σ_κ ~ truncated(Normal(0.0, 0.15), lower=0.0)
    κ_team_raw ~ filldist(Normal(1.0, 0.20), n_teams)
    κ_team = max.(0.10, κ_team_raw .* (1.0 .+ σ_κ))

    # 5. Dispersion & Outfield Dynamics Weights
    φ_h ~ truncated(Normal(1.0, 0.20), lower=0.05)
    φ_a ~ truncated(Normal(1.0, 0.20), lower=0.05)

    w_G_att ~ Normal(0.0, 0.10)
    w_G_def ~ Normal(0.0, 0.10)
    w_Outfield_att ~ Normal(0.0, 0.05)
    w_Outfield_def ~ Normal(0.0, 0.05)

    ν_xg ~ Gamma(2.0, 0.5)

    # 6. Likelihood Evaluation
    for i in 1:n_matches
        h_id = match_team_indices[i][1]
        a_id = match_team_indices[i][2]
        s_id = seasons[i]
        m_id = months[i]

        μ_val = μ_base[s_id] + δ_month[m_id]
        γ_val = γ_team[h_id]

        # Player rating contributions
        att_h = (w_G_att * h_G_c[i]) + (w_Outfield_att * h_O_c[i])
        def_h = (w_G_def * h_G_c[i]) + (w_Outfield_def * h_O_c[i])
        att_a = (w_G_att * a_G_c[i]) + (w_Outfield_att * a_O_c[i])
        def_a = (w_G_def * a_G_c[i]) + (w_Outfield_def * a_O_c[i])

        # Wealth & Manager Tactical Pace shifts
        wealth_shift = w_wealth * wealth_diff[i]
        pace_shift   = w_pace * pace_sum[i]

        log_λ_h_raw = μ_val + γ_val + att_h + def_a + wealth_shift + pace_shift
        log_λ_a_raw = μ_val         + att_a + def_h - wealth_shift + pace_shift

        # Market Anchoring (if enabled)
        if market_on
            mkt_sup_i = market_sup[i]
            mod_sup_i = log_λ_h_raw - log_λ_a_raw
            sup_err   = mod_sup_i - mkt_sup_i
            log_λ_h = log_λ_h_raw - (supremacy_weight * 0.5 * sup_err) + (smile_weight * market_smile_h[i])
            log_λ_a = log_λ_a_raw + (supremacy_weight * 0.5 * sup_err) + (smile_weight * market_smile_a[i])
        else
            log_λ_h = log_λ_h_raw
            log_λ_a = log_λ_a_raw
        end

        λ_true_h = exp(log_λ_h)
        λ_true_a = exp(log_λ_a)

        λ_goals_h = max(1e-4, κ_team[h_id] * λ_true_h)
        λ_goals_a = max(1e-4, κ_team[a_id] * λ_true_a)

        # Observed Goal Counts
        h_goals[i] ~ DoublePoisson(λ_goals_h, φ_h)
        a_goals[i] ~ DoublePoisson(λ_goals_a, φ_a)

        # Observed xG
        if !isnan(h_xg[i]) && h_xg[i] > 0.0
            h_xg[i] ~ Gamma(ν_xg, λ_true_h / ν_xg)
        end
        if !isnan(a_xg[i]) && a_xg[i] > 0.0
            a_xg[i] ~ Gamma(ν_xg, λ_true_a / ν_xg)
        end
    end
end

# ==============================================================================
# SECTION 3: BUILDER & EXTRACTOR INTERFACES
# ==============================================================================

function PreGame.build_turing_model(
    model::DynamicSmileDoublePoissonXGWealthManagerPaceModel,
    feature_set::Features.FeatureSet
)
    F = feature_set.data
    n_matches = length(F[:home_goals])

    # Extract centered player features
    h_G_c = get(F, :home_G_centered, zeros(Float64, n_matches))
    h_O_c = get(F, :home_O_centered, zeros(Float64, n_matches))
    a_G_c = get(F, :away_G_centered, zeros(Float64, n_matches))
    a_O_c = get(F, :away_O_centered, zeros(Float64, n_matches))

    wealth_diff = get(F, :wealth_diff, zeros(Float64, n_matches))
    pace_sum    = get(F, :pace_sum, zeros(Float64, n_matches))

    market_sup     = get(F, :market_supremacy, zeros(Float64, n_matches))
    market_smile_h = get(F, :market_smile_home, zeros(Float64, n_matches))
    market_smile_a = get(F, :market_smile_away, zeros(Float64, n_matches))

    return build_double_poisson_smile_xg_wealth_manager_pace_engine(
        F[:home_goals], F[:away_goals],
        F[:home_xg], F[:away_xg],
        market_sup,
        market_smile_h, market_smile_a,
        F[:match_team_indices],
        F[:n_teams],
        F[:n_time_steps],
        F[:time_indices],
        F[:seasons], F[:months],
        h_G_c, h_O_c, a_G_c, a_O_c,
        wealth_diff,
        pace_sum,
        model.w_wealth_prior,
        model.w_pace_prior,
        model.market_on,
        model.supremacy_weight,
        model.smile_weight
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
    n_teams   = F[:n_teams]
    team_map  = feature_set.metadata[:team_map]

    # 1. Extract component parameters
    inter_nt = PreGame.extract_interception(model.interception_config, chain, feature_set)
    ha_mat   = PreGame.extract_home_advantage(model.homeadvantage_config, chain, feature_set)
    kap_mat  = PreGame.extract_kappa(model.kappa_config, chain, feature_set)
    p_dyn_nt = PreGame.extract_player_dynamics(model.player_dynamics_config, chain, feature_set)
    φ_mat    = PreGame.extract_dispersion(model.dispersion_config, chain, feature_set)

    # 2. Extract global slopes
    w_wealth_samples = haskey(chain, :w_wealth) ? vec(Array(chain[:w_wealth])) : fill(0.105, n_samples)
    w_pace_samples   = haskey(chain, :w_pace) ? vec(Array(chain[:w_pace])) : fill(0.05, n_samples)

    # 3. Retrieve metadata maps
    ratings_map      = get(F, :match_ratings, Dict())
    wealth_map       = get(F, :match_wealth_diff, Dict())
    manager_z_map    = get(F, :manager_z_map, Dict())
    match_managers   = get(F, :match_managers, Dict())
    fallback_mgr     = get(F, :fallback_manager, "Unknown Manager")
    base_r           = 6.5

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

        # Wealth shift
        dw = get(wealth_map, m_id, 0.0)
        w_shift = w_wealth_samples .* dw

        # Manager Tactical Pace shift
        h_mgr, a_mgr = get(match_managers, m_id, (fallback_mgr, fallback_mgr))
        zh = get(manager_z_map, h_mgr, 0.0)
        za = get(manager_z_map, a_mgr, 0.0)
        pace_shift = w_pace_samples .* (zh + za)

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a .+ w_shift .+ pace_shift, -20.0, 20.0)
        log_λ_a = clamp.(μ_v        .+ att_a .+ def_h .- w_shift .+ pace_shift, -20.0, 20.0)

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
