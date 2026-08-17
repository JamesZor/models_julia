# current_development/scottish_wealth/l02_wealth_engines.jl
#
# LOADER: Wealth-Augmented Bayesian Engines for Scottish Lower Leagues
#
# Implements 3 Wealth-Augmented Models:
# 1. DynamicFunnelPlusMinusWealthModel (Baseline Control + Wealth)
# 2. TeamPxGGoalsAPMWealthModel       (Arm A: Proxy xG + RAPM + Wealth)
# 3. TeamFunnelPxGGoalsAPMWealthModel (Arm B: 3-Layer Volume -> Quality -> Goals + RAPM + Wealth)

using Turing
using Distributions
using DataFrames
using Dates
using Statistics
using LogExpFunctions: log1pexp

using BayesianFootball
const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

include("l01_wealth_data.jl")

# ==============================================================================
# 1. MODEL CONFIGURATION STRUCTS
# ==============================================================================

"""
    DynamicFunnelPlusMinusWealthModel
Baseline Funnel (Shots + Lineups) augmented with latent Team Wealth differential.
"""
Base.@kwdef struct DynamicFunnelPlusMinusWealthModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractTeamDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    L<:PreGame.AbstractLeagueConfig,
    P<:PreGame.AbstractPlayerDynamicsConfig,
    F<:Features.AbstractFeatureConfig,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    team_dynamics_config::T
    homeadvantage_config::H
    league_config::L
    player_dynamics_config::P
    player_ratings_feature::F
    wealth_feature::W            = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution = truncated(Normal(0.15, 0.08), lower=0.0)
    shot_scale::Float64          = 2.29
    goals_scale::Float64         = 1.0
end

"""
    TeamPxGGoalsAPMWealthModel
Arm A (Proxy xG + Lineups + Kappa Conversion) augmented with latent Team Wealth differential.
"""
Base.@kwdef struct TeamPxGGoalsAPMWealthModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractTeamDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    L<:PreGame.AbstractLeagueConfig,
    P<:PreGame.AbstractPlayerDynamicsConfig,
    K<:PreGame.AbstractKappaConfig,
    F<:Features.AbstractFeatureConfig,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    team_dynamics_config::T
    homeadvantage_config::H
    league_config::L
    player_dynamics_config::P
    kappa_config::K
    player_ratings_feature::F
    wealth_feature::W            = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution = truncated(Normal(0.15, 0.08), lower=0.0)
    ν_prior::Distribution        = truncated(Normal(3.0, 0.5), lower=0.5)
    log_kappa_sd::Float64        = 0.20
end

"""
    TeamFunnelPxGGoalsAPMWealthModel
Arm B Champion (3-Layer Volume -> Quality -> Goals + Lineups) augmented with Team Wealth.
"""
Base.@kwdef struct TeamFunnelPxGGoalsAPMWealthModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractTeamDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    L<:PreGame.AbstractLeagueConfig,
    P<:PreGame.AbstractPlayerDynamicsConfig,
    K<:PreGame.AbstractKappaConfig,
    F<:Features.AbstractFeatureConfig,
    W<:Features.AbstractFeatureConfig
} <: PreGame.AbstractTimeDecayPlayerModel
    interception_config::I
    team_dynamics_config::T
    homeadvantage_config::H
    league_config::L
    player_dynamics_config::P
    kappa_config::K
    player_ratings_feature::F
    wealth_feature::W                = ScottishTeamWealthFeature()
    w_wealth_prior::Distribution     = truncated(Normal(0.12, 0.06), lower=0.0)
    shot_scale::Float64              = 2.29
    q_base::Float64                  = -2.09
    σ_q_prior::Distribution          = truncated(Normal(0.05, 0.03), lower=0.005)
    ν_q_prior::Distribution          = truncated(Normal(30.0, 5.0), lower=5.0)
    log_kappa_sd::Float64            = 0.20
end

# ==============================================================================
# 2. REQUIRED FEATURES OVERLOADS
# ==============================================================================

function Features.required_features(model::DynamicFunnelPlusMinusWealthModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.ShotsFeature(),
        Features.TimeIndicesFeature(),
        Features.TournamentFeature(),
        model.player_ratings_feature,
        model.wealth_feature
    ]
end

function Features.required_features(model::TeamPxGGoalsAPMWealthModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        Features.TimeIndicesFeature(),
        Features.TournamentFeature(),
        model.player_ratings_feature,
        model.wealth_feature
    ]
end

function Features.required_features(model::TeamFunnelPxGGoalsAPMWealthModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.ShotsFeature(),
        Features.XGFeature(),
        Features.TimeIndicesFeature(),
        Features.TournamentFeature(),
        model.player_ratings_feature,
        model.wealth_feature
    ]
end

# ==============================================================================
# 3. TURING ENGINE DEFINITIONS
# ==============================================================================

# --- ENGINE 1: Baseline Shots Funnel + Wealth ---
@model function build_funnel_pm_wealth_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    home_shots::Vector{Int},
    away_shots::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_pm_ratings::Vector{Float64},
    away_pm_ratings::Vector{Float64},
    wealth_diff::Vector{Float64},
    shot_scale::Float64,
    goals_scale::Float64,
    config::DynamicFunnelPlusMinusWealthModel,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    n_leagues::Int
)
    # Component Builders
    inter   = Turing.@submodel PreGame.build_interception_component(config.interception_config, n_seasons, n_months)
    tdyn_a  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :attack)
    tdyn_d  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :defense)
    ha      = Turing.@submodel PreGame.build_home_advantage_component(config.homeadvantage_config, n_teams)
    league  = Turing.@submodel PreGame.build_league_component(config.league_config, n_leagues)
    pdyn    = Turing.@submodel PreGame.build_player_dynamics_component(config.player_dynamics_config)
    
    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    # Linear Predictor
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg_m  = view(league.δ_league, league_indices)
    ha_m  = view(ha, home_team_indices)
    
    att_h = pdyn.w_att .* home_pm_ratings
    def_a = pdyn.w_def .* away_pm_ratings
    att_a = pdyn.w_att .* away_pm_ratings
    def_h = pdyn.w_def .* home_pm_ratings
    
    w_shift = w_wealth .* wealth_diff
    
    log_λ_s_h = shot_scale .+ int_m .+ lg_m .+ ha_m .+ view(tdyn_a, home_team_indices) .- view(tdyn_d, away_team_indices) .+ att_h .- def_a .+ w_shift
    log_λ_s_a = shot_scale .+ int_m .+ lg_m        .+ view(tdyn_a, away_team_indices) .- view(tdyn_d, home_team_indices) .+ att_a .- def_h .- w_shift
    
    λ_s_h = exp.(clamp.(log_λ_s_h, -15.0, 15.0))
    λ_s_a = exp.(clamp.(log_λ_s_a, -15.0, 15.0))
    
    # Shots Likelihood
    home_shots ~ arraydist(Poisson.(λ_s_h))
    away_shots ~ arraydist(Poisson.(λ_s_a))
    
    # Goals Likelihood
    λ_g_h = λ_s_h .* goals_scale .* 0.11
    λ_g_a = λ_s_a .* goals_scale .* 0.11
    home_goals ~ arraydist(Poisson.(λ_g_h))
    away_goals ~ arraydist(Poisson.(λ_g_a))
end

# --- ENGINE 2: Arm A Proxy xG + Wealth ---
@model function build_pxg_apm_wealth_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_pm_ratings::Vector{Float64},
    away_pm_ratings::Vector{Float64},
    wealth_diff::Vector{Float64},
    config::TeamPxGGoalsAPMWealthModel,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    n_leagues::Int
)
    inter   = Turing.@submodel PreGame.build_interception_component(config.interception_config, n_seasons, n_months)
    tdyn_a  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :attack)
    tdyn_d  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :defense)
    ha      = Turing.@submodel PreGame.build_home_advantage_component(config.homeadvantage_config, n_teams)
    league  = Turing.@submodel PreGame.build_league_component(config.league_config, n_leagues)
    pdyn    = Turing.@submodel PreGame.build_player_dynamics_component(config.player_dynamics_config)
    
    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    # Global Finishing Kappa & Gamma Dispersion
    raw_log_kappa ~ Normal(0.0, 1.0)
    κ = exp(raw_log_kappa * config.log_kappa_sd)
    ν ~ config.ν_prior
    
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg_m  = view(league.δ_league, league_indices)
    ha_m  = view(ha, home_team_indices)
    
    att_h = pdyn.w_att .* home_pm_ratings
    def_a = pdyn.w_def .* away_pm_ratings
    att_a = pdyn.w_att .* away_pm_ratings
    def_h = pdyn.w_def .* home_pm_ratings
    
    w_shift = w_wealth .* wealth_diff
    
    log_μ_h = int_m .+ lg_m .+ ha_m .+ view(tdyn_a, home_team_indices) .- view(tdyn_d, away_team_indices) .+ att_h .- def_a .+ w_shift
    log_μ_a = int_m .+ lg_m        .+ view(tdyn_a, away_team_indices) .- view(tdyn_d, home_team_indices) .+ att_a .- def_h .- w_shift
    
    μ_h = exp.(clamp.(log_μ_h, -15.0, 15.0))
    μ_a = exp.(clamp.(log_μ_a, -15.0, 15.0))
    
    # Pillar A: Proxy xG (Gamma likelihood)
    scale_h = μ_h ./ ν
    scale_a = μ_a ./ ν
    ll_xg_h = logpdf.(Gamma.(ν, scale_h), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν, scale_a), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* xg_mask)
    
    # Pillar B: Goals (Poisson likelihood)
    home_goals ~ arraydist(Poisson.(κ .* μ_h))
    away_goals ~ arraydist(Poisson.(κ .* μ_a))
end

# --- ENGINE 3: Arm B Champion 3-Layer Funnel + Wealth ---
@model function build_funnel_pxg_apm_wealth_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    home_shots::Vector{Int},
    away_shots::Vector{Int},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    home_n_events::Vector{Int},
    away_n_events::Vector{Int},
    xg_mask::Vector{Float64},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    home_pm_ratings::Vector{Float64},
    away_pm_ratings::Vector{Float64},
    wealth_diff::Vector{Float64},
    shot_scale::Float64,
    q_base::Float64,
    config::TeamFunnelPxGGoalsAPMWealthModel,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    n_leagues::Int
)
    inter   = Turing.@submodel PreGame.build_interception_component(config.interception_config, n_seasons, n_months)
    tdyn_a  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :attack)
    tdyn_d  = Turing.@submodel PreGame.build_team_dynamics_component(config.team_dynamics_config, n_teams, :defense)
    ha      = Turing.@submodel PreGame.build_home_advantage_component(config.homeadvantage_config, n_teams)
    league  = Turing.@submodel PreGame.build_league_component(config.league_config, n_leagues)
    pdyn    = Turing.@submodel PreGame.build_player_dynamics_component(config.player_dynamics_config)
    
    # Wealth Prior
    w_wealth ~ config.w_wealth_prior

    # Global Finishing Kappa & Quality Dispersion
    raw_log_kappa ~ Normal(0.0, 1.0)
    κ = exp(raw_log_kappa * config.log_kappa_sd)
    
    σ_q ~ config.σ_q_prior
    ν_q ~ config.ν_q_prior
    
    raw_aq ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_dq ~ filldist(Normal(0.0, 1.0), n_teams)
    aq = (raw_aq .- mean(raw_aq)) .* σ_q
    dq = (raw_dq .- mean(raw_dq)) .* σ_q

    # --- Pillar 1: Shot Volume ---
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg_m  = view(league.δ_league, league_indices)
    ha_m  = view(ha, home_team_indices)
    
    att_h = pdyn.w_att .* home_pm_ratings
    def_a = pdyn.w_def .* away_pm_ratings
    att_a = pdyn.w_att .* away_pm_ratings
    def_h = pdyn.w_def .* home_pm_ratings
    
    w_shift = w_wealth .* wealth_diff
    
    log_λ_s_h = shot_scale .+ int_m .+ lg_m .+ ha_m .+ view(tdyn_a, home_team_indices) .- view(tdyn_d, away_team_indices) .+ att_h .- def_a .+ w_shift
    log_λ_s_a = shot_scale .+ int_m .+ lg_m        .+ view(tdyn_a, away_team_indices) .- view(tdyn_d, home_team_indices) .+ att_a .- def_h .- w_shift
    
    λ_s_h = exp.(clamp.(log_λ_s_h, -15.0, 15.0))
    λ_s_a = exp.(clamp.(log_λ_s_a, -15.0, 15.0))
    
    home_shots ~ arraydist(Poisson.(λ_s_h))
    away_shots ~ arraydist(Poisson.(λ_s_a))

    # --- Pillar 2: Shot Quality ---
    logit_q_h = q_base .+ view(aq, home_team_indices) .- view(dq, away_team_indices)
    logit_q_a = q_base .+ view(aq, away_team_indices) .- view(dq, home_team_indices)
    
    q_h = 1.0 ./ (1.0 .+ exp.(-logit_q_h))
    q_a = 1.0 ./ (1.0 .+ exp.(-logit_q_a))
    
    n_ev_h = max.(home_n_events, 1)
    n_ev_a = max.(away_n_events, 1)
    
    shape_q_h = ν_q .* n_ev_h
    scale_q_h = q_h ./ ν_q
    shape_q_a = ν_q .* n_ev_a
    scale_q_a = q_a ./ ν_q
    
    ll_q_h = logpdf.(Gamma.(shape_q_h, scale_q_h), home_xg)
    ll_q_a = logpdf.(Gamma.(shape_q_a, scale_q_a), away_xg)
    Turing.@addlogprob! sum((ll_q_h .+ ll_q_a) .* xg_mask)

    # --- Pillar 3: Marginal Goals ---
    λ_g_h = κ .* λ_s_h .* q_h
    λ_g_a = κ .* λ_s_a .* q_a
    
    home_goals ~ arraydist(Poisson.(λ_g_h))
    away_goals ~ arraydist(Poisson.(λ_g_a))
end

# ==============================================================================
# 4. TURING MODEL BUILDER HOOKS
# ==============================================================================

function PreGame.build_turing_model(config::DynamicFunnelPlusMinusWealthModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    return build_funnel_pm_wealth_engine(
        d[:home_team_indices], d[:away_team_indices],
        d[:home_goals], d[:away_goals],
        d[:home_shots], d[:away_shots],
        d[:season_indices], d[:month_indices], d[:league_indices],
        d[:home_player_ratings], d[:away_player_ratings],
        d[:flat_wealth_diff],
        config.shot_scale, config.goals_scale,
        config,
        d[:n_teams], d[:n_seasons], d[:n_months], d[:n_leagues]
    )
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMWealthModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    xg_mask = Float64.(!isnan.(d[:home_xg]) .& (d[:home_xg] .> 0))
    home_xg_safe = coalesce.(d[:home_xg], 1.0)
    away_xg_safe = coalesce.(d[:away_xg], 1.0)
    
    return build_pxg_apm_wealth_engine(
        d[:home_team_indices], d[:away_team_indices],
        d[:home_goals], d[:away_goals],
        home_xg_safe, away_xg_safe, xg_mask,
        d[:season_indices], d[:month_indices], d[:league_indices],
        d[:home_player_ratings], d[:away_player_ratings],
        d[:flat_wealth_diff],
        config,
        d[:n_teams], d[:n_seasons], d[:n_months], d[:n_leagues]
    )
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMWealthModel, feature_set::Features.FeatureSet)
    d = feature_set.data
    xg_mask = Float64.(!isnan.(d[:home_xg]) .& (d[:home_xg] .> 0))
    home_xg_safe = coalesce.(d[:home_xg], 1.0)
    away_xg_safe = coalesce.(d[:away_xg], 1.0)
    home_nev = fill(9, length(d[:home_goals]))
    away_nev = fill(9, length(d[:away_goals]))
    
    return build_funnel_pxg_apm_wealth_engine(
        d[:home_team_indices], d[:away_team_indices],
        d[:home_goals], d[:away_goals],
        d[:home_shots], d[:away_shots],
        home_xg_safe, away_xg_safe,
        home_nev, away_nev, xg_mask,
        d[:season_indices], d[:month_indices], d[:league_indices],
        d[:home_player_ratings], d[:away_player_ratings],
        d[:flat_wealth_diff],
        config.shot_scale, config.q_base,
        config,
        d[:n_teams], d[:n_seasons], d[:n_months], d[:n_leagues]
    )
end

# ==============================================================================
# 5. PREDICTION DISPATCH HOOKS (Poisson Score Matrix Overrides)
# ==============================================================================

const ScottishWealthModelUnion = Union{
    DynamicFunnelPlusMinusWealthModel,
    TeamPxGGoalsAPMWealthModel,
    TeamFunnelPxGGoalsAPMWealthModel
}

function Pred.extract_params(model::ScottishWealthModelUnion, chain::MCMCChains.Chains, split)
    df = DataFrame(chain)
    return (
        chain_df = df,
        w_att = mean(df[!, "pdyn.w_att"]),
        w_def = mean(df[!, "pdyn.w_def"]),
        w_wealth = hasproperty(df, :w_wealth) ? mean(df[!, :w_wealth]) : 0.0,
        raw_log_kappa = hasproperty(df, :raw_log_kappa) ? mean(df[!, :raw_log_kappa]) : 0.0
    )
end

@info "Scottish Lower Wealth Engines defined successfully"
