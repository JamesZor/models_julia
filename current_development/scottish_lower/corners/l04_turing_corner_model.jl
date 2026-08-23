# current_development/scottish_lower/corners/l04_turing_corner_model.jl
#
# Baseline 4-Way Turing Recombination Model with Corner Generation & z-Score Conversion

using Turing
using DynamicPPL
using Distributions
using DataFrames
using Dates
using Statistics
using MCMCChains
using LinearAlgebra
using SpecialFunctions: loggamma
using LogExpFunctions: log1pexp

# Reference the core types and components
using BayesianFootball
using BayesianFootball.Models
using BayesianFootball.Models.PreGame
import BayesianFootball.Models.PreGame: AbstractInterceptionConfig, HierarchicalMonthlyInterception,
                                        AbstractDynamicsConfig, TimeDecayDynamics,
                                        AbstractHomeAdvantageConfig, GlobalHomeAdvantage,
                                        AbstractTimeDecayTeamModel, to_submodel,
                                        build_interception, build_home_advantage, build_dynamics
using BayesianFootball.Features
using BayesianFootball.MyDistributions: RobustNegativeBinomial

"""
    DynamicCornerRecombModel <: AbstractTimeDecayTeamModel

Baseline 4-way goal recombination model decomposing goals into:
Open-play goals + Officiating Penalties + Accidental Own Goals + Corner Set-Piece Goals.
"""
Base.@kwdef struct DynamicCornerRecombModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig
} <: AbstractTimeDecayTeamModel
    interception_config::I   = HierarchicalMonthlyInterception()
    dynamics_config::T       = TimeDecayDynamics(days_half_life = 365.0)
    homeadvantage_config::H  = GlobalHomeAdvantage()
    name::String             = "dynamic_corner_recomb_baseline"
end

# Turing @model for Baseline 4-Way Corner Recombination Engine
@model function build_corner_recomb_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    month_indices::Vector{Int},
    league_indices::Vector{Int},
    y_open_h::Vector{Int},
    y_open_a::Vector{Int},
    corners_h::Vector{Int},
    corners_a::Vector{Int},
    corner_goals_h::Vector{Int},
    corner_goals_a::Vector{Int},
    loggamma_yh_1::Vector{Float64},
    loggamma_ya_1::Vector{Float64},
    loggamma_ch_1::Vector{Float64},
    loggamma_ca_1::Vector{Float64},
    logbinom_h::Vector{Float64},
    logbinom_a::Vector{Float64},
    mask_c_h::Vector{Float64},
    mask_c_a::Vector{Float64},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_leagues::Int,
    config::DynamicCornerRecombModel
)
    # -------------------------------------------------------------
    # 1. Open-Play Tactical Submodel
    # -------------------------------------------------------------
    inter ~ to_submodel(build_interception(config.interception_config, 1, 12))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # League fixed effects (sum-to-zero)
    δ_league_raw ~ filldist(Normal(0, 0.2), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)

    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)

    base_mu = inter.μ_base[1]
    month_eff = inter.δ_month[month_indices]
    league_eff = δ_league[league_indices]
    ha_eff = view(ha, home_team_indices)

    log_μ_open_h = base_mu .+ month_eff .+ league_eff .+ ha_eff .+ att_h .- def_a
    log_μ_open_a = base_mu .+ month_eff .+ league_eff .+           att_a .- def_h

    log_μ_open_h_clamped = clamp.(log_μ_open_h, -5.0, 4.0)
    log_μ_open_a_clamped = clamp.(log_μ_open_a, -5.0, 4.0)
    μ_open_h = exp.(log_μ_open_h_clamped)
    μ_open_a = exp.(log_μ_open_a_clamped)

    # -------------------------------------------------------------
    # 2. Corner Count Generation Submodel (Robust Negative Binomial)
    # -------------------------------------------------------------
    μ_c_base ~ Normal(1.45, 0.20)
    γ_ha_c   ~ Normal(0.13, 0.05)
    ϕ_c_inv  ~ Exponential(0.15) # 1 / r dispersion scale
    ϕ_c      = 1.0 / clamp(ϕ_c_inv, 0.01, 10.0)

    α_c_raw ~ filldist(Normal(0, 0.25), n_teams)
    β_c_raw ~ filldist(Normal(0, 0.25), n_teams)
    α_c     = α_c_raw .- mean(α_c_raw)
    β_c     = β_c_raw .- mean(β_c_raw)

    α_c_h = view(α_c, home_team_indices)
    β_c_h = view(β_c, home_team_indices)
    α_c_a = view(α_c, away_team_indices)
    β_c_a = view(β_c, away_team_indices)

    log_λ_c_h = μ_c_base .+ γ_ha_c .+ α_c_h .- β_c_a
    log_λ_c_a = μ_c_base .+           α_c_a .- β_c_h

    log_λ_c_h_clamped = clamp.(log_λ_c_h, -2.0, 4.0)
    log_λ_c_a_clamped = clamp.(log_λ_c_a, -2.0, 4.0)
    λ_c_h = exp.(log_λ_c_h_clamped)
    λ_c_a = exp.(log_λ_c_a_clamped)

    # -------------------------------------------------------------
    # 3. Corner Goal Conversion Submodel (z-Score Parameterization)
    # -------------------------------------------------------------
    const_logit_q_base = -3.23 # Fixed anchor: logit(0.038)
    σ_conv_att ~ truncated(Normal(0, 0.25), 0.0, 2.0)
    σ_conv_def ~ truncated(Normal(0, 0.25), 0.0, 2.0)

    z_att_raw ~ filldist(Normal(0, 1), n_teams)
    z_def_raw ~ filldist(Normal(0, 1), n_teams)
    z_att     = z_att_raw .- mean(z_att_raw)
    z_def     = z_def_raw .- mean(z_def_raw)

    z_att_h = view(z_att, home_team_indices)
    z_def_h = view(z_def, home_team_indices)
    z_att_a = view(z_att, away_team_indices)
    z_def_a = view(z_def, away_team_indices)

    logit_q_h = const_logit_q_base .+ σ_conv_att .* z_att_h .- σ_conv_def .* z_def_a
    logit_q_a = const_logit_q_base .+ σ_conv_att .* z_att_a .- σ_conv_def .* z_def_h

    # -------------------------------------------------------------
    # 4. Pure Vectorized Arithmetic Likelihood (0 Allocations)
    # -------------------------------------------------------------
    # 1. Open Play Goals (Poisson LogPMF: y*log(μ) - μ - log(y!))
    ll_open_h = y_open_h .* log_μ_open_h_clamped .- μ_open_h .- loggamma_yh_1
    ll_open_a = y_open_a .* log_μ_open_a_clamped .- μ_open_a .- loggamma_ya_1
    Turing.@addlogprob! sum(ll_open_h .* match_weights)
    Turing.@addlogprob! sum(ll_open_a .* match_weights)

    # 2. Corner Generation (Robust NegBin: logΓ(k+r) - logΓ(k+1) - logΓ(r) + r*log(r/(r+μ)) + k*log(μ/(r+μ)))
    log_ϕ = log(ϕ_c)
    log_r_plus_μ_h = log.(ϕ_c .+ λ_c_h)
    log_r_plus_μ_a = log.(ϕ_c .+ λ_c_a)
    loggamma_r = loggamma(ϕ_c)

    ll_c_h = loggamma.(corners_h .+ ϕ_c) .- loggamma_ch_1 .- loggamma_r .+ 
             ϕ_c .* (log_ϕ .- log_r_plus_μ_h) .+ corners_h .* (log_λ_c_h_clamped .- log_r_plus_μ_h)
    ll_c_a = loggamma.(corners_a .+ ϕ_c) .- loggamma_ca_1 .- loggamma_r .+ 
             ϕ_c .* (log_ϕ .- log_r_plus_μ_a) .+ corners_a .* (log_λ_c_a_clamped .- log_r_plus_μ_a)
    Turing.@addlogprob! sum(ll_c_h .* match_weights)
    Turing.@addlogprob! sum(ll_c_a .* match_weights)

    # 3. Corner Goal Conversion (Analytical Logit-Binomial: logbinom + k*logit_q - n*log1pexp(logit_q))
    lq_h_clamped = clamp.(logit_q_h, -10.0, 5.0)
    lq_a_clamped = clamp.(logit_q_a, -10.0, 5.0)
    ll_cg_h = logbinom_h .+ corner_goals_h .* lq_h_clamped .- corners_h .* log1pexp.(lq_h_clamped)
    ll_cg_a = logbinom_a .+ corner_goals_a .* lq_a_clamped .- corners_a .* log1pexp.(lq_a_clamped)
    Turing.@addlogprob! sum(ll_cg_h .* mask_c_h .* match_weights)
    Turing.@addlogprob! sum(ll_cg_a .* mask_c_a .* match_weights)
end

"""
    compute_4way_score_matrix(
        μ_open_h::Real, μ_open_a::Real,
        λ_c_h::Real, λ_c_a::Real,
        q_c_h::Real, q_c_a::Real;
        λ_pen::Real = 0.219, q_pen::Real = 0.78, λ_og::Real = 0.063,
        max_goals::Int = 10
    ) -> Matrix{Float64}

Executes exact 4-way discrete Poisson convolution for matchday predictions.
Guarantees sum(M) = 1.000000.
"""
function compute_4way_score_matrix(
    μ_open_h::Real, μ_open_a::Real,
    λ_c_h::Real, λ_c_a::Real,
    q_c_h::Real, q_c_a::Real;
    λ_pen::Real = 0.219, q_pen::Real = 0.78, λ_og::Real = 0.063,
    max_goals::Int = 10
)::Matrix{Float64}
    # Expected total goals per side
    μ_tot_h = μ_open_h + q_pen * λ_pen + λ_og + q_c_h * λ_c_h
    μ_tot_a = μ_open_a + q_pen * λ_pen + λ_og + q_c_a * λ_c_a

    d_h = Poisson(max(1e-5, Float64(μ_tot_h)))
    d_a = Poisson(max(1e-5, Float64(μ_tot_a)))

    p_h = [pdf(d_h, g) for g in 0:max_goals]
    p_a = [pdf(d_a, g) for g in 0:max_goals]

    # Normalize vectors
    p_h ./= sum(p_h)
    p_a ./= sum(p_a)

    return p_h * p_a'
end
