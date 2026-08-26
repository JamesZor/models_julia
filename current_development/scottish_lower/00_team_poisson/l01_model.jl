# ==============================================================================
# Model 00 — Pure Poisson Baseline : CONFIG  (stage ① of the protocol)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# This file builds `DynamicPoissonGoalsTimeDecayModel`, a pure Poisson pregame
# engine that evaluates likelihood in direct log-intensity space.
#
# Subtypes `AbstractPoissonModel` so that score-matrix evaluation automatically
# routes to `src/predictions/score_computation/poisson.jl`.
#
# Read alongside MODEL.md, which states the equations this config selects.
#
# ==============================================================================

using BayesianFootball
using Turing
using Distributions
using DataFrames
using Dates
using SpecialFunctions

const PG       = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features


# ==============================================================================
# 1. The Model Type & Engine Definition
# ==============================================================================

Base.@kwdef struct DynamicPoissonGoalsTimeDecayModel{
    I<:PG.AbstractInterceptionConfig,
    T<:PG.AbstractDynamicsConfig, 
    H<:PG.AbstractHomeAdvantageConfig
} <: BayesianFootball.TypesInterfaces.AbstractPoissonModel
    interception_config::I
    dynamics_config::T
    homeadvantage_config::H
end

function calculate_match_weights(deltas::Vector{<:Real}, half_life_days::Real)
    return 0.5 .^ (deltas ./ half_life_days)
end

function Features.required_features(model::DynamicPoissonGoalsTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), 
        Features.GoalsFeature(), 
        Features.DatesFeature(), 
        Features.MonthFeature(),
        Features.TimeIndicesFeature()
    ] 
end

@model function build_weighted_poisson_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicPoissonGoalsTimeDecayModel
)
    # 1. LOAD COMPONENTS (No dispersion)
    inter ~ to_submodel(PG.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PG.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PG.build_dynamics(config.dynamics_config, n_teams))

    # 2. VECTORIZED INDEXING
    att_h = dyn.α[home_team_indices]
    def_h = dyn.β[home_team_indices]
    att_a = dyn.α[away_team_indices]
    def_a = dyn.β[away_team_indices]
    inter_match = inter.μ_base[season_indices] .+ inter.δ_month[month_indices]
    home_adv = ha[home_team_indices]

    # 3. LINEAR PREDICTORS (LOG SPACE)
    η_h = inter_match .+ home_adv .+ att_h .+ def_a
    η_a = inter_match .+             att_a .+ def_h

    # 4. DIRECT LOG-POISSON LIKELIHOOD
    # log p(y | η) = y*η - exp(η) - log(y!)
    log_fact_h = SpecialFunctions.loggamma.(Float64.(home_goals) .+ 1.0)
    log_fact_a = SpecialFunctions.loggamma.(Float64.(away_goals) .+ 1.0)

    log_lik_h = home_goals .* η_h .- exp.(η_h) .- log_fact_h
    log_lik_a = away_goals .* η_a .- exp.(η_a) .- log_fact_a

    Turing.@addlogprob! sum(log_lik_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_a .* match_weights)
end

function PG.build_turing_model(model::DynamicPoissonGoalsTimeDecayModel, feature_set::Features.FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    n_months   = 12
    
    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_weighted_poisson_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        month_indices,
        home_goals,
        away_goals,
        match_weights,
        n_teams,
        n_seasons,
        n_months,
        model
    )
end

function PG.extract_parameters(
    model::DynamicPoissonGoalsTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_nt = PG.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PG.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = PG.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    for row in eachrow(df)
        mid = Int(row.match_id)

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        # dyn_nt.α is [Samples, Teams]
        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        
        # --- Reconstruct Interception ---
        m_idx = Dates.month(row.match_date)
        inter_match = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        λ_goals_h = exp.(inter_match .+ γ_h .+ α_h .+ β_a)
        λ_goals_a = exp.(inter_match .+        α_a .+ β_h)

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a
        )
    end

    return results
end


# ==============================================================================
# 2. Model Constructor & Reporting
# ==============================================================================

"""
    tp00_model(; kwargs...) -> DynamicPoissonGoalsTimeDecayModel

The pure Poisson baseline engine.
"""
function tp00_model(;
    half_life_days = 180.0,
    interception   = PG.GlobalInterception(μ = Normal(0.2, 0.1)),
    home_advantage = PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2)),
    sigma_att      = Gamma(2.0, 0.15),
    sigma_def      = Gamma(2.0, 0.15),
)
    dynamics = PG.TimeDecayDynamics(
        days_half_life = half_life_days,
        σ_att          = sigma_att,
        σ_def          = sigma_def,
    )

    return DynamicPoissonGoalsTimeDecayModel(
        interception_config  = interception,
        dynamics_config      = dynamics,
        homeadvantage_config = home_advantage,
    )
end

"""
    tp00_menu()

Print the component menu for Model 00.
"""
function tp00_menu()
    println("=" ^ 74)
    println("MODEL 00 — PURE POISSON COMPONENT MENU   (default marked *)")
    println("=" ^ 74)
    println("  Interception     * PG.GlobalInterception(μ)")
    println("                     PG.SeasonalInterception(μ)                  per-season level")
    println("                     PG.HierarchicalMonthlyInterception(...)     + month effects")
    println()
    println("  Dispersion         None (Pure Poisson Likelihood)")
    println()
    println("  Home advantage   * PG.GlobalHomeAdvantage(γ_global)")
    println("                     PG.HierarchicalTeamHomeAdvantage(γ_base, σ_γ)")
    println("                     PG.HierarchicalLeagueHomeAdvantage(γ_base, σ_γ)")
    println()
    println("  Dynamics         * PG.TimeDecayDynamics(days_half_life = 180)")
    println("=" ^ 74)
    return nothing
end

function tp00_component_fields(label::AbstractString, cfg)
    println("  ", rpad(label, 18), ": ", typeof(cfg).name.name)
    for fname in fieldnames(typeof(cfg))
        println("      ", rpad(String(fname), 16), " = ", getfield(cfg, fname))
    end
    return nothing
end

"""
    tp00_describe(model)

Print every hyperparameter the model carries, priors included.
"""
function tp00_describe(model::DynamicPoissonGoalsTimeDecayModel)
    println("=" ^ 74)
    println("MODEL 00 — CONFIG   [$(sl_hash(model))]")
    println("=" ^ 74)
    println("  engine            : $(typeof(model).name.name)")
    println("-" ^ 74)
    tp00_component_fields("interception",   model.interception_config)
    println("  dispersion        : None (Pure Poisson)")
    tp00_component_fields("home advantage", model.homeadvantage_config)
    tp00_component_fields("dynamics",       model.dynamics_config)
    println("-" ^ 74)
    println("  required features : ")
    for f in Features.required_features(model)
        println("      - $(typeof(f).name.name)")
    end
    println("=" ^ 74)
    return nothing
end

"""
    tp00_sampled_sites(n_teams::Int) -> Vector{String}

The chain variable names this pure Poisson configuration produces.
"""
function tp00_sampled_sites(n_teams::Int)
    sites = String["inter.μ", "ha.γ_global", "dyn.σ_a", "dyn.σ_d"]
    append!(sites, ["dyn.raw_a[$i]" for i in 1:n_teams])
    append!(sites, ["dyn.raw_d[$i]" for i in 1:n_teams])
    return sites
end
