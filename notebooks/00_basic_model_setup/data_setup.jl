using CSV, DataFrames 
using Statistics, Dates
using Turing, StatsPlots, LinearAlgebra, Distributions, Random, MCMCChains
using Base.Threads
# using Optim  # For MAP estimation
# using Printf
#
#=
 Column1
 sofa_match_id
 betfair_match_id
 minutes
 timestamp
 over_3_5
 under_3_5
 0_0
 0_1
 0_2
 0_3
 1_0
 1_1
 1_2
 1_3
 2_0
 2_1
 2_2
 2_3
 3_0
 3_1
 3_2
 3_3
 any_other_away
 any_other_draw
 any_other_home
 over_1_5
 under_1_5
 over_2_5
 under_2_5
 ht_over_1_5
 ht_under_1_5
 over_0_5
 under_0_5
 ht_home
 ht_away
 ht_draw
 ht_over_2_5
 ht_under_2_5
 seconds_to_kickoff
 ht_over_0_5
 ht_under_0_5
 home
 away
 draw

tournament_id
season_id
match_id
tournament_slug
home_team
home_team_slug
away_team
away_team_slug
home_score
away_score
home_score_ht
away_score_ht
match_date
round
status
winner_code
venue
attendance
has_xg
has_stats
match_hour
match_dayofweek
match_month


names(matches)
tournament_id
season_id
match_id
tournament_slug
home_team
home_team_slug
away_team
away_team_slug
home_score
away_score
home_score_ht
away_score_ht
match_date
round
status
winner_code
venue
attendance
has_xg
has_stats
match_hour
match_dayofweek
match_month

=#

########################################
# Data set up 
########################################

struct DataFiles 
    base_dir::String 
    match::String 
    odds::String 
    
    # Inner constructor
    function DataFiles(path::String)
        base_dir = path
        match = joinpath(path, "football_data_mixed_matches.csv")
        odds = joinpath(path, "odds.csv")
        
        # Create the new instance
        new(base_dir, match, odds)
    end
end


struct DataStore
  matches::DataFrame 
  odds:: DataFrame

  function DataStore(data_files::DataFiles)
    matches = CSV.read( data_files.match, DataFrame, header=1)
    odds = CSV.read( data_files.odds, DataFrame, header=1)

    new(matches, odds)
  end
end

function display_1x2(data::DataStore, match_id::Int)
    df = filter(:sofa_match_id => ==(match_id), data.odds) 
    return df[:, [:home, :draw, :away]] 
end

function display_scoreodds(data::DataStore, match_id::Int)
    df = filter(:sofa_match_id => ==(match_id), data.odds) 
    return df[:, [:"minutes", :"0_0",:"0_1", :"0_2", :"0_3", :"1_0", :"1_1", :"1_2", :"1_3", :"2_0", :"2_1", :"2_2", :"2_3", :"3_0", :"3_1", :"3_2", :"3_3" ]]
end


function display_team_matches(data, team_name::String) 
  df = filter(row -> row.home_team_slug == team_name || row.away_team_slug == team_name, data.matches)
  return df[:, [:home_team_slug, :away_team_slug, :home_score_ht, :away_score_ht, :home_score, :away_score, :winner_code]]
end

########################################
# features  setup 
########################################

struct DataTransformers 
  team_mapping_func::Function
  league_mapping_func::Function 

  function DataTransformers(func::Function)
    new(func, func)
  end
end



function create_list_mapping(inputs:: AbstractVector{<:AbstractString})::Dict{String, Int} 
  all_items = unique(inputs)
  items_to_idx = Dict(item => i for (i, item) in enumerate(all_items))
  return items_to_idx
end 

struct MappedData
  team::Dict{String, Int} 
  league::Dict{String, Int} 

  function MappedData(data::DataStore, transformer::DataTransformers) 
    all_teams = unique(vcat(data.matches.home_team_slug, data.matches.away_team_slug))
    team_mapping = transformer.team_mapping_func(all_teams)

    all_leagues = unique(string.(data.matches.tournament_id))
    league_mapping = transformer.league_mapping_func(all_leagues) 
    new(team_mapping, league_mapping)

  end
end


struct FeaturesBasicMaherModel 
    home_team_ids::AbstractVector{Int}
    away_team_ids::AbstractVector{Int}
    league_ids::AbstractVector{Int}
    home_goals::AbstractVector{Int} # full time - all game 0 -90
    away_goals::AbstractVector{Int}
    home_goals_ht::AbstractVector{Int} # half time - 0 -45
    away_goals_ht::AbstractVector{Int}
    home_goals_st::AbstractVector{Int} # second half 45 till 90
    away_goals_st::AbstractVector{Int}
    round::AbstractVector{Int}

    n_teams::Int
    n_leagues::Int

    function FeaturesBasicMaherModel(data::DataStore, mapping::MappedData) 
      home_teams_id = [ mapping.team[team_name] for team_name in data.matches.home_team_slug ]
      away_teams_id = [ mapping.team[team_name] for team_name in data.matches.away_team_slug ]
      leauge_ids = [mapping.league[string(league)] for league in data.matches.tournament_id] 
      home_goals_st = data.matches.home_score - data.matches.home_score_ht
      away_goals_st = data.matches.away_score - data.matches.away_score_ht

      n_teams = length(mapping.team)
      n_leagues = length(mapping.league)

      new(
        home_teams_id,
        away_teams_id,
        leauge_ids,
        data.matches.home_score, 
        data.matches.away_score,
        data.matches.home_score_ht,
        data.matches.away_score_ht,
        home_goals_st,
        away_goals_st,
        data.matches.round,
        n_teams,
        n_leagues
      )

    end
end


########################################
# model setup 
########################################

@model function basic_maher_model_raw(home_team_ids, away_team_ids, home_goals, away_goals, n_teams)
    # PRIORS (in log space for positivity constraints)
    # Attack parameters: log(αᵢ) ~ Normal(0, σ)
    σ_attack ~ truncated(Normal(0, 1), 0, Inf)
    log_α_raw ~ MvNormal(zeros(n_teams), σ_attack * I)
    
    # Defense parameters: log(βᵢ) ~ Normal(0, σ)  
    σ_defense ~ truncated(Normal(0, 1), 0, Inf)
    log_β_raw ~ MvNormal(zeros(n_teams), σ_defense * I)
    
    # Home advantage: log(γ) ~ Normal(log(1.3), 0.2) [prior belief ~30% home advantage]
    log_γ ~ Normal(log(1.3), 0.2)
    
    # IDENTIFIABILITY CONSTRAINT: Center α parameters
    # Equivalent to ∏αᵢ = 1 constraint
    log_α = log_α_raw .- mean(log_α_raw)
    log_β = log_β_raw .- mean(log_β_raw)
    
    # Transform to original scale
    α = exp.(log_α)
    β = exp.(log_β)
    γ = exp(log_γ)
    
    # LIKELIHOOD
    n_matches = length(home_goals)
    for k in 1:n_matches
        i = home_team_ids[k]  # home team index
        j = away_team_ids[k]  # away team index
        
        # Expected goals
        λ = α[i] * β[j] * γ   # Home team expected goals
        μ = α[j] * β[i]       # Away team expected goals
        
        # Observations
        home_goals[k] ~ Poisson(λ)
        away_goals[k] ~ Poisson(μ)
    end
    
    # Return parameters for convenience
    return (α = α, β = β, γ = γ)
end

struct BasicMaherModels
    ht::Turing.Model
    st::Turing.Model
    ft::Turing.Model
end

function create_maher_models(features::FeaturesBasicMaherModel)
    ht_model = basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        features.home_goals_ht,  # Half-time goals
        features.away_goals_ht,
        features.n_teams
    )
    
    st_model = basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        features.home_goals_st,  # Half-time goals
        features.away_goals_st,
        features.n_teams
    )
    ft_model = basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        features.home_goals,     # Full-time goals
        features.away_goals,
        features.n_teams
    )
    
    return BasicMaherModels(ht_model, st_model,  ft_model)
end

##################################################
# train model 
##################################################

struct ModelTrainingConfig
    steps::Int 
    bar::Bool 
    n_chains::Int
    parallel::Bool
    
    # Constructor with keyword arguments (your existing one)
    function ModelTrainingConfig( steps=2000, bar=true, n_chains=4, parallel=false)
        new(steps, bar, n_chains, parallel)
    end
    
    # Add constructor for all positional arguments
    function ModelTrainingConfig(method::DynamicPPL.Sampler, steps::Int, bar::Bool, n_chains::Int, parallel::Bool)
        new( steps, bar, n_chains, parallel)
    end
end


struct ModelChain 
  ht::Chains 
  st::Chains
  ft::Chains 
end


function train_maher_model(model::BasicMaherModels, cfg::ModelTrainingConfig) 
    if cfg.parallel
        # Multiple chains per model (sequential execution of models)
        println("Training with $(cfg.n_chains) chains per model...")
        start_time = time()
        
        ht_chain = sample(model.ht, NUTS(), MCMCThreads(), cfg.steps, cfg.n_chains, progress=cfg.bar)
        ft_chain = sample(model.full, NUTS(), MCMCThreads(), cfg.steps, cfg.n_chains, progress=cfg.bar)
        
        total_time = time() - start_time
        println("Training completed in $(round(total_time, digits=2)) seconds")
    else
        # Single chain per model (simultaneous execution)
        println("Training models simultaneously...")
        start_time = time()
        
        ht_task = Threads.@spawn sample(model.ht, NUTS(), cfg.steps, progress=cfg.bar)
        st_task = Threads.@spawn sample(model.st, NUTS(), cfg.steps, progress=cfg.bar)
        ft_task = Threads.@spawn sample(model.ft, NUTS(), cfg.steps, progress=cfg.bar)
        
        # Wait for both to complete
        ht_chain = fetch(ht_task)
        st_chain = fetch(st_task)
        ft_chain = fetch(ft_task)
        
        total_time = time() - start_time
        println("Simultaneous training completed in $(round(total_time, digits=2)) seconds")
    end
    return ModelChain(ht_chain, st_chain, ft_chain) 
end


# TODO: update for st
function extract_single_chain_with_intervals(single_chain::Chains, team_names::AbstractVector{<:AbstractString})
    n_teams = length(team_names)
    
    # Use MCMCChains functionality to extract parameters more elegantly
    log_α_raw_samples = zeros(size(single_chain, 1), n_teams)
    log_β_raw_samples = zeros(size(single_chain, 1), n_teams)
    
    for i in 1:n_teams
        # Extract using vec() to get a simple vector
        log_α_raw_samples[:, i] = vec(single_chain[Symbol("log_α_raw[$i]")])
        log_β_raw_samples[:, i] = vec(single_chain[Symbol("log_β_raw[$i]")])
    end
    
    # Apply centering and transform to original scale
    log_α_centered = log_α_raw_samples .- mean(log_α_raw_samples, dims=2)
    log_β_centered = log_β_raw_samples .- mean(log_β_raw_samples, dims=2)
    
    α_samples = exp.(log_α_centered)
    β_samples = exp.(log_β_centered)
    
    # Calculate statistics
    α_mean = mean(α_samples, dims=1)[1, :]
    β_mean = mean(β_samples, dims=1)[1, :]
    
    α_q025 = [quantile(α_samples[:, i], 0.025) for i in 1:n_teams]
    α_q975 = [quantile(α_samples[:, i], 0.975) for i in 1:n_teams]
    β_q025 = [quantile(β_samples[:, i], 0.025) for i in 1:n_teams]
    β_q975 = [quantile(β_samples[:, i], 0.975) for i in 1:n_teams]
    
    # Gamma statistics
    log_γ_samples = vec(single_chain[:log_γ])
    γ_samples = exp.(log_γ_samples)
    γ_stats = (
        mean = mean(γ_samples),
        q025 = quantile(γ_samples, 0.025),
        q975 = quantile(γ_samples, 0.975)
    )
    
    return α_mean, β_mean, α_q025, α_q975, β_q025, β_q975, γ_stats
end

function extract_team_parameters_with_intervals(chain::ModelChain, team_names::AbstractVector{<: AbstractString})
    # Extract from both chains
    α_ht, β_ht, α_ht_q025, α_ht_q975, β_ht_q025, β_ht_q975, γ_ht = extract_single_chain_with_intervals(chain.ht, team_names)
    α_ft, β_ft, α_ft_q025, α_ft_q975, β_ft_q025, β_ft_q975, γ_ft = extract_single_chain_with_intervals(chain.ft, team_names)
    
    results = DataFrame(
        team = team_names,
        attack_α_ht = α_ht,
        attack_α_ht_q025 = α_ht_q025,
        attack_α_ht_q975 = α_ht_q975,
        defense_β_ht = β_ht,
        defense_β_ht_q025 = β_ht_q025,
        defense_β_ht_q975 = β_ht_q975,
        attack_α_ft = α_ft,
        attack_α_ft_q025 = α_ft_q025,
        attack_α_ft_q975 = α_ft_q975,
        defense_β_ft = β_ft,
        defense_β_ft_q025 = β_ft_q025,
        defense_β_ft_q975 = β_ft_q975,
        attack_rank_ht = sortperm(α_ht, rev=true),
        defense_rank_ht = sortperm(β_ht),
        attack_rank_ft = sortperm(α_ft, rev=true),
        defense_rank_ft = sortperm(β_ft)
    )
    
    return results, (ht = γ_ht, ft = γ_ft)
end


function extract_team_parameters_essential(chain::ModelChain, team_names::AbstractVector{<:AbstractString})
    function extract_single_chain_essential(single_chain)
        n_teams = length(team_names)
        
        # Extract and transform parameters (same as before)
        log_α_raw_samples = zeros(size(single_chain, 1), n_teams)
        log_β_raw_samples = zeros(size(single_chain, 1), n_teams)
        
        for i in 1:n_teams
            log_α_raw_samples[:, i] = vec(single_chain[Symbol("log_α_raw[$i]")])
            log_β_raw_samples[:, i] = vec(single_chain[Symbol("log_β_raw[$i]")])
        end
        
        log_α_centered = log_α_raw_samples .- mean(log_α_raw_samples, dims=2)
        log_β_centered = log_β_raw_samples .- mean(log_β_raw_samples, dims=2)
        
        α_samples = exp.(log_α_centered)
        β_samples = exp.(log_β_centered)
        
        # Essential statistics only
        α_median = [quantile(α_samples[:, i], 0.5) for i in 1:n_teams]
        β_median = [quantile(β_samples[:, i], 0.5) for i in 1:n_teams]
        
        # 95% credible intervals
        α_ci_lower = [quantile(α_samples[:, i], 0.025) for i in 1:n_teams]
        α_ci_upper = [quantile(α_samples[:, i], 0.975) for i in 1:n_teams]
        β_ci_lower = [quantile(β_samples[:, i], 0.025) for i in 1:n_teams]
        β_ci_upper = [quantile(β_samples[:, i], 0.975) for i in 1:n_teams]
        
        return α_median, β_median, α_ci_lower, α_ci_upper, β_ci_lower, β_ci_upper
    end

    function categorize_uncertainty(median_val, ci_lower, ci_upper)
        ci_width = ci_upper - ci_lower
        relative_width = ci_width / median_val
        return relative_width 
        
    end
    
    α_ht, β_ht, α_ht_low, α_ht_high, β_ht_low, β_ht_high = extract_single_chain_essential(chain.ht)
    α_ft, β_ft, α_ft_low, α_ft_high, β_ft_low, β_ft_high = extract_single_chain_essential(chain.ft)
    
    results = DataFrame(
        team = team_names,
        attack_ht = α_ht,
        attack_ht_ci = ["[$(round(α_ht_low[i], digits=2)), $(round(α_ht_high[i], digits=2))]" for i in 1:length(team_names)],
        defense_ht = β_ht,
        defense_ht_ci = ["[$(round(β_ht_low[i], digits=2)), $(round(β_ht_high[i], digits=2))]" for i in 1:length(team_names)],
        attack_ft = α_ft,
        attack_ft_ci = ["[$(round(α_ft_low[i], digits=2)), $(round(α_ft_high[i], digits=2))]" for i in 1:length(team_names)],
        defense_ft = β_ft,
        defense_ft_ci = ["[$(round(β_ft_low[i], digits=2)), $(round(β_ft_high[i], digits=2))]" for i in 1:length(team_names)],
        attack_rank_ht = sortperm(α_ht, rev=true),
        attack_rank_ft = sortperm(α_ft, rev=true)
    )
    results.attack_confidence_ht = [categorize_uncertainty(results.attack_ht[i], α_ht_low[i], α_ht_high[i]) for i in 1:nrow(results)]
    
    return results
end

#### better one 
function extract_team_parameters_essential(chain::ModelChain, mapping::MappedData)
    # Get teams in the SAME ORDER as they were trained
    teams_in_training_order = sort(collect(keys(mapping.team)), by = team -> mapping.team[team])
    
    function extract_single_chain_essential(single_chain)
        n_teams = length(teams_in_training_order)
        
        # Extract parameters using the training indices (1, 2, 3, ...)
        log_α_raw_samples = zeros(size(single_chain, 1), n_teams)
        log_β_raw_samples = zeros(size(single_chain, 1), n_teams)
        
        for i in 1:n_teams
            # i corresponds directly to the model parameter index
            log_α_raw_samples[:, i] = vec(single_chain[Symbol("log_α_raw[$i]")])
            log_β_raw_samples[:, i] = vec(single_chain[Symbol("log_β_raw[$i]")])
        end
        
        # Apply centering and transform
        log_α_centered = log_α_raw_samples .- mean(log_α_raw_samples, dims=2)
        log_β_centered = log_β_raw_samples .- mean(log_β_raw_samples, dims=2)
        
        α_samples = exp.(log_α_centered)
        β_samples = exp.(log_β_centered)
        
        # Calculate medians and confidence intervals
        α_median = [quantile(α_samples[:, i], 0.5) for i in 1:n_teams]
        β_median = [quantile(β_samples[:, i], 0.5) for i in 1:n_teams]
        
        α_ci_lower = [quantile(α_samples[:, i], 0.025) for i in 1:n_teams]
        α_ci_upper = [quantile(α_samples[:, i], 0.975) for i in 1:n_teams]
        β_ci_lower = [quantile(β_samples[:, i], 0.025) for i in 1:n_teams]
        β_ci_upper = [quantile(β_samples[:, i], 0.975) for i in 1:n_teams]
        
        return α_median, β_median, α_ci_lower, α_ci_upper, β_ci_lower, β_ci_upper
    end
    
    α_ht, β_ht, α_ht_low, α_ht_high, β_ht_low, β_ht_high = extract_single_chain_essential(chain.ht)
    α_st, β_st, α_st_low, α_st_high, β_st_low, β_st_high = extract_single_chain_essential(chain.st)
    α_ft, β_ft, α_ft_low, α_ft_high, β_ft_low, β_ft_high = extract_single_chain_essential(chain.ft)

    attack_rank_ht = invperm(sortperm(α_ht, rev=true))
    defense_rank_ht = invperm(sortperm(β_ht))  # Lower β = better defense
    attack_rank_st = invperm(sortperm(α_st, rev=true))
    defense_rank_st = invperm(sortperm(β_st))  # Lower β = better defense
    attack_rank_ft = invperm(sortperm(α_ft, rev=true))
    defense_rank_ft = invperm(sortperm(β_ft))
    
    results = DataFrame(
        team = teams_in_training_order,
        attack_ht = α_ht,
        attack_ht_ci = ["[$(round(α_ht_low[i], digits=2)), $(round(α_ht_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        defense_ht = β_ht,
        defense_ht_ci = ["[$(round(β_ht_low[i], digits=2)), $(round(β_ht_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        attack_st = α_st,
        attack_st_ci = ["[$(round(α_st_low[i], digits=2)), $(round(α_st_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        defense_st = β_st,
        defense_st_ci = ["[$(round(β_st_low[i], digits=2)), $(round(β_st_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        attack_ft = α_ft,
        attack_ft_ci = ["[$(round(α_ft_low[i], digits=2)), $(round(α_ft_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        defense_ft = β_ft,
        defense_ft_ci = ["[$(round(β_ft_low[i], digits=2)), $(round(β_ft_high[i], digits=2))]" for i in 1:length(teams_in_training_order)],
        attack_rank_ht = attack_rank_ht,
        defense_rank_ht = defense_rank_ht,
        attack_rank_st = attack_rank_st,
        defense_rank_st = defense_rank_st,
        attack_rank_ft = attack_rank_ft,
        defense_rank_ft = defense_rank_ft
    )
    
    return results
end

function extract_team_parameters(chain::ModelChain, mapping::MappedData)
    # Get teams in the SAME ORDER as they were trained
    teams_in_training_order = sort(collect(keys(mapping.team)), by = team -> mapping.team[team])
    
    function extract_single_chain_basic(single_chain)
        n_teams = length(teams_in_training_order)
        
        # Extract parameters using the training indices (1, 2, 3, ...)
        log_α_raw_samples = zeros(size(single_chain, 1), n_teams)
        log_β_raw_samples = zeros(size(single_chain, 1), n_teams)
        
        for i in 1:n_teams
            # i corresponds directly to the model parameter index
            log_α_raw_samples[:, i] = vec(single_chain[Symbol("log_α_raw[$i]")])
            log_β_raw_samples[:, i] = vec(single_chain[Symbol("log_β_raw[$i]")])
        end
        
        # Apply centering and transform
        log_α_centered = log_α_raw_samples .- mean(log_α_raw_samples, dims=2)
        log_β_centered = log_β_raw_samples .- mean(log_β_raw_samples, dims=2)
        
        α_samples = exp.(log_α_centered)
        β_samples = exp.(log_β_centered)
        
        # Calculate medians and confidence intervals
        α_median = [quantile(α_samples[:, i], 0.5) for i in 1:n_teams]
        β_median = [quantile(β_samples[:, i], 0.5) for i in 1:n_teams]

        # gamma 
        log_γ_samples = vec(single_chain[:log_γ])
        γ_samples = exp.(log_γ_samples)
        γ_median = quantile(γ_samples, 0.5) 
        
        return α_median, β_median, γ_median
    end
    
    α_ht, β_ht, γ_ht = extract_single_chain_basic(chain.ht)
    α_st, β_st, γ_st = extract_single_chain_basic(chain.st)
    α_ft, β_ft, γ_ft = extract_single_chain_basic(chain.ft)
    
    results = DataFrame(
        team = teams_in_training_order,
        attack_ht = α_ht,
        defense_ht = β_ht,
        home_ht = γ_ht,
        attack_st = α_st,
        defense_st = β_st,
        home_st = γ_st,
        attack_ft = α_ft,
        defense_ft = β_ft,
        home_ft = γ_ft,
    )
    return results
end



##################################################
# Predictions
##################################################




# old
function predict_match(team1, team2, α, β, γ, team_to_idx; max_goals=5)
    """Predict probabilities for team1 (home) vs team2 (away)"""
    i = team_to_idx[team1]
    j = team_to_idx[team2] 
    
    λ = α[i] * β[j] * γ  # Expected home goals
    μ = α[j] * β[i]      # Expected away goals
    
    # Calculate score probabilities
    probs = zeros(max_goals+1, max_goals+1)
    for h in 0:max_goals
        for a in 0:max_goals
            probs[h+1, a+1] = pdf(Poisson(λ), h) * pdf(Poisson(μ), a)
        end
    end
    
    # Match outcome probabilities
    home_win = sum(probs[i, j] for i in 2:(max_goals+1), j in 1:(i-1))
    draw = sum(probs[i, i] for i in 1:(max_goals+1))
    away_win = sum(probs[i, j] for i in 1:max_goals, j in (i+1):(max_goals+1))
    
    return (
        λ = λ, μ = μ,
        home_win_prob = home_win,
        draw_prob = draw, 
        away_win_prob = away_win,
        score_probs = probs
    )
end

# Add uncertainty categories to your results





##################################################
seasonid_to_year_map = Dict(
    # Championship
    62411 => "24/25",
    52606 => "23/24",
    41958 => "22/23",
    37030 => "21/22",
    29268 => "20/21",
    23988 => "19/20",
    17366 => "18/19",
    13458 => "17/18",
    11765 => "16/17",
    
    # Premiership
    77128 => "25/26",
    62408 => "24/25",
    52588 => "23/24",
    41957 => "22/23",
    37029 => "21/22",
    28212 => "20/21",
    23987 => "19/20",
    17364 => "18/19",
    13448 => "17/18",
    11743 => "16/17"
) ;


##################################################
# Models 
##################################################


