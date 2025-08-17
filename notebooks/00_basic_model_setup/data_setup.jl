using CSV, DataFrames 
using Statistics, Dates
using Turing, StatsPlots, LinearAlgebra, Distributions, Random, MCMCChains
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
    home_goals::AbstractVector{Int}
    away_goals::AbstractVector{Int}
    home_goals_ht::AbstractVector{Int}
    away_goals_ht::AbstractVector{Int}
    round::AbstractVector{Int}

    n_teams::Int
    n_leagues::Int

    function FeaturesBasicMaherModel(data::DataStore, mapping::MappedData) 
      home_teams_id = [ mapping.team[team_name] for team_name in data.matches.home_team_slug ]
      away_teams_id = [ mapping.team[team_name] for team_name in data.matches.away_team_slug ]
      leauge_ids = [mapping.league[string(league)] for league in data.matches.tournament_id] 

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
        data.matches.round,
        n_teams,
        n_leagues
      )

    end

end


########################################
# model setup 
########################################

@model function basic_maher_model(home_team_ids, away_team_ids, home_goals, away_goals, n_teams)
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

@model function basic_maher_model(features::FeaturesBasicMaherModel) 
  return basic_maher_model(
        features.home_team_ids,
        features.away_team_ids,
        features.home_goals,
        features.away_goals,
        features.n_teams  
    )
end


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
)


##################################################
# Models 
##################################################


