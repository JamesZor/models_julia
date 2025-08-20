using CSV, DataFrames 
using Statistics, Dates
using Turing, StatsPlots, LinearAlgebra, Distributions, Random, MCMCChains
using Base.Threads


#= 
General Idea - lifting map (category theory) 
consider a 'morphism' f such that
 f: X -> Y 
 where X is the data files, and Y is the predictive output data. 
 Then we can compose it by a series of 'morphism'. 

 f = g * h (X) 

functions:
mapping - map items to index ( tho separate as this is global) 

cv_splitting: X -> S  ( splits X into groups) 
feature_map: S -> F  ( create the features )


columns: 
  Matches:
tournament_id 
season_id 
season 
match_id 
tournament_slug 
home_team 
away_team 
home_score 
away_score 
home_score_ht 
away_score_ht 
match_date 
round 
winner_code 
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
  # odds:: DataFrame

  function DataStore(data_files::DataFiles)
    matches = CSV.read( data_files.match, DataFrame, header=1)
    # HACK: Add odds once we finished sorting them
    # odds = CSV.read( data_files.odds, DataFrame, header=1)

    new(matches)
  end
end



##################################################
# Mapping 
##################################################
struct MappingFunctions 
  team_mapping_func::Function
  league_mapping_func::Function 
  function MappingFunctions(func::Function)
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

  function MappedData(data::DataStore, transformer::MappingFunctions) 
    all_teams = unique(vcat(data.matches.home_team, data.matches.away_team))
    team_mapping = transformer.team_mapping_func(all_teams)

    all_leagues = unique(string.(data.matches.tournament_id))
    league_mapping = transformer.league_mapping_func(all_leagues) 
    new(team_mapping, league_mapping)

  end
end


##################################################
# Cv splits  
##################################################
struct TimeSeriesSplitsConfig 
  base_seasons::AbstractVector{AbstractString}
  target_season::AbstractString
  round_col::Symbol
end

 
struct TimeSeriesSplits
    df::DataFrame
    base_seasons::Vector
    target_season
    round_col::Symbol
    
    base_indices::Vector{Int}
    target_rounds::Vector
    
    function TimeSeriesSplits(df, base_seasons, target_season, round_col=:round)
        # Pre-compute base data indices
        base_indices = findall(row -> row.season in base_seasons, eachrow(df))
        
        # Pre-compute target season rounds
        target_data = filter(row -> row.season == target_season, df)
        target_rounds = sort(unique(target_data[!, round_col]))
        
        new(df, base_seasons, target_season, round_col, 
            base_indices, target_rounds)
    end
end

# Make it iterable
Base.length(splits::TimeSeriesSplits) = length(splits.target_rounds) + 1

function Base.iterate(splits::TimeSeriesSplits, state=1)
    if state > length(splits)
        return nothing
    end
    
    if state == 1
        # First split: just base data
        train_view = @view splits.df[splits.base_indices, :]
        round_info = "base_only"
        return ((train_view, round_info), state + 1)
    else
        # Subsequent splits: base + incremental rounds
        round_idx = state - 1
        current_rounds = splits.target_rounds[1:round_idx]
        
        # Get indices for target rounds from original DataFrame
        target_indices = findall(eachrow(splits.df)) do row
            row.season == splits.target_season && row[splits.round_col] in current_rounds
        end
        
        # Combine base and target indices  
        combined_indices = vcat(splits.base_indices, target_indices)
        
        train_view = @view splits.df[combined_indices, :]
        round_info = round_idx == 1 ? 
                    "base + round_$(splits.target_rounds[round_idx])" :
                    "base + rounds_1-$(splits.target_rounds[round_idx])"
        
        return ((train_view, round_info), state + 1)
    end
end

# Convenience constructor function
function time_series_splits(data::DataStore, cv_config::TimeSeriesSplitsConfig)
    return TimeSeriesSplits(data.matches, cv_config.base_seasons, cv_config.target_season, cv_config.round_col)
end


##################################################
# features map
##################################################
function feature_map_basic_maher_model( data::SubDataFrame, mapping::MappedData) 
    home_teams_id = [ mapping.team[team_name] for team_name in data.home_team ]
    away_teams_id = [ mapping.team[team_name] for team_name in data.away_team ]
    leauge_ids = [mapping.league[string(league)] for league in data.tournament_id] 
    home_goals_st = data.home_score - data.home_score_ht
    away_goals_st = data.away_score - data.away_score_ht
    n_teams = length(unique(home_teams_id) )
    n_leagues = length(unique(leauge_ids)) 

    return (
        home_team_ids=home_teams_id,
        away_team_ids=away_teams_id,
        leauge_ids=leauge_ids,
        goals_home_ft=data.home_score, 
        goals_away_ft=data.away_score,
        goals_home_ht=data.home_score_ht,
        goals_away_ht=data.away_score_ht,
        goals_home_st=home_goals_st,
        goals_away_st=away_goals_st,
        round=data.round,
        n_teams=n_teams,
        n_leauges=n_leagues,
       )
end

##################################################
# model
##################################################

struct ModelConfig 
  model::Function 
  feature_map::Function
end


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
    # st::Turing.Model # TODO: add the ST into it, skip to save time
    ft::Turing.Model
end


function create_maher_models(model_config::ModelConfig, train_data::SubDataFrame)::BasicMaherModels
    features::NamedTuple =  model_config.feature_map(train_data)

    ht_model = model_config.basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        features.goals_home_ht,  # Half-time goals
        features.goals_away_ht,
        features.n_teams
    )

    # st_model = model_config.basic_maher_model_raw(
    #     features.home_team_ids,
    #     features.away_team_ids,
    #     features.goals_home_st,  # Half-time goals
    #     features.goals_away_st,
    #     features.n_teams
    # )

    ft_model = model_config.basic_maher_model_raw(
        features.home_team_ids,
        features.away_team_ids,
        features.goals_home_ft,  # Half-time goals
        features.goals_away_ft,
        features.n_teams
    )
    return BasicMaherModels(ht_model, st_model,  ft_model)
end

##################################################
#  training
##################################################

struct ModelSampleConfig
    steps::Int 
    bar::Bool 
    # Constructor with keyword arguments (your existing one)
    function ModelSampleConfig( steps=2000, bar=true)
        new(steps, bar)
    end
end


struct ModelChain 
  ht::Chains 
  ft::Chains 
end

function sample_basic_maher_model(model::BasicMaherModels, train_config::ModelTrainConfig)::ModelChain
    ht_chain = sample(model.ht, NUTS(), MCMCThreads(), train_config.steps, progress=train_config.bar)
    ft_chain = sample(model.ft, NUTS(), MCMCThreads(), train_config.steps, progress=train_config.bar)
    return ModelChain(ht_chain, ft_chain)
end


struct ModelTrainingConfig
  cv_functions::Function 
  feature_map_functions::Function 

end


  function train_model(cfg::ModelTrainingConfig, data::DataStore, ) 




###
splits = time_series_splits(data_store, cv_config)
for (i, (train_data, round_info)) in enumerate(splits)
    println("Split $i ($round_info): $(nrow(train_data)) rows")
    feature_map_basic_maher_model( train_data, mapping) 
end

##### running part 

data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)
mapping_functions = MappingFunctions(create_list_mapping) 
mapping = MappedData(data_store, mapping_functions)


cv_config = TimeSeriesSplitsConfig(["20/21", "21/22", "22/23", "23/24"], "24/25", :round)

model_conf = ModelConfig(basic_maher_model_raw, feature_map_basic_maher_model)
splits = time_series_splits(data_store, cv_config)
for (i, (train_data, round_info)) in enumerate(splits)
    println("Split $i ($round_info): $(nrow(train_data)) rows")
    a = feature_map_basic_maher_model( train_data, mapping) 
    println(typeof(a))
    # println(a)
    # println( first(sort(train_data[:, [:season, :home_team, :away_team, :match_date, :home_score, :away_score]], :match_date, rev=true)))
end



