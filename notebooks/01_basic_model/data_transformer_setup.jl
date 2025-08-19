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
function time_series_splits(df, base_seasons, target_season, round_col=:round)
    return TimeSeriesSplits(df, base_seasons, target_season, round_col)
end
