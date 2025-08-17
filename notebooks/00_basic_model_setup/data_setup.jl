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
=#
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

