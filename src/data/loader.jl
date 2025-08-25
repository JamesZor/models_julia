# Data loading functionality
function DataFiles(path::String)
    base_dir = path
    match = joinpath(path, "football_data_mixed_matches.csv")
    odds = joinpath(path, "odds.csv")
    incidents = joinpath(path, "football_data_mixed_incidents.csv")
    return DataFiles(base_dir, match, odds, incidents)
end

function DataStore(data_files::DataFiles)
    # Define types for the incidents DataFrame
    # Using String31 as shown in your data output for string columns
    incidents_types = Dict(
      :tournament_id => Int64,
      :season_id => Int64,
      :match_id => Int64,
      :incident_type => String15,
      :time => Int64,
      :is_home => Union{Missing, Bool},
      :is_live => Union{Missing, Bool},
      :period_text => Union{Missing, String3}, 
      :player_in_name => Union{Missing, String31}, 
      :player_out_name => Union{Missing, String31},
      :is_injury => Union{Missing, Bool},
      :player_name => Union{Missing, String31},
      :card_type => Union{Missing, String15},
      :assist1_name => Union{Missing, String31},
      :assist2_name => Union{Missing, String31},
      :home_score => Union{Missing, Float64},
      :away_score => Union{Missing, Float64},
      :added_time => Union{Missing, Float64},
      :time_seconds => Union{Missing, Float64},
      :reversed_period_time => Union{Missing, Float64},
      :reversed_period_time_seconds => Union{Missing, Float64},
      :period_time_seconds => Union{Missing, Float64},
      :injury_time_length => Union{Missing, Float64},
      :incident_class => Union{Missing, String31},  
      :reason => Union{Missing, String31},         
      :rescinded => Union{Missing, Bool},
      :var_confirmed => Union{Missing, Bool},
      :var_decision => Union{Missing, String31},  
      :var_reason => Union{Missing, String31},   
      :penalty_reason => Union{Missing, String31}
    )

    # Define types for the matches DataFrame
    matches_types = Dict(
        :tournament_id => Int64,
        :season_id => Int64,
        :season => String7,  # Based on your data showing String7
        :match_id => Int64,
        :tournament_slug => String15,  # Based on your data
        :home_team => String31,  # Based on your data
        :away_team => String31,
        :home_score => Int64,
        :away_score => Int64,
        :home_score_ht => Int64,
        :away_score_ht => Int64,
        :match_date => Date,  # Parse as Date type
        :round => Int64,
        :winner_code => Int64,
        :has_xg => Bool,
        :has_stats => Bool,
        :match_hour => Int64,
        :match_dayofweek => Int64,
        :match_month => Int64
    )

    # Define types for the odds DataFrame
    odds_types = Dict(
        :sofa_match_id => Int64,
        :betfair_match_id => Int64,
        :minutes => Int64,
        :timestamp => DateTime,  # Parse as DateTime type
        # All betting odds columns as Union{Missing, Float64}
        :over_3_5 => Union{Missing, Float64},
        :under_3_5 => Union{Missing, Float64},
        Symbol("0_0") => Union{Missing, Float64},
        Symbol("0_1") => Union{Missing, Float64},
        Symbol("0_2") => Union{Missing, Float64},
        Symbol("0_3") => Union{Missing, Float64},
        Symbol("1_0") => Union{Missing, Float64},
        Symbol("1_1") => Union{Missing, Float64},
        Symbol("1_2") => Union{Missing, Float64},
        Symbol("1_3") => Union{Missing, Float64},
        Symbol("2_0") => Union{Missing, Float64},
        Symbol("2_1") => Union{Missing, Float64},
        Symbol("2_2") => Union{Missing, Float64},
        Symbol("2_3") => Union{Missing, Float64},
        Symbol("3_0") => Union{Missing, Float64},
        Symbol("3_1") => Union{Missing, Float64},
        Symbol("3_2") => Union{Missing, Float64},
        Symbol("3_3") => Union{Missing, Float64},
        :any_other_away => Union{Missing, Float64},
        :any_other_draw => Union{Missing, Float64},
        :any_other_home => Union{Missing, Float64},
        :over_1_5 => Union{Missing, Float64},
        :under_1_5 => Union{Missing, Float64},
        :home => Union{Missing, Float64},
        :away => Union{Missing, Float64},
        :draw => Union{Missing, Float64},
        :over_2_5 => Union{Missing, Float64},
        :under_2_5 => Union{Missing, Float64},
        :ht_home => Union{Missing, Float64},
        :ht_away => Union{Missing, Float64},
        :ht_draw => Union{Missing, Float64},
        :ht_over_0_5 => Union{Missing, Float64},
        :ht_under_0_5 => Union{Missing, Float64},
        :ht_over_1_5 => Union{Missing, Float64},
        :ht_under_1_5 => Union{Missing, Float64},
        :ht_over_2_5 => Union{Missing, Float64},
        :ht_under_2_5 => Union{Missing, Float64}
    )

    # Read the CSV files with appropriate date formats for each file
    matches = CSV.read(data_files.match, DataFrame; 
        types=matches_types, 
        dateformat=Dict(:match_date => dateformat"yyyy-mm-dd"))
    
    odds = CSV.read(data_files.odds, DataFrame; 
        types=odds_types, 
        dateformat=Dict(:timestamp => dateformat"yyyy-mm-dd HH:MM:SS"))
    
    incidents = CSV.read(data_files.incidents, DataFrame; types=incidents_types)

    return DataStore(matches, odds, incidents)
end
