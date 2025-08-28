
using BayesianFootball
using DataFrames 

module Odds
  # Type aliases matching Results module
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Float64}
  
  struct MatchHTPredictions
    home::Float64
    draw::Float64
    away::Float64
    correct_score::CorrectScore
    under_05::Float64
    under_15::Float64
    under_25::Float64
  end
  
  struct MatchFTPredictions
    home::Float64
    draw::Float64
    away::Float64
    correct_score::CorrectScore
    under_05::Float64
    under_15::Float64
    under_25::Float64
    under_35::Float64
    btts::Float64
  end
  
  struct MatchLinePredictions
    ht::MatchHTPredictions
    ft::MatchFTPredictions
  end
end

"""
    get_game_line_odds(data_store::DataStore, match_id::Int)
    
Extracts the kickoff odds (at minutes=0) for a given match from the odds DataFrame.
Returns Odds.MatchLinePredictions or nothing if not found.
"""
function get_game_line_odds(data_store::DataStore, match_id::Int)
    # Filter for kickoff odds (minutes = 0)
    odds_rows = filter(row -> row.sofa_match_id == match_id && row.minutes == 0, data_store.odds)
    
    if isempty(odds_rows)
        @warn "No kickoff odds found for match_id: $match_id"
        return nothing
    end
    
    odds_row = odds_rows[1, :]
    
    # Helper to safely extract values
    function get_val(col_name::String)
        if hasproperty(odds_row, Symbol(col_name))
            val = odds_row[Symbol(col_name)]
            return ismissing(val) ? NaN : Float64(val)
        else
            return NaN
        end
    end
    
    # Extract HT correct scores
    ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:2, a in 0:2
        ht_correct_scores[(h, a)] = get_val("ht_$(h)_$(a)")
    end
    ht_correct_scores["any_unquoted"] = get_val("ht_any_unquoted")
    
    # Extract FT correct scores
    ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:3, a in 0:3
        ft_correct_scores[(h, a)] = get_val("$(h)_$(a)")
    end
    ft_correct_scores["other_home_win"] = get_val("any_other_home")
    ft_correct_scores["other_away_win"] = get_val("any_other_away")
    ft_correct_scores["other_draw"] = get_val("any_other_draw")
    
    # Build HT odds
    ht_odds = Odds.MatchHTPredictions(
        get_val("ht_home"),
        get_val("ht_draw"),
        get_val("ht_away"),
        ht_correct_scores,
        get_val("ht_under_0_5"),  # Note: under_05 = 1/over_05 if not directly available
        get_val("ht_under_1_5"),
        get_val("ht_under_2_5")
    )
    
    # Build FT odds
    ft_odds = Odds.MatchFTPredictions(
        get_val("home"),
        get_val("draw"),
        get_val("away"),
        ft_correct_scores,
        get_val("under_0_5"),
        get_val("under_1_5"),
        get_val("under_2_5"),
        get_val("under_3_5"),
        get_val("btts_yes")
    )
    
    return Odds.MatchLinePredictions(ht_odds, ft_odds)
end

"""
    process_matches_odds(data_store::DataStore, target_matches::DataFrame)
    
Process odds for multiple matches.
Returns Dict{Int64, Odds.MatchLinePredictions}
"""
function process_matches_odds(data_store::DataStore, target_matches::DataFrame)
    results = Dict{Int64, Odds.MatchLinePredictions}()
    
    for match_id in target_matches.match_id
        odds = get_game_line_odds(data_store, match_id)
        if !isnothing(odds)
            results[match_id] = odds
        end
    end
    
    return results
end
