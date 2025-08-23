module Eval 
  struct Odds1x2
    home::Float64
    draw::Float64
    away::Float64
  end

  struct PredOdds_1x2 
    ht:: Odds1x2
    ft:: Odds1x2
  end
end 


using Statistics
function get_1x2_odds(match::MatchHalfChainPredicts) 
  h = 1 / mean( match.home) 
  d = 1 / mean( match.draw )
  a = 1 / mean( match.away )
  return Eval.Odds1x2(h,d,a)
end

function get_1x2_odds(match::MatchPredict) 
  return Eval.PredOdds_1x2(get_1x2_odds(match.ht), get_1x2_odds(match.ft))
end
######
module Eval 
    struct Odds1x2
        home::Float64
        draw::Float64
        away::Float64
    end
    
    struct PredOdds_1x2 
        ht::Odds1x2
        ft::Odds1x2
    end
    
    # Market odds from data source
    struct MarketOdds_1x2
        ht::Union{Odds1x2, Nothing}  # Nothing if too many missing values
        ft::Union{Odds1x2, Nothing}
    end
    
    # Comparison results
    struct OddsComparison
        model_odds::PredOdds_1x2
        market_odds::MarketOdds_1x2
        ht_differences::Union{Odds1x2, Nothing}  # model - market
        ft_differences::Union{Odds1x2, Nothing}
        ht_ratios::Union{Odds1x2, Nothing}      # model / market
        ft_ratios::Union{Odds1x2, Nothing}
    end
end

using Statistics

# Your existing functions
function get_1x2_odds(match::MatchHalfChainPredicts) 
    h = 1 / mean(match.home) 
    d = 1 / mean(match.draw)
    a = 1 / mean(match.away)
    return Eval.Odds1x2(h, d, a)
end

function get_1x2_odds(match::MatchPredict) 
    return Eval.PredOdds_1x2(get_1x2_odds(match.ht), get_1x2_odds(match.ft))
end

# Helper function to estimate missing odds using implied probabilities
function estimate_missing_odds(home::Union{Float64, Missing}, 
                              draw::Union{Float64, Missing}, 
                              away::Union{Float64, Missing})
    
    # Count non-missing values
    non_missing = sum([!ismissing(home), !ismissing(draw), !ismissing(away)])
    
    if non_missing < 2
        return nothing  # Can't estimate with less than 2 values
    end
    
    if non_missing == 3
        return Eval.Odds1x2(home, draw, away)  # All present
    end
    
    # Estimate the missing one using implied probabilities
    if ismissing(home)
        prob_draw = 1 / draw
        prob_away = 1 / away
        prob_home = 1 - prob_draw - prob_away
        home_est = prob_home > 0 ? 1 / prob_home : 999.0  # Cap at high odds
        return Eval.Odds1x2(home_est, draw, away)
    elseif ismissing(draw)
        prob_home = 1 / home
        prob_away = 1 / away
        prob_draw = 1 - prob_home - prob_away
        draw_est = prob_draw > 0 ? 1 / prob_draw : 999.0
        return Eval.Odds1x2(home, draw_est, away)
    else  # ismissing(away)
        prob_home = 1 / home
        prob_draw = 1 / draw
        prob_away = 1 - prob_home - prob_draw
        away_est = prob_away > 0 ? 1 / prob_away : 999.0
        return Eval.Odds1x2(home, draw, away_est)
    end
end

# Get market odds for a match, handling missing values
function get_market_odds(match_id::Int64, data_store, 
                        time_range::UnitRange{Int64} = 0:5,
                        score_timeline = nothing)
    
    # Filter odds data for the time range
    odds_data = sort(
        filter(row -> row.sofa_match_id == match_id && 
                     row.minutes in time_range, 
               data_store.odds)[:, [:minutes, :ht_home, :ht_draw, :ht_away, 
                                  :home, :draw, :away]], 
        :minutes
    )
    
    if isempty(odds_data)
        return Eval.MarketOdds_1x2(nothing, nothing)
    end
    
    # Check if any goals occurred in this time range (if score_timeline provided)
    goals_in_range = false
    if score_timeline !== nothing
        goals_in_range = any(minute -> minute in time_range, 
                           keys(score_timeline.score_timeline))
    end
    
    if goals_in_range
        @warn "Goals occurred in time range $(time_range) - odds may be distorted"
    end
    
    # Average the odds over the time period
    ht_home_avg = ismissing(mean(skipmissing(odds_data.ht_home))) ? missing : mean(skipmissing(odds_data.ht_home))
    ht_draw_avg = ismissing(mean(skipmissing(odds_data.ht_draw))) ? missing : mean(skipmissing(odds_data.ht_draw))
    ht_away_avg = ismissing(mean(skipmissing(odds_data.ht_away))) ? missing : mean(skipmissing(odds_data.ht_away))
    
    ft_home_avg = ismissing(mean(skipmissing(odds_data.home))) ? missing : mean(skipmissing(odds_data.home))
    ft_draw_avg = ismissing(mean(skipmissing(odds_data.draw))) ? missing : mean(skipmissing(odds_data.draw))
    ft_away_avg = ismissing(mean(skipmissing(odds_data.away))) ? missing : mean(skipmissing(odds_data.away))
    
    # Estimate missing odds using implied probabilities
    ht_odds = estimate_missing_odds(ht_home_avg, ht_draw_avg, ht_away_avg)
    ft_odds = estimate_missing_odds(ft_home_avg, ft_draw_avg, ft_away_avg)
    
    return Eval.MarketOdds_1x2(ht_odds, ft_odds)
end

# Compare model odds with market odds
function compare_odds(model_odds::Eval.PredOdds_1x2, market_odds::Eval.MarketOdds_1x2)
    
    # Calculate differences (model - market)
    ht_diff = if market_odds.ht !== nothing
        Eval.Odds1x2(
            model_odds.ht.home - market_odds.ht.home,
            model_odds.ht.draw - market_odds.ht.draw,
            model_odds.ht.away - market_odds.ht.away
        )
    else
        nothing
    end
    
    ft_diff = if market_odds.ft !== nothing
        Eval.Odds1x2(
            model_odds.ft.home - market_odds.ft.home,
            model_odds.ft.draw - market_odds.ft.draw,
            model_odds.ft.away - market_odds.ft.away
        )
    else
        nothing
    end
    
    # Calculate ratios (model / market)
    ht_ratio = if market_odds.ht !== nothing
        Eval.Odds1x2(
            market_odds.ht.home / model_odds.ht.home,
            market_odds.ht.draw / model_odds.ht.draw,
            market_odds.ht.away / model_odds.ht.away 
        )
    else
        nothing
    end
    
    ft_ratio = if market_odds.ft !== nothing
        Eval.Odds1x2(
            market_odds.ft.home / model_odds.ft.home,
            market_odds.ft.draw / model_odds.ft.draw,
            market_odds.ft.away / model_odds.ft.away 
        )
    else
        nothing
    end
    
    return Eval.OddsComparison(model_odds, market_odds, ht_diff, ft_diff, ht_ratio, ft_ratio)
end

# Main function to analyze a match
function analyze_match_odds(match_id::Int64, matches_results, data_store, 
                           score_timeline = nothing, time_range = 0:5)
    
    # Get model predictions
    model_odds = get_1x2_odds(matches_results[match_id])
    
    # Get market odds
    market_odds = get_market_odds(match_id, data_store, time_range, score_timeline)
    
    # Compare them
    comparison = compare_odds(model_odds, market_odds)
    
    return comparison
end


# Determine winning outcomes based on score
function get_winning_outcomes(score_timeline)
    if isempty(score_timeline.score_timeline)
        return (ht_winner = "unknown", ft_winner = "unknown")
    end
    
    final_score = score_timeline.final_score
    
    # Full time winner
    ft_winner = if final_score[1] > final_score[2]
        "home"
    elseif final_score[1] < final_score[2] 
        "away"
    else
        "draw"
    end
    
    # Half time winner (score at 45 minutes or last goal before 46)
    ht_goals = filter(minute -> minute <= 45, keys(score_timeline.score_timeline))
    ht_winner = if isempty(ht_goals)
        "draw"  # 0-0 at half time
    else
        last_ht_minute = maximum(ht_goals)
        ht_score = score_timeline.score_timeline[last_ht_minute]
        if ht_score[1] > ht_score[2]
            "home"
        elseif ht_score[1] < ht_score[2]
            "away" 
        else
            "draw"
        end
    end
    
    return (ht_winner = ht_winner, ft_winner = ft_winner)
end
function check_hedging_opportunities(score_timeline, time_range)
    goals_in_range = filter(minute -> minute in time_range, collect(keys(score_timeline.score_timeline)))
    
    if isempty(goals_in_range)
        return (has_hedge = false, description = "No goals in odds window")
    end
    
    hedge_events = String[]
    for minute in sort(goals_in_range)
        score_before = if minute == minimum(collect(keys(score_timeline.score_timeline)))
            (0, 0)
        else
            prev_minutes = filter(m -> m < minute, collect(keys(score_timeline.score_timeline)))
            if isempty(prev_minutes)
                (0, 0)
            else
                score_timeline.score_timeline[maximum(prev_minutes)]
            end
        end
        
        score_after = score_timeline.score_timeline[minute]
        
        if score_after[1] > score_before[1]  # Home goal
            push!(hedge_events, "$(minute)': Home goal ($(score_before[1])-$(score_before[2]) → $(score_after[1])-$(score_after[2])) - Home odds would shorten")
        else  # Away goal
            push!(hedge_events, "$(minute)': Away goal ($(score_before[1])-$(score_before[2]) → $(score_after[1])-$(score_after[2])) - Away odds would shorten")
        end
    end
    
    return (has_hedge = true, description = join(hedge_events, "; "))
end

# Check for hedging opportunities based on goals scored
function check_hedging_opportunities(score_timeline, time_range)
    goals_in_range = filter(minute -> minute in time_range, keys(score_timeline.score_timeline))
    
    if isempty(goals_in_range)
        return (has_hedge = false, description = "No goals in odds window")
    end
    
    hedge_events = String[]
    for minute in sort(goals_in_range)
        score_before = if minute == minimum(keys(score_timeline.score_timeline))
            (0, 0)
        else
            prev_minutes = filter(m -> m < minute, keys(score_timeline.score_timeline))
            if isempty(prev_minutes)
                (0, 0)
            else
                score_timeline.score_timeline[maximum(prev_minutes)]
            end
        end
        
        score_after = score_timeline.score_timeline[minute]
        
        if score_after[1] > score_before[1]  # Home goal
            push!(hedge_events, "$(minute)': Home goal ($(score_before[1])-$(score_before[2]) → $(score_after[1])-$(score_after[2])) - Home odds would shorten")
        else  # Away goal
            push!(hedge_events, "$(minute)': Away goal ($(score_before[1])-$(score_before[2]) → $(score_after[1])-$(score_after[2])) - Away odds would shorten")
        end
    end
    
    return (has_hedge = true, description = join(hedge_events, "; "))
end

# Enhanced table printing with score information
function print_comparison_table(comparison::Eval.OddsComparison, score_timeline = nothing, time_range = 0:5)
    println("="^90)
    println("ODDS COMPARISON TABLE WITH MATCH ANALYSIS")
    println("="^90)
    
    # Match summary
    if score_timeline !== nothing
        winners = get_winning_outcomes(score_timeline)
        hedge_info = check_hedging_opportunities(score_timeline, time_range)
        
        println("\nMATCH SUMMARY:")
        println("-"^50)
        println("Final Score: $(score_timeline.final_score[1])-$(score_timeline.final_score[2])")
        println("Half Time Winner: $(uppercase(winners.ht_winner))")
        println("Full Time Winner: $(uppercase(winners.ft_winner))")
        
        if !isempty(score_timeline.home_goals)
            println("Home Goals: $(join(score_timeline.home_goals, ", "))' mins")
        end
        if !isempty(score_timeline.away_goals)
            println("Away Goals: $(join(score_timeline.away_goals, ", "))' mins")
        end
        
        println("\nHEDGING ANALYSIS ($(time_range) mins):")
        println("-"^50)
        if hedge_info.has_hedge
            println("⚠️  HEDGING OPPORTUNITY: $(hedge_info.description)")
        else
            println("✅ Clean window: $(hedge_info.description)")
        end
    end
    
    # Half Time
    println("\nHALF TIME ODDS:")
    println("-"^70)
    if comparison.market_odds.ht !== nothing
        @printf "%-12s %8s %8s %8s %8s %8s\n" "Outcome" "Model" "Market" "Diff" "EV Ratio" "Result"
        println("-"^70)
        
        ht_result = score_timeline !== nothing ? get_winning_outcomes(score_timeline).ht_winner : ""
        
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Home" comparison.model_odds.ht.home comparison.market_odds.ht.home comparison.ht_differences.home comparison.ht_ratios.home (ht_result == "home" ? "✅ WON" : "")
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Draw" comparison.model_odds.ht.draw comparison.market_odds.ht.draw comparison.ht_differences.draw comparison.ht_ratios.draw (ht_result == "draw" ? "✅ WON" : "")
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Away" comparison.model_odds.ht.away comparison.market_odds.ht.away comparison.ht_differences.away comparison.ht_ratios.away (ht_result == "away" ? "✅ WON" : "")
        
        # Highlight EV bets
        ev_bets_ht = String[]
        if comparison.ht_ratios.home > 1.0
            push!(ev_bets_ht, "Home ($(round(comparison.ht_ratios.home, digits=3)))")
        end
        if comparison.ht_ratios.draw > 1.0
            push!(ev_bets_ht, "Draw ($(round(comparison.ht_ratios.draw, digits=3)))")
        end
        if comparison.ht_ratios.away > 1.0
            push!(ev_bets_ht, "Away ($(round(comparison.ht_ratios.away, digits=3)))")
        end
        
        if !isempty(ev_bets_ht)
            println("\n🎯 HT EV BETS: $(join(ev_bets_ht, ", "))")
        end
    else
        println("Market odds not available for Half Time")
    end
    
    # Full Time  
    println("\nFULL TIME ODDS:")
    println("-"^70)
    if comparison.market_odds.ft !== nothing
        @printf "%-12s %8s %8s %8s %8s %8s\n" "Outcome" "Model" "Market" "Diff" "EV Ratio" "Result"
        println("-"^70)
        
        ft_result = score_timeline !== nothing ? get_winning_outcomes(score_timeline).ft_winner : ""
        
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Home" comparison.model_odds.ft.home comparison.market_odds.ft.home comparison.ft_differences.home comparison.ft_ratios.home (ft_result == "home" ? "✅ WON" : "")
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Draw" comparison.model_odds.ft.draw comparison.market_odds.ft.draw comparison.ft_differences.draw comparison.ft_ratios.draw (ft_result == "draw" ? "✅ WON" : "")
        @printf "%-12s %8.2f %8.2f %8.2f %8.3f %8s\n" "Away" comparison.model_odds.ft.away comparison.market_odds.ft.away comparison.ft_differences.away comparison.ft_ratios.away (ft_result == "away" ? "✅ WON" : "")
        
        # Highlight EV bets
        ev_bets_ft = String[]
        if comparison.ft_ratios.home > 1.0
            push!(ev_bets_ft, "Home ($(round(comparison.ft_ratios.home, digits=3)))")
        end
        if comparison.ft_ratios.draw > 1.0
            push!(ev_bets_ft, "Draw ($(round(comparison.ft_ratios.draw, digits=3)))")
        end
        if comparison.ft_ratios.away > 1.0
            push!(ev_bets_ft, "Away ($(round(comparison.ft_ratios.away, digits=3)))")
        end
        
        if !isempty(ev_bets_ft)
            println("\n🎯 FT EV BETS: $(join(ev_bets_ft, ", "))")
        end
    else
        println("Market odds not available for Full Time")
    end
    
    println("\n" * "="^90)
    println("Note: Diff = Model - Market, EV Ratio = Market / Model")
    println("EV Ratio > 1.0 means positive expected value bet")
    println("✅ = Winning outcome, 🎯 = EV betting opportunity")
    println("="^90)
end
####
module test
  struct MatchScoreTimeline
      home_goals::Vector{Int64}      # Minutes when home team scored
      away_goals::Vector{Int64}      # Minutes when away team scored
      score_timeline::Dict{Int64, Tuple{Int64, Int64}}  # minute => (home_score, away_score)
      final_score::Tuple{Int64, Int64}  # Final (home, away) score
  end
end

function match_score_times(match_id::Int64, data::DataStore)
    # Get all goals for this match, sorted by time
    match_goals = sort(
        filter(row -> row.match_id == match_id && row.incident_type == "goal", 
               data.incidents)[:, [:time, :is_home, :home_score, :away_score]], 
        :time
    )
    
    # Initialize containers
    home_goals = Int64[]
    away_goals = Int64[]
    score_timeline = Dict{Int64, Tuple{Int64, Int64}}()
    
    # Process each goal
    for row in eachrow(match_goals)
        minute = row.time
        home_score = Int64(row.home_score)
        away_score = Int64(row.away_score)
        
        # Record when each team scored
        if row.is_home
            push!(home_goals, minute)
        else
            push!(away_goals, minute)
        end
        
        # Record the score at this minute
        score_timeline[minute] = (home_score, away_score)
    end
    
    # Get final score (from last goal, or 0-0 if no goals)
    final_score = if isempty(match_goals)
        (0, 0)
    else
        last_row = match_goals[end, :]
        (Int64(last_row.home_score), Int64(last_row.away_score))
    end
    sorted_timeline = Dict(sort(collect(score_timeline)))
    return test.MatchScoreTimeline(home_goals, away_goals, sorted_timeline, final_score)
end
