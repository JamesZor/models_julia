using BayesianFootball
using Statistics

##################################################
# Basic set up - loading 
##################################################
### Load the saved data for the basic maher model 
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
# extract the results and config 
result = r1[1] 
experiment_config = r1[2]
mapping = result.mapping
# After running your experiment, you have:
# - result (ExperimentResult)
# - data_store (DataStore)
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

# Get the target season 
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)
# compute the chains based off the models chains
matches_results = predict_target_season(target_matches, result, mapping)
keys(matches_results)
##################################################
# Eval
##################################################
# De-constructed 
match_id = 12476901 
m1::MatchPredict = matches_results[match_id]



####
module Eval
  struct Odds1x2
      home::Float64
      draw::Float64
      away::Float64
  end

  struct MarketOdds_1x2
      ht::Union{Odds1x2, Nothing}  # Nothing if too many missing values
      ft::Union{Odds1x2, Nothing}
  end

  struct MatchScoreTimeline
      home_goals::Vector{Int64}      # Minutes when home team scored
      away_goals::Vector{Int64}      # Minutes when away team scored
      score_timeline::Dict{Int64, Tuple{Int64, Int64}}  # minute => (home_score, away_score)
      final_score::Tuple{Int64, Int64}  # Final (home, away) score
  end


  struct ProbValuePeriod_1x2 
    home::Float64 
    draw::Float64 
    away::Float64 
  end 

  struct ProbValue_1x2 
    ht::Union{ProbValuePeriod_1x2, Nothing}
    ft::Union{ProbValuePeriod_1x2, Nothing}
  end


  struct BetDetails
    placed::Bool 
    odds::Float64 
    won::Bool 
    roi::Float64
  end 

  struct BetPeriod_1x2 
    home::BetDetails
    draw::BetDetails
    away::BetDetails
  end 

  struct Bet_1x2 
    ht::BetPeriod_1x2
    ft::BetPeriod_1x2
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
    return Eval.MatchScoreTimeline(home_goals, away_goals, sorted_timeline, final_score)
end

function match_score_times(match_id::Int64, data::Union{SubDataFrame, DataFrame})
    # Get all goals for this match, sorted by time
    match_goals = sort(
        filter(row -> row.match_id == match_id && row.incident_type == "goal", 
               data)[:, [:time, :is_home, :home_score, :away_score]], 
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
    return Eval.MatchScoreTimeline(home_goals, away_goals, sorted_timeline, final_score)
end

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
        return evalMarketOdds_1x2(nothing, nothing)
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


function get_market_odds(match_id::Int64, data_store::DataStore, 
                        time_range::UnitRange{Int64} = 0:5,
                        score_timeline = nothing)
    
    # Adjust time range based on goals
    adjusted_time_range = adjust_time_range_for_goals(time_range, score_timeline)
    
    # Filter odds data for the adjusted time range
    odds_data = sort(
        filter(row -> row.sofa_match_id == match_id && 
                     row.minutes in adjusted_time_range, 
               data_store.odds)[:, [:minutes, :ht_home, :ht_draw, :ht_away, 
                                  :home, :draw, :away]], 
        :minutes
    )
    
    if isempty(odds_data)
        return Eval.MarketOdds_1x2(nothing, nothing)
    end
    
    # Average the odds over the time period
    ht_odds_avg = calculate_average_odds(odds_data.ht_home, odds_data.ht_draw, odds_data.ht_away)
    ft_odds_avg = calculate_average_odds(odds_data.home, odds_data.draw, odds_data.away)
    
    # Estimate missing odds using implied probabilities
    ht_odds = estimate_missing_odds(ht_odds_avg...)
    ft_odds = estimate_missing_odds(ft_odds_avg...)
    
    return Eval.MarketOdds_1x2(ht_odds, ft_odds)
end
function get_market_odds(match_id::Int64, data::AbstractDataFrame, 
                        time_range::UnitRange{Int64} = 0:5,
                        score_timeline = nothing)
    
    # Adjust time range based on goals
    adjusted_time_range = adjust_time_range_for_goals(time_range, score_timeline)
    
    # Filter odds data for the adjusted time range
    odds_data = sort(
        filter(row -> row.sofa_match_id == match_id && 
                     row.minutes in adjusted_time_range, 
               data)[:, [:minutes, :ht_home, :ht_draw, :ht_away, 
                                  :home, :draw, :away]], 
        :minutes
    )
    
    if isempty(odds_data)
        return Eval.MarketOdds_1x2(nothing, nothing)
    end
    
    # Average the odds over the time period
    ht_odds_avg = calculate_average_odds(odds_data.ht_home, odds_data.ht_draw, odds_data.ht_away)
    ft_odds_avg = calculate_average_odds(odds_data.home, odds_data.draw, odds_data.away)
    
    # Estimate missing odds using implied probabilities
    ht_odds = estimate_missing_odds(ht_odds_avg...)
    ft_odds = estimate_missing_odds(ft_odds_avg...)
    
    return Eval.MarketOdds_1x2(ht_odds, ft_odds)
end


function adjust_time_range_for_goals(time_range::UnitRange{Int64}, score_timeline)
    if score_timeline === nothing
        return time_range
    end
    
    # Find goals within the original time range
    goals_in_range = filter(minute -> minute in time_range, 
                           keys(score_timeline.score_timeline))
    
    if isempty(goals_in_range)
        # No goals in range, use original window
        return time_range
    else
        # Goals found, adjust end of window to earliest goal
        earliest_goal = minimum(goals_in_range)
        adjusted_range = time_range.start:(earliest_goal - 1)
        
        @info "Adjusted time range from $(time_range) to $(adjusted_range) due to goal at minute $(earliest_goal)"
        
        # Return at least a 1-minute window if the goal is at the start
        return length(adjusted_range) > 0 ? adjusted_range : time_range.start:time_range.start
    end
end

function calculate_average_odds(home_odds, draw_odds, away_odds)
    home_avg = isempty(skipmissing(home_odds)) ? missing : mean(skipmissing(home_odds))
    draw_avg = isempty(skipmissing(draw_odds)) ? missing : mean(skipmissing(draw_odds))
    away_avg = isempty(skipmissing(away_odds)) ? missing : mean(skipmissing(away_odds))
    
    return (home_avg, draw_avg, away_avg)
end

#####
# runner section - notepad 
# matches_results = predict_target_season(target_matches, result, mapping)
match_id =rand(keys(matches_results))
filter(row -> row.match_id == match_id, target_matches)[:, [:home_team, :away_team, :round, :match_hour, :match_dayofweek, :match_month]]
m1::MatchPredict = matches_results[match_id];
m_score = match_score_times(match_id, data_store)
m_score.final_score
sort(m_score.score_timeline)

market_odds = get_market_odds(match_id, data_store, 0:5, m_score)
market_odds.ht

market_odds.ft

sum(m1.ft.home .* market_odds.ft.home .> 1) / length(m1.ft.home)
sum(m1.ft.away .* market_odds.ft.away .> 1) / length(m1.ft.away)
sum(m1.ft.draw .* market_odds.ft.draw .> 1) / length(m1.ft.draw)


sum(m1.ht.home .* market_odds.ht.home .> 1) / length(m1.ht.home)
sum(m1.ht.away .* market_odds.ht.away .> 1) / length(m1.ht.away)
sum(m1.ht.draw .* market_odds.ht.draw .> 1) / length(m1.ht.draw)
mean(1 ./ m1.ft.home)
mean(1 ./ m1.ft.draw)
mean(1 ./ m1.ft.away)


mean( m1.ft.home + m1.ft.draw + m1.ft.away)

1 /market_odds.ft.home  + 1/ market_odds.ft.draw + 1/market_odds.ft.away 

##############################
# Process 
##############################
# matches_results = predict_target_season(target_matches, result, mapping)
# matches_results is a dict match_id => chains..
#
using DataFrames
using Base.Threads
# matches_score a dict match_id to matchscoretimeline 
function get_target_score_time_lines(target::Union{SubDataFrame, DataFrame}, data_store::DataStore)
    match_ids = unique(target.match_id)
    reduce = filter(row -> row.match_id in match_ids, data_store.incidents)
    grouped_incidents = groupby(reduce, :match_id)
    thread_results = Vector{Dict{Int64, Eval.MatchScoreTimeline}}(undef, nthreads())
    
    @threads for i in 1:nthreads() 
        local_dict = Dict{Int64, Eval.MatchScoreTimeline}()  
        for idx in i:nthreads():length(match_ids) 
            timeline = match_score_times(match_ids[idx], grouped_incidents[(match_id=match_ids[idx],)])
            local_dict[match_ids[idx]] = timeline 
        end 
        thread_results[i] = local_dict
    end 
    
    return merge(thread_results...)
end

matches_scorelines = get_target_score_time_lines(target_matches, data_store)


id1 = match_ids[2]
ms = matches_scorelines[id1]
get_market_odds(id1, data_store, 0:10, ms)


function get_target_market_odds(target::Union{SubDataFrame, DataFrame}, 
                               data_store::DataStore, 
                               matches_scorelines::Dict{Int64,Eval.MatchScoreTimeline},
                               time_range::UnitRange{Int64} = 0:5)
    match_ids = unique(target.match_id)
    thread_results = Vector{Dict{Int64, Union{Eval.MarketOdds_1x2, Nothing}}}(undef, nthreads())
    
    @threads for i in 1:nthreads()
        local_dict = Dict{Int64, Union{Eval.MarketOdds_1x2, Nothing}}()
        for idx in i:nthreads():length(match_ids)  # Fixed indexing
            m_id = match_ids[idx]
            # Get scoreline if available, otherwise pass nothing
            scoreline = get(matches_scorelines, m_id, nothing)
            market_odds = get_market_odds(m_id, data_store, time_range, scoreline) 
            local_dict[m_id] = market_odds 
        end 
        thread_results[i] = local_dict 
    end 
    return merge(thread_results...)
end

matches_market_odds = get_target_market_odds(target_matches, data_store, matches_scorelines)
matches_market_odds[id1]

####

id1 = match_ids[100]
ms = matches_scorelines[id1]
mo = matches_market_odds[id1]
mp = matches_results[id1];

sum(mp.ft.home .* mo.ft.home .> 1) / length(mp.ft.home)
sum(mp.ft.away .* mo.ft.away .> 1) / length(mp.ft.away)
sum(mp.ft.draw .* mo.ft.draw .> 1) / length(mp.ft.draw)


sum(mp.ht.home .* mo.ht.home .> 1) / length(mp.ht.home)
sum(mp.ht.away .* mo.ht.away .> 1) / length(mp.ht.away)
sum(mp.ht.draw .* mo.ht.draw .> 1) / length(mp.ht.draw)
mean(1 ./ mp.ft.home)
mean(1 ./ mp.ft.draw)
mean(1 ./ mp.ft.away)
ms.final_score
sort(ms.score_timeline)

function compute_probability_value_1x2(model::MatchPredict, market::Main.Eval.MarketOdds_1x2) 
    # Helper function to safely compute probability value
    function safe_prob_value(model_probs, market_odds_struct)
        if market_odds_struct === nothing
            return 0.0
        end
        # market_odds_struct is an Odds1x2, so we can access .home, .draw, .away
        home_val = sum(model_probs.home .* market_odds_struct.home .> 1) / length(model_probs.home)
        draw_val = sum(model_probs.draw .* market_odds_struct.draw .> 1) / length(model_probs.draw)
        away_val = sum(model_probs.away .* market_odds_struct.away .> 1) / length(model_probs.away)
        return (home_val, draw_val, away_val)
    end
    
    # Compute FT probabilities
    if market.ft === nothing
        ft = Eval.ProbValuePeriod_1x2(0.0, 0.0, 0.0)
    else
        h_ft = sum(model.ft.home .* market.ft.home .> 1) / length(model.ft.home)
        d_ft = sum(model.ft.draw .* market.ft.draw .> 1) / length(model.ft.draw)
        a_ft = sum(model.ft.away .* market.ft.away .> 1) / length(model.ft.away)
        ft = Eval.ProbValuePeriod_1x2(h_ft, d_ft, a_ft)
    end
    
    # Compute HT probabilities
    if market.ht === nothing
        ht = Eval.ProbValuePeriod_1x2(0.0, 0.0, 0.0)
    else
        h_ht = sum(model.ht.home .* market.ht.home .> 1) / length(model.ht.home)
        d_ht = sum(model.ht.draw .* market.ht.draw .> 1) / length(model.ht.draw)
        a_ht = sum(model.ht.away .* market.ht.away .> 1) / length(model.ht.away)
        ht = Eval.ProbValuePeriod_1x2(h_ht, d_ht, a_ht)
    end
    
    return Eval.ProbValue_1x2(ht, ft)
end

id1 = match_ids[100]
ms = matches_scorelines[id1]
mo = matches_market_odds[id1]
mp = matches_results[id1];
compute_probability_value_1x2(mp, mo)


function get_probability_values(target::AbstractDataFrame, market_odds::AbstractDict, model_preds::AbstractDict)
    match_ids = unique(target.match_id)

    prob_values = Dict{Int64, Eval.ProbValue_1x2}()

    for m_id in match_ids
      prob_values[m_id] = compute_probability_value_1x2(model_preds[m_id], market_odds[m_id])
    end
    return prob_values
  end

matches_probs = get_probability_values(target_matches, matches_market_odds, matches_results)


## bet

function process_bets_1x2(matches_scorelines::Dict{Int64, Eval.MatchScoreTimeline},
                         matches_probs::Dict{Int64, Eval.ProbValue_1x2},
                         matches_market_odds::Dict{Int64, Union{Eval.MarketOdds_1x2, Nothing}},
                         c_threshold::Float64)
    
    results = Dict{Int64, Eval.Bet_1x2}()
    
    for match_id in keys(matches_probs)
        prob = matches_probs[match_id]
        market = matches_market_odds[match_id]
        scoreline = matches_scorelines[match_id]
        
        # Process FT bets (check if ft period exists)
        if prob.ft !== nothing && market.ft !== nothing
            ft_home = process_bet_line(prob.ft.home, market.ft.home, 
                                      check_ft_home_win(scoreline), c_threshold)
            ft_draw = process_bet_line(prob.ft.draw, market.ft.draw, 
                                      check_ft_draw_win(scoreline), c_threshold)
            ft_away = process_bet_line(prob.ft.away, market.ft.away, 
                                      check_ft_away_win(scoreline), c_threshold)
        else
            # No FT data available
            ft_home = Eval.BetDetails(false, 0.0, false, 0.0)
            ft_draw = Eval.BetDetails(false, 0.0, false, 0.0)
            ft_away = Eval.BetDetails(false, 0.0, false, 0.0)
        end
        ft_period = Eval.BetPeriod_1x2(ft_home, ft_draw, ft_away)
        
        # Process HT bets (check if ht period exists)
        if prob.ht !== nothing && market.ht !== nothing
            ht_home = process_bet_line(prob.ht.home, market.ht.home, 
                                      check_ht_home_win(scoreline), c_threshold)
            ht_draw = process_bet_line(prob.ht.draw, market.ht.draw, 
                                      check_ht_draw_win(scoreline), c_threshold)
            ht_away = process_bet_line(prob.ht.away, market.ht.away, 
                                      check_ht_away_win(scoreline), c_threshold)
        else
            # No HT data available
            ht_home = Eval.BetDetails(false, 0.0, false, 0.0)
            ht_draw = Eval.BetDetails(false, 0.0, false, 0.0)
            ht_away = Eval.BetDetails(false, 0.0, false, 0.0)
        end
        ht_period = Eval.BetPeriod_1x2(ht_home, ht_draw, ht_away)
        
        results[match_id] = Eval.Bet_1x2(ht_period, ft_period)
    end
    
    return results
end

function process_bet_line(prob_value::Float64, market_odds::Float64, 
                         actual_won::Bool, threshold::Float64)
    if prob_value > threshold
        # Place bet
        placed = true
        odds = market_odds
        won = actual_won
        roi = won ? (odds - 1.0) * 1.0* (1-0.02) : -1.0  # Win: (odds-1)*stake*(1-commission), Loss: -stake
    else
        # Don't place bet
        placed = false
        odds = 0.0
        won = false
        roi = 0.0
    end
    
    return Eval.BetDetails(placed, odds, won, roi)
end

# Helper functions remain the same
function check_ft_home_win(scoreline::Eval.MatchScoreTimeline)
    return scoreline.final_score[1] > scoreline.final_score[2]
end

function check_ft_draw_win(scoreline::Eval.MatchScoreTimeline)
    return scoreline.final_score[1] == scoreline.final_score[2]
end

function check_ft_away_win(scoreline::Eval.MatchScoreTimeline)
    return scoreline.final_score[1] < scoreline.final_score[2]
end

function check_ht_home_win(scoreline::Eval.MatchScoreTimeline)
    ht_score = get_ht_score(scoreline)
    return ht_score[1] > ht_score[2]
end

function check_ht_draw_win(scoreline::Eval.MatchScoreTimeline)
    ht_score = get_ht_score(scoreline)
    return ht_score[1] == ht_score[2]
end

function check_ht_away_win(scoreline::Eval.MatchScoreTimeline)
    ht_score = get_ht_score(scoreline)
    return ht_score[1] < ht_score[2]
end

function get_ht_score(scoreline::Eval.MatchScoreTimeline)
    ht_minutes = filter(min -> min <= 45, keys(scoreline.score_timeline))
    if isempty(ht_minutes)
        return (0, 0)
    else
        last_ht_minute = maximum(ht_minutes)
        return scoreline.score_timeline[last_ht_minute]
    end
end




matches_bets = process_bets_1x2(matches_scorelines, matches_probs, 
                               matches_market_odds, 0.6)


# Example analysis
total_bets = 0
total_roi = 0.0
for bet in values(matches_bets)
    for period in [bet.ht, bet.ft]
        for line in [period.home, period.draw, period.away]
            if line.placed
                total_bets += 1
                total_roi += line.roi
            end
        end
    end
end
println("Total bets: $total_bets, Total ROI: $total_roi")


####
using Plots 
using Printf

function analyze_roi_vs_threshold(matches_scorelines, matches_probs, matches_market_odds, 
                                c_range = 0.0:0.1:1.0)
    
    c_values = collect(c_range)
    n_threads = nthreads()
    thread_results = Vector{Dict{Float64, Dict}}(undef, n_threads)
    
    # Initialize thread-local dictionaries
    @threads for i in 1:n_threads
        thread_results[i] = Dict{Float64, Dict}()
    end
    
    @threads for i in 1:length(c_values)
        thread_id = threadid()
        c = c_values[i]
        
        println("Thread $(thread_id) processing threshold: $c")
        
        matches_bets = process_bets_1x2(matches_scorelines, matches_probs, 
                                       matches_market_odds, c)
        
        # Detailed analysis for this threshold
        analysis = analyze_bets_detailed(matches_bets)
        thread_results[thread_id][c] = analysis
    end
    
    # Merge results from all threads
    results = Dict{Float64, Dict}()
    for thread_dict in thread_results
        merge!(results, thread_dict)
    end
    
    return results
end

function analyze_bets_detailed(matches_bets)
    stats = Dict(
        :total => Dict(:bets => 0, :roi => 0.0, :wins => 0),
        :ht => Dict(
            :total => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :home => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :draw => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :away => Dict(:bets => 0, :roi => 0.0, :wins => 0)
        ),
        :ft => Dict(
            :total => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :home => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :draw => Dict(:bets => 0, :roi => 0.0, :wins => 0),
            :away => Dict(:bets => 0, :roi => 0.0, :wins => 0)
        )
    )
    
    for bet in values(matches_bets)
        # HT Analysis
        for (line_name, line) in [(:home, bet.ht.home), (:draw, bet.ht.draw), (:away, bet.ht.away)]
            if line.placed
                stats[:ht][line_name][:bets] += 1
                stats[:ht][line_name][:roi] += line.roi
                stats[:ht][line_name][:wins] += line.won ? 1 : 0
                
                stats[:ht][:total][:bets] += 1
                stats[:ht][:total][:roi] += line.roi
                stats[:ht][:total][:wins] += line.won ? 1 : 0
                
                stats[:total][:bets] += 1
                stats[:total][:roi] += line.roi
                stats[:total][:wins] += line.won ? 1 : 0
            end
        end
        
        # FT Analysis
        for (line_name, line) in [(:home, bet.ft.home), (:draw, bet.ft.draw), (:away, bet.ft.away)]
            if line.placed
                stats[:ft][line_name][:bets] += 1
                stats[:ft][line_name][:roi] += line.roi
                stats[:ft][line_name][:wins] += line.won ? 1 : 0
                
                stats[:ft][:total][:bets] += 1
                stats[:ft][:total][:roi] += line.roi
                stats[:ft][:total][:wins] += line.won ? 1 : 0
                
                stats[:total][:bets] += 1
                stats[:total][:roi] += line.roi
                stats[:total][:wins] += line.won ? 1 : 0
            end
        end
    end
    
    return stats
end

function print_analysis_table(results)
    println("\n" * "="^120)
    println("BETTING ANALYSIS - ROI vs THRESHOLD")
    println("="^120)
    
    # Header
    header = @sprintf("%-6s │ %-8s %-8s %-8s │ %-25s │ %-25s │ %-25s", 
                     "Thresh", "Tot_Bets", "Tot_ROI", "Tot_WR", 
                     "HT (Bets/ROI/WR)", "FT (Bets/ROI/WR)", "Breakdown")
    println(header)
    println("─"^120)
    
    # Sort thresholds
    sorted_thresholds = sort(collect(keys(results)))
    
    for c in sorted_thresholds
        stats = results[c]
        
        # Total stats
        total_bets = stats[:total][:bets]
        total_roi = stats[:total][:roi]
        total_wr = total_bets > 0 ? stats[:total][:wins] / total_bets * 100 : 0.0
        
        # HT stats
        ht_bets = stats[:ht][:total][:bets]
        ht_roi = stats[:ht][:total][:roi]
        ht_wr = ht_bets > 0 ? stats[:ht][:total][:wins] / ht_bets * 100 : 0.0
        
        # FT stats
        ft_bets = stats[:ft][:total][:bets]
        ft_roi = stats[:ft][:total][:roi]
        ft_wr = ft_bets > 0 ? stats[:ft][:total][:wins] / ft_bets * 100 : 0.0
        
        # Breakdown
        ht_breakdown = @sprintf("H:%d/%.1f/%.1f D:%d/%.1f/%.1f A:%d/%.1f/%.1f",
                               stats[:ht][:home][:bets], stats[:ht][:home][:roi], 
                               stats[:ht][:home][:bets] > 0 ? stats[:ht][:home][:wins]/stats[:ht][:home][:bets]*100 : 0,
                               stats[:ht][:draw][:bets], stats[:ht][:draw][:roi],
                               stats[:ht][:draw][:bets] > 0 ? stats[:ht][:draw][:wins]/stats[:ht][:draw][:bets]*100 : 0,
                               stats[:ht][:away][:bets], stats[:ht][:away][:roi],
                               stats[:ht][:away][:bets] > 0 ? stats[:ht][:away][:wins]/stats[:ht][:away][:bets]*100 : 0)
        
        ft_breakdown = @sprintf("H:%d/%.1f/%.1f D:%d/%.1f/%.1f A:%d/%.1f/%.1f",
                               stats[:ft][:home][:bets], stats[:ft][:home][:roi],
                               stats[:ft][:home][:bets] > 0 ? stats[:ft][:home][:wins]/stats[:ft][:home][:bets]*100 : 0,
                               stats[:ft][:draw][:bets], stats[:ft][:draw][:roi],
                               stats[:ft][:draw][:bets] > 0 ? stats[:ft][:draw][:wins]/stats[:ft][:draw][:bets]*100 : 0,
                               stats[:ft][:away][:bets], stats[:ft][:away][:roi],
                               stats[:ft][:away][:bets] > 0 ? stats[:ft][:away][:wins]/stats[:ft][:away][:bets]*100 : 0)
        
        row = @sprintf("%-6.1f │ %-8d %-8.2f %-8.1f │ %-8d/%-7.2f/%-6.1f%% │ %-8d/%-7.2f/%-6.1f%% │ HT: %s",
                      c, total_bets, total_roi, total_wr,
                      ht_bets, ht_roi, ht_wr,
                      ft_bets, ft_roi, ft_wr,
                      ht_breakdown)
        println(row)
        
        # FT breakdown on next line
        ft_row = @sprintf("%-6s │ %-8s %-8s %-8s │ %-25s │ %-25s │ FT: %s",
                         "", "", "", "", "", "", ft_breakdown)
        println(ft_row)
        println("─"^120)
    end
end

function plot_roi_curve(results)
    
    # Extract data for plotting
    thresholds = sort(collect(keys(results)))
    total_roi = [results[c][:total][:roi] for c in thresholds]
    ht_roi = [results[c][:ht][:total][:roi] for c in thresholds]
    ft_roi = [results[c][:ft][:total][:roi] for c in thresholds]
    total_bets = [results[c][:total][:bets] for c in thresholds]
    
    # Create the main ROI plot
    p1 = plot(thresholds, total_roi, 
              label="Total ROI", 
              linewidth=3, 
              marker=:circle,
              title="ROI vs Confidence Threshold",
              xlabel="Confidence Threshold",
              ylabel="Total ROI",
              grid=true,
              legend=:topright)
    
    plot!(p1, thresholds, ht_roi, 
          label="HT ROI", 
          linewidth=2, 
          marker=:square,
          linestyle=:dash)
    
    plot!(p1, thresholds, ft_roi, 
          label="FT ROI", 
          linewidth=2, 
          marker=:diamond,
          linestyle=:dash)
    
    # Add horizontal line at ROI = 0
    hline!(p1, [0], color=:red, linestyle=:dot, label="Break-even")
    
    # Create secondary plot showing number of bets
    p2 = plot(thresholds, total_bets,
              label="Total Bets",
              linewidth=2,
              marker=:circle,
              title="Number of Bets vs Threshold",
              xlabel="Confidence Threshold", 
              ylabel="Number of Bets",
              grid=true,
              color=:orange)
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main analysis function
println("Starting betting analysis...")
    
# Run analysis across thresholds
results = analyze_roi_vs_threshold(matches_scorelines, matches_probs, matches_market_odds, 0:0.1:1)

# Print detailed table
print_analysis_table(results)

# Create plots
roi_plot = plot_roi_curve(results)

# Find optimal threshold
best_threshold = 0.0
best_roi = -Inf
for (c, stats) in results
    if stats[:total][:roi] > best_roi
        best_roi = stats[:total][:roi]
        best_threshold = c
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("Best threshold: $best_threshold with ROI: $best_roi")
println("Total matches analyzed: $(length(matches_scorelines))")

####
using DataFrames, Dates, Plots, Statistics

function simulate_cumulative_wealth(target_matches, matches_bets, starting_wealth::Float64 = 100.0)
    # Sort matches by date
    sorted_matches = sort(target_matches, :match_date)
    
    # Initialize tracking arrays
    dates = Date[]
    cumulative_wealth = Float64[]
    daily_roi = Float64[]
    daily_bets_count = Int[]
    match_ids = Int[]
    
    current_wealth = starting_wealth
    
    for row in eachrow(sorted_matches)
        match_id = row.match_id
        match_date = Date(row.match_date)
        
        # Get bets for this match if they exist
        if haskey(matches_bets, match_id)
            bet = matches_bets[match_id]
            
            # Calculate total ROI and bets for this match
            match_roi = 0.0
            match_bet_count = 0
            
            # Check all betting lines
            for period in [bet.ht, bet.ft]
                for line in [period.home, period.draw, period.away]
                    if line.placed
                        match_roi += line.roi
                        match_bet_count += 1
                    end
                end
            end
            
            # Update wealth (assuming unit stake of 1 per bet)
            current_wealth += match_roi
            
            push!(dates, match_date)
            push!(cumulative_wealth, current_wealth)
            push!(daily_roi, match_roi)
            push!(daily_bets_count, match_bet_count)
            push!(match_ids, match_id)
        else
            # No bets for this match
            push!(dates, match_date)
            push!(cumulative_wealth, current_wealth)
            push!(daily_roi, 0.0)
            push!(daily_bets_count, 0)
            push!(match_ids, match_id)
        end
    end
    
    # Create summary DataFrame
    wealth_df = DataFrame(
        date = dates,
        match_id = match_ids,
        cumulative_wealth = cumulative_wealth,
        daily_roi = daily_roi,
        daily_bets = daily_bets_count
    )
    
    return wealth_df
end

function plot_wealth_evolution(wealth_df, starting_wealth::Float64 = 100.0)
    # Main wealth plot
    p1 = plot(wealth_df.date, wealth_df.cumulative_wealth,
              label="Cumulative Wealth",
              linewidth=2,
              title="Betting Wealth Evolution Over Time",
              xlabel="Date",
              ylabel="Wealth",
              grid=true,
              color=:blue)
    
    # Add starting wealth reference line
    hline!(p1, [starting_wealth], 
           color=:red, 
           linestyle=:dash, 
           label="Starting Wealth ($starting_wealth)")
    
    # Daily ROI subplot
    p2 = scatter(wealth_df.date, wealth_df.daily_roi,
                label="Daily ROI",
                markersize=2,
                title="Daily ROI",
                xlabel="Date",
                ylabel="Daily ROI",
                grid=true,
                color=:green,
                alpha=0.6)
    
    # Add zero line for break-even
    hline!(p2, [0], color=:red, linestyle=:dot, label="Break-even")
    
    # Daily bets count
    p3 = scatter(wealth_df.date, wealth_df.daily_bets,
                label="Daily Bets",
                markersize=2,
                title="Number of Bets Per Match",
                xlabel="Date", 
                ylabel="Number of Bets",
                grid=true,
                color=:orange,
                alpha=0.6)
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(1000, 800))
    
    return combined_plot
end

function wealth_summary_stats(wealth_df, starting_wealth::Float64 = 100.0)
    final_wealth = last(wealth_df.cumulative_wealth)
    total_roi = final_wealth - starting_wealth
    total_return_pct = (total_roi / starting_wealth) * 100
    
    # Calculate some key metrics
    winning_days = sum(wealth_df.daily_roi .> 0)
    losing_days = sum(wealth_df.daily_roi .< 0)
    break_even_days = sum(wealth_df.daily_roi .== 0)
    
    total_bets = sum(wealth_df.daily_bets)
    avg_daily_roi = mean(wealth_df.daily_roi)
    max_daily_gain = maximum(wealth_df.daily_roi)
    max_daily_loss = minimum(wealth_df.daily_roi)
    
    # Calculate maximum drawdown
    peak_wealth = [maximum(wealth_df.cumulative_wealth[1:i]) for i in 1:length(wealth_df.cumulative_wealth)]
    drawdown = (wealth_df.cumulative_wealth .- peak_wealth) ./ peak_wealth * 100
    max_drawdown = minimum(drawdown)
    
    # Volatility (standard deviation of daily returns)
    volatility = std(wealth_df.daily_roi)
    
    println("="^60)
    println("WEALTH SIMULATION SUMMARY")
    println("="^60)
    println("Starting Wealth: \$$(starting_wealth)")
    println("Final Wealth: \$$(round(final_wealth, digits=2))")
    println("Total ROI: \$$(round(total_roi, digits=2))")
    println("Total Return: $(round(total_return_pct, digits=2))%")
    println()
    println("Match Statistics:")
    println("  Total Matches: $(length(wealth_df.date))")
    println("  Matches with Bets: $(sum(wealth_df.daily_bets .> 0))")
    println("  Total Bets Placed: $total_bets")
    println()
    println("Daily Performance:")
    println("  Winning Days: $winning_days")
    println("  Losing Days: $losing_days") 
    println("  Break-even Days: $break_even_days")
    println("  Win Rate: $(round(winning_days/(winning_days+losing_days)*100, digits=1))%")
    println()
    println("Risk Metrics:")
    println("  Average Daily ROI: \$$(round(avg_daily_roi, digits=3))")
    println("  Best Day: +\$$(round(max_daily_gain, digits=2))")
    println("  Worst Day: \$$(round(max_daily_loss, digits=2))")
    println("  Maximum Drawdown: $(round(max_drawdown, digits=2))%")
    println("  Volatility (Daily): $(round(volatility, digits=3))")
    
    return Dict(
        :final_wealth => final_wealth,
        :total_roi => total_roi,
        :total_return_pct => total_return_pct,
        :winning_days => winning_days,
        :losing_days => losing_days,
        :total_bets => total_bets,
        :max_drawdown => max_drawdown,
        :volatility => volatility
    )
end

function run_wealth_simulation(target_matches, matches_bets, starting_wealth::Float64 = 100.0)
    println("Running wealth simulation...")
    
    # Generate wealth evolution data
    wealth_df = simulate_cumulative_wealth(target_matches, matches_bets, starting_wealth)
    
    # Create plots
    wealth_plot = plot_wealth_evolution(wealth_df, starting_wealth)
    
    # Print summary statistics
    summary_stats = wealth_summary_stats(wealth_df, starting_wealth)
    
    # Additional: Plot wealth evolution with different perspectives
    p_monthly = plot_monthly_performance(wealth_df)
    
    return wealth_df, wealth_plot, summary_stats, p_monthly
end

function plot_monthly_performance(wealth_df)
    # Group by month and calculate monthly returns
    wealth_df_copy = copy(wealth_df)
    wealth_df_copy.year_month = Dates.format.(wealth_df_copy.date, "yyyy-mm")
    
    monthly_data = combine(groupby(wealth_df_copy, :year_month)) do group
        monthly_roi = sum(group.daily_roi)
        start_wealth = first(group.cumulative_wealth) - first(group.daily_roi)
        end_wealth = last(group.cumulative_wealth)
        monthly_return_pct = (monthly_roi / start_wealth) * 100
        
        return (
            start_date = minimum(group.date),
            end_date = maximum(group.date),
            monthly_roi = monthly_roi,
            monthly_return_pct = monthly_return_pct,
            num_bets = sum(group.daily_bets)
        )
    end
    
    # Create monthly performance bar chart
    colors = [roi > 0 ? :green : :red for roi in monthly_data.monthly_roi]
    
    p_monthly = bar(1:nrow(monthly_data), monthly_data.monthly_roi,
                   title="Monthly ROI Performance",
                   xlabel="Month",
                   ylabel="Monthly ROI (\$)",
                   color=colors,
                   alpha=0.7,
                   grid=true,
                   xticks=(1:nrow(monthly_data), monthly_data.year_month),
                   xrotation=45)
    
    hline!(p_monthly, [0], color=:black, linestyle=:dash, linewidth=1)
    
    return p_monthly
end

# First, get your bets for a specific threshold (e.g., 0.6)
matches_bets = process_bets_1x2(matches_scorelines, matches_probs, 
                               matches_market_odds, 0.05)

# Run the complete wealth simulation
starting_amount = 100.0  # Starting with $100
wealth_df, wealth_plot, summary_stats, monthly_plot = run_wealth_simulation(
    target_matches, matches_bets, starting_amount)

# Display the plots
display(wealth_plot)
display(monthly_plot)

# The wealth_df DataFrame contains all the detailed data:
println(first(wealth_df, 10))  # Show first 10 rows


#### Compare multiple thresholds
thresholds = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
wealth_plots = []

for threshold in thresholds
    bets = process_bets_1x2(matches_scorelines, matches_probs, 
                           matches_market_odds, threshold)
    wealth_data = simulate_cumulative_wealth(target_matches, bets, 100.0)
    
    p = plot!(wealth_data.date, wealth_data.cumulative_wealth,
              label="Threshold $threshold",
              linewidth=2)
    push!(wealth_plots, p)
end

# Plot all thresholds together
plot(title="Wealth Evolution Comparison",
     xlabel="Date", 
     ylabel="Cumulative Wealth",
     legend=:topleft)

for (i, threshold) in enumerate(thresholds)
    bets = process_bets_1x2(matches_scorelines, matches_probs, 
                           matches_market_odds, threshold)
    wealth_data = simulate_cumulative_wealth(target_matches, bets, 100.0)
    plot!(wealth_data.date, wealth_data.cumulative_wealth,
          label="Threshold $threshold",
          linewidth=2)
end

hline!([100], color=:red, linestyle=:dash, label="Starting Wealth")
display(current())
