using BayesianFootball

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
match_id = 12476824
m1::MatchPredict = matches_results[match_id]

# 1x2 lines 
ft = get_1x2_odds(m1.ft)
ht = get_1x2_odds(m1.ht)
p_1x2 = get_1x2_odds(m1)

sort(filter(row -> row.sofa_match_id==match_id && row.minutes ≥ 0 && row.minutes ≤ 5, data_store.odds)[:, [:minutes, :ht_home, :ht_draw, :ht_away, :home, :draw, :away]], :minutes)

m_score = match_score_times(match_id, data_store)
sort(m_score.score_timeline)
# get match results
#=
5×4 DataFrame
 Row │ time   is_home  home_score  away_score 
     │ Int64  Bool?    Float64?    Float64?   
─────┼────────────────────────────────────────
   1 │     6    false         0.0         1.0
   2 │    25    false         0.0         2.0
   3 │    59     true         1.0         2.0
   4 │    81     true         2.0         2.0
   5 │    90    false         2.0         3.0
=#


model_odds = get_1x2_odds(matches_results[match_id])
market_odds = get_market_odds(match_id, data_store, 0:5, m_score)
c = compare_odds(model_odds, market_odds)
print_comparison_table(c, m_score, 0:120)


##################################################
#  notes on 1x2 lines 
##################################################
# 1. BASIC BET STRUCTURE
struct Bet1x2
    match_id::Int64
    selection::Symbol  # :home, :draw, :away
    period::Symbol     # :ft or :ht
    model_prob::Float64
    model_odds::Float64
    market_odds::Float64
    ev_ratio::Float64  # market_odds / model_odds
    stake::Float64
end

# 2. BET RESULT
struct BetResult
    bet::Bet1x2
    won::Bool
    pnl::Float64  # profit/loss
    actual_result::Symbol  # what actually happened (:home, :draw, :away)
end

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

# 3. SIMPLE CONFIG
struct EvalConfig
    ev_threshold::Float64  # default 1.0
    stake::Float64         # default 1.0
    time_window::UnitRange{Int64}  # 0:5 for market odds
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
    return MatchScoreTimeline(home_goals, away_goals, sorted_timeline, final_score)
end

##############################
# funcs 
##############################

# STEP 1: Extract model probabilities (start with mean)
function get_model_probs_1x2(match::MatchPredict)
    ft_probs = (
        home = mean(match.ft.home),
        draw = mean(match.ft.draw),
        away = mean(match.ft.away)
    )
    
    ht_probs = (
        home = mean(match.ht.home),
        draw = mean(match.ht.draw),
        away = mean(match.ht.away)
    )
    
    return (ft = ft_probs, ht = ht_probs)
end

####################
# STEP 2: Find EV opportunities for a single match
####################

function find_ev_bets_1x2(match_id::Int64, 
                          model_pred::MatchPredict,
                          market_odds::MarketOdds_1x2,
                          config::EvalConfig)
    
    bets = Bet1x2[]
    model_probs = get_model_probs_1x2(model_pred)
    
    # Check FT markets
    if market_odds.ft !== nothing
        for (selection, prob) in pairs(model_probs.ft)
            model_odds = 1.0 / prob
            mkt_odds = getfield(market_odds.ft, selection)
            ev_ratio = mkt_odds / model_odds
            
            if ev_ratio >= config.ev_threshold
                push!(bets, Bet1x2(match_id, selection, :ft, 
                                  prob, model_odds, mkt_odds, 
                                  ev_ratio, config.stake))
            end
        end
    end
    
    # Repeat for HT markets...
    if market_odds.ht !== nothing
        for (selection, prob) in pairs(model_probs.ht)
            model_odds = 1.0 / prob
            mkt_odds = getfield(market_odds.ht, selection)
            ev_ratio = mkt_odds / model_odds
            
            if ev_ratio >= config.ev_threshold
                push!(bets, Bet1x2(match_id, selection, :ht, 
                                  prob, model_odds, mkt_odds, 
                                  ev_ratio, config.stake))
            end
        end
    end
    return bets
end

# STEP 3: Check outcomes
function check_bet_outcome(bet::Bet1x2, match_score::MatchScoreTimeline)
    actual_result = if bet.period == :ft
        determine_result(match_score.final_score)
    else  # :ht
        determine_ht_result(match_score)
    end
    
    won = (bet.selection == actual_result)
    pnl = won ? (bet.stake * bet.market_odds - bet.stake) : -bet.stake
    
    return BetResult(bet, won, pnl, actual_result)
end




########################################
# phase 3: simple evaluation loop
########################################

function evaluate_1x2(matches_results::Dict, 
                      data_store::DataStore,
                      config::EvalConfig = EvalConfig(1.0, 1.0, 0:5))
    
    all_bets = Bet1x2[]
    all_results = BetResult[]
    
    for (match_id, model_pred) in matches_results
        # Get market odds
        market_odds = get_market_odds(match_id, data_store, config.time_window)
        
        # Find EV bets
        bets = find_ev_bets_1x2(match_id, model_pred, market_odds, config)
        append!(all_bets, bets)
        
        # Get match outcome
        match_score = match_score_times(match_id, data_store)
        
        # Check results
        for bet in bets
            result = check_bet_outcome(bet, match_score)
            push!(all_results, result)
        end
    end
    
    return (bets = all_bets, results = all_results)
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
        return Odds1x2(home, draw, away)  # All present
    end
    
    # Estimate the missing one using implied probabilities
    if ismissing(home)
        prob_draw = 1 / draw
        prob_away = 1 / away
        prob_home = 1 - prob_draw - prob_away
        home_est = prob_home > 0 ? 1 / prob_home : 999.0  # Cap at high odds
        return Odds1x2(home_est, draw, away)
    elseif ismissing(draw)
        prob_home = 1 / home
        prob_away = 1 / away
        prob_draw = 1 - prob_home - prob_away
        draw_est = prob_draw > 0 ? 1 / prob_draw : 999.0
        return Odds1x2(home, draw_est, away)
    else  # ismissing(away)
        prob_home = 1 / home
        prob_draw = 1 / draw
        prob_away = 1 - prob_home - prob_draw
        away_est = prob_away > 0 ? 1 / prob_away : 999.0
        return Odds1x2(home, draw, away_est)
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
        return MarketOdds_1x2(nothing, nothing)
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
    
    return MarketOdds_1x2(ht_odds, ft_odds)
end

########################################
# phase 4: Basic Analysis
########################################
struct Performance1x2
    total_bets::Int64
    total_staked::Float64
    total_pnl::Float64
    roi::Float64
    win_rate::Float64
    by_selection::Dict  # :home, :draw, :away breakdown
    by_period::Dict     # :ft, :ht breakdown
end

function analyze_performance(results::Vector{BetResult})
    total_bets = length(results)
    total_staked = sum(r.bet.stake for r in results)
    total_pnl = sum(r.pnl for r in results)
    roi = total_pnl / total_staked
    win_rate = count(r.won for r in results) / total_bets
    
    # Break down by selection type
    by_selection = Dict()
    for sel in [:home, :draw, :away]
        sel_results = filter(r -> r.bet.selection == sel, results)
        if !isempty(sel_results)
            by_selection[sel] = (
                count = length(sel_results),
                pnl = sum(r.pnl for r in sel_results),
                win_rate = count(r.won for r in sel_results) / length(sel_results)
            )
        end
    end
    
    return Performance1x2(total_bets, total_staked, total_pnl, 
                         roi, win_rate, by_selection, Dict())
end
matches_results = predict_target_season(target_matches, result, mapping)
config = EvalConfig(1.0, 1.0, 0:5)  # EV threshold = 1.0


b1 = get_model_probs_1x2(m1)
market_odds = get_market_odds(match_id, data_store, config.time_window)
bet  = find_ev_bets_1x2(match_id, m1, market_odds, config)
check_bet_outcome(bet[1], m_score.score_timeline)

m_score = match_score_times(match_id, data_store)
evaluation = evaluate_1x2(matches_results, data_store, config)



###### 
# v2 
#####


# ===== STRUCTS =====
struct Bet1x2
    match_id::Int64
    selection::Symbol  # :home, :draw, :away
    period::Symbol     # :ft or :ht
    model_prob::Float64
    model_odds::Float64
    market_odds::Float64
    ev_ratio::Float64  # market_odds / model_odds
    stake::Float64
end

struct BetResult
    bet::Bet1x2
    won::Bool
    pnl::Float64  # profit/loss
    actual_result::Symbol  # what actually happened
end

struct EvalConfig
    ev_threshold::Float64  # default 1.0
    stake::Float64         # default 1.0
    time_window::UnitRange{Int64}  # 0:5 for market odds
end

# ===== COMPLETE WORKING FUNCTIONS =====

function get_model_probs_1x2(match::MatchPredict)
    ft_probs = (
        home = mean(match.ft.home),
        draw = mean(match.ft.draw),
        away = mean(match.ft.away)
    )
    
    ht_probs = (
        home = mean(match.ht.home),
        draw = mean(match.ht.draw),
        away = mean(match.ht.away)
    )
    
    return (ft = ft_probs, ht = ht_probs)
end

function find_ev_bets_1x2(match_id::Int64, 
                          model_pred::MatchPredict,
                          market_odds::Eval.MarketOdds_1x2,
                          config::EvalConfig)
    
    bets = Bet1x2[]
    model_probs = get_model_probs_1x2(model_pred)
    
    # Check FT markets
    if market_odds.ft !== nothing
        for selection in [:home, :draw, :away]
            prob = getfield(model_probs.ft, selection)
            model_odds = 1.0 / prob
            mkt_odds = getfield(market_odds.ft, selection)
            ev_ratio = mkt_odds / model_odds
            
            if ev_ratio >= config.ev_threshold
                push!(bets, Bet1x2(match_id, selection, :ft, 
                                  prob, model_odds, mkt_odds, 
                                  ev_ratio, config.stake))
            end
        end
    end
    
    # Check HT markets
    if market_odds.ht !== nothing
        for selection in [:home, :draw, :away]
            prob = getfield(model_probs.ht, selection)
            model_odds = 1.0 / prob
            mkt_odds = getfield(market_odds.ht, selection)
            ev_ratio = mkt_odds / model_odds
            
            if ev_ratio >= config.ev_threshold
                push!(bets, Bet1x2(match_id, selection, :ht, 
                                  prob, model_odds, mkt_odds, 
                                  ev_ratio, config.stake))
            end
        end
    end
    
    return bets
end

# Use a new name to avoid the old definition
function check_bet_outcome_v2(bet::Bet1x2, match_score)
    # Determine actual result based on the period
    actual_result = if bet.period == :ft
        # Full time result
        score = match_score.final_score
        if score[1] > score[2]
            :home
        elseif score[1] < score[2]
            :away
        else
            :draw
        end
    else  # :ht
        # Half time result - find score at minute 45 or last goal before that
        ht_minutes = filter(m -> m <= 45, keys(match_score.score_timeline))
        
        if isempty(ht_minutes)
            :draw  # 0-0 at half time
        else
            last_ht_minute = maximum(ht_minutes)
            ht_score = match_score.score_timeline[last_ht_minute]
            if ht_score[1] > ht_score[2]
                :home
            elseif ht_score[1] < ht_score[2]
                :away
            else
                :draw
            end
        end
    end
    
    # Check if bet won
    won = (bet.selection == actual_result)
    
    # Calculate profit/loss
    pnl = won ? (bet.stake * bet.market_odds - bet.stake) : -bet.stake
    
    return BetResult(bet, won, pnl, actual_result)
end

# Update evaluate function to use the new name
function evaluate_1x2_v2(matches_results::Dict, 
                      data_store::DataStore,
                      config::EvalConfig = EvalConfig(1.0, 1.0, 0:5))
    
    all_bets = Bet1x2[]
    all_results = BetResult[]
    
    for (match_id, model_pred) in matches_results
        # Get market odds using your existing function
        market_odds = get_market_odds(match_id, data_store, config.time_window)
        
        # Find EV bets
        bets = find_ev_bets_1x2(match_id, model_pred, market_odds, config)
        append!(all_bets, bets)
        
        # Get match outcome using your existing function
        match_score = match_score_times(match_id, data_store)
        
        # Check results  
        for bet in bets
            result = check_bet_outcome_v2(bet, match_score)  # Use v2 here
            push!(all_results, result)
        end
    end
    
    return (bets = all_bets, results = all_results)
end

function analyze_performance(results::Vector{BetResult})
    if isempty(results)
        println("No bets placed!")
        return nothing
    end
    
    total_bets = length(results)
    total_staked = sum(r.bet.stake for r in results)
    total_pnl = sum(r.pnl for r in results)
    roi = total_pnl / total_staked
    win_rate = count(r.won for r in results) / total_bets
    
    println("="^50)
    println("PERFORMANCE SUMMARY")
    println("="^50)
    println("Total Bets: $total_bets")
    println("Total Staked: $(round(total_staked, digits=2))")
    println("Total P&L: $(round(total_pnl, digits=2))")
    println("ROI: $(round(roi * 100, digits=2))%")
    println("Win Rate: $(round(win_rate * 100, digits=2))%")
    println()
    
    # Break down by selection
    println("BY SELECTION:")
    for sel in [:home, :draw, :away]
        sel_results = filter(r -> r.bet.selection == sel, results)
        if !isempty(sel_results)
            sel_pnl = sum(r.pnl for r in sel_results)
            sel_wr = count(r.won for r in sel_results) / length(sel_results)
            println("  $sel: $(length(sel_results)) bets, P&L: $(round(sel_pnl, digits=2)), WR: $(round(sel_wr*100, digits=1))%")
        end
    end
    
    # Break down by period
    println("\nBY PERIOD:")
    for period in [:ft, :ht]
        period_results = filter(r -> r.bet.period == period, results)
        if !isempty(period_results)
            period_pnl = sum(r.pnl for r in period_results)
            period_wr = count(r.won for r in period_results) / length(period_results)
            println("  $period: $(length(period_results)) bets, P&L: $(round(period_pnl, digits=2)), WR: $(round(period_wr*100, digits=1))%")
        end
    end
    
    return (roi = roi, win_rate = win_rate, total_pnl = total_pnl)
end



config = EvalConfig(1.2, 1.0, 0:5)  # EV threshold = 1.0, stake = 1.0
evaluation = evaluate_1x2_v2(matches_results, data_store, config)

performance = analyze_performance(evaluation.results)


function analyze_performance_detailed(results::Vector{BetResult})
    if isempty(results)
        println("No bets placed!")
        return nothing
    end
    
    total_bets = length(results)
    total_staked = sum(r.bet.stake for r in results)
    total_pnl = sum(r.pnl for r in results)
    roi = total_pnl / total_staked
    win_rate = count(r.won for r in results) / total_bets
    
    # Per-bet metrics
    avg_stake = total_staked / total_bets
    avg_pnl_per_bet = total_pnl / total_bets
    avg_odds_when_won = mean([r.bet.market_odds for r in results if r.won])
    
    println("="^60)
    println("DETAILED PERFORMANCE ANALYSIS")
    println("="^60)
    println("OVERALL METRICS:")
    println("  Total Bets: $total_bets")
    println("  Total Staked: $(round(total_staked, digits=2))")
    println("  Total P&L: $(round(total_pnl, digits=2))")
    println("  Overall ROI: $(round(roi * 100, digits=2))%")
    println("  Win Rate: $(round(win_rate * 100, digits=2))%")
    println()
    println("PER-BET METRICS:")
    println("  Average P&L per bet: $(round(avg_pnl_per_bet, digits=3))")
    println("  Average odds when won: $(round(avg_odds_when_won, digits=2))")
    println("  Expected value per bet: $(round(avg_pnl_per_bet/avg_stake * 100, digits=2))%")
    
    return (
        roi = roi, 
        win_rate = win_rate, 
        total_pnl = total_pnl,
        avg_pnl_per_bet = avg_pnl_per_bet,
        ev_per_bet = avg_pnl_per_bet/avg_stake
    )
end

# Run detailed analysis
detailed = analyze_performance_detailed(evaluation.results)



using Printf

function test_ev_thresholds(matches_results, data_store, thresholds=[0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    results = []
    
    for threshold in thresholds
        config = EvalConfig(threshold, 1.0, 0:5)
        eval = evaluate_1x2_v2(matches_results, data_store, config)
        perf = analyze_performance(eval.results)
        
        if perf !== nothing
            push!(results, (
                threshold = threshold,
                num_bets = length(eval.results),
                roi = perf.roi,
                win_rate = perf.win_rate,
                total_pnl = perf.total_pnl
            ))
        end
    end
    
    println("\n" * "="^60)
    println("EV THRESHOLD ANALYSIS")
    println("="^60)
    println("Threshold | Bets | ROI % | Win % | P&L")
    println("-"^60)
    for r in results
        @printf "%.2f      | %4d | %6.2f | %5.2f | %7.2f\n" r.threshold r.num_bets r.roi*100 r.win_rate*100 r.total_pnl
    end
    
    return results
end

threshold_results = test_ev_thresholds(matches_results, data_store)


function analyze_ev_distribution(evaluation)
    if isempty(evaluation.bets)
        return nothing
    end
    
    # Group by EV buckets
    ev_buckets = Dict(
        "1.0-1.1" => filter(b -> 1.0 <= b.ev_ratio < 1.1, evaluation.bets),
        "1.1-1.2" => filter(b -> 1.1 <= b.ev_ratio < 1.2, evaluation.bets),
        "1.2-1.3" => filter(b -> 1.2 <= b.ev_ratio < 1.3, evaluation.bets),
        "1.3-1.5" => filter(b -> 1.3 <= b.ev_ratio < 1.5, evaluation.bets),
        "1.5+" => filter(b -> b.ev_ratio >= 1.5, evaluation.bets)
    )
    
    println("\n" * "="^60)
    println("EV RATIO DISTRIBUTION")
    println("="^60)
    
    for (bucket, bets) in sort(collect(ev_buckets))
        if !isempty(bets)
            # Find results for these bets
            bet_ids = Set([(b.match_id, b.selection, b.period) for b in bets])
            matching_results = filter(r -> (r.bet.match_id, r.bet.selection, r.bet.period) in bet_ids, evaluation.results)
            
            if !isempty(matching_results)
                wins = count(r.won for r in matching_results)
                pnl = sum(r.pnl for r in matching_results)
                roi = pnl / length(matching_results)
                
                println("$bucket: $(length(bets)) bets, $(wins) wins, ROI: $(round(roi*100, digits=2))%")
            end
        end
    end
end

# Run it
analyze_ev_distribution(evaluation)


using Plots

function plot_cumulative_wealth(evaluation, initial_wealth=100.0)
    if isempty(evaluation.results)
        println("No results to plot")
        return
    end
    
    # Sort results by match_id (as proxy for time)
    sorted_results = sort(evaluation.results, by=r->r.bet.match_id)
    
    # Calculate cumulative P&L for all bets
    cumsum_all = cumsum([r.pnl for r in sorted_results])
    wealth_all = initial_wealth .+ cumsum_all
    
    # Separate FT and HT results
    ft_results = filter(r -> r.bet.period == :ft, sorted_results)
    ht_results = filter(r -> r.bet.period == :ht, sorted_results)
    
    # Calculate cumulative P&L for FT and HT
    cumsum_ft = cumsum([r.pnl for r in ft_results])
    cumsum_ht = cumsum([r.pnl for r in ht_results])
    
    wealth_ft = initial_wealth .+ cumsum_ft
    wealth_ht = initial_wealth .+ cumsum_ht
    
    # Create the plot
    p = plot(
        title="Cumulative Wealth Over Bets",
        xlabel="Bet Number",
        ylabel="Wealth",
        legend=:topleft,
        size=(1000, 600),
        grid=true,
        linewidth=2
    )
    
    # Plot total wealth
    plot!(p, 1:length(wealth_all), wealth_all, 
          label="Total", color=:blue, linewidth=2.5)
    
    # Plot FT wealth
    plot!(p, 1:length(wealth_ft), wealth_ft, 
          label="Full Time Only", color=:green, linewidth=1.5, linestyle=:dash)
    
    # Plot HT wealth  
    plot!(p, 1:length(wealth_ht), wealth_ht,
          label="Half Time Only", color=:orange, linewidth=1.5, linestyle=:dash)
    
    # Add initial wealth line
    hline!(p, [initial_wealth], label="Initial Wealth", 
           color=:red, linestyle=:dot, linewidth=1)
    
    # Calculate and display key metrics - FIXED THIS PART
    final_wealth = wealth_all[end]
    max_wealth = maximum(wealth_all)
    min_wealth = minimum(wealth_all)
    
    # Correct max drawdown calculation
    running_max = accumulate(max, wealth_all)
    drawdowns = running_max .- wealth_all
    max_drawdown = maximum(drawdowns)
    
    # Add text box with metrics
    annotate!(p, 
        length(wealth_all) * 0.7, 
        initial_wealth + (max_wealth - initial_wealth) * 0.2,
        text(
            "Final: $(round(final_wealth, digits=2))\n" *
            "Max DD: $(round(max_drawdown, digits=2))\n" *
            "ROI: $(round((final_wealth/initial_wealth - 1)*100, digits=2))%",
            :left, 10
        )
    )
    
    return p
end


p = plot_cumulative_wealth(evaluation, 100.0)


function create_wealth_animation(evaluation, initial_wealth=100.0)
    sorted_results = sort(evaluation.results, by=r->r.bet.match_id)
    cumsum_pnl = cumsum([r.pnl for r in sorted_results])
    wealth = initial_wealth .+ cumsum_pnl
    
    anim = @animate for i in 1:length(wealth)
        plot(1:i, wealth[1:i], 
             title="Wealth Evolution (Bet $i of $(length(wealth)))",
             xlabel="Bet Number",
             ylabel="Wealth",
             ylim=(minimum(wealth)*0.95, maximum(wealth)*1.05),
             xlim=(1, length(wealth)),
             linewidth=2,
             legend=false,
             grid=true)
        hline!([initial_wealth], color=:red, linestyle=:dot)
        
        # Add marker for current bet
        if sorted_results[i].won
            scatter!([i], [wealth[i]], color=:green, markersize=6)
        else
            scatter!([i], [wealth[i]], color=:red, markersize=4)
        end
    end
    
    return anim
end


anim = create_wealth_animation(evaluation, 100.0)

gif(anim, "wealth_evolution.gif", fps=10)


function analyze_hedging_opportunities(evaluation, data_store)
    hedge_scenarios = []
    
    for result in evaluation.results
        match_score = match_score_times(result.bet.match_id, data_store)
        
        if !isempty(match_score.score_timeline)
            # Check if our selection scored first or took the lead
            for (minute, score) in match_score.score_timeline
                if minute <= 70  # Reasonable hedging window
                    leader = score[1] > score[2] ? :home : (score[2] > score[1] ? :away : :draw)
                    
                    if leader == result.bet.selection && !result.won
                        # We had the lead but lost - hedging opportunity
                        push!(hedge_scenarios, (
                            bet = result.bet,
                            lead_minute = minute,
                            final_result = result.actual_result,
                            potential_hedge_profit = result.bet.stake * 0.5  # Conservative estimate
                        ))
                    end
                end
            end
        end
    end
    
    println("\nHEDGING OPPORTUNITY ANALYSIS")
    println("="^50)
    println("Total losing bets: $(count(!r.won for r in evaluation.results))")
    println("Hedgeable losses: $(length(hedge_scenarios))")
    println("Hedge rate: $(round(length(hedge_scenarios)/count(!r.won for r in evaluation.results)*100, digits=1))%")
    
    potential_hedge_recovery = sum(h.potential_hedge_profit for h in hedge_scenarios)
    current_losses = sum(r.pnl for r in evaluation.results if !r.won)
    
    println("\nFinancial Impact:")
    println("Current losses: $(round(current_losses, digits=2))")
    println("Potential hedge recovery: $(round(potential_hedge_recovery, digits=2))")
    println("New losses with hedging: $(round(current_losses + potential_hedge_recovery, digits=2))")
    
    # New metrics with hedging
    total_pnl_with_hedge = sum(r.pnl for r in evaluation.results) + potential_hedge_recovery
    new_roi = total_pnl_with_hedge / length(evaluation.results)
    
    println("\nROI Impact:")
    println("Current ROI: $(round(sum(r.pnl for r in evaluation.results)/length(evaluation.results)*100, digits=2))%")
    println("ROI with hedging: $(round(new_roi*100, digits=2))%")
    
    return hedge_scenarios
end


analyze_hedging_opportunities(evaluation, data_store)

function analyze_partial_hedging(evaluation, data_store, hedge_ratios=[0.0, 0.25, 0.5, 0.75, 1.0])
    # First identify all hedgeable scenarios
    hedge_scenarios = []
    
    for result in evaluation.results
        if !result.won  # Only look at losses
            match_score = match_score_times(result.bet.match_id, data_store)
            
            if !isempty(match_score.score_timeline)
                for (minute, score) in match_score.score_timeline
                    if minute <= 70
                        leader = score[1] > score[2] ? :home : (score[2] > score[1] ? :away : :draw)
                        
                        if leader == result.bet.selection
                            # Estimate realistic hedge odds based on minute and score
                            hedge_odds_reduction = 0.3 + (minute / 90) * 0.4  # Odds drop 30-70%
                            implied_hedge_return = result.bet.market_odds * (1 - hedge_odds_reduction) - 1
                            
                            push!(hedge_scenarios, (
                                original_bet = result.bet,
                                lead_minute = minute,
                                score_when_leading = score,
                                implied_hedge_return = implied_hedge_return,
                                original_loss = result.pnl
                            ))
                            break  # Only count first lead
                        end
                    end
                end
            end
        end
    end
    
    results_by_ratio = []
    
    for h in hedge_ratios
        if h == 0
            # No hedging - current situation
            total_pnl = sum(r.pnl for r in evaluation.results)
            variance = var([r.pnl for r in evaluation.results])
        else
            # Calculate P&L with partial hedging
            modified_pnl = []
            
            for result in evaluation.results
                # Check if this bet had a hedge opportunity
                hedge_opp = findfirst(hs -> hs.original_bet.match_id == result.bet.match_id && 
                                           hs.original_bet.selection == result.bet.selection &&
                                           hs.original_bet.period == result.bet.period, 
                                     hedge_scenarios)
                
                if hedge_opp !== nothing && !result.won
                    hs = hedge_scenarios[hedge_opp]
                    # Partial hedge calculation
                    hedge_stake = result.bet.stake * h
                    hedge_cost = -hedge_stake  # Cost of placing hedge bet
                    hedge_return = hedge_stake * hs.implied_hedge_return  # Return if hedge wins
                    
                    # Net P&L = original loss + hedge profit
                    net_pnl = result.pnl * (1 - h) + hedge_return
                    push!(modified_pnl, net_pnl)
                else
                    push!(modified_pnl, result.pnl)
                end
            end
            
            total_pnl = sum(modified_pnl)
            variance = var(modified_pnl)
        end
        
        roi = total_pnl / length(evaluation.results)
        sharpe = roi / sqrt(variance) * sqrt(length(evaluation.results))
        
        push!(results_by_ratio, (
            h = h,
            total_pnl = total_pnl,
            roi = roi,
            variance = variance,
            sharpe = sharpe,
            hedged_bets = h > 0 ? length(hedge_scenarios) : 0
        ))
    end
    
    println("\nPARTIAL HEDGING ANALYSIS")
    println("="^70)
    println("Hedge % | Total P&L |   ROI   | Variance | Sharpe | Hedged Bets")
    println("-"^70)
    
    for r in results_by_ratio
      formatted = Printf.format(Printf.Format("%6.0f%% | %9.2f | %7.2f%% | %8.2f | %6.3f | %11d\n"), 
                                r.h*100, r.total_pnl, r.roi*100, r.variance, r.sharpe, r.hedged_bets)
      print(formatted)
    end
    
    return results_by_ratio
end

# Run analysis
partial_results = analyze_partial_hedging(evaluation, data_store)
