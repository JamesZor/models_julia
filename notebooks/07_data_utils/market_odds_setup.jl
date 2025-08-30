using BayesianFootball
using DataFrames 

module Odds
  const MaybeOdds = Union{Nothing, Float64}
  # Type aliases matching Results module
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, Float64}
  
  struct MatchHTOdds
    home::MaybeOdds
    draw::MaybeOdds
    away::MaybeOdds
    correct_score::CorrectScore
    under_05::MaybeOdds
    over_05::MaybeOdds
    under_15::MaybeOdds
    over_15::MaybeOdds
    under_25::MaybeOdds
    over_25::MaybeOdds
  end
  
  struct MatchFTOdds
    home::MaybeOdds
    draw::MaybeOdds
    away::MaybeOdds
    correct_score::CorrectScore
    under_05::MaybeOdds
    over_05::MaybeOdds
    under_15::MaybeOdds
    over_15::MaybeOdds
    under_25::MaybeOdds
    over_25::MaybeOdds
    under_35::MaybeOdds
    over_35::MaybeOdds
    btts_yes::MaybeOdds
    btts_no::MaybeOdds
  end
  
  struct MatchLineOdds
    ht::MatchHTOdds
    ft::MatchFTOdds
  end
end

MARKET_GROUPS = Dict(
        "ft_1x2" => ["home", "draw", "away"],
        "ht_1x2" => ["ht_home", "ht_draw", "ht_away"],
        "ft_ou_05" => ["over_0_5", "under_0_5"],
        "ft_ou_15" => ["over_1_5", "under_1_5"],
        "ft_ou_25" => ["over_2_5", "under_2_5"],
        "ft_ou_35" => ["over_3_5", "under_3_5"],
        "ht_ou_05" => ["ht_over_0_5", "ht_under_0_5"],
        "ht_ou_15" => ["ht_over_1_5", "ht_under_1_5"],
        "ht_ou_25" => ["ht_over_2_5", "ht_under_2_5"],
        "btts" => ["btts_yes", "btts_no"],
        "ft_cs" => ["0_0", "0_1", "0_2", "0_3", "1_1", "1_2", "1_3","2_1", "2_2", "2_3","3_1", "3_2", "3_3","1_0", "2_0", "3_0", "any_other_home", "any_other_away", "any_other_draw"], 
        "ht_cs" => ["ht_0_0", "ht_0_1", "ht_0_2", "ht_1_1", "ht_1_2","ht_2_1", "ht_2_2","ht_1_0", "ht_2_0", "ht_any_unquoted"], 
       )

"""
    get_game_line_odds(data_store::DataStore, match_id::Int)
    
Extracts the kickoff odds (at minutes=0) for a given match from the odds DataFrame.
Returns Odds.MatchLineOdds or nothing if not found.
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
    ht_odds = Odds.MatchHTOdds(
        get_val("ht_home"),
        get_val("ht_draw"),
        get_val("ht_away"),
        ht_correct_scores,
        get_val("ht_under_0_5"),  # Note: under_05 = 1/over_05 if not directly available
        get_val("ht_over_0_5"),
        get_val("ht_under_1_5"),
        get_val("ht_over_1_5"),
        get_val("ht_under_2_5"),
        get_val("ht_over_2_5"),
    )
    
    # Build FT odds
    ft_odds = Odds.MatchFTOdds(
        get_val("home"),
        get_val("draw"),
        get_val("away"),
        ft_correct_scores,
        get_val("under_0_5"),
        get_val("over_0_5"),
        get_val("under_1_5"),
        get_val("over_1_5"),
        get_val("under_2_5"),
        get_val("over_2_5"),
        get_val("under_3_5"),
        get_val("over_3_5"),
        get_val("btts_yes"),
        get_val("btts_no")
    )
    
    return Odds.MatchLineOdds(ht_odds, ft_odds)
end

"""
    process_matches_odds(data_store::DataStore, target_matches::DataFrame)
    
Process odds for multiple matches.
Returns Dict{Int64, Odds.MatchLineOdds}
"""
function process_matches_odds(data_store::DataStore, target_matches::DataFrame)
    results = Dict{Int64, Odds.MatchLineOdds}()
    
    for match_id in target_matches.match_id
        odds = get_game_line_odds(data_store, match_id)
        if !isnothing(odds)
            results[match_id] = odds
        end
    end
    
    return results
end


##################################################
#  Odds Processing Pipeline Architecture
##################################################


using DataFrames
using Statistics
using StatsBase

# ============================================
# Core Types and Configuration
# ============================================

"""
Configuration for odds processing pipeline
"""
module OddsProcessing
  struct OddsConfig
      # Window parameters
      window_minutes_before::Int
      window_minutes_after::Int
      min_window_size::Int  # Minimum minutes needed for valid window
      
      # Aggregation method
      aggregation::Symbol  # :mean, :median, :weighted_mean, :last_valid
      outlier_threshold::Float64  # For filtering jumps (e.g., 2.0 = 100% change)
      
      # Market completion
      complete_missing::Bool
      completion_method::Symbol  # :probability_sum, :similar_markets, :regression
      
      # Normalization
      normalize_groups::Bool
      target_overround::Float64  # e.g., 1.05 for 5% overround
      normalization_method::Symbol  # :proportional, :power, :margin_weights
      
      # Validation
      min_liquidity_threshold::Float64  # Minimum implied probability sum to consider valid
      max_staleness_minutes::Int  # Max minutes since last trade to consider valid
      
      # Market groups definition
      market_groups::Dict{String, Vector{String}}
  end
end # module

"""
Default configuration for typical use case
"""
function default_config()
    market_groups = Dict(
        "ft_1x2" => ["home", "draw", "away"],
        "ht_1x2" => ["ht_home", "ht_draw", "ht_away"],
        "ft_ou_0.5" => ["over_0_5", "under_0_5"],
        "ft_ou_1.5" => ["over_1_5", "under_1_5"],
        "ft_ou_2.5" => ["over_2_5", "under_2_5"],
        "ft_ou_3.5" => ["over_3_5", "under_3_5"],
        "ht_ou_0.5" => ["ht_over_0_5", "ht_under_0_5"],
        "ht_ou_1.5" => ["ht_over_1_5", "ht_under_1_5"],
        "ht_ou_2.5" => ["ht_over_2_5", "ht_under_2_5"],
        "btts" => ["btts_yes", "btts_no"],
        "ft_cs" => ["0_0", "0_1", "0_2", "0_3", "1_0", "1_1", "1_2", "1_3",
                    "2_0", "2_1", "2_2", "2_3", "3_0", "3_1", "3_2", "3_3",
                    "any_other_home", "any_other_away", "any_other_draw"],
        "ht_cs" => ["ht_0_0", "ht_0_1", "ht_0_2", "ht_1_0", "ht_1_1", "ht_1_2",
                    "ht_2_0", "ht_2_1", "ht_2_2", "ht_any_unquoted"]
    )
    
    return OddsProcessing.OddsConfig(
        10, 4, 3,  # Window: 5 mins before/after, min 3 mins
        :weighted_mean, 1.4,  # Aggregation with 100% jump filter
        true, :probability_sum,  # Complete missing via probability
        true, 1.001, :proportional,  # Normalize to 1% overround
        0.8, 10,  # Min 80% liquidity, max 10 min staleness
        market_groups
    )
end

# ============================================
# Pipeline Components (Composable Functions)
# ============================================

"""
Filter odds data to relevant time window, accounting for goals
"""
function apply_time_window(odds_df::DataFrame, incidents_df::DataFrame, 
                          match_id::Int, target_minute::Int, config::OddsProcessing.OddsConfig)
    # Get match odds
    match_odds = filter(row -> row.sofa_match_id == match_id, odds_df)
    
    if isempty(match_odds)
        return DataFrame()
    end
    
    # Check for goals in the window
    match_incidents = filter(row -> row.match_id == match_id, incidents_df)
    goals = filter(row -> row.incident_type == "goal", match_incidents)
    
    # Adjust window if goals occurred
    window_start = target_minute - config.window_minutes_before
    window_end = target_minute + config.window_minutes_after
    
    for goal in eachrow(goals)
        goal_time = goal.time
        if window_start <= goal_time <= window_end
            # Shrink window to exclude goal
            if goal_time < target_minute
                window_start = goal_time + 1
            else
                window_end = goal_time - 1
            end
        end
    end
    
    # Check minimum window size
    if window_end - window_start < config.min_window_size
        @warn "Window too small after goal adjustment for match $match_id at minute $target_minute"
        return DataFrame()
    end
    
    # Filter to window
    return filter(row -> window_start <= row.minutes <= window_end, match_odds)
end

"""
Remove outliers based on price jumps between consecutive observations
"""
function filter_outliers(odds_df::DataFrame, columns::Vector{String}, config::OddsProcessing.OddsConfig)
    if nrow(odds_df) <= 1
        return odds_df
    end
    
    odds_sorted = sort(odds_df, :minutes)
    valid_rows = trues(nrow(odds_sorted))
    
    for col in columns
        if !hasproperty(odds_sorted, Symbol(col))
            continue
        end
        
        values = odds_sorted[!, Symbol(col)]
        for i in 2:length(values)
            if !ismissing(values[i]) && !ismissing(values[i-1])
                # Convert odds to probabilities for comparison
                prob_curr = 1.0 / values[i]
                prob_prev = 1.0 / values[i-1]
                
                # Check relative change
                if abs(prob_curr - prob_prev) / prob_prev > config.outlier_threshold
                    valid_rows[i] = false
                end
            end
        end
    end
    
    return odds_sorted[valid_rows, :]
end

"""
Aggregate odds using specified method
"""
function aggregate_odds(odds_df::DataFrame, columns::Vector{String}, config::OddsProcessing.OddsConfig)
    result = Dict{String, Float64}()
    
    for col in columns
        if !hasproperty(odds_df, Symbol(col))
            result[col] = NaN
            continue
        end
        
        values = skipmissing(odds_df[!, Symbol(col)])
        
        if isempty(values)
            result[col] = NaN
            continue
        end
        
        if config.aggregation == :mean
            result[col] = mean(values)
        elseif config.aggregation == :median
            result[col] = median(values)
        elseif config.aggregation == :weighted_mean
            # Weight by recency
            minutes = odds_df.minutes
            weights = exp.(-0.1 * (maximum(minutes) .- minutes))
            valid_idx = .!ismissing.(odds_df[!, Symbol(col)])
            result[col] = sum(values .* weights[valid_idx]) / sum(weights[valid_idx])
        elseif config.aggregation == :last_valid
            result[col] = last(values)
        else
            result[col] = mean(values)  # Default
        end
    end
    
    return result
end

"""
Complete missing odds in a market group
"""
function complete_market_group(odds_dict::Dict{String, Float64}, 
                              group_columns::Vector{String}, 
                              config::OddsProcessing.OddsConfig)
    # Count available odds
    available = [col for col in group_columns if !isnan(get(odds_dict, col, NaN))]
    missing = [col for col in group_columns if isnan(get(odds_dict, col, NaN))]
    
    if isempty(missing)
        return odds_dict
    end
    
    if config.completion_method == :probability_sum && length(missing) == 1
        # Complete single missing odd using probability sum
        available_probs = [1.0 / odds_dict[col] for col in available]
        prob_sum = sum(available_probs)
        
        if prob_sum < config.target_overround - 0.2  # Leave room for missing
            missing_prob = config.target_overround - prob_sum
            odds_dict[missing[1]] = 1.0 / missing_prob
        end
    elseif config.completion_method == :similar_markets
        # Use ratios from similar complete markets
        # This would require historical data - placeholder for now
        @warn "Similar markets completion not yet implemented"
    end
    
    return odds_dict
end

"""
Normalize market group probabilities to target overround
"""
function normalize_market_group(odds_dict::Dict{String, Float64}, 
                               group_columns::Vector{String}, 
                               config::OddsProcessing.OddsConfig)
    # Get available odds
    available = [col for col in group_columns if !isnan(get(odds_dict, col, NaN))]
    
    if length(available) < 2
        return odds_dict  # Can't normalize with less than 2 odds
    end
    
    # Calculate current probability sum
    probs = [1.0 / odds_dict[col] for col in available]
    current_sum = sum(probs)
    
    if current_sum < config.min_liquidity_threshold
        @warn "Probability sum too low: $current_sum"
        return odds_dict
    end


    if current_sum < config.target_overround
    # Apply normalization
      if config.normalization_method == :proportional
          # Simple proportional scaling
          scale_factor = config.target_overround / current_sum
          for col in available
              odds_dict[col] = odds_dict[col] / scale_factor
          end
      elseif config.normalization_method == :power
          # Power scaling (preserves favorites/longshots relationship better)
          power = log(config.target_overround) / log(current_sum)
          for col in available
              prob = 1.0 / odds_dict[col]
              new_prob = prob^power
              odds_dict[col] = 1.0 / new_prob
          end
      elseif config.normalization_method == :margin_weights
          # Weighted by distance from fair (more margin on longshots)
          fair_probs = probs ./ current_sum
          margins = (config.target_overround - 1.0) .* (1 .- fair_probs)
          new_probs = fair_probs .+ margins
          
          for (i, col) in enumerate(available)
              odds_dict[col] = 1.0 / new_probs[i]
          end
      end
    end

    
    return odds_dict
end

"""
Process correct scores as a special case
"""
function process_correct_scores(odds_dict::Dict{String, Float64}, 
                               prefix::String,  # "ft" or "ht"
                               config::OddsProcessing.OddsConfig)
    if prefix == "ht"
        scores = [(i,j) for i in 0:2 for j in 0:2]
        other_key = "ht_any_unquoted"
    else
        scores = [(i,j) for i in 0:3 for j in 0:3]
        other_keys = ["any_other_home", "any_other_away", "any_other_draw"]
    end
    
    # Collect available scores
    total_prob = 0.0
    for (h, a) in scores
        key = prefix == "ht" ? "ht_$(h)_$(a)" : "$(h)_$(a)"
        if !isnan(get(odds_dict, key, NaN))
            total_prob += 1.0 / odds_dict[key]
        end
    end
    
    # Add "other" probabilities
    if prefix == "ht"
        if !isnan(get(odds_dict, other_key, NaN))
            total_prob += 1.0 / odds_dict[other_key]
        end
    else
        for key in other_keys
            if !isnan(get(odds_dict, key, NaN))
                total_prob += 1.0 / odds_dict[key]
            end
        end
    end
    
    # Normalize if needed
    if config.normalize_groups && total_prob > config.min_liquidity_threshold
        scale = config.target_overround / total_prob
        
        for (h, a) in scores
            key = prefix == "ht" ? "ht_$(h)_$(a)" : "$(h)_$(a)"
            if !isnan(get(odds_dict, key, NaN))
                odds_dict[key] = odds_dict[key] / scale
            end
        end
        
        if prefix == "ht"
            if !isnan(get(odds_dict, other_key, NaN))
                odds_dict[other_key] = odds_dict[other_key] / scale
            end
        else
            for key in other_keys
                if !isnan(get(odds_dict, key, NaN))
                    odds_dict[key] = odds_dict[key] / scale
                end
            end
        end
    end
    
    return odds_dict
end

# ============================================
# Main Pipeline Function
# ============================================

"""
Main pipeline to process odds for a match at a specific minute
"""
function process_odds_pipeline(data_store::DataStore, 
                              match_id::Int, 
                              target_minute::Int,
                              config::OddsProcessing.OddsConfig = default_config())
    
    # Step 1: Get time window
    windowed_odds = apply_time_window(
        data_store.odds, 
        data_store.incidents,
        match_id, 
        target_minute, 
        config
    )
    
    if isempty(windowed_odds)
        @warn "No odds data in window for match $match_id at minute $target_minute"
        return nothing
    end
    
    # Step 2: Filter outliers for each market group
    all_columns = Set{String}()
    for (_, cols) in config.market_groups
        union!(all_columns, cols)
    end
    
    filtered_odds = filter_outliers(
        windowed_odds,
        collect(all_columns),
        config
    )
    
    # Step 3: Aggregate odds
    odds_dict = aggregate_odds(
        filtered_odds,
        collect(all_columns),
        config
    )
    
    # Step 4: Process each market group
    for (group_name, group_cols) in config.market_groups
        # Complete missing odds
        if config.complete_missing
            odds_dict = complete_market_group(odds_dict, group_cols, config)
        end
        
        # Normalize group
        if config.normalize_groups
            odds_dict = normalize_market_group(odds_dict, group_cols, config)
        end
    end
    
    # Step 5: Process correct scores
    odds_dict = process_correct_scores(odds_dict, "ht", config)
    odds_dict = process_correct_scores(odds_dict, "ft", config)
    
    return odds_dict
end

"""
Convert processed odds dictionary to Odds.MatchLineOdds structure
"""
function dict_to_predictions(odds_dict::Dict{String, Float64})
    # Helper to safely get values
    get_val(key) = get(odds_dict, key, NaN)
    
    # Build HT correct scores
    ht_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:2, a in 0:2
        ht_correct_scores[(h, a)] = get_val("ht_$(h)_$(a)")
    end
    ht_correct_scores["any_unquoted"] = get_val("ht_any_unquoted")
    
    # Build FT correct scores
    ft_correct_scores = Dict{Union{Tuple{Int,Int}, String}, Float64}()
    for h in 0:3, a in 0:3
        ft_correct_scores[(h, a)] = get_val("$(h)_$(a)")
    end
    ft_correct_scores["other_home_win"] = get_val("any_other_home")
    ft_correct_scores["other_away_win"] = get_val("any_other_away")
    ft_correct_scores["other_draw"] = get_val("any_other_draw")
    
    # Build predictions
    ht_pred = Odds.MatchHTOdds(
        get_val("ht_home"),
        get_val("ht_draw"),
        get_val("ht_away"),
        ht_correct_scores,
        get_val("ht_under_0_5"),
        get_val("ht_over_0_5"),
        get_val("ht_under_1_5"),
        get_val("ht_over_1_5"),
        get_val("ht_under_2_5"),
        get_val("ht_over_2_5"),
    )
    
    ft_pred = Odds.MatchFTOdds(
        get_val("home"),
        get_val("draw"),
        get_val("away"),
        ft_correct_scores,
        get_val("under_0_5"),
        get_val("over_0_5"),
        get_val("under_1_5"),
        get_val("over_1_5"),
        get_val("under_2_5"),
        get_val("over_2_5"),
        get_val("under_3_5"),
        get_val("over_3_5"),
        get_val("btts_yes"),
        get_val("btts_no")
    )
    
    return Odds.MatchLineOdds(ht_pred, ft_pred)
end

"""
Get processed game line odds (at kickoff)
"""
function get_processed_game_line_odds(data_store::DataStore, 
                                     match_id::Int,
                                     config::OddsProcessing.OddsConfig = default_config())
    odds_dict = process_odds_pipeline(data_store, match_id, 0, config)
    
    if isnothing(odds_dict)
        return nothing
    end
    
    return dict_to_predictions(odds_dict)
end

"""
Process multiple matches in parallel
"""
function process_matches_odds(data_store::DataStore, 
                             target_matches::DataFrame,
                             target_minute::Int = 0,
                             config::OddsProcessing.OddsConfig = default_config())
    match_ids = target_matches.match_id
    n_matches = length(match_ids)
    n_threads = Threads.nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, Odds.MatchLineOdds}}(undef, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        local_dict = Dict{Int64, Odds.MatchLineOdds}()
        
        for idx in tid:n_threads:n_matches
            match_id = match_ids[idx]
            try
                odds_dict = process_odds_pipeline(data_store, match_id, target_minute, config)
                if !isnothing(odds_dict)
                    local_dict[match_id] = dict_to_predictions(odds_dict)
                end
            catch e
                @warn "Failed to process odds for match $match_id" exception=e
            end
        end
        
        thread_results[tid] = local_dict
    end
    
    return merge(thread_results...)
end



##################################################################
##################################################################
##################################################################

using DataFrames
using Printf
using Statistics

##################################################
#  Odds Display and Testing Functions
##################################################

"""
Convert odds to implied probability percentage
"""
odds_to_prob(odds::Float64) = isnan(odds) || odds == 0 ? NaN : 100.0 / odds
prob_to_odds(prob::Float64) = isnan(prob) || prob == 0 ? NaN : 100.0 / prob

"""
Display a market group comparison between raw and processed odds
"""
function display_market_group(raw_odds::Union{Odds.MatchLineOdds, Nothing}, 
                            processed_odds::Union{Odds.MatchLineOdds, Nothing},
                            market_name::String;
                            show_probs::Bool = true)
    
    # Helper to extract values based on market
    function get_market_values(odds_struct, market)
        if isnothing(odds_struct)
            return Dict{String, Float64}()
        end
        
        values = Dict{String, Float64}()
        
        if market == "ft_1x2"
            values["Home"] = isnothing(odds_struct.ft.home) ? NaN : odds_struct.ft.home
            values["Draw"] = isnothing(odds_struct.ft.draw) ? NaN : odds_struct.ft.draw
            values["Away"] = isnothing(odds_struct.ft.away) ? NaN : odds_struct.ft.away
        elseif market == "ht_1x2"
            values["Home"] = isnothing(odds_struct.ht.home) ? NaN : odds_struct.ht.home
            values["Draw"] = isnothing(odds_struct.ht.draw) ? NaN : odds_struct.ht.draw
            values["Away"] = isnothing(odds_struct.ht.away) ? NaN : odds_struct.ht.away
        elseif market == "ft_ou_05"
            values["Over 0.5"] = isnothing(odds_struct.ft.over_05) ? NaN : odds_struct.ft.over_05
            values["Under 0.5"] = isnothing(odds_struct.ft.under_05) ? NaN : odds_struct.ft.under_05
        elseif market == "ft_ou_15"
            values["Over 1.5"] = isnothing(odds_struct.ft.over_15) ? NaN : odds_struct.ft.over_15
            values["Under 1.5"] = isnothing(odds_struct.ft.under_15) ? NaN : odds_struct.ft.under_15
        elseif market == "ft_ou_25"
            values["Over 2.5"] = isnothing(odds_struct.ft.over_25) ? NaN : odds_struct.ft.over_25
            values["Under 2.5"] = isnothing(odds_struct.ft.under_25) ? NaN : odds_struct.ft.under_25
        elseif market == "ft_ou_35"
            values["Over 3.5"] = isnothing(odds_struct.ft.over_35) ? NaN : odds_struct.ft.over_35
            values["Under 3.5"] = isnothing(odds_struct.ft.under_35) ? NaN : odds_struct.ft.under_35
        elseif market == "ht_ou_05"
            values["Over 0.5"] = isnothing(odds_struct.ht.over_05) ? NaN : odds_struct.ht.over_05
            values["Under 0.5"] = isnothing(odds_struct.ht.under_05) ? NaN : odds_struct.ht.under_05
        elseif market == "ht_ou_15"
            values["Over 1.5"] = isnothing(odds_struct.ht.over_15) ? NaN : odds_struct.ht.over_15
            values["Under 1.5"] = isnothing(odds_struct.ht.under_15) ? NaN : odds_struct.ht.under_15
        elseif market == "ht_ou_25"
            values["Over 2.5"] = isnothing(odds_struct.ht.over_25) ? NaN : odds_struct.ht.over_25
            values["Under 2.5"] = isnothing(odds_struct.ht.under_25) ? NaN : odds_struct.ht.under_25
        elseif market == "btts"
            values["BTTS Yes"] = isnothing(odds_struct.ft.btts_yes) ? NaN : odds_struct.ft.btts_yes
            values["BTTS No"] = isnothing(odds_struct.ft.btts_no) ? NaN : odds_struct.ft.btts_no
        elseif market == "ft_cs"
            # Full-time correct scores
            cs = odds_struct.ft.correct_score
            for h in 0:3, a in 0:3
                values["$h-$a"] = get(cs, (h,a), NaN)
            end
            values["Other Home"] = get(cs, "other_home_win", NaN)
            values["Other Away"] = get(cs, "other_away_win", NaN)
            values["Other Draw"] = get(cs, "other_draw", NaN)
        elseif market == "ht_cs"
            # Half-time correct scores
            cs = odds_struct.ht.correct_score
            for h in 0:2, a in 0:2
                values["HT $h-$a"] = get(cs, (h,a), NaN)
            end
            values["HT Other"] = get(cs, "any_unquoted", NaN)
        end
        
        return values
    end
    
    raw_values = get_market_values(raw_odds, market_name)
    proc_values = get_market_values(processed_odds, market_name)
    
    # Build DataFrame
    selections = collect(keys(raw_values))
    if isempty(selections) && !isempty(proc_values)
        selections = collect(keys(proc_values))
    end
    
    df = DataFrame(
        Selection = selections,
        Raw_Odds = [get(raw_values, sel, NaN) for sel in selections],
        Raw_Prob = [odds_to_prob(get(raw_values, sel, NaN)) for sel in selections],
        Proc_Odds = [get(proc_values, sel, NaN) for sel in selections],
        Proc_Prob = [odds_to_prob(get(proc_values, sel, NaN)) for sel in selections],
        Diff_Odds = [get(proc_values, sel, NaN) - get(raw_values, sel, NaN) for sel in selections],
        Diff_Prob = [odds_to_prob(get(proc_values, sel, NaN)) - odds_to_prob(get(raw_values, sel, NaN)) for sel in selections]
    )
    
    # Add totals row
    raw_total = sum(skipmissing(df.Raw_Prob))
    proc_total = sum(skipmissing(df.Proc_Prob))
    
    push!(df, ("TOTAL", NaN, raw_total, NaN, proc_total, NaN, proc_total - raw_total))
    
    println("\n=== $market_name ===")
    
    if show_probs
        select!(df, :Selection, :Raw_Odds, :Raw_Prob, :Proc_Odds, :Proc_Prob, :Diff_Prob)
    else
        select!(df, :Selection, :Raw_Odds, :Proc_Odds, :Diff_Odds)
    end
    
    return df
end

"""
Compare odds across multiple configurations
"""
function compare_configurations(data_store::DataStore, 
                               match_id::Int,
                               configs::Dict{String, OddsProcessing.OddsConfig})
    
    # Get raw odds
    raw_odds = get_game_line_odds(data_store, match_id)
    
    # Process with each config
    results = Dict{String, Odds.MatchLineOdds}()
    for (name, config) in configs
        processed = get_processed_game_line_odds(data_store, match_id, config)
        if !isnothing(processed)
            results[name] = processed
        end
    end
    
    # Build comparison DataFrame for main markets
    comparison_data = []
    
    # FT 1X2
    for (config_name, odds) in results
        push!(comparison_data, (
            Config = config_name,
            Market = "FT 1X2",
            Home = odds.ft.home,
            Draw = odds.ft.draw,
            Away = odds.ft.away,
            Total_Prob = odds_to_prob(odds.ft.home) + odds_to_prob(odds.ft.draw) + odds_to_prob(odds.ft.away),
            Overround = (odds_to_prob(odds.ft.home) + odds_to_prob(odds.ft.draw) + odds_to_prob(odds.ft.away)) - 100.0
        ))
    end
    
    # Add raw odds
    if !isnothing(raw_odds)
        push!(comparison_data, (
            Config = "RAW",
            Market = "FT 1X2",
            Home = raw_odds.ft.home,
            Draw = raw_odds.ft.draw,
            Away = raw_odds.ft.away,
            Total_Prob = odds_to_prob(raw_odds.ft.home) + odds_to_prob(raw_odds.ft.draw) + odds_to_prob(raw_odds.ft.away),
            Overround = (odds_to_prob(raw_odds.ft.home) + odds_to_prob(raw_odds.ft.draw) + odds_to_prob(raw_odds.ft.away)) - 100.0
        ))
    end
    
    return DataFrame(comparison_data)
end

"""
Display detailed odds analysis for a single match
"""
function analyze_match_odds(data_store::DataStore, match_id::Int; 
                          config::Union{OddsProcessing.OddsConfig, Nothing} = nothing)
    
    if isnothing(config)
        config = default_config()
    end
    
    # Get both raw and processed odds
    raw_odds = get_game_line_odds(data_store, match_id)
    processed_odds = get_processed_game_line_odds(data_store, match_id, config)
    
    println("\n" * "="^60)
    println("MATCH ODDS ANALYSIS - Match ID: $match_id")
    println("="^60)
    
    # Main markets
    markets = ["ft_1x2", "ht_1x2", "ft_ou_2.5", "ft_ou_1.5", "btts"]
    
    dfs = Dict{String, DataFrame}()
    for market in markets
        df = display_market_group(raw_odds, processed_odds, market)
        dfs[market] = df
        display(df)
    end
    
    # Show configuration used
    println("\n=== Configuration ===")
    println("Aggregation: $(config.aggregation)")
    println("Target Overround: $(config.target_overround)")
    println("Normalization: $(config.normalization_method)")
    println("Window: -$(config.window_minutes_before) to +$(config.window_minutes_after) minutes")
    
    return dfs
end

"""
Show raw odds timeline for a match with event detection
"""
function display_odds_timeline(data_store::DataStore, match_id::Int; 
                              minutes_range::Tuple{Int,Int} = (-5, 90))
    
    # Filter odds for this match
    match_odds = filter(row -> row.sofa_match_id == match_id && 
                              minutes_range[1] <= row.minutes <= minutes_range[2],
                       data_store.odds)
    
    if isempty(match_odds)
        println("No odds found for match $match_id")
        return nothing
    end
    
    # Sort by time
    sort!(match_odds, :minutes)
    
    # Create summary DataFrame
    timeline_df = DataFrame(
        Minute = match_odds.minutes,
        Home = match_odds.home,
        Draw = match_odds.draw,
        Away = match_odds.away,
        Home_Prob = odds_to_prob.(match_odds.home),
        Draw_Prob = odds_to_prob.(match_odds.draw),
        Away_Prob = odds_to_prob.(match_odds.away),
        Total_Prob = odds_to_prob.(match_odds.home) .+ 
                    odds_to_prob.(match_odds.draw) .+ 
                    odds_to_prob.(match_odds.away),
        O25 = [ismissing(u) ? NaN : prob_to_odds(100.0 - odds_to_prob(u)) 
               for u in match_odds.under_2_5],
        U25 = match_odds.under_2_5
    )
    
    println("\n=== Odds Timeline for Match $match_id ===")
    display(timeline_df)
    
    # Calculate statistics
    println("\n=== Statistics ===")
    println("Number of odds updates: $(nrow(timeline_df))")
    println("Time range: $(minimum(timeline_df.Minute)) to $(maximum(timeline_df.Minute)) minutes")
    
    # Check for large jumps (potential goals)
    home_values = collect(skipmissing(timeline_df.Home))
    if length(home_values) > 1
        for i in 2:length(home_values)
            prob_change = odds_to_prob(home_values[i]) - odds_to_prob(home_values[i-1])
            if abs(prob_change) > 10.0  # More than 10% probability change
                minute_idx = findfirst(x -> !ismissing(x) && x == home_values[i], timeline_df.Home)
                if !isnothing(minute_idx) && minute_idx > 1
                    event_minute = timeline_df.Minute[minute_idx]
                    prev_minute = timeline_df.Minute[minute_idx-1]
                    println("\n⚠️  Large odds movement detected between minutes $prev_minute and $event_minute")
                    println("   Home: $(home_values[i-1]) → $(home_values[i]) ($(round(prob_change, digits=1))% prob change)")
                    if prob_change > 10
                        println("   Likely HOME GOAL")
                    elseif prob_change < -10
                        println("   Likely AWAY GOAL")
                    end
                end
            end
        end
    end
    
    # Average total probability
    avg_total = mean(skipmissing(timeline_df.Total_Prob))
    println("\n=== Market Quality ===")
    println("Average total probability: $(@sprintf("%.2f%%", avg_total))")
    println("Average overround: $(@sprintf("%.2f%%", avg_total - 100))")
    
    # Check for missing data
    n_missing_home = sum(ismissing.(match_odds.home))
    n_missing_draw = sum(ismissing.(match_odds.draw))
    n_missing_away = sum(ismissing.(match_odds.away))
    if n_missing_home + n_missing_draw + n_missing_away > 0
        println("Missing data: Home=$n_missing_home, Draw=$n_missing_draw, Away=$n_missing_away")
    end
    
    return timeline_df
end

"""
Test different configurations and show their effects
"""
function test_configurations(data_store::DataStore, match_id::Int)
    
    configs = Dict{String, OddsProcessing.OddsConfig}()
    
    # Default configuration
    configs["Default"] = default_config()
    
    # Conservative configuration (less normalization)
    configs["Conservative"] = OddsProcessing.OddsConfig(
        3, 3, 2,  # Smaller window
        :median, 1.5,  # Median, tighter outlier filter
        false, :probability_sum,  # No completion
        false, 1.0, :proportional,  # No normalization
        0.8, 10,
        configs["Default"].market_groups
    )
    
    # Aggressive normalization
    configs["Aggressive"] = OddsProcessing.OddsConfig(
        10, 10, 5,  # Larger window
        :weighted_mean, 3.0,  # Weighted mean, looser outlier filter
        true, :probability_sum,  # Complete missing
        true, 1.03, :power,  # Normalize to 3% overround with power method
        0.7, 15,
        configs["Default"].market_groups
    )
    
    # Bookmaker-like (higher margin)
    configs["Bookmaker"] = OddsProcessing.OddsConfig(
        5, 5, 3,
        :weighted_mean, 2.0,
        true, :probability_sum,
        true, 1.08, :margin_weights,  # 8% overround with margin weights
        0.8, 10,
        configs["Default"].market_groups
    )
    
    comparison_df = compare_configurations(data_store, match_id, configs)
    
    println("\n" * "="^80)
    println("CONFIGURATION COMPARISON - Match ID: $match_id")
    println("="^80)
    display(comparison_df)
    
    # Detailed analysis for each config
    for (name, config) in configs
        println("\n\n" * "-"^40)
        println("Configuration: $name")
        println("-"^40)
        
        processed = get_processed_game_line_odds(data_store, match_id, config)
        if !isnothing(processed)
            # Show FT 1X2 details
            println("\nFT 1X2:")
            @printf("  Home: %.2f (%.1f%%)\n", processed.ft.home, odds_to_prob(processed.ft.home))
            @printf("  Draw: %.2f (%.1f%%)\n", processed.ft.draw, odds_to_prob(processed.ft.draw))
            @printf("  Away: %.2f (%.1f%%)\n", processed.ft.away, odds_to_prob(processed.ft.away))
            total = odds_to_prob(processed.ft.home) + odds_to_prob(processed.ft.draw) + odds_to_prob(processed.ft.away)
            @printf("  Total: %.1f%% (Overround: %.1f%%)\n", total, total - 100)
        end
    end
    
    return configs
end

"""
Quick summary function to check odds quality with incident correlation
"""
function odds_quality_check(data_store::DataStore, match_id::Int)
    println("\n" * "="^60)
    println("ODDS QUALITY CHECK - Match ID: $match_id")
    println("="^60)
    
    # Check for goals in incidents
    if hasproperty(data_store, :incidents)
        match_incidents = filter(row -> row.match_id == match_id, data_store.incidents)
        goals = filter(row -> !ismissing(row.incident_type) && row.incident_type == "goal", match_incidents)
        if nrow(goals) > 0
            println("\n=== Goals in this match ===")
            for goal in eachrow(goals)
                println("Goal at minute $(goal.time): Score $(goal.home_score)-$(goal.away_score)")
            end
        end
    end
    
    # Get raw timeline
    timeline = display_odds_timeline(data_store, match_id, minutes_range=(-10, 10))
    
    # Test configurations
    configs = test_configurations(data_store, match_id)
    
    # Detailed analysis with default config
    analyze_match_odds(data_store, match_id)
    
    return nothing
end




