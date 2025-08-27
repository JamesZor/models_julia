
using BayesianFootball
using Statistics
using DataFrames
using Plots
using Printf

using Accessors

##################################################
# Extended Types for Full Market Coverage
##################################################

module Eval 
  const MaybeFloat = Union{Float64, Nothing}
  const MaybeBool = Union{Bool, Nothing}

  struct MatchPeriodLine 
    home::MaybeFloat
    draw::MaybeFloat
    away::MaybeFloat

    correct_score::AbstractArray{MaybeFloat}
    other_home_win_score::MaybeFloat 
    other_away_win_score::MaybeFloat 
    other_draw_score::MaybeFloat 

    under_05::MaybeFloat 
    under_15::MaybeFloat 
    under_25::MaybeFloat 
    under_35::MaybeFloat 
  end 
  
  struct MatchLines
    ht::MatchPeriodLine
    ft::MatchPeriodLine
  end

  struct MatchPeriodResults 
    home::MaybeBool
    draw::MaybeBool
    away::MaybeBool

    # order for cs 
 # 0-0, 1-0, 2-0, 3-0, 0-1, 1-1, 2-1, 3-1, 0-2, 1-2, 2-2, 3-2, 0-3, 1-3, 2-3, 3-3 
 # 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16. 
    correct_score::AbstractArray{MaybeBool}
    other_home_win_score::MaybeBool 
    other_away_win_score::MaybeBool 
    other_draw_score::MaybeBool 

    under_05::MaybeBool 
    under_15::MaybeBool 
    under_25::MaybeBool 
    under_35::MaybeBool 
  end 
  
  struct MatchLinesResults
    ht::MatchPeriodResults
    ft::MatchPeriodResults
  end

  const MaybeInt = Union{Int, Nothing}


  # Struct to store the minutes for the 1x2 outcomes, now vectors to track changes
  struct OutcomeMinutes
      home::Vector{Int}
      draw::Vector{Int}
      away::Vector{Int}
  end

  # Struct to store the minutes when the over/under lines become 'active' (passed)
  struct OverUnderMinutes
      over_05::MaybeInt
      over_15::MaybeInt
      over_25::MaybeInt
      over_35::MaybeInt
  end

  # A new struct to store the line active minutes for a single period
  struct MatchPeriodActiveMinutes
      outcome::OutcomeMinutes
      over_under::OverUnderMinutes
      # A dictionary mapping score tuples to a vector of minutes
      correct_score::Dict{Tuple{Int, Int}, Vector{Int}}
  end

  # The main final struct, similar to the one from get_match_results
  struct MatchLinesActive
      ht::MatchPeriodActiveMinutes
      ft::MatchPeriodActiveMinutes
  end
struct PeriodOdds
    # 1x2 odds
    home::MaybeFloat
    draw::MaybeFloat
    away::MaybeFloat

    # Over/Under odds
    over_0_5::MaybeFloat
    under_0_5::MaybeFloat
    over_1_5::MaybeFloat
    under_1_5::MaybeFloat
    over_2_5::MaybeFloat
    under_2_5::MaybeFloat
    over_3_5::MaybeFloat
    under_3_5::MaybeFloat

    # Correct Score odds (a dictionary for flexibility)
    correct_score::Dict{Tuple{Int, Int}, MaybeFloat}
    any_other_home::MaybeFloat
    any_other_draw::MaybeFloat
    any_other_away::MaybeFloat

    # Both teams to score (BTTS) odds - may be Nothing for HT
    btts_yes::MaybeFloat
    btts_no::MaybeFloat
end
# Main struct to hold results for both half-time and full-time
struct MatchOdds
    sofa_match_id::Int
    ht::PeriodOdds
    ft::PeriodOdds
end

# Structs for the probability check results
struct ProbabilityCheckResults
    sum::MaybeFloat
    missing_count::Int
end

struct OverUnderProbabilityResults
    over_0_5::ProbabilityCheckResults
    over_1_5::ProbabilityCheckResults
    over_2_5::ProbabilityCheckResults
    over_3_5::ProbabilityCheckResults
end

struct CorrectScoreProbabilityResults
    sum::MaybeFloat
    missing_count::Int
end

struct PeriodProbabilityChecks
    # 1x2 probability check results
    _1x2::ProbabilityCheckResults
    # Over/Under probability checks
    over_under::OverUnderProbabilityResults
    # Correct Score probability check
    correct_score::CorrectScoreProbabilityResults
    # Both teams to score (BTTS) probability check
    btts::ProbabilityCheckResults
end

struct MatchProbabilityChecks
    sofa_match_id::Int
    ht::PeriodProbabilityChecks
    ft::PeriodProbabilityChecks
end

# New structs for the rescaled odds
struct RescaledPeriodOdds
    home::MaybeFloat
    draw::MaybeFloat
    away::MaybeFloat
    over_0_5::MaybeFloat
    under_0_5::MaybeFloat
    over_1_5::MaybeFloat
    under_1_5::MaybeFloat
    over_2_5::MaybeFloat
    under_2_5::MaybeFloat
    over_3_5::MaybeFloat
    under_3_5::MaybeFloat
    correct_score::Dict{Tuple{Int, Int}, MaybeFloat}
    any_other_home::MaybeFloat
    any_other_draw::MaybeFloat
    any_other_away::MaybeFloat
    btts_yes::MaybeFloat
    btts_no::MaybeFloat
end

struct RescaledMatchOdds
    sofa_match_id::Int
    ht::RescaledPeriodOdds
    ft::RescaledPeriodOdds
end
end



##################################################
# Extract the informations from the incidents 
##################################################
"""
    get_match_results(df::DataFrame, match_id::Int)

Processes a DataFrame of match incidents to determine half-time and full-time
results based on the scores at the end of each period.

Returns an instance of `Eval.MatchLinesResults` populated with boolean
indicators for win/loss/draw, correct score, and over/under markets.
"""
function get_match_results(df::DataFrame, match_id::Int)
    # 1. Filter and sort the DataFrame for the given match
    match_incidents = sort(
        filter(row -> row.match_id == match_id, df),
        :time
    )

    # 2. Extract scores for half-time and full-time
    ht_row = findfirst(
        row -> !ismissing(row.time) && row.time == 45 && row.incident_type == "period",
        eachrow(match_incidents)
    )

    ft_row = findfirst(
        row -> !ismissing(row.time) && row.time == 90 && row.incident_type == "period",
        eachrow(match_incidents)
    )

    # 3. Helper function to process scores for a given period
    function process_period_data(score_row)
        if isnothing(score_row)
            return Eval.MatchPeriodResults(
                nothing, nothing, nothing,
                Vector{Eval.MaybeBool}(nothing, 16),
                nothing, nothing, nothing,
                nothing, nothing, nothing, nothing
            )
        else
            home_score = score_row.home_score
            away_score = score_row.away_score

            if ismissing(home_score) || ismissing(away_score)
                return Eval.MatchPeriodResults(
                    nothing, nothing, nothing,
                    Vector{Eval.MaybeBool}(nothing, 16),
                    nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing
                )
            end

            home_win = home_score > away_score
            away_win = away_score > home_score
            draw = home_score == away_score

            total_goals = home_score + away_score
            under_05 = total_goals < 0.5
            under_15 = total_goals < 1.5
            under_25 = total_goals < 2.5
            under_35 = total_goals < 3.5

            correct_score_vec = Vector{Eval.MaybeBool}(fill(false, 16))
            other_home = false
            other_away = false
            other_draw = false

            if home_score <= 3 && away_score <= 3
                idx = Int(home_score * 4 + away_score + 1)
                correct_score_vec[idx] = true
            else
                if home_score > away_score
                    other_home = true
                elseif away_score > home_score
                    other_away = true
                else
                    other_draw = true
                end
            end

            return Eval.MatchPeriodResults(
                home_win, draw, away_win,
                correct_score_vec,
                other_home, other_away, other_draw,
                under_05, under_15, under_25, under_35
            )
        end
    end

    ht_results = process_period_data(isnothing(ht_row) ? nothing : match_incidents[ht_row, :])
    ft_results = process_period_data(isnothing(ft_row) ? nothing : match_incidents[ft_row, :])

    return Eval.MatchLinesResults(ht_results, ft_results)
end

"""
    get_line_active_minutes(df::DataFrame, match_id::Int)

Processes a DataFrame of match incidents to track every minute at which a given
betting line (1x2, over/under, correct score) becomes the leading or active outcome.

Returns an instance of `Eval.MatchLinesResults` containing a breakdown of
active minutes for both half-time and full-time periods.
"""
function get_line_active_minutes(df::DataFrame, match_id::Int)
    match_incidents = sort(
        filter(row -> !ismissing(row.home_score) && !ismissing(row.away_score), df[df.match_id .== match_id, :]),
        :time
    )

    function process_period(incidents_df::DataFrame)
        outcome_minutes = Eval.OutcomeMinutes(Int[], Int[], Int[])
        over_under_minutes = Eval.OverUnderMinutes(nothing, nothing, nothing, nothing)
        correct_score_minutes = Dict{Tuple{Int, Int}, Vector{Int}}()

        push!(outcome_minutes.draw, 0)
        
        prev_home_score = 0
        prev_away_score = 0

        for row in eachrow(incidents_df)
            current_home_score = Int(row.home_score)
            current_away_score = Int(row.away_score)
            current_time = row.time

            if current_home_score == prev_home_score && current_away_score == prev_away_score
                continue
            end
            
            if current_home_score > current_away_score && !(prev_home_score > prev_away_score)
                push!(outcome_minutes.home, current_time)
            elseif current_away_score > current_home_score && !(prev_away_score > prev_home_score)
                push!(outcome_minutes.away, current_time)
            elseif current_home_score == current_away_score && !(prev_home_score == prev_away_score)
                push!(outcome_minutes.draw, current_time)
            end

            current_total_goals = current_home_score + current_away_score
            prev_total_goals = prev_home_score + prev_away_score

            if current_total_goals >= 0.5 && prev_total_goals < 0.5 && isnothing(over_under_minutes.over_05)
                over_under_minutes = @set over_under_minutes.over_05 = current_time
            end
            if current_total_goals >= 1.5 && prev_total_goals < 1.5 && isnothing(over_under_minutes.over_15)
                over_under_minutes = @set over_under_minutes.over_15 = current_time
            end
            if current_total_goals >= 2.5 && prev_total_goals < 2.5 && isnothing(over_under_minutes.over_25)
                over_under_minutes = @set over_under_minutes.over_25 = current_time
            end
            if current_total_goals >= 3.5 && prev_total_goals < 3.5 && isnothing(over_under_minutes.over_35)
                over_under_minutes = @set over_under_minutes.over_35 = current_time
            end

            score_tuple = (current_home_score, current_away_score)
            if !haskey(correct_score_minutes, score_tuple)
                correct_score_minutes[score_tuple] = Int[]
            end
            push!(correct_score_minutes[score_tuple], current_time)

            prev_home_score = current_home_score
            prev_away_score = current_away_score
        end

        return Eval.MatchPeriodActiveMinutes(
            outcome_minutes,
            over_under_minutes,
            correct_score_minutes
        )
    end

    ht_incidents = filter(row -> row.time <= 45, match_incidents)
    ft_incidents = filter(row -> row.time <= 90, match_incidents)

    ht_results = process_period(ht_incidents)
    ft_results = process_period(ft_incidents)

    return Eval.MatchLinesActive(ht_results, ft_results)
end

"""
    get_game_line_odds(df::DataFrame, match_id::Int)

Extracts the kickoff odds (at `minutes=0`) for a given match from the odds DataFrame.
"""
function get_game_line_odds(df::DataFrame, match_id::Int)
    odds_row = nothing
    try
        odds_row = filter(row -> row.sofa_match_id == match_id && row.minutes == 0, df)
    catch e
        @error "Error filtering DataFrame: $e"
        return nothing
    end
    
    if isempty(odds_row)
        @warn "No kickoff odds found for match_id: $match_id"
        empty_period_odds = Eval.PeriodOdds(
            nothing, nothing, nothing,
            nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing,
            Dict{Tuple{Int, Int}, Eval.MaybeFloat}(),
            nothing, nothing, nothing,
            nothing, nothing
        )
        return Eval.MatchOdds(match_id, empty_period_odds, empty_period_odds)
    end
    
    odds_row = odds_row[1, :]

    function extract_period_odds(row, prefix="")
        function get_val(col_name)
            full_col_name = prefix * col_name
            if hasproperty(row, Symbol(full_col_name))
                val = row[Symbol(full_col_name)]
                return ismissing(val) ? nothing : Float64(val)
            else
                return nothing
            end
        end

        cs_odds = Dict{Tuple{Int, Int}, Eval.MaybeFloat}()
        scores = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3), (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (3,3)]
        for (h, a) in scores
            key = (h, a)
            val = get_val("$(h)_$(a)")
            if !isnothing(val)
                cs_odds[key] = val
            end
        end
        
        Eval.PeriodOdds(
            get_val("home"), get_val("draw"), get_val("away"),
            get_val("over_0_5"), get_val("under_0_5"),
            get_val("over_1_5"), get_val("under_1_5"),
            get_val("over_2_5"), get_val("under_2_5"),
            get_val("over_3_5"), get_val("under_3_5"),
            cs_odds,
            get_val("any_other_home"), get_val("any_other_draw"), get_val("any_other_away"),
            get_val("btts_yes"), get_val("btts_no")
        )
    end

    ht_odds = extract_period_odds(odds_row, "ht_")
    ft_odds = extract_period_odds(odds_row, "")

    return Eval.MatchOdds(match_id, ht_odds, ft_odds)
end

"""
    check_market_probabilities(mo::MatchOdds)

Calculates the implied probability sums for key betting markets.
"""
function check_market_probabilities(mo::Eval.MatchOdds)
    function check_period_probabilities(period_odds::Eval.PeriodOdds)
        _1x2_odds = [period_odds.home, period_odds.draw, period_odds.away]
        _1x2_sum = 0.0
        _1x2_missing_count = 0
        for odds in _1x2_odds
            if !isnothing(odds)
                _1x2_sum += 1 / odds
            else
                _1x2_missing_count += 1
            end
        end
        _1x2_results = Eval.ProbabilityCheckResults(
            _1x2_missing_count == length(_1x2_odds) ? nothing : _1x2_sum,
            _1x2_missing_count
        )

        over_under_results_dict = Dict{String, Eval.ProbabilityCheckResults}()
        over_under_pairs = [
            ("over_0_5", "under_0_5"), ("over_1_5", "under_1_5"),
            ("over_2_5", "under_2_5"), ("over_3_5", "under_3_5")
        ]
        
        for (over_key, under_key) in over_under_pairs
            over_odds = getproperty(period_odds, Symbol(over_key))
            under_odds = getproperty(period_odds, Symbol(under_key))
            sum_val = 0.0
            missing_count = 0
            
            if !isnothing(over_odds)
                sum_val += 1 / over_odds
            else
                missing_count += 1
            end
            
            if !isnothing(under_odds)
                sum_val += 1 / under_odds
            else
                missing_count += 1
            end

            over_under_results_dict[over_key] = Eval.ProbabilityCheckResults(
                missing_count == 2 ? nothing : sum_val,
                missing_count
            )
        end

        over_under_results = Eval.OverUnderProbabilityResults(
            over_under_results_dict["over_0_5"],
            over_under_results_dict["over_1_5"],
            over_under_results_dict["over_2_5"],
            over_under_results_dict["over_3_5"]
        )

        cs_sum = 0.0
        cs_missing_count = 0
        all_cs_odds = vcat(
            collect(values(period_odds.correct_score)),
            [period_odds.any_other_home, period_odds.any_other_draw, period_odds.any_other_away]
        )
        for odds in all_cs_odds
            if !isnothing(odds)
                cs_sum += 1 / odds
            else
                cs_missing_count += 1
            end
        end
        cs_results = Eval.CorrectScoreProbabilityResults(
            cs_missing_count == length(all_cs_odds) ? nothing : cs_sum,
            cs_missing_count
        )

        btts_odds = [period_odds.btts_yes, period_odds.btts_no]
        btts_sum = 0.0
        btts_missing_count = 0
        for odds in btts_odds
            if !isnothing(odds)
                btts_sum += 1 / odds
            else
                btts_missing_count += 1
            end
        end
        btts_results = Eval.ProbabilityCheckResults(
            btts_missing_count == length(btts_odds) ? nothing : btts_sum,
            btts_missing_count
        )
        
        return Eval.PeriodProbabilityChecks(
            _1x2_results,
            over_under_results,
            cs_results,
            btts_results
        )
    end

    ht_checks = check_period_probabilities(mo.ht)
    ft_checks = check_period_probabilities(mo.ft)

    return Eval.MatchProbabilityChecks(mo.sofa_match_id, ht_checks, ft_checks)
end

"""
    impute_and_rescale_odds(data_store, match_id::Int; target_sum::Float64=1.0)

Fetches odds for a match, imputes single missing values in market groups,
and rescales the odds to a target implied probability sum.

# Arguments
- `data_store`: The main data store containing the odds DataFrame.
- `match_id`: The `sofa_match_id` of the match to process.
- `target_sum`: The desired sum of implied probabilities for each market (e.g., 1.0 for a 'fair' book).

# Returns
- A `RescaledMatchOdds` struct with the processed odds. Returns `nothing` if initial odds can't be fetched.
"""
function impute_and_rescale_odds(data_store, match_id::Int; target_sum::Float64=1.0)
    
    # First, get the raw odds for the match
    mo = get_game_line_odds(data_store.odds, match_id)
    if isnothing(mo)
        return nothing
    end

    # --- Helper function to process a group of odds ---
    function process_group(odds_array::Vector{<:Union{Float64, Nothing}})
        # Find missing values
        missing_indices = findall(isnothing, odds_array)
        num_missing = length(missing_indices)
        
        local completed_odds_array
        
        if num_missing == 0
            # No missing values, just rescale
            completed_odds_array = convert(Vector{Float64}, odds_array)
        elseif num_missing == 1
            # One missing value, try to impute
            existing_odds = filter(!isnothing, odds_array)
            sum_existing_probs = sum(1.0 ./ existing_odds)
            
            # Imputed probability must be positive
            if sum_existing_probs >= 1.0
                return odds_array # Cannot impute, return original
            end
            
            imputed_prob = 1.0 - sum_existing_probs
            imputed_odd = 1.0 / imputed_prob
            
            # Create the completed array
            completed_odds_array = Vector{Float64}(undef, length(odds_array))
            valid_idx = 1
            for i in 1:length(odds_array)
                if isnothing(odds_array[i])
                    completed_odds_array[i] = imputed_odd
                else
                    completed_odds_array[i] = odds_array[i]
                    valid_idx += 1
                end
            end
        else
            # Too many missing values, cannot process
            return odds_array
        end
        
        # --- Rescale the completed odds array ---
        prob_sum = sum(1.0 ./ completed_odds_array)
        if prob_sum <= 0
            return fill(nothing, length(completed_odds_array)) # Avoid division by zero
        end

        # The factor to adjust the current sum to the target sum
        scaling_factor = prob_sum / target_sum
        rescaled_odds = completed_odds_array .* scaling_factor
        
        return rescaled_odds
    end

    # --- Helper function to process all markets in a single period ---
    function process_period_odds(period_odds)
        # 1x2 odds
        rescaled_1x2 = process_group([period_odds.home, period_odds.draw, period_odds.away])
        
        # Over/Under odds
        rescaled_ou_05 = process_group([period_odds.over_0_5, period_odds.under_0_5])
        rescaled_ou_15 = process_group([period_odds.over_1_5, period_odds.under_1_5])
        rescaled_ou_25 = process_group([period_odds.over_2_5, period_odds.under_2_5])
        rescaled_ou_35 = process_group([period_odds.over_3_5, period_odds.under_3_5])

        # Correct Score odds
        all_cs_keys = sort(collect(keys(period_odds.correct_score)))
        cs_odds_values = [period_odds.correct_score[k] for k in all_cs_keys]
        all_cs_odds_array = vcat(cs_odds_values, [period_odds.any_other_home, period_odds.any_other_draw, period_odds.any_other_away])
        
        rescaled_cs_values = process_group(all_cs_odds_array)

        rescaled_cs_dict = Dict{Tuple{Int, Int}, Eval.MaybeFloat}()
        num_cs_scores = length(all_cs_keys)
        
        # Check if rescaling was successful before populating the dict
        if !any(isnothing, rescaled_cs_values)
            for i in 1:num_cs_scores
                rescaled_cs_dict[all_cs_keys[i]] = rescaled_cs_values[i]
            end
            any_other_home = rescaled_cs_values[num_cs_scores + 1]
            any_other_draw = rescaled_cs_values[num_cs_scores + 2]
            any_other_away = rescaled_cs_values[num_cs_scores + 3]
        else
            # If rescaling failed, populate with nothings
            for k in all_cs_keys
                rescaled_cs_dict[k] = nothing
            end
            any_other_home, any_other_draw, any_other_away = nothing, nothing, nothing
        end


        # BTTS odds
        rescaled_btts = process_group([period_odds.btts_yes, period_odds.btts_no])

        return Eval.RescaledPeriodOdds(
            rescaled_1x2[1], rescaled_1x2[2], rescaled_1x2[3],
            rescaled_ou_05[1], rescaled_ou_05[2],
            rescaled_ou_15[1], rescaled_ou_15[2],
            rescaled_ou_25[1], rescaled_ou_25[2],
            rescaled_ou_35[1], rescaled_ou_35[2],
            rescaled_cs_dict,
            any_other_home, any_other_draw, any_other_away,
            rescaled_btts[1], rescaled_btts[2]
        )
    end
    
    # Process both HT and FT periods
    ht_rescaled = process_period_odds(mo.ht)
    ft_rescaled = process_period_odds(mo.ft)

    return Eval.RescaledMatchOdds(mo.sofa_match_id, ht_rescaled, ft_rescaled)
end


