
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

    # TODO: Change 0_5 to _05
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



module ExtendedEval
  # Kelly criterion structures
  struct KellyFraction
      outcome::Symbol  # :home, :draw, :away, etc.
      fraction::Float64
      confidence_interval::Tuple{Float64, Float64}  # CI from Bayesian posterior
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


function get_match_results(df::DataFrame, match_id::Int)
    # 1. Filter and sort the DataFrame for the given match
    match_incidents = sort(
        filter(row -> row.match_id == match_id, df),
        :time
    )

    # 2. Extract scores for half-time and full-time
    # Find the row corresponding to the end of the first half (time 45)
    # The `period` incident type contains the end-of-period score.
    ht_row = findfirst(
        row -> !ismissing(row.time) && row.time == 45 && row.incident_type == "period",
        eachrow(match_incidents)
    )

    # Find the row corresponding to the end of the second half (time 90)
    ft_row = findfirst(
        row -> !ismissing(row.time) && row.time == 90 && row.incident_type == "period",
        eachrow(match_incidents)
    )

    # 3. Helper function to process scores for a given period
    function process_period_data(score_row)
        if isnothing(score_row)
            # Return a struct with all `nothing` if the data isn't found
            return Eval.MatchPeriodResults(
                nothing, nothing, nothing,
                Vector{Eval.MaybeBool}(nothing, 16),
                nothing, nothing, nothing,
                nothing, nothing, nothing, nothing
            )
        else
            home_score = score_row.home_score
            away_score = score_row.away_score

            # Handle cases where scores might be missing from the period row
            if ismissing(home_score) || ismissing(away_score)
                return Eval.MatchPeriodResults(
                    nothing, nothing, nothing,
                    Vector{Eval.MaybeBool}(nothing, 16),
                    nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing
                )
            end

            # Determine win/draw/loss
            home_win = home_score > away_score
            away_win = away_score > home_score
            draw = home_score == away_score

            # Calculate over/under
            total_goals = home_score + away_score
            under_05 = total_goals < 0.5
            under_15 = total_goals < 1.5
            under_25 = total_goals < 2.5
            under_35 = total_goals < 3.5

            # Correct Score (4x4 matrix, flattened)
            correct_score_vec = Vector{Eval.MaybeBool}(fill(false, 16))
            other_home = false
            other_away = false
            other_draw = false

            if home_score <= 3 && away_score <= 3
                # Map scores to the flattened index (1-16)
                idx = Int(home_score * 4 + away_score + 1)
                correct_score_vec[idx] = true
            else
                # Score is outside the 4x4 grid
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


    # 4. Process the data for each period
    ht_results = process_period_data(isnothing(ht_row) ? nothing : match_incidents[ht_row, :])
    ft_results = process_period_data(isnothing(ft_row) ? nothing : match_incidents[ft_row, :])

    # 5. Return the final composite struct
    return Eval.MatchLinesResults(ht_results, ft_results)
end





#### Claudes junk 
##################################################
# Kelly Criterion Structures for All Markets
##################################################

module KellyEval
    using Statistics
    
    const MaybeKellyFraction = Union{Main.ExtendedEval.KellyFraction, Nothing}
    
    # Structure to hold Kelly fractions for a period's 1x2 market
    struct Kelly1x2
        home::MaybeKellyFraction
        draw::MaybeKellyFraction
        away::MaybeKellyFraction
    end
    
    # Structure to hold Kelly fractions for over/under markets
    struct KellyOverUnder
        under_05::MaybeKellyFraction
        over_05::MaybeKellyFraction
        under_15::MaybeKellyFraction
        over_15::MaybeKellyFraction
        under_25::MaybeKellyFraction
        over_25::MaybeKellyFraction
        under_35::MaybeKellyFraction
        over_35::MaybeKellyFraction
    end
    
    # Structure to hold Kelly fractions for correct score markets
    struct KellyCorrectScore
        scores::Dict{Tuple{Int, Int}, MaybeKellyFraction}
        other_home::MaybeKellyFraction
        other_draw::MaybeKellyFraction
        other_away::MaybeKellyFraction
    end
    
    # Structure for a single period (HT or FT)
    struct PeriodKellyFractions
        one_x_two::Kelly1x2
        over_under::KellyOverUnder
        correct_score::KellyCorrectScore
    end
    
    # Main structure for match Kelly fractions
    struct MatchKellyFractions
        ht::PeriodKellyFractions
        ft::PeriodKellyFractions
    end
end

##################################################
# Safe Kelly Calculation Helper
##################################################

"""
Calculate Bayesian Kelly fraction for a single outcome
"""
function calculate_bayesian_kelly(
    model_probs::Vector{Float64},  # Posterior samples
    market_odds::Float64,
    fractional_kelly::Float64 = 0.25  # Conservative fraction
)
    kelly_fractions = Float64[]
    
    for p in model_probs
        # Standard Kelly formula: f = (p*o - 1) / (o - 1)
        # where p is probability, o is decimal odds
        edge = p * market_odds - 1.0
        
        if edge > 0
            f = edge / (market_odds - 1.0)
            push!(kelly_fractions, f)
        else
            push!(kelly_fractions, 0.0)
        end
    end
    
    if isempty(kelly_fractions) || all(f -> f == 0, kelly_fractions)
        return ExtendedEval.KellyFraction(:none, 0.0, (0.0, 0.0))
    end
    
    # Use percentiles for conservative betting
    fraction = quantile(kelly_fractions, 0.5) * fractional_kelly  # Median * safety factor
    ci = (quantile(kelly_fractions, 0.25), quantile(kelly_fractions, 0.75))
    
    return ExtendedEval.KellyFraction(:single, fraction, ci)
end

"""
    safe_calculate_kelly(model_probs, market_odds, fractional_kelly)

Safely calculates Kelly fraction, handling nothing values and invalid inputs.
"""
function safe_calculate_kelly(
    model_probs::Union{Vector{Float64}, Nothing},
    market_odds::Union{Float64, Nothing},
    fractional_kelly::Float64 = 1.0
)
    # Check for nothing values
    if isnothing(model_probs) || isnothing(market_odds)
        return nothing
    end
    
    # Check for invalid odds
    if market_odds <= 1.0
        return nothing
    end
    
    # Check for empty or all-zero probabilities
    if isempty(model_probs) || all(p -> p <= 0, model_probs)
        return nothing
    end
    
    return calculate_bayesian_kelly(model_probs, market_odds, fractional_kelly)
end

##################################################
# Period Kelly Calculation
##################################################

"""
    calculate_period_kelly_fractions(
        model_predictions,
        normalized_odds,
        raw_odds,
        period::Symbol;
        fractional_kelly::Float64 = 1.0
    )

Calculate Kelly fractions for all markets in a period (HT or FT).
"""
function calculate_period_kelly_fractions(
    model_predictions,
    normalized_odds,
    raw_odds,
    period::Symbol;
    fractional_kelly::Float64 = 1.0
)
    # Select the appropriate period data
    model_period = getfield(model_predictions, period)
    norm_odds_period = getfield(normalized_odds, period)
    raw_odds_period = getfield(raw_odds, period)
    
    # Calculate 1x2 Kelly fractions
    kelly_1x2 = KellyEval.Kelly1x2(
        safe_calculate_kelly(model_period.home, norm_odds_period.home, fractional_kelly),
        safe_calculate_kelly(model_period.draw, norm_odds_period.draw, fractional_kelly),
        safe_calculate_kelly(model_period.away, norm_odds_period.away, fractional_kelly)
    )
    
    # Calculate over/under Kelly fractions
    kelly_ou = KellyEval.KellyOverUnder(
        safe_calculate_kelly(model_period.under_05, norm_odds_period.under_0_5, fractional_kelly),
        safe_calculate_kelly(1 .- model_period.under_05, norm_odds_period.over_0_5, fractional_kelly),
        safe_calculate_kelly(model_period.under_15, norm_odds_period.under_1_5, fractional_kelly),
        safe_calculate_kelly(1 .- model_period.under_15, norm_odds_period.over_1_5, fractional_kelly),
        safe_calculate_kelly(model_period.under_25, norm_odds_period.under_2_5, fractional_kelly),
        safe_calculate_kelly(1 .- model_period.under_25, norm_odds_period.over_2_5, fractional_kelly),
        safe_calculate_kelly(model_period.under_35, norm_odds_period.under_3_5, fractional_kelly),
        safe_calculate_kelly(1 .- model_period.under_35, norm_odds_period.over_3_5, fractional_kelly)
    )
    
    # Calculate correct score Kelly fractions
    kelly_cs_dict = Dict{Tuple{Int, Int}, KellyEval.MaybeKellyFraction}()
    
    # Standard 4x4 grid of scores  
    score_grid = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (0,3), (1,3), (2,3), (3,3)
    ]
    
    for (idx, (home_score, away_score)) in enumerate(score_grid)
        # Get model probabilities for this score
        model_probs = if !isnothing(model_period.correct_score) && 
                        size(model_period.correct_score, 2) >= idx
            model_period.correct_score[:, idx]
        else
            nothing
        end
        
        # Get market odds for this score
        market_odds = get(raw_odds_period.correct_score, (home_score, away_score), nothing)
        
        # Calculate Kelly fraction
        kelly_cs_dict[(home_score, away_score)] = safe_calculate_kelly(
            model_probs, 
            market_odds, 
            fractional_kelly
        )
    end
    
    # Handle "other" scores
    kelly_other_home = safe_calculate_kelly(
        model_period.other_home_win_score,
        raw_odds_period.any_other_home,
        fractional_kelly
    )
    
    kelly_other_draw = safe_calculate_kelly(
        model_period.other_draw_score,
        raw_odds_period.any_other_draw,
        fractional_kelly
    )
    
    kelly_other_away = safe_calculate_kelly(
        model_period.other_away_win_score,
        raw_odds_period.any_other_away,
        fractional_kelly
    )
    
    kelly_cs = KellyEval.KellyCorrectScore(
        kelly_cs_dict,
        kelly_other_home,
        kelly_other_draw,
        kelly_other_away
    )
    
    return KellyEval.PeriodKellyFractions(kelly_1x2, kelly_ou, kelly_cs)
end

##################################################
# Main Function to Calculate All Kelly Fractions
##################################################

"""
    calculate_match_kelly_fractions(
        model_predictions,
        data_store,
        match_id::Int;
        target_sum::Float64 = 1.005,
        fractional_kelly::Float64 = 1.0
    )

Calculate Kelly fractions for all markets and periods for a match.
"""
function calculate_match_kelly_fractions(
    model_predictions,
    data_store,
    match_id::Int;
    target_sum::Float64 = 1.005,
    fractional_kelly::Float64 = 1.0
)
    # Get normalized odds
    normalized_odds = impute_and_rescale_odds(data_store, match_id, target_sum=target_sum)
    if isnothing(normalized_odds)
        @warn "Could not get normalized odds for match_id: $match_id"
        return nothing
    end
    
    # Get raw odds for correct scores
    raw_odds = get_game_line_odds(data_store.odds, match_id)
    if isnothing(raw_odds)
        @warn "Could not get raw odds for match_id: $match_id"
        return nothing
    end
    
    # Calculate Kelly fractions for both periods
    ht_kelly = calculate_period_kelly_fractions(
        model_predictions,
        normalized_odds,
        raw_odds,
        :ht,
        fractional_kelly=fractional_kelly
    )
    
    ft_kelly = calculate_period_kelly_fractions(
        model_predictions,
        normalized_odds,
        raw_odds,
        :ft,
        fractional_kelly=fractional_kelly
    )
    
    return KellyEval.MatchKellyFractions(ht_kelly, ft_kelly)
end


##################################################
# Display Functions
##################################################

using Printf

"""
    display_kelly_fractions(kelly::KellyEval.MatchKellyFractions, match_id::Int, data_store, match_results::Eval.MatchLinesResults)

Display Kelly fractions with win/loss indicators and profit calculations.
"""
function display_kelly_fractions_v2(kelly::KellyEval.MatchKellyFractions, match_id::Int, data_store, match_results::Eval.MatchLinesResults)
    # Helper to format Kelly fraction
    function fmt_kelly(kf::Union{ExtendedEval.KellyFraction, Nothing})
        isnothing(kf) ? "     -     " : @sprintf("%11.4f", kf.fraction)
    end
    
    # Helper to format confidence interval
    function fmt_ci(kf::Union{ExtendedEval.KellyFraction, Nothing})
        isnothing(kf) ? "        -        " : @sprintf("[%6.3f, %6.3f]", kf.confidence_interval[1], kf.confidence_interval[2])
    end
    
    # Helper to calculate profit/loss for a Kelly bet
    function calc_profit(kf::Union{ExtendedEval.KellyFraction, Nothing}, odds::Union{Float64, Nothing}, won::Union{Bool, Nothing})
        if isnothing(kf) || isnothing(odds) || isnothing(won) || kf.fraction <= 0
            return 0.0
        end
        stake = kf.fraction  # As fraction of bankroll
        return won ? stake * (odds - 1) : -stake
    end
    
    # Helper to format profit
    function fmt_profit(profit::Float64)
        profit == 0 ? "       -       " : @sprintf("%+14.4f", profit)
    end
    
    # Extract match info from data_store
    match_row = filter(row -> row.match_id == match_id, data_store.matches)
    
    println("\n╔" * "═"^80 * "╗")
    println("║" * " "^25 * "KELLY CRITERION ANALYSIS" * " "^31 * "║")
    println("╠" * "═"^80 * "╣")
    
    if !isempty(match_row)
        match_row = match_row[1, :]
        teams = "$(match_row.home_team) vs $(match_row.away_team)"
        scores = "FT: $(match_row.home_score)-$(match_row.away_score), HT: $(match_row.home_score_ht)-$(match_row.away_score_ht)"
        println("║ Match: $(rpad(teams, 72)) ║")
        println("║ Score: $(rpad(scores, 72)) ║")
    else
        println("║ Match ID: $(rpad(string(match_id), 68)) ║")
    end
    println("╚" * "═"^80 * "╝")
    
    # Get odds for profit calculation
    norm_odds = impute_and_rescale_odds(data_store, match_id, target_sum=1.005)
    raw_odds = get_game_line_odds(data_store.odds, match_id)
    
    # Track profit by market group
    profit_1x2_ft = 0.0
    profit_1x2_ht = 0.0
    profit_ou_ft = 0.0
    profit_ou_ht = 0.0
    profit_cs_ft = 0.0
    profit_cs_ht = 0.0
    total_profit = 0.0
    
    # 1X2 Markets with results
    println("\n┌" * "─"^94 * "┐")
    println("│" * " "^42 * "1X2 MARKETS" * " "^41 * "│")
    println("├" * "─"^94 * "┤")
    println("│ Period │  Market  │    Kelly    │       95% CI       │  Result  │   Profit/Loss   │")
    println("├────────┼──────────┼─────────────┼────────────────────┼──────────┼─────────────────┤")
    
    # FT 1X2
    ft_1x2 = [
        (:home, "Home", kelly.ft.one_x_two.home, match_results.ft.home, norm_odds.ft.home),
        (:draw, "Draw", kelly.ft.one_x_two.draw, match_results.ft.draw, norm_odds.ft.draw),
        (:away, "Away", kelly.ft.one_x_two.away, match_results.ft.away, norm_odds.ft.away)
    ]
    
    for (field, name, kf, won, odds) in ft_1x2
        profit = calc_profit(kf, odds, won)
        profit_1x2_ft += profit
        result_str = isnothing(won) ? "    -    " : (won ? "   WIN   " : "   LOSS  ")
        println("│   FT   │ $(rpad(name, 8)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(fmt_profit(profit)) │")
    end
    
    println("├────────┼──────────┼─────────────┼────────────────────┼──────────┼─────────────────┤")
    
    # HT 1X2
    ht_1x2 = [
        (:home, "Home", kelly.ht.one_x_two.home, match_results.ht.home, norm_odds.ht.home),
        (:draw, "Draw", kelly.ht.one_x_two.draw, match_results.ht.draw, norm_odds.ht.draw),
        (:away, "Away", kelly.ht.one_x_two.away, match_results.ht.away, norm_odds.ht.away)
    ]
    
    for (field, name, kf, won, odds) in ht_1x2
        profit = calc_profit(kf, odds, won)
        profit_1x2_ht += profit
        result_str = isnothing(won) ? "    -    " : (won ? "   WIN   " : "   LOSS  ")
        println("│   HT   │ $(rpad(name, 8)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(fmt_profit(profit)) │")
    end
    
    println("└" * "─"^94 * "┘")
    
    # Over/Under Markets with results
    println("\n┌" * "─"^94 * "┐")
    println("│" * " "^38 * "OVER/UNDER MARKETS" * " "^38 * "│")
    println("├" * "─"^94 * "┤")
    println("│ Period │  Line  │   Type   │    Kelly    │       95% CI       │  Result  │   Profit   │")
    println("├────────┼────────┼──────────┼─────────────┼────────────────────┼──────────┼────────────┤")
    
    # FT Over/Under
    ft_ou_markets = [
        ("0.5", "Under", kelly.ft.over_under.under_05, match_results.ft.under_05, norm_odds.ft.under_0_5),
        ("0.5", "Over", kelly.ft.over_under.over_05, !match_results.ft.under_05, norm_odds.ft.over_0_5),
        ("1.5", "Under", kelly.ft.over_under.under_15, match_results.ft.under_15, norm_odds.ft.under_1_5),
        ("1.5", "Over", kelly.ft.over_under.over_15, !match_results.ft.under_15, norm_odds.ft.over_1_5),
        ("2.5", "Under", kelly.ft.over_under.under_25, match_results.ft.under_25, norm_odds.ft.under_2_5),
        ("2.5", "Over", kelly.ft.over_under.over_25, !match_results.ft.under_25, norm_odds.ft.over_2_5),
        ("3.5", "Under", kelly.ft.over_under.under_35, match_results.ft.under_35, norm_odds.ft.under_3_5),
        ("3.5", "Over", kelly.ft.over_under.over_35, !match_results.ft.under_35, norm_odds.ft.over_3_5)
    ]
    
    for (line, type, kf, won, odds) in ft_ou_markets
        profit = calc_profit(kf, odds, won)
        profit_ou_ft += profit
        result_str = isnothing(won) ? "    -    " : (won ? "   WIN   " : "   LOSS  ")
        profit_str = profit == 0 ? "     -      " : @sprintf("%+11.4f", profit)
        println("│   FT   │  $(line)  │ $(rpad(type, 8)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(profit_str) │")
    end
    
    println("├────────┼────────┼──────────┼─────────────┼────────────────────┼──────────┼────────────┤")
    
    # HT Over/Under
    ht_ou_markets = [
        ("0.5", "Under", kelly.ht.over_under.under_05, match_results.ht.under_05, norm_odds.ht.under_0_5),
        ("0.5", "Over", kelly.ht.over_under.over_05, !match_results.ht.under_05, norm_odds.ht.over_0_5),
        ("1.5", "Under", kelly.ht.over_under.under_15, match_results.ht.under_15, norm_odds.ht.under_1_5),
        ("1.5", "Over", kelly.ht.over_under.over_15, !match_results.ht.under_15, norm_odds.ht.over_1_5),
        ("2.5", "Under", kelly.ht.over_under.under_25, match_results.ht.under_25, norm_odds.ht.under_2_5),
        ("2.5", "Over", kelly.ht.over_under.over_25, !match_results.ht.under_25, norm_odds.ht.over_2_5)
    ]
    
    for (line, type, kf, won, odds) in ht_ou_markets
        profit = calc_profit(kf, odds, won)
        profit_ou_ht += profit
        result_str = isnothing(won) ? "    -    " : (won ? "   WIN   " : "   LOSS  ")
        profit_str = profit == 0 ? "     -      " : @sprintf("%+11.4f", profit)
        println("│   HT   │  $(line)  │ $(rpad(type, 8)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(profit_str) │")
    end
    
    println("└" * "─"^94 * "┘")
    
    # Correct Score Markets
    println("\n┌" * "─"^94 * "┐")
    println("│" * " "^34 * "CORRECT SCORE MARKETS" * " "^38 * "│")
    println("├" * "─"^94 * "┤")
    println("│ Period │  Score  │    Kelly    │       95% CI       │  Result  │   Profit/Loss   │")
    println("├────────┼─────────┼─────────────┼────────────────────┼──────────┼─────────────────┤")
    
    # FT Correct Scores - show ALL scores from the 4x4 grid
    score_grid = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (0,3), (1,3), (2,3), (3,3)
    ]
    
    for (idx, (h, a)) in enumerate(score_grid)
        kf = get(kelly.ft.correct_score.scores, (h, a), nothing)
        
        # Check if this score actually happened
        won = if !isnothing(match_results.ft.correct_score) && idx <= length(match_results.ft.correct_score)
            match_results.ft.correct_score[idx]
        else
            false
        end
        
        odds = get(raw_odds.ft.correct_score, (h, a), nothing)
        profit = calc_profit(kf, odds, won)
        profit_cs_ft += profit
        
        score_str = "$h-$a"
        result_str = won ? "   WIN   " : "   LOSS  "
        println("│   FT   │   $(rpad(score_str, 5)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(fmt_profit(profit)) │")
    end
    
    # Other scores for FT
    other_scores = [
        ("Other H", kelly.ft.correct_score.other_home, match_results.ft.other_home_win_score, raw_odds.ft.any_other_home),
        ("Other D", kelly.ft.correct_score.other_draw, match_results.ft.other_draw_score, raw_odds.ft.any_other_draw),
        ("Other A", kelly.ft.correct_score.other_away, match_results.ft.other_away_win_score, raw_odds.ft.any_other_away)
    ]
    
    for (name, kf, won, odds) in other_scores
        profit = calc_profit(kf, odds, won)
        profit_cs_ft += profit
        result_str = isnothing(won) ? "    -    " : (won ? "   WIN   " : "   LOSS  ")
        println("│   FT   │ $(rpad(name, 7)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(fmt_profit(profit)) │")
    end
    
    println("├────────┼─────────┼─────────────┼────────────────────┼──────────┼─────────────────┤")
    
    # HT Correct Scores
    for (idx, (h, a)) in enumerate(score_grid)
        kf = get(kelly.ht.correct_score.scores, (h, a), nothing)
        
        won = if !isnothing(match_results.ht.correct_score) && idx <= length(match_results.ht.correct_score)
            match_results.ht.correct_score[idx]
        else
            false
        end
        
        odds = get(raw_odds.ht.correct_score, (h, a), nothing)
        profit = calc_profit(kf, odds, won)
        profit_cs_ht += profit
        
        score_str = "$h-$a"
        result_str = won ? "   WIN   " : "   LOSS  "
        println("│   HT   │   $(rpad(score_str, 5)) │ $(fmt_kelly(kf)) │ $(fmt_ci(kf)) │ $(result_str) │ $(fmt_profit(profit)) │")
    end
    
    println("└" * "─"^94 * "┘")
    
    # Calculate total profit and ROI
    total_profit = profit_1x2_ft + profit_1x2_ht + profit_ou_ft + profit_ou_ht + profit_cs_ft + profit_cs_ht
    
    # Count total stake
    total_stake = 0.0
    
    # Function to add stake if bet was placed
    function add_stake(kf)
        if !isnothing(kf) && kf.fraction > 0.0001
            return kf.fraction
        end
        return 0.0
    end
    
    # Calculate total stake from all markets
    # 1X2
    for kf in [kelly.ft.one_x_two.home, kelly.ft.one_x_two.draw, kelly.ft.one_x_two.away,
               kelly.ht.one_x_two.home, kelly.ht.one_x_two.draw, kelly.ht.one_x_two.away]
        total_stake += add_stake(kf)
    end
    
    # Over/Under
    for kf in [kelly.ft.over_under.under_05, kelly.ft.over_under.over_05,
               kelly.ft.over_under.under_15, kelly.ft.over_under.over_15,
               kelly.ft.over_under.under_25, kelly.ft.over_under.over_25,
               kelly.ft.over_under.under_35, kelly.ft.over_under.over_35,
               kelly.ht.over_under.under_05, kelly.ht.over_under.over_05,
               kelly.ht.over_under.under_15, kelly.ht.over_under.over_15,
               kelly.ht.over_under.under_25, kelly.ht.over_under.over_25]
        total_stake += add_stake(kf)
    end
    
    # Correct Scores
    for (_, kf) in kelly.ft.correct_score.scores
        total_stake += add_stake(kf)
    end
    for (_, kf) in kelly.ht.correct_score.scores
        total_stake += add_stake(kf)
    end
    
    roi = total_stake > 0 ? (total_profit / total_stake * 100) : 0.0
    
    # Summary Statistics with P/L breakdown
    println("\n┌" * "─"^60 * "┐")
    println("│" * " "^21 * "PROFIT/LOSS BREAKDOWN" * " "^17 * "│")
    println("├" * "─"^60 * "┤")
    println("│ Market Group │ Period │       P/L      │      Stake      │")
    println("├──────────────┼────────┼────────────────┼─────────────────┤")
    
    # Calculate stakes for each group
    stake_1x2_ft = add_stake(kelly.ft.one_x_two.home) + add_stake(kelly.ft.one_x_two.draw) + add_stake(kelly.ft.one_x_two.away)
    stake_1x2_ht = add_stake(kelly.ht.one_x_two.home) + add_stake(kelly.ht.one_x_two.draw) + add_stake(kelly.ht.one_x_two.away)
    
    stake_ou_ft = add_stake(kelly.ft.over_under.under_05) + add_stake(kelly.ft.over_under.over_05) +
                  add_stake(kelly.ft.over_under.under_15) + add_stake(kelly.ft.over_under.over_15) +
                  add_stake(kelly.ft.over_under.under_25) + add_stake(kelly.ft.over_under.over_25) +
                  add_stake(kelly.ft.over_under.under_35) + add_stake(kelly.ft.over_under.over_35)
    
    stake_ou_ht = add_stake(kelly.ht.over_under.under_05) + add_stake(kelly.ht.over_under.over_05) +
                  add_stake(kelly.ht.over_under.under_15) + add_stake(kelly.ht.over_under.over_15) +
                  add_stake(kelly.ht.over_under.under_25) + add_stake(kelly.ht.over_under.over_25)
    
    stake_cs_ft = sum(add_stake(kf) for (_, kf) in kelly.ft.correct_score.scores)
    stake_cs_ht = sum(add_stake(kf) for (_, kf) in kelly.ht.correct_score.scores)
    
    println("│     1X2      │   FT   │ $(@sprintf("%+14.4f", profit_1x2_ft)) │ $(@sprintf("%15.4f", stake_1x2_ft)) │")
    println("│     1X2      │   HT   │ $(@sprintf("%+14.4f", profit_1x2_ht)) │ $(@sprintf("%15.4f", stake_1x2_ht)) │")
    println("│  Over/Under  │   FT   │ $(@sprintf("%+14.4f", profit_ou_ft)) │ $(@sprintf("%15.4f", stake_ou_ft)) │")
    println("│  Over/Under  │   HT   │ $(@sprintf("%+14.4f", profit_ou_ht)) │ $(@sprintf("%15.4f", stake_ou_ht)) │")
    println("│ Correct Score│   FT   │ $(@sprintf("%+14.4f", profit_cs_ft)) │ $(@sprintf("%15.4f", stake_cs_ft)) │")
    println("│ Correct Score│   HT   │ $(@sprintf("%+14.4f", profit_cs_ht)) │ $(@sprintf("%15.4f", stake_cs_ht)) │")
    println("├──────────────┴────────┴────────────────┴─────────────────┤")
    println("│ Total P/L (% of bankroll): $(@sprintf("%+8.4f%%", total_profit * 100)) " * " "^22 * "│")
    println("│ Total Stake (% of bankroll): $(@sprintf("%7.4f%%", total_stake * 100)) " * " "^20 * "│")
    println("│ ROI: $(@sprintf("%+8.2f%%", roi)) " * " "^42 * "│")
    println("└" * "─"^60 * "┘")
   # _won = 0
    
    # Check all markets
    all_bets = [
        (kelly.ft.one_x_two.home, match_results.ft.home),
        (kelly.ft.one_x_two.draw, match_results.ft.draw),
        (kelly.ft.one_x_two.away, match_results.ft.away),
        (kelly.ht.one_x_two.home, match_results.ht.home),
        (kelly.ht.one_x_two.draw, match_results.ht.draw),
        (kelly.ht.one_x_two.away, match_results.ht.away),
        (kelly.ft.over_under.under_05, match_results.ft.under_05),
        (kelly.ft.over_under.over_05, !match_results.ft.under_05),
        (kelly.ft.over_under.under_15, match_results.ft.under_15),
        (kelly.ft.over_under.over_15, !match_results.ft.under_15),
        (kelly.ft.over_under.under_25, match_results.ft.under_25),
        (kelly.ft.over_under.over_25, !match_results.ft.under_25),
        (kelly.ft.over_under.under_35, match_results.ft.under_35),
        (kelly.ft.over_under.over_35, !match_results.ft.under_35),
        (kelly.ht.over_under.under_05, match_results.ht.under_05),
        (kelly.ht.over_under.over_05, !match_results.ht.under_05),
        (kelly.ht.over_under.under_15, match_results.ht.under_15),
        (kelly.ht.over_under.over_15, !match_results.ht.under_15),
        (kelly.ht.over_under.under_25, match_results.ht.under_25),
        (kelly.ht.over_under.over_25, !match_results.ht.under_25)
    ]
   
    bets_placed = 0
    bets_won =0
    for (kf, won) in all_bets
        if !isnothing(kf) && kf.fraction > 0.0001
            bets_placed += 1
            if !isnothing(won) && won
                bets_won += 1
            end
        end
    end
    
    win_rate = bets_placed > 0 ? (bets_won / bets_placed * 100) : 0.0
    println("│ Bets Placed: $(lpad(string(bets_placed), 2))  |  Bets Won: $(lpad(string(bets_won), 2))  |  Win Rate: $(@sprintf("%5.1f%%", win_rate)) │")
    println("└" * "─"^50 * "┘")
end

"""
    get_positive_kelly_bets(kelly::KellyEval.MatchKellyFractions; min_kelly=0.01)

Extract all positive Kelly fractions above threshold.
"""
function get_positive_kelly_bets(kelly::KellyEval.MatchKellyFractions; min_kelly::Float64=0.01)
    bets = []
    
    # Check FT 1X2
    for (market, field) in [("FT_Home", :home), ("FT_Draw", :draw), ("FT_Away", :away)]
        kf = getfield(kelly.ft.one_x_two, field)
        if !isnothing(kf) && kf.fraction >= min_kelly
            push!(bets, (market=market, kelly=kf.fraction, ci=kf.confidence_interval))
        end
    end
    
    # Check HT 1X2
    for (market, field) in [("HT_Home", :home), ("HT_Draw", :draw), ("HT_Away", :away)]
        kf = getfield(kelly.ht.one_x_two, field)
        if !isnothing(kf) && kf.fraction >= min_kelly
            push!(bets, (market=market, kelly=kf.fraction, ci=kf.confidence_interval))
        end
    end
    
    # Check Over/Under markets
    for period in [:ft, :ht]
        period_str = uppercase(string(period))
        ou = getfield(kelly, period).over_under
        
        for (line, under_field, over_field) in [
            ("0.5", :under_05, :over_05),
            ("1.5", :under_15, :over_15),
            ("2.5", :under_25, :over_25),
            ("3.5", :under_35, :over_35)
        ]
            kf_under = getfield(ou, under_field)
            kf_over = getfield(ou, over_field)
            
            if !isnothing(kf_under) && kf_under.fraction >= min_kelly
                push!(bets, (market="$(period_str)_Under_$(line)", kelly=kf_under.fraction, ci=kf_under.confidence_interval))
            end
            if !isnothing(kf_over) && kf_over.fraction >= min_kelly
                push!(bets, (market="$(period_str)_Over_$(line)", kelly=kf_over.fraction, ci=kf_over.confidence_interval))
            end
        end
    end
    
    # Sort by Kelly fraction (highest first)
    sort!(bets, by=x->x.kelly, rev=true)
    
    return bets
end

