using BayesianFootball
using DataFrames 

##################################################
#  Types 
##################################################

module Results
  const MaybeFloat = Union{Float64, Nothing}
  const MaybeBool = Union{Bool, Nothing}

  # Define correct score dictionary types
  const CorrectScore = Dict{Union{Tuple{Int,Int}, String}, MaybeBool}

  struct MatchFTResults 
    home::MaybeBool
    draw::MaybeBool
    away::MaybeBool
    correct_score::CorrectScore 
    # under
    under_05::MaybeBool 
    under_15::MaybeBool 
    under_25::MaybeBool 
    under_35::MaybeBool 
    # btts
    btts::MaybeBool
    injury_time::MaybeFloat  # Added
  end 
  
  struct MatchHTResults 
    home::MaybeBool
    draw::MaybeBool
    away::MaybeBool
    correct_score::CorrectScore
    # under
    under_05::MaybeBool 
    under_15::MaybeBool 
    under_25::MaybeBool 
    injury_time::MaybeFloat  # Added
  end 
  
  struct MatchLinesResults
    ht::MatchHTResults
    ft::MatchFTResults
  end
end 

##################################################
# Extract the informations from the incidents 
##################################################
# Helper function to get injury time for a period
function _get_injury_time_for_period(df::AbstractDataFrame, period::AbstractString)
    period_to_minute = Dict("ht" => 45, "ft" => 90)
    target_minute = period_to_minute[period]
    
    # Find injury time record right after the period end
    injury_row = findfirst(
        row -> !ismissing(row.time) && 
               row.time == target_minute && 
               row.incident_type == "injuryTime" &&
               !ismissing(row.injury_time_length),
        eachrow(df)
    )
    
    if isnothing(injury_row)
        return nothing
    else
        return df[injury_row, :injury_time_length]
    end
end
# helper function to get time
function _get_scores_for_period(df::AbstractDataFrame, period::AbstractString)
  period_to_minute::Dict{AbstractString,Int64} = Dict("ht"=>45, "ft"=>90)
return findfirst(
                  row -> !ismissing(row.time) && row.time == period_to_minute[period] && row.incident_type == "period",
      eachrow(df)
  )
end

function _return_nothing_period(period::AbstractString)
  if period == "ht"
    cs_ht = Dict{Union{Tuple{Int,Int}, String}, Results.MaybeBool}(
      (0,0) => nothing, (1,0) => nothing, (2,0) => nothing,
      (0,1) => nothing, (1,1) => nothing, (2,1) => nothing,
      (0,2) => nothing, (1,2) => nothing, (2,2) => nothing,
      "any_unquoted" => nothing
    )
    return Results.MatchHTResults(
      nothing, nothing, nothing, cs_ht,
      nothing, nothing, nothing,
      nothing  # injury_time
    )
  else  # ft
    cs_ft = Dict{Union{Tuple{Int,Int}, String}, Results.MaybeBool}(
      (0,0) => nothing, (1,0) => nothing, (2,0) => nothing, (3,0) => nothing,
      (0,1) => nothing, (1,1) => nothing, (2,1) => nothing, (3,1) => nothing,
      (0,2) => nothing, (1,2) => nothing, (2,2) => nothing, (3,2) => nothing,
      (0,3) => nothing, (1,3) => nothing, (2,3) => nothing, (3,3) => nothing,
      "other_home_win" => nothing, "other_away_win" => nothing, "other_draw" => nothing
    )
    return Results.MatchFTResults(
      nothing, nothing, nothing, cs_ft,
      nothing, nothing, nothing, nothing, nothing,
      nothing  # injury_time
    )
  end
end

function process_period_data_ht(score_row, injury_time::Union{Float64, Nothing})
  if isnothing(score_row)
    return _return_nothing_period("ht")
  end

  home_score = score_row.home_score
  away_score = score_row.away_score

  if ismissing(home_score) || ismissing(away_score)
    return _return_nothing_period("ht")
  end

  home_win = home_score > away_score
  away_win = away_score > home_score
  draw = home_score == away_score
  total_goals = home_score + away_score
  under_05 = total_goals < 0.5
  under_15 = total_goals < 1.5
  under_25 = total_goals < 2.5

  correct_score = Dict{Union{Tuple{Int,Int}, String}, Results.MaybeBool}(
    (0,0) => false, (1,0) => false, (2,0) => false,
    (0,1) => false, (1,1) => false, (2,1) => false,
    (0,2) => false, (1,2) => false, (2,2) => false,
    "any_unquoted" => false
  )
  
  if home_score <= 2 && away_score <= 2
    correct_score[(Integer(home_score),Integer(away_score))] = true
  else
    correct_score["any_unquoted"] = true
  end

  return Results.MatchHTResults(
    home_win, draw, away_win,
    correct_score,
    under_05, under_15, under_25,
    injury_time
  )
end

function process_period_data_ft(score_row, injury_time::Union{Float64, Nothing})
  if isnothing(score_row)
    return _return_nothing_period("ft")
  end

  home_score = score_row.home_score
  away_score = score_row.away_score

  if ismissing(home_score) || ismissing(away_score)
    return _return_nothing_period("ft")
  end

  home_win = home_score > away_score
  away_win = away_score > home_score
  draw = home_score == away_score
  total_goals = home_score + away_score
  under_05 = total_goals < 0.5
  under_15 = total_goals < 1.5
  under_25 = total_goals < 2.5
  under_35 = total_goals < 3.5
  btts = home_score > 0 && away_score > 0

  correct_score = Dict{Union{Tuple{Int,Int}, String}, Results.MaybeBool}(
    (0,0) => false, (1,0) => false, (2,0) => false, (3,0) => false,
    (0,1) => false, (1,1) => false, (2,1) => false, (3,1) => false,
    (0,2) => false, (1,2) => false, (2,2) => false, (3,2) => false,
    (0,3) => false, (1,3) => false, (2,3) => false, (3,3) => false,
    "other_home_win" => false, "other_away_win" => false, "other_draw" => false
  )
  
  if home_score <= 3 && away_score <= 3
    correct_score[(Integer(home_score), Integer(away_score))] = true
  else
    if home_score > away_score
      correct_score["other_home_win"] = true
    elseif away_score > home_score
      correct_score["other_away_win"] = true
    else
      correct_score["other_draw"] = true
    end
  end

  return Results.MatchFTResults(
    home_win, draw, away_win,
    correct_score,
    under_05, under_15, under_25, under_35,
    btts,
    injury_time
  )
end



"""
    get_match_results(df::DataFrame, match_id::Int)

Processes a DataFrame of match incidents to determine half-time and full-time
results based on the scores at the end of each period.

Returns an instance of `Results.MatchLinesResults` populated with boolean
indicators for win/loss/draw, correct score, and over/under markets.
"""
function get_match_results(data_store::DataStore, match_id::Int)
    match_incidents = sort(
        filter(row -> row.match_id == match_id, data_store.incidents),
        :time
    )
    
    ht_row_idx = _get_scores_for_period(match_incidents, "ht")
    ft_row_idx = _get_scores_for_period(match_incidents, "ft")
    
    # Get injury times
    ht_injury_time = _get_injury_time_for_period(match_incidents, "ht")
    ft_injury_time = _get_injury_time_for_period(match_incidents, "ft")
    
    ht_results = if isnothing(ht_row_idx)
        _return_nothing_period("ht")
    else
        process_period_data_ht(match_incidents[ht_row_idx, :], ht_injury_time)
    end
    
    ft_results = if isnothing(ft_row_idx)
        _return_nothing_period("ft")
    else
        process_period_data_ft(match_incidents[ft_row_idx, :], ft_injury_time)
    end
    
    return Results.MatchLinesResults(ht_results, ft_results)
end

using Base.Threads

"""
    process_matches_threaded(data_store::DataStore, target_matches::DataFrame)

Process multiple matches in parallel using threading.
Returns a Dict{Int64, Results.MatchLinesResults} mapping match_id to results.
"""
function process_matches_results(data_store::DataStore, target_matches::DataFrame)
    match_ids = target_matches.match_id
    n_matches = length(match_ids)
    n_threads = nthreads()
    
    # Each thread gets its own dictionary
    thread_results = Vector{Dict{Int64, Results.MatchLinesResults}}(undef, n_threads)
    
    @threads for tid in 1:n_threads
        # Each thread creates its own dictionary
        local_dict = Dict{Int64, Results.MatchLinesResults}()
        
        # Thread tid processes every n_threads-th match
        for idx in tid:n_threads:n_matches
            match_id = match_ids[idx]
            try
                match_result = get_match_results(data_store, match_id)
                local_dict[match_id] = match_result
            catch e
                @warn "Failed to process match $match_id" exception=e
            end
        end
        
        thread_results[tid] = local_dict
    end
    
    # Merge all thread dictionaries
    return merge(thread_results...)
end
