##################################################
# Load data and run analysis
##################################################

# Load the saved data
println("Loading experiment data...")
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

# Get target season matches
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)
println("Found $(nrow(target_matches)) matches in target season")

##################################################
# Later  
##################################################
# Run the Kelly analysis with custom configuration
config = Dict(
    :odds_config => OddsExtractionConfig(
        0.25,           # 25% max jump threshold
        3,              # Min 3 samples
        1.0,            # Normalize to 1.0
        :interpolate,   # Handle missing by interpolation
        :filter,        # Filter out jumps
        true,           # Normalize odds
        0:5,            # 0-5 minute window
        true            # Adjust for goals
    ),
    :betting_config => Dict(
        :fractional_kelly => 0.25,  # Use 25% Kelly
        :min_edge => 0.02,          # Minimum 2% edge
        :min_confidence => 0.7,      # 70% confidence in odds quality
        :use_simultaneous => true    # Allow simultaneous bets
    )
)

# Run the analysis
results = run_kelly_analysis(
    target_matches,
    result,
    data_store,
    1000.0,  # Starting bankroll
    config
)

println("\n" * "="*80)
println("Analysis complete!")
println("="*80)

######### de com need funcs 
#
using Base.Threads




module Eval
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
    return Eval.MatchScoreTimeline(home_goals, away_goals, sorted_timeline, final_score)
end

function match_score_times(match_id::Int64, data::AbstractDataFrame)
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

function get_target_score_time_lines(target::AbstractDataFrame, data_store::DataStore)
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

##################################################
# de construct 
##################################################
odds_config = get(config, :odds_config, DEFAULT_ODDS_CONFIG)
betting_config = get(config, :betting_config, Dict(
    :fractional_kelly => 0.25,
    :min_edge => 0.02,
    :min_confidence => 0.7,
    :use_simultaneous => true
))


matches_predictions = predict_target_season(target_matches, result, result.mapping)


matches_scorelines = get_target_score_time_lines(target_matches, data_store)

matches_odds = Dict{Int64, ExtendedEval.MarketOdds}()
quality_issues = Dict{Symbol, Int}()

m_id = first(keys(matches_predictions))

scoreline = get(matches_scorelines, m_id, nothing)

odds = extract_market_odds(m_id, data_store, odds_config, scoreline)

matches_odds[m_id] = odds
        
haskey(odds.extraction_metadata, :large_jumps)


jumps = odds.extraction_metadata[:large_jumps]

if !isempty(jumps)
    quality_issues[:large_jumps] = get(quality_issues, :large_jumps, 0) + 1
end
odds
    for match_id in keys(matches_predictions)
        scoreline = get(matches_scorelines, match_id, nothing)
        odds = extract_market_odds(match_id, data_store, odds_config, scoreline)
        matches_odds[match_id] = odds
        
        # Track quality issues
        if haskey(odds.extraction_metadata, :large_jumps)
            jumps = odds.extraction_metadata[:large_jumps]
            if !isempty(jumps)
                quality_issues[:large_jumps] = get(quality_issues, :large_jumps, 0) + 1
            end
        end
    end
    filter(row -> row.sofa_match_id== m_id && row.minutes ==0, data_store.odds)
