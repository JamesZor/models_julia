# src/Data/betfair_util.jl

using DataFrames
using Statistics

# Define a trait-based system for price estimation
abstract type PriceEstimator end

struct MedianEstimator <: PriceEstimator end
struct MeanEstimator   <: PriceEstimator end
struct TWAEstimator    <: PriceEstimator end # Time-Weighted Average

# 1. Median: Robust to outliers
estimate_price(::MedianEstimator, prices, mins) = median(prices)

# 2. Mean: Simple average
estimate_price(::MeanEstimator, prices, mins) = mean(prices)

# 3. TWA: Weights prices by how long they stayed active
function estimate_price(::TWAEstimator, prices, mins)
    # Sort to ensure chronological order
    p = sortperm(mins)
    sorted_p = prices[p]
    sorted_m = mins[p]
    
    if length(sorted_m) < 2
        return sorted_p[1]
    end

    durations = Base.diff(sorted_m)
    
    # Assume the last price was valid for the average duration of previous ticks.
    push!(durations, mean(durations)) 
    
    return sum(sorted_p .* durations) / sum(durations)
end

"""
Summarizes the long_df into a ds.odds-style structure using a specific window and estimator.
"""
function summarize_odds(
    df_long::DataFrame, 
    estimator::PriceEstimator; 
    window::Tuple{Float64, Float64} = (-10.0, 0.0), # e.g., 10 mins before KO to KO
    min_ticks::Int = 1,
    overround_limits::Tuple{Float64, Float64} = (0.9, 1.10) # Efficiency filter
)
    # 1. Filter for the time window
    df_window = filter(row -> window[1] <= row.minutes_to_kickoff <= window[2], df_long)

    # 2. Group by match and selection to estimate the price
    summary = combine(groupby(df_window, [:match_id, :market_name, :market_line, :selection])) do sdf
        if nrow(sdf) < min_ticks
            return (odds = (missing),)
        end
        
        # Dispatch to our abstract estimator
        price = estimate_price(estimator, sdf.traded_price, sdf.minutes_to_kickoff)
        return (odds = price,)
    end

    # 3. Efficiency Filter: Cross-selection correlation check
    valid_matches = combine(groupby(summary, [:match_id, :market_name, :market_line])) do sdf
        overround = sum(1.0 ./ skipmissing(sdf.odds))
        is_sane = overround_limits[1] <= overround <= overround_limits[2]
        return (is_sane = is_sane, overround = overround)
    end

    # Join back and filter out the "insane" markets
    final_df = leftjoin(summary, valid_matches, on = [:match_id, :market_name, :market_line])
    return filter(row -> row.is_sane, final_df)
end

"""
Evaluates a betting selection against the actual match result.
Returns true if the bet won, false if it lost, and missing if no result exists.
"""
function grade_selection(
    market_name::String, 
    market_line::Float64, 
    selection::Symbol, 
    home_score::Union{Missing, Integer},
    away_score::Union{Missing, Integer}
)
    if ismissing(home_score) || ismissing(away_score)
        return missing
    end

    total_goals = home_score + away_score
    sel_str = string(selection)

    if market_name == "1X2"
        if selection == :home
            return home_score > away_score
        elseif selection == :draw
            return home_score == away_score
        elseif selection == :away
            return home_score < away_score
        end
    elseif market_name == "OverUnder"
        if startswith(sel_str, "over_")
            return total_goals > market_line
        elseif startswith(sel_str, "under_")
            return total_goals < market_line
        end
    elseif market_name == "BTTS"
        if selection == :btts_yes
            return home_score > 0 && away_score > 0
        elseif selection == :btts_no
            return home_score == 0 || away_score == 0
        end
    elseif market_name == "CorrectScore"
        if occursin(r"^cs_\d\d$", sel_str)
            h = parse(Int, sel_str[4])
            a = parse(Int, sel_str[5])
            return home_score == h && away_score == a
        elseif sel_str == "cs_any_other_home"
            return home_score > away_score && (home_score >= 4 || away_score >= 4)
        elseif sel_str == "cs_any_other_away"
            return home_score < away_score && (home_score >= 4 || away_score >= 4)
        elseif sel_str == "cs_any_other_draw"
            return home_score == away_score && home_score >= 4
        end
    elseif market_name == "DoubleChance"
        if selection == :DC_1X
            return home_score >= away_score
        elseif selection == :DC_X2
            return away_score >= home_score
        elseif selection == :DC_12
            return home_score != away_score
        end
    elseif market_name == "DrawNoBet"
        if home_score == away_score
            return missing            # draw -> stake refunded (push)
        elseif selection == :dnb_home
            return home_score > away_score
        elseif selection == :dnb_away
            return away_score > home_score
        end
    elseif market_name == "AsianHandicap"
        side, L = Markets.parse_ah_selection(selection)
        # v1: quarter lines split the stake -> not settleable with a Bool. Grade
        # missing so they are excluded from the backtest.
        Markets.ah_is_quarter(L) && return missing
        margin = home_score - away_score
        adj = (side === :home ? margin : -margin) + L
        return adj > 0 ? true : adj < 0 ? false : missing   # ==0 -> push
    end

    return missing
end

"""
Merges match results into the odds dataframe and calculates the `is_winner` flag.
"""
function append_match_results(odds_df::DataFrame, matches_df::DataFrame)
    target_cols = [:match_id, :home_score, :away_score, :match_date]
    merged_df = leftjoin(odds_df, matches_df[:, target_cols], on = :match_id)

    merged_df.is_winner = grade_selection.(
        merged_df.market_name, 
        merged_df.market_line, 
        merged_df.selection, 
        merged_df.home_score, 
        merged_df.away_score
    )

    rename!(merged_df, :match_date => :date)
    select!(merged_df, Not([:home_score, :away_score]))
    return merged_df
end

"""
    summarize_betfair_market(ds::DataStore; kwargs...) -> DataFrame
Takes the entire DataStore and produces a closed, summarized DataFrame of Betfair odds
matching the ds.odds schema, fully graded with match results.
"""
function summarize_betfair_market(
    ds::DataStore; 
    open_window=(-1440.0, -1380.0), 
    close_window=(-20.0, 0.0),      
    estimator=TWAEstimator()
)
    df_long = ds.betfair_odds
    
    if nrow(df_long) == 0
        return DataFrame()
    end

    # 1. Generate Open and Close summaries
    df_open = summarize_odds(df_long, estimator, window=open_window)
    rename!(df_open, :odds => :odds_open, :overround => :overround_open)
    
    df_close = summarize_odds(df_long, estimator, window=close_window)
    rename!(df_close, :odds => :odds_close, :overround => :overround_close)

    # 2. Join them
    final_df = innerjoin(
        df_open[:, [:match_id, :market_name, :market_line, :selection, :odds_open, :overround_open]],
        df_close[:, [:match_id, :market_name, :market_line, :selection, :odds_close, :overround_close]],
        on = [:match_id, :market_name, :market_line, :selection]
    )

    # 3. Calculate implied probabilities
    final_df.prob_implied_open  = 1.0 ./ final_df.odds_open
    final_df.prob_implied_close = 1.0 ./ final_df.odds_close

    # 4. Calculate Fair Probabilities (removing the vig)
    final_df.prob_fair_open  = final_df.prob_implied_open ./ final_df.overround_open
    final_df.prob_fair_close = final_df.prob_implied_close ./ final_df.overround_close

    # 5. Calculate Fair Odds
    final_df.fair_odds_open  = 1.0 ./ final_df.prob_fair_open
    final_df.fair_odds_close = 1.0 ./ final_df.prob_fair_close

    # 6. Calculate Vig (Vig = Overround - 1)
    final_df.vig_open  = final_df.overround_open .- 1.0
    final_df.vig_close = final_df.overround_close .- 1.0

    # 7. Append match results and is_winner
    return append_match_results(final_df, ds.matches)
end

"""
    check_betfair_coverage(ds::DataStore) -> NamedTuple
Analyzes the coverage of Betfair odds for the DataStore.
Returns a NamedTuple containing two DataFrames:
- `match_level`: Coverage stats per match (how many markets, total ticks, etc.)
- `market_level`: Coverage stats per market selection (coverage %, avg ticks per match, etc.)
"""
function check_betfair_coverage(ds::DataStore)
    if isempty(ds.matches) || isempty(ds.betfair_odds)
        @warn "DataStore matches or betfair_odds are empty."
        return (match_level=DataFrame(), market_level=DataFrame())
    end

    total_matches = nrow(ds.matches)

    # 1. MATCH LEVEL COVERAGE
    # Get tick counts per match and selection
    tick_counts = combine(groupby(ds.betfair_odds, [:match_id, :selection]), nrow => :ticks)
    
    match_coverage = combine(groupby(tick_counts, :match_id)) do sdf
        (
            markets_covered = nrow(sdf),
            total_ticks = sum(sdf.ticks),
            avg_ticks_per_market = mean(sdf.ticks)
        )
    end

    matches_summary = select(ds.matches, :match_id, :match_date, :home_team, :away_team)
    match_level = leftjoin(matches_summary, match_coverage, on=:match_id)
    
    match_level.has_odds = .!ismissing.(match_level.markets_covered)
    match_level.markets_covered = coalesce.(match_level.markets_covered, 0)
    match_level.total_ticks = coalesce.(match_level.total_ticks, 0)
    match_level.avg_ticks_per_market = coalesce.(match_level.avg_ticks_per_market, 0.0)

    # 2. MARKET LEVEL COVERAGE
    market_level = combine(groupby(tick_counts, :selection)) do sdf
        matches_with_market = nrow(sdf)
        (
            matches_covered = matches_with_market,
            coverage_pct = round(matches_with_market / total_matches * 100, digits=1),
            avg_ticks_per_match = round(mean(sdf.ticks), digits=1),
            min_ticks = minimum(sdf.ticks),
            max_ticks = maximum(sdf.ticks)
        )
    end
    sort!(market_level, :coverage_pct, rev=true)

    return (match_level = match_level, market_level = market_level)
end
