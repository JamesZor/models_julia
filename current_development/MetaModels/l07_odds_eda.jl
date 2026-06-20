# current_development/MetaModels/l07_odds_eda.jl
#
# Utilities for Exploratory Data Analysis (EDA) to compare 
# Bookmaker (Sofascore) vs Exchange (Betfair) odds coverage and pricing.

using DataFrames
using Statistics

"""
    compare_odds_coverage(sofa_odds::DataFrame, bf_odds::DataFrame, target_selections::Vector{Symbol})

Analyzes the coverage drop-off when moving from Sofascore to Betfair odds,
and calculates the distributional differences in price and implied probability.
Returns `(coverage_df, joined_df, stats_df)`.
"""
function compare_odds_coverage(sofa_odds::DataFrame, bf_odds::DataFrame, target_selections::Vector{Symbol}; total_matches::Int=0)
    # 1. Filter both DataFrames to our target markets
    sofa_filtered = subset(sofa_odds, :selection => ByRow(s -> s in target_selections))
    bf_filtered   = subset(bf_odds,   :selection => ByRow(s -> s in target_selections))
    
    # 2. Calculate Coverage (How many matches survive?)
    sofa_matches = combine(groupby(sofa_filtered, :selection), :match_id => length => :sofa_count)
    bf_matches   = combine(groupby(bf_filtered, :selection), :match_id => length => :bf_count)
    
    coverage_df = outerjoin(sofa_matches, bf_matches, on=:selection)
    coverage_df.bf_count   = coalesce.(coverage_df.bf_count, 0)
    coverage_df.sofa_count = coalesce.(coverage_df.sofa_count, 0)
    coverage_df.total_matches .= total_matches
    
    # Prevent division by zero
    coverage_df.bf_true_coverage_pct = [
        total_matches == 0 ? 0.0 : round((b / total_matches) * 100, digits=2) 
        for b in coverage_df.bf_count
    ]
    
    coverage_df.sofa_true_coverage_pct = [
        total_matches == 0 ? 0.0 : round((s / total_matches) * 100, digits=2) 
        for s in coverage_df.sofa_count
    ]
    
    println("\n" * "="^65)
    println("  ODDS COVERAGE REPORT (Betfair vs Sofascore)")
    println("="^65)
    display(coverage_df)
    
    # 3. Join the DataFrames to compare prices directly
    # We use makeunique=true to handle duplicate column names automatically
    joined = innerjoin(
        sofa_filtered[!, [:match_id, :selection, :odds_close, :prob_implied_close]],
        bf_filtered[!, [:match_id, :selection, :odds_close, :prob_implied_close]],
        on = [:match_id, :selection],
        makeunique = true
    )
    
    rename!(joined, 
        :odds_close => :sofa_odds, 
        :odds_close_1 => :bf_odds,
        :prob_implied_close => :sofa_prob,
        :prob_implied_close_1 => :bf_prob
    )
    
    # 4. Calculate Deltas
    joined.odds_diff = joined.bf_odds .- joined.sofa_odds
    joined.prob_diff = joined.bf_prob .- joined.sofa_prob
    
    println("\n" * "="^65)
    println("  ODDS DIFFERENCE DISTRIBUTION (Betfair - Sofascore)")
    println("="^65)
    
    stats_df = combine(groupby(joined, :selection),
        :sofa_odds => mean => :avg_sofa_odds,
        :bf_odds   => mean => :avg_bf_odds,
        :odds_diff => mean => :mean_odds_diff,
        :odds_diff => median => :median_odds_diff,
        :odds_diff => std => :std_odds_diff,
        :prob_diff => mean => :mean_prob_diff
    )
    
    # Format cleanly
    for col in names(stats_df)
        if col != "selection"
            stats_df[!, col] = round.(stats_df[!, col], digits=4)
        end
    end
    
    display(stats_df)
    println("="^65 * "\n")
    
    return coverage_df, joined, stats_df
end
