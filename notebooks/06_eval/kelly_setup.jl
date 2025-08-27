using BayesianFootball
using Statistics
using DataFrames
using Plots
using Printf

##################################################
# Configuration and Setup
##################################################

# Configuration for odds extraction pipeline
struct OddsExtractionConfig
    # Data quality thresholds
    max_jump_threshold::Float64      # Max allowed jump between consecutive odds
    min_samples_required::Int        # Min samples needed to consider valid
    normalization_target::Float64    # Target for odds normalization (typically 1.0)
    
    # Processing pipeline options
    handle_missing::Symbol           # :interpolate, :drop, :passthrough
    handle_jumps::Symbol            # :filter, :smooth, :passthrough
    normalize_odds::Bool            # Whether to normalize to sum to target
    
    # Time window settings
    default_time_range::UnitRange{Int64}
    adjust_for_goals::Bool
end

# Default configuration
const DEFAULT_ODDS_CONFIG = OddsExtractionConfig(
    0.3,                    # 30% max jump
    3,                      # Need at least 3 samples
    1.0,                    # Normalize to 1.0
    :interpolate,           # Interpolate missing values
    :filter,                # Filter out jumps
    true,                   # Normalize odds
    0:5,                    # 0-5 minutes window
    true                    # Adjust window for goals
)

##################################################
# Extended Types for Full Market Coverage
##################################################

module ExtendedEval
    # Core odds structures
    struct Odds1x2
        home::Float64
        draw::Float64
        away::Float64
        confidence::Float64  # Quality indicator
    end
    
    struct OddsOverUnder
        over::Float64
        under::Float64
        line::Float64
        confidence::Float64
    end
    
    struct CorrectScoreOdds
        scores::Dict{Tuple{Int,Int}, Float64}  # (home, away) => odds
        any_other_home::Float64
        any_other_draw::Float64
        any_other_away::Float64
        confidence::Float64
    end
    
    # Complete market odds
    struct MarketOdds
        match_id::Int64
        ht_1x2::Union{Odds1x2, Nothing}
        ft_1x2::Union{Odds1x2, Nothing}
        ht_ou_05::Union{OddsOverUnder, Nothing}
        ht_ou_15::Union{OddsOverUnder, Nothing}
        ft_ou_15::Union{OddsOverUnder, Nothing}
        ft_ou_25::Union{OddsOverUnder, Nothing}
        ft_ou_35::Union{OddsOverUnder, Nothing}
        correct_score::Union{CorrectScoreOdds, Nothing}
        extraction_metadata::Dict{Symbol, Any}  # Track data quality issues
    end
    
    # Kelly criterion structures
    struct KellyFraction
        outcome::Symbol  # :home, :draw, :away, etc.
        fraction::Float64
        confidence_interval::Tuple{Float64, Float64}  # CI from Bayesian posterior
    end
    
    struct OptimalAllocation
        match_id::Int64
        period::Symbol  # :ht or :ft
        market::Symbol   # :match_1x2, :over_under, :correct_score
        allocations::Vector{KellyFraction}
        total_stake::Float64
        expected_growth::Float64
    end
    
    # Betting decision structures  
    struct BetDecision
        placed::Bool
        outcome::Symbol
        stake::Float64  # Now variable based on Kelly
        odds::Float64
        expected_value::Float64
        kelly_fraction::Float64
    end
    
    struct BetResult
        decision::BetDecision
        won::Bool
        pnl::Float64  # Profit/loss accounting for stake size
        roi::Float64
    end
    
    struct MatchBets
        match_id::Int64
        ht_bets::Vector{BetResult}
        ft_bets::Vector{BetResult}
        total_stake::Float64
        total_pnl::Float64
    end
end

##################################################
# Enhanced Odds Extraction with Data Quality Handling
##################################################

"""
Enhanced odds extraction with configurable data quality pipeline
"""
function extract_market_odds(
    match_id::Int64,
    data::Union{DataStore, AbstractDataFrame},
    config::OddsExtractionConfig = DEFAULT_ODDS_CONFIG,
    score_timeline = nothing
)
    # Get the raw odds data
    odds_df = if isa(data, DataStore)
        data.odds
    else
        data
    end
    
    # Adjust time range for goals if needed
    time_range = config.adjust_for_goals ? 
        adjust_time_range_for_goals(config.default_time_range, score_timeline) : 
        config.default_time_range
    
    # Filter for this match and time range
    match_odds = filter(row -> row.sofa_match_id == match_id && 
                              row.minutes in time_range, odds_df)
    
    if isempty(match_odds)
        return ExtendedEval.MarketOdds(
            match_id, nothing, nothing, nothing, nothing, 
            nothing, nothing, nothing, nothing,
            Dict(:error => "No odds data found")
        )
    end
    
    # Sort by time
    sort!(match_odds, :minutes)
    
    # Extract metadata about data quality
    metadata = analyze_odds_quality(match_odds, config)
    
    # Process each market type
    ft_1x2 = process_1x2_odds(match_odds, [:home, :draw, :away], config)
    ht_1x2 = process_1x2_odds(match_odds, [:ht_home, :ht_draw, :ht_away], config)
    
    # Process over/under markets
    ht_ou_05 = process_ou_odds(match_odds, :ht_over_0_5, :ht_under_0_5, 0.5, config)
    ht_ou_15 = process_ou_odds(match_odds, :ht_over_1_5, :ht_under_1_5, 1.5, config)
    ft_ou_15 = process_ou_odds(match_odds, :over_1_5, :under_1_5, 1.5, config)
    ft_ou_25 = process_ou_odds(match_odds, :over_2_5, :under_2_5, 2.5, config)
    ft_ou_35 = process_ou_odds(match_odds, :over_3_5, :under_3_5, 3.5, config)
    
    # Process correct score
    correct_score = process_correct_score_odds(match_odds, config)
    
    return ExtendedEval.MarketOdds(
        match_id, ht_1x2, ft_1x2, ht_ou_05, ht_ou_15,
        ft_ou_15, ft_ou_25, ft_ou_35, correct_score, metadata
    )
end

"""
Process 1x2 odds with quality checks and normalization
"""
function process_1x2_odds(
    odds_data::DataFrame,
    columns::Vector{Symbol},
    config::OddsExtractionConfig
)
    # Check if columns exist
    if !all(col -> col in names(odds_data), columns)
        return nothing
    end
    
    # Extract odds series
    odds_series = Dict(col => Float64[] for col in columns)
    
    for row in eachrow(odds_data)
        for col in columns
            val = row[col]
            if !ismissing(val) && !isnan(val)
                push!(odds_series[col], float(val))
            end
        end
    end
    
    # Check minimum samples
    if any(length(v) < config.min_samples_required for v in values(odds_series))
        if config.handle_missing == :passthrough
            return nothing
        end
    end
    
    # Clean the odds series
    cleaned_series = Dict{Symbol, Vector{Float64}}()
    for (col, series) in odds_series
        cleaned = clean_odds_series(series, config)
        if !isnothing(cleaned)
            cleaned_series[col] = cleaned
        end
    end
    
    if length(cleaned_series) < length(columns)
        return nothing
    end
    
    # Calculate representative odds (median more robust than mean)
    final_odds = Dict(col => median(cleaned_series[col]) for col in columns)
    
    # Calculate confidence based on data quality
    confidence = calculate_confidence(cleaned_series, config)
    
    # Normalize if requested
    if config.normalize_odds
        final_odds = normalize_odds_dict(final_odds, config.normalization_target)
    end
    
    return ExtendedEval.Odds1x2(
        final_odds[columns[1]],  # home
        final_odds[columns[2]],  # draw  
        final_odds[columns[3]],  # away
        confidence
    )
end

"""
Clean odds series by removing jumps and outliers
"""
function clean_odds_series(series::Vector{Float64}, config::OddsExtractionConfig)
    if isempty(series)
        return nothing
    end
    
    if length(series) == 1
        return series
    end
    
    cleaned = Float64[series[1]]
    
    for i in 2:length(series)
        # Check for jumps
        relative_change = abs(series[i] - series[i-1]) / series[i-1]
        
        if config.handle_jumps == :filter && relative_change > config.max_jump_threshold
            # Skip this value
            continue
        elseif config.handle_jumps == :smooth && relative_change > config.max_jump_threshold
            # Use moving average
            push!(cleaned, mean([series[i-1], series[i]]))
        else
            push!(cleaned, series[i])
        end
    end
    
    return length(cleaned) >= config.min_samples_required ? cleaned : nothing
end

"""
Normalize odds to sum to target (accounting for margin)
"""
function normalize_odds_dict(odds_dict::Dict, target::Float64)
    implied_probs = Dict(k => 1/v for (k, v) in odds_dict)
    total_prob = sum(values(implied_probs))
    
    if total_prob ≈ 0
        return odds_dict
    end
    
    # Scale probabilities to sum to target
    scaled_probs = Dict(k => v * target / total_prob for (k, v) in implied_probs)
    
    # Convert back to odds
    return Dict(k => 1/v for (k, v) in scaled_probs)
end

"""
Process over/under odds
"""
function process_ou_odds(
    odds_data::DataFrame,
    over_col::Symbol,
    under_col::Symbol,
    line::Float64,
    config::OddsExtractionConfig
)
    if !(over_col in names(odds_data)) || !(under_col in names(odds_data))
        return nothing
    end
    
    ou_odds = process_1x2_odds(
        odds_data,
        [over_col, under_col, :dummy],  # dummy for 3rd position
        config
    )
    
    if isnothing(ou_odds)
        return nothing
    end
    
    return ExtendedEval.OddsOverUnder(
        ou_odds.home,   # over
        ou_odds.draw,    # under
        line,
        ou_odds.confidence
    )
end

"""
Process correct score odds
"""
function process_correct_score_odds(odds_data::DataFrame, config::OddsExtractionConfig)
    score_cols = [Symbol("$(h)_$(a)") for h in 0:3 for a in 0:3]
    available_cols = filter(col -> col in names(odds_data), score_cols)
    
    if length(available_cols) < 4  # Need at least some scores
        return nothing
    end
    
    scores_dict = Dict{Tuple{Int,Int}, Float64}()
    
    for col in available_cols
        parts = split(String(col), "_")
        h, a = parse(Int, parts[1]), parse(Int, parts[2])
        
        series = Float64[]
        for val in odds_data[!, col]
            if !ismissing(val) && !isnan(val)
                push!(series, float(val))
            end
        end
        
        if !isempty(series)
            cleaned = clean_odds_series(series, config)
            if !isnothing(cleaned) && !isempty(cleaned)
                scores_dict[(h, a)] = median(cleaned)
            end
        end
    end
    
    # Process "any other" outcomes
    any_other_home = process_single_odds_column(odds_data, :any_other_home, config)
    any_other_draw = process_single_odds_column(odds_data, :any_other_draw, config)
    any_other_away = process_single_odds_column(odds_data, :any_other_away, config)
    
    confidence = calculate_confidence(
        Dict(:scores => collect(values(scores_dict))),
        config
    )
    
    return ExtendedEval.CorrectScoreOdds(
        scores_dict,
        something(any_other_home, 999.0),
        something(any_other_draw, 999.0),
        something(any_other_away, 999.0),
        confidence
    )
end

function process_single_odds_column(data::DataFrame, col::Symbol, config::OddsExtractionConfig)
    if !(col in names(data))
        return nothing
    end
    
    series = Float64[]
    for val in data[!, col]
        if !ismissing(val) && !isnan(val)
            push!(series, float(val))
        end
    end
    
    if isempty(series)
        return nothing
    end
    
    cleaned = clean_odds_series(series, config)
    return !isnothing(cleaned) && !isempty(cleaned) ? median(cleaned) : nothing
end

"""
Calculate confidence score based on data quality
"""
function calculate_confidence(series_dict::Dict, config::OddsExtractionConfig)
    if isempty(series_dict)
        return 0.0
    end
    
    scores = Float64[]
    
    for (_, series) in series_dict
        if !isempty(series)
            # Penalize high variance
            cv = std(series) / mean(series)  # Coefficient of variation
            cv_score = 1.0 / (1.0 + cv)
            
            # Penalize low sample count
            sample_score = min(1.0, length(series) / (2 * config.min_samples_required))
            
            push!(scores, 0.7 * cv_score + 0.3 * sample_score)
        end
    end
    
    return isempty(scores) ? 0.0 : mean(scores)
end
using BayesianFootball
using Statistics
using DataFrames
using Plots
using Printf

##################################################
# Configuration and Setup
##################################################

# Configuration for odds extraction pipeline
struct OddsExtractionConfig
    # Data quality thresholds
    max_jump_threshold::Float64      # Max allowed jump between consecutive odds
    min_samples_required::Int        # Min samples needed to consider valid
    normalization_target::Float64    # Target for odds normalization (typically 1.0)
    
    # Processing pipeline options
    handle_missing::Symbol           # :interpolate, :drop, :passthrough
    handle_jumps::Symbol            # :filter, :smooth, :passthrough
    normalize_odds::Bool            # Whether to normalize to sum to target
    
    # Time window settings
    default_time_range::UnitRange{Int64}
    adjust_for_goals::Bool
end

# Default configuration
const DEFAULT_ODDS_CONFIG = OddsExtractionConfig(
    0.3,                    # 30% max jump
    3,                      # Need at least 3 samples
    1.0,                    # Normalize to 1.0
    :interpolate,           # Interpolate missing values
    :filter,                # Filter out jumps
    true,                   # Normalize odds
    0:5,                    # 0-5 minutes window
    true                    # Adjust window for goals
)

##################################################
# Extended Types for Full Market Coverage
##################################################

module ExtendedEval
    # Core odds structures
    struct Odds1x2
        home::Float64
        draw::Float64
        away::Float64
        confidence::Float64  # Quality indicator
    end
    
    struct OddsOverUnder
        over::Float64
        under::Float64
        line::Float64
        confidence::Float64
    end
    
    struct CorrectScoreOdds
        scores::Dict{Tuple{Int,Int}, Float64}  # (home, away) => odds
        any_other_home::Float64
        any_other_draw::Float64
        any_other_away::Float64
        confidence::Float64
    end
    
    # Complete market odds
    struct MarketOdds
        match_id::Int64
        ht_1x2::Union{Odds1x2, Nothing}
        ft_1x2::Union{Odds1x2, Nothing}
        ht_ou_05::Union{OddsOverUnder, Nothing}
        ht_ou_15::Union{OddsOverUnder, Nothing}
        ft_ou_15::Union{OddsOverUnder, Nothing}
        ft_ou_25::Union{OddsOverUnder, Nothing}
        ft_ou_35::Union{OddsOverUnder, Nothing}
        correct_score::Union{CorrectScoreOdds, Nothing}
        extraction_metadata::Dict{Symbol, Any}  # Track data quality issues
    end
    
    # Kelly criterion structures
    struct KellyFraction
        outcome::Symbol  # :home, :draw, :away, etc.
        fraction::Float64
        confidence_interval::Tuple{Float64, Float64}  # CI from Bayesian posterior
    end
    
    struct OptimalAllocation
        match_id::Int64
        period::Symbol  # :ht or :ft
        market::Symbol   # :match_1x2, :over_under, :correct_score
        allocations::Vector{KellyFraction}
        total_stake::Float64
        expected_growth::Float64
    end
    
    # Betting decision structures  
    struct BetDecision
        placed::Bool
        outcome::Symbol
        stake::Float64  # Now variable based on Kelly
        odds::Float64
        expected_value::Float64
        kelly_fraction::Float64
    end
    
    struct BetResult
        decision::BetDecision
        won::Bool
        pnl::Float64  # Profit/loss accounting for stake size
        roi::Float64
    end
    
    struct MatchBets
        match_id::Int64
        ht_bets::Vector{BetResult}
        ft_bets::Vector{BetResult}
        total_stake::Float64
        total_pnl::Float64
    end
end

##################################################
# Enhanced Odds Extraction with Data Quality Handling
##################################################

"""
Enhanced odds extraction with configurable data quality pipeline
"""
function extract_market_odds(
    match_id::Int64,
    data::Union{DataStore, AbstractDataFrame},
    config::OddsExtractionConfig = DEFAULT_ODDS_CONFIG,
    score_timeline = nothing
)
    # Get the raw odds data
    odds_df = if isa(data, DataStore)
        data.odds
    else
        data
    end
    
    # Adjust time range for goals if needed
    time_range = config.adjust_for_goals ? 
        adjust_time_range_for_goals(config.default_time_range, score_timeline) : 
        config.default_time_range
    
    # Filter for this match and time range
    match_odds = filter(row -> row.sofa_match_id == match_id && 
                              row.minutes in time_range, odds_df)
    
    if isempty(match_odds)
        return ExtendedEval.MarketOdds(
            match_id, nothing, nothing, nothing, nothing, 
            nothing, nothing, nothing, nothing,
            Dict(:error => "No odds data found")
        )
    end
    
    # Sort by time
    sort!(match_odds, :minutes)
    
    # Extract metadata about data quality
    metadata = analyze_odds_quality(match_odds, config)
    
    # Process each market type
    ft_1x2 = process_1x2_odds(match_odds, [:home, :draw, :away], config)
    ht_1x2 = process_1x2_odds(match_odds, [:ht_home, :ht_draw, :ht_away], config)
    
    # Process over/under markets
    ht_ou_05 = process_ou_odds(match_odds, :ht_over_0_5, :ht_under_0_5, 0.5, config)
    ht_ou_15 = process_ou_odds(match_odds, :ht_over_1_5, :ht_under_1_5, 1.5, config)
    ft_ou_15 = process_ou_odds(match_odds, :over_1_5, :under_1_5, 1.5, config)
    ft_ou_25 = process_ou_odds(match_odds, :over_2_5, :under_2_5, 2.5, config)
    ft_ou_35 = process_ou_odds(match_odds, :over_3_5, :under_3_5, 3.5, config)
    
    # Process correct score
    correct_score = process_correct_score_odds(match_odds, config)
    
    return ExtendedEval.MarketOdds(
        match_id, ht_1x2, ft_1x2, ht_ou_05, ht_ou_15,
        ft_ou_15, ft_ou_25, ft_ou_35, correct_score, metadata
    )
end

"""
Process 1x2 odds with quality checks and normalization
"""
function process_1x2_odds(
    odds_data::DataFrame,
    columns::Vector{Symbol},
    config::OddsExtractionConfig
)
    # Check if columns exist
    if !all(col -> col in names(odds_data), columns)
        return nothing
    end
    
    # Extract odds series
    odds_series = Dict(col => Float64[] for col in columns)
    
    for row in eachrow(odds_data)
        for col in columns
            val = row[col]
            if !ismissing(val) && !isnan(val)
                push!(odds_series[col], float(val))
            end
        end
    end
    
    # Check minimum samples
    if any(length(v) < config.min_samples_required for v in values(odds_series))
        if config.handle_missing == :passthrough
            return nothing
        end
    end
    
    # Clean the odds series
    cleaned_series = Dict{Symbol, Vector{Float64}}()
    for (col, series) in odds_series
        cleaned = clean_odds_series(series, config)
        if !isnothing(cleaned)
            cleaned_series[col] = cleaned
        end
    end
    
    if length(cleaned_series) < length(columns)
        return nothing
    end
    
    # Calculate representative odds (median more robust than mean)
    final_odds = Dict(col => median(cleaned_series[col]) for col in columns)
    
    # Calculate confidence based on data quality
    confidence = calculate_confidence(cleaned_series, config)
    
    # Normalize if requested
    if config.normalize_odds
        final_odds = normalize_odds_dict(final_odds, config.normalization_target)
    end
    
    return ExtendedEval.Odds1x2(
        final_odds[columns[1]],  # home
        final_odds[columns[2]],  # draw  
        final_odds[columns[3]],  # away
        confidence
    )
end

"""
Clean odds series by removing jumps and outliers
"""
function clean_odds_series(series::Vector{Float64}, config::OddsExtractionConfig)
    if isempty(series)
        return nothing
    end
    
    if length(series) == 1
        return series
    end
    
    cleaned = Float64[series[1]]
    
    for i in 2:length(series)
        # Check for jumps
        relative_change = abs(series[i] - series[i-1]) / series[i-1]
        
        if config.handle_jumps == :filter && relative_change > config.max_jump_threshold
            # Skip this value
            continue
        elseif config.handle_jumps == :smooth && relative_change > config.max_jump_threshold
            # Use moving average
            push!(cleaned, mean([series[i-1], series[i]]))
        else
            push!(cleaned, series[i])
        end
    end
    
    return length(cleaned) >= config.min_samples_required ? cleaned : nothing
end

"""
Normalize odds to sum to target (accounting for margin)
"""
function normalize_odds_dict(odds_dict::Dict, target::Float64)
    implied_probs = Dict(k => 1/v for (k, v) in odds_dict)
    total_prob = sum(values(implied_probs))
    
    if total_prob ≈ 0
        return odds_dict
    end
    
    # Scale probabilities to sum to target
    scaled_probs = Dict(k => v * target / total_prob for (k, v) in implied_probs)
    
    # Convert back to odds
    return Dict(k => 1/v for (k, v) in scaled_probs)
end

"""
Process over/under odds
"""
function process_ou_odds(
    odds_data::DataFrame,
    over_col::Symbol,
    under_col::Symbol,
    line::Float64,
    config::OddsExtractionConfig
)
    if !(over_col in names(odds_data)) || !(under_col in names(odds_data))
        return nothing
    end
    
    ou_odds = process_1x2_odds(
        odds_data,
        [over_col, under_col, :dummy],  # dummy for 3rd position
        config
    )
    
    if isnothing(ou_odds)
        return nothing
    end
    
    return ExtendedEval.OddsOverUnder(
        ou_odds.home,   # over
        ou_odds.draw,    # under
        line,
        ou_odds.confidence
    )
end

"""
Process correct score odds
"""
function process_correct_score_odds(odds_data::DataFrame, config::OddsExtractionConfig)
    score_cols = [Symbol("$(h)_$(a)") for h in 0:3 for a in 0:3]
    available_cols = filter(col -> col in names(odds_data), score_cols)
    
    if length(available_cols) < 4  # Need at least some scores
        return nothing
    end
    
    scores_dict = Dict{Tuple{Int,Int}, Float64}()
    
    for col in available_cols
        parts = split(String(col), "_")
        h, a = parse(Int, parts[1]), parse(Int, parts[2])
        
        series = Float64[]
        for val in odds_data[!, col]
            if !ismissing(val) && !isnan(val)
                push!(series, float(val))
            end
        end
        
        if !isempty(series)
            cleaned = clean_odds_series(series, config)
            if !isnothing(cleaned) && !isempty(cleaned)
                scores_dict[(h, a)] = median(cleaned)
            end
        end
    end
    
    # Process "any other" outcomes
    any_other_home = process_single_odds_column(odds_data, :any_other_home, config)
    any_other_draw = process_single_odds_column(odds_data, :any_other_draw, config)
    any_other_away = process_single_odds_column(odds_data, :any_other_away, config)
    
    confidence = calculate_confidence(
        Dict(:scores => collect(values(scores_dict))),
        config
    )
    
    return ExtendedEval.CorrectScoreOdds(
        scores_dict,
        something(any_other_home, 999.0),
        something(any_other_draw, 999.0),
        something(any_other_away, 999.0),
        confidence
    )
end

function process_single_odds_column(data::DataFrame, col::Symbol, config::OddsExtractionConfig)
    if !(col in names(data))
        return nothing
    end
    
    series = Float64[]
    for val in data[!, col]
        if !ismissing(val) && !isnan(val)
            push!(series, float(val))
        end
    end
    
    if isempty(series)
        return nothing
    end
    
    cleaned = clean_odds_series(series, config)
    return !isnothing(cleaned) && !isempty(cleaned) ? median(cleaned) : nothing
end

"""
Calculate confidence score based on data quality
"""
function calculate_confidence(series_dict::Dict, config::OddsExtractionConfig)
    if isempty(series_dict)
        return 0.0
    end
    
    scores = Float64[]
    
    for (_, series) in series_dict
        if !isempty(series)
            # Penalize high variance
            cv = std(series) / mean(series)  # Coefficient of variation
            cv_score = 1.0 / (1.0 + cv)
            
            # Penalize low sample count
            sample_score = min(1.0, length(series) / (2 * config.min_samples_required))
            
            push!(scores, 0.7 * cv_score + 0.3 * sample_score)
        end
    end
    
    return isempty(scores) ? 0.0 : mean(scores)
end

function analyze_odds_quality(odds_data::DataFrame, config::OddsExtractionConfig)
    metadata = Dict{Symbol, Any}()
    
    # Count missing values
    missing_counts = Dict{Symbol, Int}()
    for col in names(odds_data)
        if col ∉ [:sofa_match_id, :betfair_match_id, :minutes, :timestamp]
            missing_counts[Symbol(col)] = count(ismissing, odds_data[!, col])
        end
    end
    metadata[:missing_counts] = missing_counts
    
    # Identify large jumps (only for numeric columns)
    jumps = Dict{Symbol, Vector{Float64}}()
    for col in names(odds_data)
        if col ∉ [:sofa_match_id, :betfair_match_id, :minutes, :timestamp]
            col_jumps = Float64[]
            vals = odds_data[!, col]
            
            # Check if column contains numeric data
            for i in 2:length(vals)
                if !ismissing(vals[i]) && !ismissing(vals[i-1])
                    # Try to convert to Float64, skip if not numeric
                    try
                        v1 = float(vals[i-1])
                        v2 = float(vals[i])
                        
                        # Check for valid odds values (should be > 1)
                        if v1 > 1.0 && v2 > 1.0
                            change = abs(v2 - v1) / v1
                            if change > config.max_jump_threshold
                                push!(col_jumps, change)
                            end
                        end
                    catch
                        # Skip non-numeric values
                        continue
                    end
                end
            end
            
            if !isempty(col_jumps)
                jumps[Symbol(col)] = col_jumps
            end
        end
    end
    metadata[:large_jumps] = jumps
    
    # Record time range used
    metadata[:time_range] = (minimum(odds_data.minutes), maximum(odds_data.minutes))
    metadata[:num_samples] = nrow(odds_data)
    
    return metadata
end

##################################################
# Kelly Criterion Implementation
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
Simultaneous Kelly for multiple correlated outcomes in same market
"""
function calculate_simultaneous_kelly(
    model_probs::Dict{Symbol, Vector{Float64}},  # :home => probs, :draw => probs, etc.
    market_odds::Dict{Symbol, Float64},
    bankroll::Float64 = 1.0,
    fractional_kelly::Float64 = 0.25
)
    n_samples = length(first(values(model_probs)))
    outcomes = collect(keys(model_probs))
    
    # Calculate Kelly fractions for each sample
    sample_allocations = Vector{Dict{Symbol, Float64}}()
    
    for i in 1:n_samples
        # Get probabilities for this sample
        probs = Dict(out => model_probs[out][i] for out in outcomes)
        
        # Solve for optimal allocation
        allocation = solve_kelly_system(probs, market_odds)
        push!(sample_allocations, allocation)
    end
    
    # Aggregate across samples
    final_allocations = ExtendedEval.KellyFraction[]
    total_stake = 0.0
    
    for outcome in outcomes
        fractions = [alloc[outcome] for alloc in sample_allocations]
        
        # Use conservative percentile
        median_frac = quantile(fractions, 0.5) * fractional_kelly
        ci = (quantile(fractions, 0.25), quantile(fractions, 0.75))
        
        if median_frac > 0.001  # Minimum threshold
            push!(final_allocations, ExtendedEval.KellyFraction(outcome, median_frac, ci))
            total_stake += median_frac
        end
    end
    
    # Calculate expected growth rate
    expected_growth = calculate_expected_growth(final_allocations, model_probs, market_odds)
    
    return ExtendedEval.OptimalAllocation(
        0,  # match_id to be set later
        :unknown,  # period to be set later
        :match_1x2,
        final_allocations,
        total_stake,
        expected_growth
    )
end

"""
Solve Kelly system for simultaneous bets (simplified version)
"""
function solve_kelly_system(probs::Dict{Symbol, Float64}, odds::Dict{Symbol, Float64})
    # This is a simplified implementation
    # For full implementation, you'd solve the system of equations
    
    allocations = Dict{Symbol, Float64}()
    
    for (outcome, p) in probs
        if haskey(odds, outcome)
            o = odds[outcome]
            edge = p * o - 1.0
            if edge > 0
                f = edge / (o - 1.0)
                allocations[outcome] = f
            else
                allocations[outcome] = 0.0
            end
        else
            allocations[outcome] = 0.0
        end
    end
    
    # Normalize if total stake exceeds 1
    total = sum(values(allocations))
    if total > 1.0
        for outcome in keys(allocations)
            allocations[outcome] /= total
        end
    end
    
    return allocations
end

function calculate_expected_growth(
    allocations::Vector{ExtendedEval.KellyFraction},
    model_probs::Dict{Symbol, Vector{Float64}},
    market_odds::Dict{Symbol, Float64}
)
    if isempty(allocations)
        return 0.0
    end
    
    # Simplified expected log growth calculation
    growth = 0.0
    for alloc in allocations
        if alloc.fraction > 0
            p_mean = mean(model_probs[alloc.outcome])
            o = market_odds[alloc.outcome]
            
            # E[log(1 + f*(o-1)*X)] where X is Bernoulli(p)
            growth += p_mean * log(1 + alloc.fraction * (o - 1)) + 
                     (1 - p_mean) * log(1 - alloc.fraction)
        end
    end
    
    return growth
end

##################################################
# Enhanced Betting Decision Process
##################################################

"""
Make optimal betting decisions using Kelly criterion
"""
function make_kelly_betting_decisions(
    match_predictions::MatchPredict,
    market_odds::ExtendedEval.MarketOdds,
    bankroll::Float64,
    config::Dict{Symbol, Any} = Dict()
)
    decisions = ExtendedEval.MatchBets(
        market_odds.match_id,
        ExtendedEval.BetResult[],
        ExtendedEval.BetResult[],
        0.0,
        0.0
    )
    
    # Get configuration parameters
    fractional_kelly = get(config, :fractional_kelly, 0.25)
    min_edge = get(config, :min_edge, 0.02)  # Minimum 2% edge required
    min_confidence = get(config, :min_confidence, 0.7)  # Minimum odds confidence
    use_simultaneous = get(config, :use_simultaneous, true)
    
    # Process HT bets
    if !isnothing(market_odds.ht_1x2) && market_odds.ht_1x2.confidence >= min_confidence
        ht_decisions = process_1x2_kelly(
            Dict(:home => match_predictions.ht.home,
                 :draw => match_predictions.ht.draw,
                 :away => match_predictions.ht.away),
            Dict(:home => market_odds.ht_1x2.home,
                 :draw => market_odds.ht_1x2.draw,
                 :away => market_odds.ht_1x2.away),
            bankroll,
            fractional_kelly,
            min_edge,
            use_simultaneous
        )
        append!(decisions.ht_bets, ht_decisions)
    end
    
    # Process FT bets
    if !isnothing(market_odds.ft_1x2) && market_odds.ft_1x2.confidence >= min_confidence
        ft_decisions = process_1x2_kelly(
            Dict(:home => match_predictions.ft.home,
                 :draw => match_predictions.ft.draw,
                 :away => match_predictions.ft.away),
            Dict(:home => market_odds.ft_1x2.home,
                 :draw => market_odds.ft_1x2.draw,
                 :away => market_odds.ft_1x2.away),
            bankroll,
            fractional_kelly,
            min_edge,
            use_simultaneous
        )
        append!(decisions.ft_bets, ft_decisions)
    end
    
    # Calculate totals
    decisions = ExtendedEval.MatchBets(
        market_odds.match_id,
        decisions.ht_bets,
        decisions.ft_bets,
        sum(b.decision.stake for b in [decisions.ht_bets; decisions.ft_bets]),
        sum(b.pnl for b in [decisions.ht_bets; decisions.ft_bets])
    )
    
    return decisions
end

function process_1x2_kelly(
    model_probs::Dict{Symbol, Vector{Float64}},
    market_odds::Dict{Symbol, Float64},
    bankroll::Float64,
    fractional_kelly::Float64,
    min_edge::Float64,
    use_simultaneous::Bool
)
    decisions = ExtendedEval.BetResult[]
    
    if use_simultaneous
        # Calculate simultaneous Kelly allocation
        allocation = calculate_simultaneous_kelly(
            model_probs, market_odds, bankroll, fractional_kelly
        )
        
        for kelly_frac in allocation.allocations
            if kelly_frac.fraction > 0
                stake = bankroll * kelly_frac.fraction
                
                decision = ExtendedEval.BetDecision(
                    true,
                    kelly_frac.outcome,
                    stake,
                    market_odds[kelly_frac.outcome],
                    mean(model_probs[kelly_frac.outcome]) * market_odds[kelly_frac.outcome] - 1,
                    kelly_frac.fraction
                )
                
                # Create placeholder result (actual outcome unknown at decision time)
                push!(decisions, ExtendedEval.BetResult(decision, false, 0.0, 0.0))
            end
        end
    else
        # Traditional approach: bet on single best outcome
        best_outcome = nothing
        best_kelly = ExtendedEval.KellyFraction(:none, 0.0, (0.0, 0.0))
        
        for (outcome, probs) in model_probs
            kelly = calculate_bayesian_kelly(probs, market_odds[outcome], fractional_kelly)
            
            if kelly.fraction > best_kelly.fraction
                best_kelly = kelly
                best_outcome = outcome
            end
        end
        
        if !isnothing(best_outcome) && best_kelly.fraction > 0
            stake = bankroll * best_kelly.fraction
            
            decision = ExtendedEval.BetDecision(
                true,
                best_outcome,
                stake,
                market_odds[best_outcome],
                mean(model_probs[best_outcome]) * market_odds[best_outcome] - 1,
                best_kelly.fraction
            )
            
            push!(decisions, ExtendedEval.BetResult(decision, false, 0.0, 0.0))
        end
    end
    
    return decisions
end

##################################################
# Update bet results with actual outcomes
##################################################

function update_bet_results!(
    bets::ExtendedEval.MatchBets,
    scoreline::Eval.MatchScoreTimeline
)
    # Update HT results
    ht_score = get_ht_score(scoreline)
    for i in 1:length(bets.ht_bets)
        bet = bets.ht_bets[i]
        won = check_outcome_win(bet.decision.outcome, ht_score, :ht)
        
        pnl = won ? 
            bet.decision.stake * (bet.decision.odds - 1) * 0.98 :  # 2% commission
            -bet.decision.stake
        
        roi = pnl / bet.decision.stake
        
        bets.ht_bets[i] = ExtendedEval.BetResult(bet.decision, won, pnl, roi)
    end
    
    # Update FT results
    for i in 1:length(bets.ft_bets)
        bet = bets.ft_bets[i]
        won = check_outcome_win(bet.decision.outcome, scoreline.final_score, :ft)
        
        pnl = won ? 
            bet.decision.stake * (bet.decision.odds - 1) * 0.98 :
            -bet.decision.stake
        
        roi = pnl / bet.decision.stake
        
        bets.ft_bets[i] = ExtendedEval.BetResult(bet.decision, won, pnl, roi)
    end
    
    # Update totals
    all_bets = [bets.ht_bets; bets.ft_bets]
    bets = ExtendedEval.MatchBets(
        bets.match_id,
        bets.ht_bets,
        bets.ft_bets,
        sum(b.decision.stake for b in all_bets),
        sum(b.pnl for b in all_bets)
    )
    
    return bets
end

function check_outcome_win(outcome::Symbol, score::Tuple{Int,Int}, period::Symbol)
    home_score, away_score = score
    
    if outcome == :home
        return home_score > away_score
    elseif outcome == :draw
        return home_score == away_score
    elseif outcome == :away
        return home_score < away_score
    else
        # Handle other outcome types (over/under, correct score, etc.)
        return false
    end
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

##################################################
# Main Analysis Pipeline
##################################################

"""
Run complete Kelly-based betting analysis
"""
function run_kelly_analysis(
    target_matches::DataFrame,
    result::ExperimentResult,
    data_store::DataStore,
    initial_bankroll::Float64 = 1000.0,
    config::Dict{Symbol, Any} = Dict()
)
    println("="^80)
    println("KELLY CRITERION BETTING ANALYSIS")
    println("="^80)
    
    # Set up configuration
    odds_config = get(config, :odds_config, DEFAULT_ODDS_CONFIG)
    betting_config = get(config, :betting_config, Dict(
        :fractional_kelly => 0.25,
        :min_edge => 0.02,
        :min_confidence => 0.7,
        :use_simultaneous => true
    ))
    
    # Step 1: Generate predictions
    println("\n1. Generating match predictions...")
    matches_predictions = predict_target_season(target_matches, result, result.mapping)
    println("   Generated predictions for $(length(matches_predictions)) matches")
    
    # Step 2: Extract scorelines
    println("\n2. Extracting match scorelines...")
    matches_scorelines = get_target_score_time_lines(target_matches, data_store)
    println("   Extracted scorelines for $(length(matches_scorelines)) matches")
    
    # Step 3: Extract market odds with quality checks
    println("\n3. Extracting market odds with quality controls...")
    matches_odds = Dict{Int64, ExtendedEval.MarketOdds}()
    quality_issues = Dict{Symbol, Int}()
    
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
    
    println("   Extracted odds for $(length(matches_odds)) matches")
    if !isempty(quality_issues)
        println("   Quality issues detected:")
        for (issue, count) in quality_issues
            println("     - $issue: $count matches")
        end
    end
    
    # Step 4: Make betting decisions
    println("\n4. Making optimal betting decisions...")
    bankroll = initial_bankroll
    all_bets = ExtendedEval.MatchBets[]
    bankroll_history = Float64[bankroll]
    
    sorted_matches = sort(target_matches, :match_date)
    
    for row in eachrow(sorted_matches)
        match_id = row.match_id
        
        if haskey(matches_predictions, match_id) && haskey(matches_odds, match_id)
            predictions = matches_predictions[match_id]
            odds = matches_odds[match_id]
            
            # Make Kelly-based decisions
            match_bets = make_kelly_betting_decisions(
                predictions, odds, bankroll, betting_config
            )
            
            # Get actual results
            if haskey(matches_scorelines, match_id)
                update_bet_results!(match_bets, matches_scorelines[match_id])
            end
            
            # Update bankroll
            bankroll += match_bets.total_pnl
            push!(bankroll_history, bankroll)
            push!(all_bets, match_bets)
        end
    end
    
    println("   Processed $(length(all_bets)) matches with bets")
    
    # Step 5: Analysis and reporting
    println("\n5. Performance Analysis:")
    println("="^80)
    
    # Calculate summary statistics
    total_bets_placed = sum(length(mb.ht_bets) + length(mb.ft_bets) for mb in all_bets)
    total_stake = sum(mb.total_stake for mb in all_bets)
    total_pnl = sum(mb.total_pnl for mb in all_bets)
    final_bankroll = bankroll
    total_return = (final_bankroll - initial_bankroll) / initial_bankroll * 100
    
    # Win rate calculation
    wins = 0
    losses = 0
    for mb in all_bets
        for bet in [mb.ht_bets; mb.ft_bets]
            if bet.won
                wins += 1
            else
                losses += 1
            end
        end
    end
    win_rate = wins / (wins + losses) * 100
    
    # Calculate Sharpe ratio (simplified)
    returns = diff(bankroll_history) ./ bankroll_history[1:end-1]
    sharpe = if length(returns) > 0
        mean(returns) / std(returns) * sqrt(252)  # Annualized
    else
        0.0
    end
    
    # Maximum drawdown
    peak = bankroll_history[1]
    max_dd = 0.0
    for value in bankroll_history
        if value > peak
            peak = value
        end
        dd = (peak - value) / peak
        if dd > max_dd
            max_dd = dd
        end
    end
    
    println("Initial Bankroll:    \$$(round(initial_bankroll, digits=2))")
    println("Final Bankroll:      \$$(round(final_bankroll, digits=2))")
    println("Total P&L:           \$$(round(total_pnl, digits=2))")
    println("Total Return:        $(round(total_return, digits=2))%")
    println()
    println("Total Bets Placed:   $total_bets_placed")
    println("Total Stake:         \$$(round(total_stake, digits=2))")
    println("Win Rate:            $(round(win_rate, digits=1))%")
    println()
    println("Sharpe Ratio:        $(round(sharpe, digits=3))")
    println("Max Drawdown:        $(round(max_dd * 100, digits=2))%")
    
    # Plot results
    println("\n6. Generating performance plots...")
    plot_kelly_performance(bankroll_history, all_bets, sorted_matches.match_date)
    
    return Dict(
        :bets => all_bets,
        :bankroll_history => bankroll_history,
        :initial_bankroll => initial_bankroll,
        :final_bankroll => final_bankroll,
        :total_return => total_return,
        :win_rate => win_rate,
        :sharpe_ratio => sharpe,
        :max_drawdown => max_dd
    )
end

function plot_kelly_performance(
    bankroll_history::Vector{Float64},
    all_bets::Vector{ExtendedEval.MatchBets},
    dates::Vector
)
    # Create bankroll evolution plot
    p1 = plot(
        1:length(bankroll_history),
        bankroll_history,
        title = "Bankroll Evolution (Kelly Criterion)",
        xlabel = "Match Number",
        ylabel = "Bankroll (\$)",
        linewidth = 2,
        color = :blue,
        legend = :topleft,
        label = "Bankroll"
    )
    
    # Add starting bankroll reference
    hline!(p1, [bankroll_history[1]], 
           color = :red, 
           linestyle = :dash, 
           label = "Starting Bankroll")
    
    # Stake sizes over time
    stakes = [mb.total_stake for mb in all_bets]
    p2 = scatter(
        1:length(stakes),
        stakes,
        title = "Bet Stake Sizes (Dynamic Kelly)",
        xlabel = "Match Number",
        ylabel = "Stake (\$)",
        markersize = 2,
        alpha = 0.6,
        color = :green,
        label = "Stake"
    )
    
    # P&L distribution
    pnls = [mb.total_pnl for mb in all_bets]
    p3 = histogram(
        pnls,
        title = "P&L Distribution",
        xlabel = "P&L (\$)",
        ylabel = "Frequency",
        bins = 30,
        alpha = 0.7,
        color = :orange,
        label = "P&L"
    )
    
    # Kelly fraction distribution
    kelly_fracs = Float64[]
    for mb in all_bets
        for bet in [mb.ht_bets; mb.ft_bets]
            if bet.decision.placed
                push!(kelly_fracs, bet.decision.kelly_fraction)
            end
        end
    end
    
    p4 = histogram(
        kelly_fracs,
        title = "Kelly Fraction Distribution",
        xlabel = "Kelly Fraction",
        ylabel = "Frequency",
        bins = 30,
        alpha = 0.7,
        color = :purple,
        label = "Fraction"
    )
    
    # Combine plots
    combined = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 800))
    display(combined)
    
    return combined
end

