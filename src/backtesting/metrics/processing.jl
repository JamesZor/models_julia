# src/backtesting/metrics/processing.jl 

using DataFrames
using Statistics

export generate_tearsheet


GROUPBY_SYMBOLS = [:model_name, :model_parameters, :signal_name, :signal_params, :selection]
METRICS_VECTOR::Vector{AbstractWealthMetric} = [
          CumulativeWealth(),
          SharpeRatio(),
          SortinoRatio(),
          CalmarRatio(),
          BurkeRatio(),
          SterlingRatio()
]
DISTRIBUTIONAL_VECTOR::Vector{AbstractDistributionalMetric} = [
          BernoulliGammaHurdle()
]

"""
    generate_tearsheet(ledger::BacktestLedger, groupby_cols::Vector{Symbol}; 
                       metrics::WealthMetricsConfig=DEFAULT_WEALTH_METRICS)

Generates a summary table. Uses optimized DataFrame aggregation.
"""
function generate_tearsheet(ledger::BacktestLedger;
                            groupby_cols::Vector{Symbol} = GROUPBY_SYMBOLS,
                            metrics::Vector{<:AbstractWealthMetric} = METRICS_VECTOR,
                            dist_metrics::Vector{<:AbstractDistributionalMetric} = DISTRIBUTIONAL_VECTOR
                            )
    
    df = ledger.df
    
    # 1. Grouping (Low cost view creation)
    gdf = groupby(df, groupby_cols)

    # 2. Aggregation 
    # `combine` is highly optimized in DataFrames.jl
    results = combine(gdf) do sub_df
        # A. Basic Accounting (Vectorized)
        stats = _compute_basic_stats(sub_df)
        
        # B. Wealth Metrics (equity curve → scalar)
        wealth = _compute_wealth_metrics(sub_df.pnl, metrics)
        
        # C. Distributional Metrics (per-bet data → NamedTuple)
        dist = _compute_distributional_metrics(sub_df, dist_metrics)
        
        # Merge all three into one row (chained — merge only takes 2 args for NamedTuple)
        merge(merge(stats, wealth), dist)
    end
    
    return results
end

# --- Helpers ---

function _compute_basic_stats(df::AbstractDataFrame)
    # --- 1. Activity Metrics ---
    # Opportunities: Total number of match/market combinations seen
    _seen_bets = nrow(df)
    
    # Active Bets: Where we actually put money down (Stake > 0)
    # Using a tolerance (1e-6) is safer for Floats than strict > 0
    _active_mask = df.stake .> 1e-6
    _number_of_bets = count(_active_mask)
    
    # Activity Rate: % of opportunities we bet on
    _bet_pct = _seen_bets > 0 ? (_number_of_bets / _seen_bets * 100.0) : 0.0

    # --- 2. Financial Metrics ---
    _profit = sum(df.pnl)
    _stake = sum(df.stake)
    
    # ROI: Return on Investment (Profit / Total Staked)
    _roi = _stake > 1e-6 ? (_profit / _stake * 100.0) : 0.0

    # --- 3. Skill Metrics (Win Rate) ---
    # We only care about win rate for bets we actually PLACED.
    # Logic: Must be an active bet AND be a winner.
    # We use coalesce(val, false) to handle potential 'missing' values safely.
    _winning_bets = count(i -> _active_mask[i] && coalesce(df.is_winner[i], false), eachindex(_active_mask))
    
    _win_rate = _number_of_bets > 0 ? (_winning_bets / _number_of_bets * 100.0) : 0.0

    # --- Return Clean NamedTuple ---
    return (
        opportunities = _seen_bets,
        bets_placed   = _number_of_bets,
        activity_pct  = round(_bet_pct, digits=1),
        turnover      = round(_stake, digits=2),
        profit        = round(_profit, digits=2),
        roi_pct       = round(_roi, digits=2),
        win_rate_pct  = round(_win_rate, digits=1)
    )
end


function _compute_wealth_metrics(pnl_vector::AbstractVector, config::Vector{<:AbstractWealthMetric})
    # 1. Build Curve (Allocates once)
    # Assuming arithmetic cumulative sum for PnL
    equity_curve = cumsum(pnl_vector)
    
    # 2. Iterate Config
    # We build a generic Dict to hold dynamic columns
    results = Dict{Symbol, Float64}()
    
    for metric in config
    results[metric_symbol(metric)] = round(compute_metric(metric, equity_curve), digits=3)
    end
    
    return results
end

"""
    _compute_distributional_metrics(sub_df, config)

Runs each AbstractDistributionalMetric over the raw per-bet data.
Each metric returns a NamedTuple; all are merged into a single Dict.
"""
function _compute_distributional_metrics(sub_df::AbstractDataFrame, config::Vector{<:AbstractDistributionalMetric})
    results = Dict{Symbol, Any}()
    
    for metric in config
        nt = compute_distributional_metric(metric, sub_df)
        for (k, v) in pairs(nt)
            results[k] = v
        end
    end
    
    return results
end
