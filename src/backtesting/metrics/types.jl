"""
    AbstractWealthMetric
Root type for all advanced portfolio metrics (Sharpe, Drawdown, etc.).
Operates on equity curves and returns a single scalar.
"""
abstract type AbstractWealthMetric end

"""
    AbstractDistributionalMetric
Metrics that fit a distributional model to per-bet ROI data.
Returns a NamedTuple of fitted parameters rather than a single scalar.
Operates on the raw sub_df (needs :stake, :pnl columns).
"""
abstract type AbstractDistributionalMetric end

