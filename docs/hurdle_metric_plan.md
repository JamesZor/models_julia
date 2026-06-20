# Bernoulli-Gamma Hurdle Metric — Audited Implementation Plan

## 1. Architecture Audit

### Current Metric Interface

The existing `compute_metric` contract is:

```julia
compute_metric(metric::AbstractWealthMetric, equity_curve::AbstractVector{<:Number}) → Float64
```

All 6 existing implementations ([sharpe.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/sharpe.jl), [sortino.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/sortino.jl), [calmar.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/calmar.jl), [burke.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/burke.jl), [sterling.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/sterling.jl), [cumulative_wealth.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/implentations/cumulative_wealth.jl)) follow this contract and return a single scalar.

The Hurdle metric breaks this contract in two ways:
1. **Input**: Needs per-bet `stake` and `pnl` columns, not the cumulative equity curve.
2. **Output**: Returns a NamedTuple of ~15 fitted parameters, not a single scalar.

### Recommended Approach: **New Abstract Type, Not Interface Refactoring**

> [!IMPORTANT]
> My original plan suggested refactoring the `compute_metric` signature to accept `sub_df`. After auditing the code, **I recommend against this**. Here's why:

- The 6 existing metrics are clean, tested, and operate on a fundamentally different object (a time-ordered equity curve). Changing their signature would require touching all 6 files for zero functional benefit.
- The Hurdle metric is conceptually different — it is a **distributional fit** on per-bet ROI data, not a risk-adjusted return ratio on a portfolio curve.
- Julia's multiple dispatch makes this trivial: we define a new abstract type and a new dispatch of the processing pipeline, keeping the existing code completely untouched.

```
AbstractWealthMetric          (existing — equity curve → scalar)
  ├─ SharpeRatio
  ├─ SortinoRatio
  ├─ ...
  
AbstractDistributionalMetric  (new — sub_df → NamedTuple)
  └─ BernoulliGammaHurdle
```

---

## 2. Mathematical Audit of Prototype

Reviewing the `fit_hurdle_roi` function from `l06_metrics.jl`:

### ✅ What's Correct
- The Bernoulli-Gamma decomposition: `R ~ p · Gamma(α, β) + (1-p) · δ(-1)` is mathematically sound for sports betting ROI.
- The Method-of-Moments fallback when `fit(Gamma, ...)` fails is essential for small samples.
- Empirical growth rate via `G_emp = exp(mean(log(1 + stake_i * roi_i))) - 1` correctly estimates the geometric growth rate.

### ⚠️ Suggested Improvements

#### A. Growth Rate Calculation — Stake Heterogeneity

The parametric growth rate currently uses `avg_stake` as a fixed fraction:

```julia
# Current (prototype)
g_param = (1-p) * log(1 - avg_stake) + p * mean(log(1 + avg_stake * y_samples))
```

This assumes all bets have the same stake size. In practice with Kelly staking, stakes vary wildly per bet. Two options:

1. **Simple fix**: Keep `avg_stake` but document the assumption.
2. **Better fix**: Sample from the empirical stake distribution alongside the Gamma ROI samples. This gives a more honest `G_param` that accounts for stake heterogeneity.

**Recommendation**: Option 1 for the initial implementation (simpler, still useful for ranking models). Add a `use_empirical_stakes::Bool = false` flag to the struct for future Option 2.

#### B. Edge Case: Zero-Variance Positive ROIs

```julia
# Current fallback
if var_pos == 0.0
    var_pos = 1e-4
end
```

When all winning bets have identical ROI (common with flat staking on similar odds), the Gamma fit degenerates. The `1e-4` fallback creates an artificially tight Gamma. 

**Recommendation**: Use `var_pos = max(var(pos_rois), (0.01 * μ_pos)^2)` — floor the variance at 1% of the mean squared. This scales with the actual ROI magnitude rather than being a fixed constant.

#### C. MC Seed Determinism

```julia
rng = Random.MersenneTwister(42)
```

Good for reproducibility. However, if `compute_metric` is called inside a `combine(gdf)` loop, each group will use the same seed and same samples. This is actually fine here since we're fitting independent distributions per group, but worth documenting.

**Recommendation**: Keep it. Deterministic results are more important than independent samples across groups for a diagnostic metric.

#### D. Numerical Protection on `log`

```julia
log_wealth_increments = log.(max.(1e-8, 1.0 .+ active_stakes .* rois))
```

If `active_stakes .* rois < -1.0` (losing more than the stake — shouldn't happen with standard betting but could with leveraged positions), this silently clamps to `log(1e-8) ≈ -18.4`. 

**Recommendation**: Add an assertion or warning: `@assert all(rois .>= -1.0) "ROI below -1.0 detected — check stake/pnl data integrity"`.

---

## 3. File-by-File Change Manifest

### Files to CREATE

| File | Purpose |
|------|---------|
| [types.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/types.jl) | Add `AbstractDistributionalMetric` abstract type |
| [interfaces.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/interfaces.jl) | Add `compute_distributional_metric` interface |
| `metrics/implentations/hurdle_roi.jl` | `BernoulliGammaHurdle` struct + `compute_distributional_metric` implementation |

### Files to MODIFY

| File | Change |
|------|--------|
| [processing.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/metrics/processing.jl) | Add distributional metrics vector + merge into `generate_tearsheet` pipeline |
| [backtesting-module.jl](file:///home/james/bet_project/BayesianFootball/src/backtesting/backtesting-module.jl) | Add `include("./metrics/implentations/hurdle_roi.jl")` |

### Files NOT touched
All 6 existing metric implementations remain untouched.

---

## 4. Detailed Design

### 4.1 New Type Hierarchy (`types.jl` — append)

```julia
"""
    AbstractDistributionalMetric
Metrics that fit a distributional model to per-bet ROI data.
Returns a NamedTuple of fitted parameters rather than a single scalar.
"""
abstract type AbstractDistributionalMetric end
```

### 4.2 New Interface (`interfaces.jl` — append)

```julia
"""
    compute_distributional_metric(metric, sub_df) → NamedTuple

Fits a distributional model to the per-bet data in `sub_df`.
Returns a NamedTuple whose keys become columns in the tearsheet.
Expects sub_df to have columns: :stake, :pnl
"""
function compute_distributional_metric(metric::AbstractDistributionalMetric, sub_df::AbstractDataFrame)
    error("Implementation missing for metric: $(typeof(metric))")
end
```

### 4.3 Implementation (`hurdle_roi.jl`)

```julia
Base.@kwdef struct BernoulliGammaHurdle <: AbstractDistributionalMetric
    mc_samples::Int = 100_000
    min_bets::Int = 5           # Minimum active bets to attempt fit
end
```

**Output columns** (prefixed with `hurdle_` to avoid collisions):

| Column | Type | Description |
|--------|------|-------------|
| `hurdle_p` | Float64 | Win probability (Bernoulli parameter) |
| `hurdle_shape` | Float64 | Gamma shape α |
| `hurdle_scale` | Float64 | Gamma scale β |
| `hurdle_E_R` | Float64 | Parametric expected ROI: `p·μ_pos - (1-p)` |
| `hurdle_sharpe` | Float64 | Parametric Sharpe: `E[R] / σ[R]` |
| `hurdle_G` | Float64 | **Parametric geometric growth rate** (primary metric of interest) |
| `hurdle_G_emp` | Float64 | Empirical geometric growth rate |
| `hurdle_n_bets` | Int | Number of active bets in this group |
| `hurdle_avg_stake` | Float64 | Mean stake size (for context) |

> [!TIP]
> The `hurdle_` prefix keeps the tearsheet scannable and avoids name collisions with the existing `win_rate_pct`, `roi_pct`, etc.

### 4.4 Processing Integration (`processing.jl`)

Add a second metrics vector and a new helper:

```julia
DISTRIBUTIONAL_VECTOR::Vector{AbstractDistributionalMetric} = [
    BernoulliGammaHurdle()
]
```

Add `_compute_distributional_metrics(sub_df, config)` — loops over the vector, calls `compute_distributional_metric`, and merges all returned NamedTuples into a single Dict.

Update `generate_tearsheet` to merge basic stats + wealth metrics + distributional metrics:

```julia
results = combine(gdf) do sub_df
    stats   = _compute_basic_stats(sub_df)
    wealth  = _compute_wealth_metrics(sub_df.pnl, metrics)
    dist    = _compute_distributional_metrics(sub_df, dist_metrics)
    merge(stats, wealth, dist)
end
```

---

## 5. Summary of Audit Corrections vs Original Plan

| Original Plan | Audit Correction | Rationale |
|---|---|---|
| Refactor `compute_metric` signature to accept `sub_df` | New `AbstractDistributionalMetric` type with its own dispatch | Zero disruption to existing 6 metrics; cleaner separation of concerns |
| Update all 6 existing metrics | Don't touch them | They work; new type hierarchy avoids the need |
| Use `var_pos = 1e-4` fallback | Use `max(var, (0.01 * μ)^2)` | Scales with ROI magnitude instead of fixed constant |
| Fixed `avg_stake` for G_param | Keep for v1 + add `use_empirical_stakes` flag | Pragmatic; documents the assumption explicitly |
| No data validation | Add `@assert rois >= -1.0` guard | Catches corrupt stake/pnl data early |
