# BRIEFING: UNIFIED EVALUATION & METRICS FRAMEWORK (`08_unified_evaluation_framework`)

> **Objective:** Build, test, and verify the `Unified Evaluation & Metrics Framework` in `current_development/08_unified_evaluation_framework/`. This prototype modernizes `src/evaluation/` by replacing slow, untyped DataFrame joins and redundant feature re-extractions with high-speed typed metric kernels operating directly on `06_typed_posterior_latents` and `07_unified_inference_framework` (`Fit`), integrating first-class MCMC convergence gating, while guaranteeing **100% backward compatibility** for legacy `evaluate_experiments` and `compute_metric` callers.

---

## 1. Problem Statement & Motivation

### Why `src/evaluation/` Must Be Modernized:
1. **Redundant Feature Re-Extraction**: `evaluate_experiments` calls `extract_oos_predictions(ds, exp)` on every run, which re-derives boundaries from the `DataStore` and rebuilds feature sets instead of reading the already-extracted posterior latents.
2. **Slow DataFrame Inner Joins**: Every metric (`logloss.jl`, `lpd.jl`, `crps.jl`, `glm_edge.jl`) converts posterior distributions into a temporary DataFrame (`model_features`), joins with `ds.odds` on 4 columns (`:match_id, :market_name, :market_line, :selection`), and calls `dropmissing!`.
3. **No Convergence Gating**: Evaluation currently computes metrics on runs even if MCMC failed to converge ($\hat{R} > 1.10$, ESS $< 50$), allowing uncalibrated garbage chains into model comparison leaderboards.
4. **Disjoint Model Inference**: Metrics unpack `latents_raw.df` row by row rather than evaluating score grids and probabilities via the zero-allocation SIMD kernels from `06_typed_posterior_latents`.

---

## 2. Target Directory & File Structure

Build the following modular files in `current_development/08_unified_evaluation_framework/`:

```
current_development/08_unified_evaluation_framework/
├── l01_types.jl            # Clean type hierarchy (AbstractScoringRule, LogLoss, LPD, CRPS, RQR, GLMEdge, MIQ, MetricScorecard)
├── l02_scoring_rules.jl    # High-performance typed metric kernels over AbstractPosteriorLatents & Fit
├── l03_batch_runner.jl     # evaluate_fits(metrics, fits, ds) with convergence filtering & rich summary display tables
├── l04_compat_bridge.jl    # 100% backward-compatibility bridge for legacy evaluate_experiments & compute_metric
├── l05_parity.jl           # Mathematical parity harness vs legacy src/evaluation/ kernels
├── r01_demo.jl             # Deterministic verification runner (exercises all gates, benchmarks, parity, and backward compatibility)
└── README.md               # Complete architecture documentation & migration guide
```

---

## 3. Detailed Component Contracts

### 3.1 Type Hierarchy (`l01_types.jl`)

```julia
abstract type AbstractScoringRule end
abstract type AbstractEvaluationResult end

# Metric Triggers:
struct LogLoss <: AbstractScoringRule
    markets::Vector{<:Data.AbstractMarket}
end
LogLoss() = LogLoss([Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()])

struct LPD <: AbstractScoringRule end
struct CRPS <: AbstractScoringRule end
Base.@kwdef struct RQR <: AbstractScoringRule
    n_sims::Int = 1000
    seed::Int = 42
end
Base.@kwdef struct GLMEdge <: AbstractScoringRule
    target_selection::Symbol = :all   # :home, :draw, :away, or :all
    min_edge::Float64 = 0.0
end
struct MIQ <: AbstractScoringRule end

# Metric Result Structs:
struct LogLossResult <: AbstractEvaluationResult ... end
struct LPDResult <: AbstractEvaluationResult ... end
struct CRPSResult <: AbstractEvaluationResult ... end
struct RQRResult <: AbstractEvaluationResult ... end
struct GLMEdgeResult <: AbstractEvaluationResult ... end
struct MIQResult <: AbstractEvaluationResult ... end
```

### 3.2 High-Performance Metric Kernels (`l02_scoring_rules.jl`)
- Implement `compute_metric(metric, fit::Fit, ds::DataStore)` and `compute_metric(metric, latents::AbstractPosteriorLatents, odds_df, matches_df)`.
- **LogLoss**: Multi-class cross-entropy on model probabilities vs Betfair/Bet365 closing fair odds ($-\sum y_i \log p_i$) across 1X2, Over/Under, BTTS.
- **LPD / ELPD**: Log posterior density across realised match goal counts $(g_h, g_a)$:
  $$\text{LPD}_i = \log \left( \frac{1}{S} \sum_{s=1}^S P(G_h = g_{h,i}, G_a = g_{a,i} \mid \theta^{(s)}) \right)$$
  Evaluated directly against `CountLatents` / `RecombLatents` score grids.
- **CRPS**: Continuous Ranked Probability Score on home and away goal distributions.
- **RQR**: Randomized Quantile Residuals testing distributional calibration and skewness.
- **GLMEdge**: Logistic regression assessing whether model edge systematically predicts market mispricing.
- **MIQ**: Mutual Information Quality.

### 3.3 Batch Evaluation & Convergence Gating (`l03_batch_runner.jl`)
- `evaluate_fits(metrics::Vector{<:AbstractScoringRule}, fits::Vector{<:Fit}, ds::DataStore; require_converged::Bool = true)`:
  - Inspects `fit.diagnostics.passed` for each model fit.
  - Automatically flags or filters unconverged runs if `require_converged = true`.
  - Runs metric computations in parallel across fits (`Threads.@threads`).
  - Returns a unified `MetricScorecard` / `DataFrame` and displays formatted summary comparison tables.
- `display_summary_metric(scorecard, metric_family)`: Curated terminal output.

### 3.4 Backward Compatibility Bridge (`l04_compat_bridge.jl`)
- Preserves all legacy signatures:
  - `evaluate_experiments(metrics, experiments::Vector{ExperimentResults}, ds::DataStore)`
  - `compute_metric(metric, exp::ExperimentResults, ds::DataStore, latents_raw)`
  - `to_dataframe_row(exp, metric, result)`
- Callers passing legacy `ExperimentResults` or `LatentStates` continue to work without modification.

### 3.5 Mathematical Parity Harness (`l05_parity.jl`)
- Verifies exact mathematical agreement ($|\Delta| < 10^{-12}$) between the new typed metric kernels and the legacy `src/evaluation/` kernels across all metrics.

### 3.6 Verification Runner (`r01_demo.jl`)
- Deterministic, zero-database runner verifying:
  1. All 6 scoring rules (LogLoss, LPD, CRPS, RQR, GLMEdge, MIQ) computing correct values.
  2. Mathematical parity against legacy `src/evaluation/` methods.
  3. Convergence gating (`require_converged = true` correctly filtering or warning on unconverged fits).
  4. Memory and timing benchmarks showing speedups over legacy DataFrame joins.
  5. 100% backward compatibility with legacy `evaluate_experiments` and `compute_metric` calls.

---

## 4. Execution Rules
- **Loader/Runner Architecture**: Code definitions in `l01`–`l05`, execution solely in `r01_demo.jl`.
- **Fast & Deterministic**: Run `r01_demo.jl` with synthetic priors/fixtures so it completes in seconds with clean ASCII tables and exits 0.
- **Zero Allocations on Hot Scoring Paths**: Use typed latents without allocating intermediate DataFrames where possible.
