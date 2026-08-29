# BRIEFING: UNIFIED PORTFOLIO & STAKING FRAMEWORK (`09_unified_portfolio_framework`)

> **Objective:** Build, test, and verify the `Unified Portfolio & Staking Framework` in `current_development/09_unified_portfolio_framework/`. This prototype modernizes `src/Portfolio/` by replacing slow DataFrame row-by-row unboxing and per-match $(12 \times 12 \times 3200)$ tensor allocations with zero-allocation score grids (`06`) and direct `Fit` integration (`07`), adding first-class MCMC convergence gating, while guaranteeing **100% backward compatibility** for legacy `Portfolio` callers.

---

## 1. Problem Statement & Motivation

### Why `src/Portfolio/` Must Be Modernized:
1. **Slow DataFrame Unboxing**: In `src/Portfolio/book.jl` (`build_book` / `build_books`), the hot loop iterates over `latents.df` row by row, unboxing `Vector{Any}` from DataFrame cells on every match via `Predictions.extract_params`.
2. **Massive Per-Fixture Allocations**: For every match in a 500-match fold, `Predictions.compute_score_matrix` allocates a new $(12 \times 12 \times 3200)$ Float64 tensor on the heap, and dynamically allocates dictionary lookups (`Dict(string(m) => ...)`), causing heavy Garbage Collection (GC) pauses.
3. **No Convergence Gating**: The portfolio pipeline currently stakes and simulates models even if MCMC failed to converge ($\hat{R} > 1.10$, ESS $< 50$), staking real bankroll on uncalibrated chains.
4. **Disjoint Pricing Kernels**: Does not leverage the typed dense containers (`CountLatents`, `RecombLatents`, `SmileLatents`) and SIMD in-place score grid kernels (`compute_score_grid!`, `price_market!`) from `06_typed_posterior_latents`.

---

## 2. Target Directory & File Structure

Build the following modular files in `current_development/09_unified_portfolio_framework/`:

```
current_development/09_unified_portfolio_framework/
├── l01_types.jl            # Clean type hierarchy (BookSpec, PolicySpec, Selection, MatchBook, PortfolioResult, DailyState, Slate, Allocator configs, Risk models)
├── l02_book_builder.jl     # High-performance 0-allocation build_book & build_books over AbstractPosteriorLatents & Fit
├── l03_stake_and_simulate.jl # Kelly allocation solvers, shrinkage, multi-slate chronological bankroll simulation & metrics (CAGR, MDD, Sharpe, bootstrap CIs)
├── l04_compat_bridge.jl    # 100% backward-compatibility bridge for legacy Portfolio functions & types
├── l05_parity.jl           # Mathematical parity harness vs legacy src/Portfolio/ kernels
├── r01_demo.jl             # Deterministic verification runner (exercises all gates, 0-allocation benchmarks, parity, and backward compatibility)
└── README.md               # Complete architecture documentation & migration guide
```

---

## 3. Detailed Component Contracts

### 3.1 Type Hierarchy (`l01_types.jl`)
- Clean, concretely-typed specifications:
  - `BookSpec(markets, price, exec, shrink, allocator)`
  - `PolicySpec(allocator, caps, commission, risk, shrinkage, trust)`
  - `Selection(family, group, line, sel, odds_close, odds_settle, prob_model, prob_market)`
  - `MatchBook(match_id, date, selections, p_grid, payoff_matrix, settle_vector, raw_alloc, shrink_k, kkt, converged)`
  - `Slate(date, books::Vector{MatchBook})`
  - `PortfolioResult(daily_states, summary, metrics, bootstrap_ci)`

### 3.2 High-Performance 0-Allocation Book Builder (`l02_book_builder.jl`)
- Implements `build_books(spec::BookSpec, fit::Fit, odds_df, ds; require_result = true, require_converged = true)` and `build_books(spec::BookSpec, latents::AbstractPosteriorLatents, odds_df, matches_df)`.
- Pre-allocates a single `GridWorkspace` and `alloc_score_grid(latents)` across all matches:
  - Evaluates `compute_score_grid!(S, ws, latents, match_idx)` (**0 bytes allocated**, ~200 µs).
  - Evaluates `price_market!(book, S, market)` (**0 bytes allocated**, ~20 µs).
- Builds `MatchBook` and payoff matrix $R$ efficiently without unboxing DataFrames.
- **Convergence Gating**: Checks `fit.diagnostics.passed`. If `require_converged = true` and `fit` failed convergence, refuses to build books or flags with a clear error.

### 3.3 Kelly Staking & Multi-Slate Simulation (`l03_stake_and_simulate.jl`)
- `allocate`: Solves convex optimization for simultaneous multi-asset Kelly staking across correlated outcomes with shrinkage, caps, and commission.
- `simulate_portfolio(policy::PolicySpec, books::Vector{MatchBook}; initial_bankroll = 1000.0)`:
  - Groups books into chronological slates (`slates.jl`).
  - Simulates compounding bankroll over time with fractional Kelly and friction.
  - Calculates summary metrics: CAGR, Growth per Slate, Max Drawdown (MDD), Sharpe, Sortino, Win Rate, Full-Book ROI, 1X2 ROI, and 95% Bootstrap Confidence Intervals.

### 3.4 Backward Compatibility Bridge (`l04_compat_bridge.jl`)
- `module UnifiedPortfolio` exporting all legacy names and function signatures:
  - `build_books(spec, latents_df, expr, odds_df, ds)`
  - `build_book(spec, latents_row, expr, odds_df, fixtures)`
  - `simulate(spec, books)` / `run_portfolio_simulation`
  - `extract_selections`, `fixture_table`, `is_settled`
- Legacy call sites and scripts execute completely unmodified.

### 3.5 Mathematical Parity Harness (`l05_parity.jl`)
- Verifies exact mathematical parity ($|\Delta| < 10^{-12}$ / 0 ULP) against legacy `src/Portfolio/` across:
  1. Score grid distributions $p_{\text{grid}}$.
  2. Selection model probabilities and market fair prices.
  3. Payoff matrices $R$ and settlement vectors.
  4. Solved Kelly stakes $a$ and shrinkage factors $k$.
  5. Multi-slate simulation trajectories and final bankroll.

### 3.6 Deterministic Verification Runner (`r01_demo.jl`)
- Fast, zero-database runner verifying:
  1. 0-allocation book building (`@allocated build_book(...) == 0` or minimal memory).
  2. Mathematical parity against legacy `src/Portfolio/`.
  3. Multi-slate bankroll simulation and bootstrap confidence intervals.
  4. Convergence gating (`require_converged = true` refusing/warning on unconverged fits).
  5. 100% backward compatibility for legacy callers.

---

## 4. Execution Rules
- **Include Chain**: `l04` includes `l03` includes `l02` includes `l01` (which loads `08_unified_evaluation_framework` $\to$ `07_unified_inference_framework` $\to$ `06_typed_posterior_latents` $\to$ `05_composable_count_builder`).
- **Fast & Deterministic**: Run `r01_demo.jl` with synthetic fixtures/odds so it runs in seconds with clean ASCII summary tables and exits 0.
- **Zero Allocations on Hot Scoring Paths**: Leverage typed latents without allocating intermediate DataFrames.
