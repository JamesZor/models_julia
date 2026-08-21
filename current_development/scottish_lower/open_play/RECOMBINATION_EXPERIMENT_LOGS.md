# Scottish Lower: Recombination Experiment Logs & Implementation Track

This file tracks the active implementation logs, AD performance profiling, smoke tests, and walk-forward evaluations for the Two-Stage Recombination Models.

---

## 1. Experiment Grid Registry

| Experiment Tag | Model Family | Likelihood | Recombination Branch | Concurrency / Hardware | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `goals_negbin_ctl_hl365_hs2` | Baseline | NB2 | None (Gross goals) | 16 threads (`mcmc-beast`) | ✅ Complete (1h 34m) | Incumbent baseline control |
| `goals_negbin_open_play_hl365_hs2` | Treatment | NB2 | None (Clean $y_{\text{np\_nog}}$) | 16 threads (`mcmc-beast`) | ✅ Complete (1h 58m) | Pure open-play, un-recombined |
| `goals_pois_ctl_hl365_hs2` | Fast Baseline | Poisson | None (Gross goals) | 16 threads (`mcmc-beast`) | ⏳ Pending | Poisson benchmark control |
| `recomb_pois_empirical_bayes` | Branch A | Poisson | Analytical EB Shrinkage | 16 threads (`mcmc-beast`) | ⏳ Pending | Open-play + EB ref/team rate |
| `recomb_pois_integrated_bayes`| Branch B | Poisson | Co-Trained Turing Engine | 16 threads (`mcmc-beast`) | ⏳ Pending | Integrated MCMC ref penalty |
| `recomb_negbin_empirical_bayes` | Branch A (Scaled) | NB2 | Analytical EB Shrinkage | 16 threads (`mcmc-beast`) | ⏳ Queued | Full Negative Binomial scaling |
| `recomb_negbin_integrated_bayes`| Branch B (Scaled) | NB2 | Co-Trained Turing Engine | 16 threads (`mcmc-beast`) | ⏳ Queued | Full Negative Binomial scaling |

---

## 2. AD Performance & Profiling Standards (`docs/turing_ad_performance_guide.md`)

All Turing engines must strictly satisfy the following criteria before launching 40-fold walk-forward sampling:

- [x] **Vectorized Broadcast Operations**: `logpdf.(Poisson.(λ), y)` wrapped in a single `TrackedArray` node.
- [x] **No Scalar Loops**: Zero for-loops inside `@model`.
- [x] **No Dynamic Conditionals**: Binary masks (`xg_mask`, `ref_mask`) used for optional/missing values.
- [x] **Zero-Copy Views for Parameters**: `view(gamma_ref, ref_indices)` to prevent intermediate array allocations.
- [x] **Continuous Numerical Bounds**: `clamp.(log_λ, -20.0, 20.0)` and `1e-6` rate floors.
- [ ] **Gradient Benchmark Target**: `@belapsed ReverseDiff.gradient! < 1.0ms` on compiled gradient tape.

---

## 3. Active Phase Log & Next Steps

- **2026-08-21 15:25**: Completed 40-fold walk-forward grid of `goals_negbin_open_play_hl365_hs2` (1h 58m).
- **2026-08-21 15:30**: Executed out-of-sample head-to-head comparison (`r03_compare_open_play_vs_all_goals.jl`). Verified that pure open play produces $+4.04\%$ ROI and sharper away 1X2 Log Loss ($0.0045$ vs $0.0049$).
- **2026-08-21 15:34**: Created comprehensive mathematical reference [`EXPERIMENT_SETUP_AND_MATH.md`](file:///home/james/bet_project/BayesianFootball/current_development/scottish_lower/open_play/EXPERIMENT_SETUP_AND_MATH.md) and initialized this execution log.
- **Next Up**: Build `l03_recombination_models.jl` (Branch A & Branch B engines) and benchmark AD gradient performance in `r04_benchmark_ad_recomb.jl`.
