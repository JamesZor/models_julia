# SPECIFICATION: pxG & RAPM / pxG-APM UNIFIED COVARIATES

> **Branch:** `feat/pxg-rapm-unified-covariates`  
> **Target:** Modernize and integrate BBC LiveText proxy expected goals (`pxG`) and stint-based Regularized Adjusted Plus-Minus (`RAPM` / `pxG-APM`) into the `BayesianFootball.jl` Unified V2 Composable Count Model Builder (`src/models/pregame/builder/`).

---

## 1. Context & Reference Codebases

The codebase has historical exploratory research in:
1. `current_development/bbc_xg_proxy/`:
   - `l01_xg_proxy.jl`, `l03_funnel_cascade.jl`: Shot commentary parser, event weighting (box/distance/header/big chance/penalties), and calibrated expected goals ($pxG$).
   - `r08_graduation_verify.jl`: Historical verification of the proxy model.
2. `current_development/plus_minus_ratings/`:
   - `l01_segments.jl`: Substitution and red card stint segmentation from `ds.lineups` and `ds.incidents`.
   - `l02_shot_parser.jl`: Shot-level $pxG$ mapping to stints.
   - `l04_ridge_apm.jl`: Ridge / Bayesian APM regression solving offensive and defensive player ratings ($RAPM_{\text{off}}, RAPM_{\text{def}}$).
   - `r04_ridge_fit.jl`, `r08_reliability.jl`: Empirical reliability and shrinkage evaluation.
3. `src/features/`:
   - `src/features/rapm.jl`: Existing legacy RAPM implementation.
   - `src/features/squad_wealth.jl`: Point-in-time reference for squad-level aggregation from starting lineups.
4. `src/models/pregame/builder/`:
   - `components.jl`: `AbstractCovariateConfig`, `ProductionWealthCovariate`, `WealthCovariate`, `DistanceCovariate`.
   - `engine.jl`: ReverseDiff tape construction with unrolled covariate tuples.
   - `builder.jl`: Composable pipeline validation and dispatch.

---

## 2. Core Architectural Objectives

### A. PxG Features (`PxGFeature`, `PxGCovariate`)
1. Point-in-time calculation of team-level offensive and defensive proxy xG from `ds.bbc_events` (or fallback to match statistics/goals when commentary is sparse).
2. `PxGFeature <: AbstractFeature` with configurable lookback window ($k$-matches or exponential decay).
3. `PxGCovariate <: AbstractCovariateConfig` supporting:
   - Supremacy Role: $(pxG_{\text{home, att}} - pxG_{\text{away, def}}) - (pxG_{\text{away, att}} - pxG_{\text{home, def}})$
   - Level Role: Baseline match volume adjustment.

### B. RAPM / pxG-APM Features (`PxGRapmFeature`, `PxGRapmCovariate`)
1. Point-in-time calculation of stint-level design matrices from `ds.lineups` and `ds.incidents` (or `ds.segments`).
2. Ridge / Bayesian shrinkage on stint goal/pxG differentials strictly evaluated on historical data prior to each fold cutoff (`stamp < kickoff`).
3. Aggregation across starting-XI players on matchday:
   $$x_{\text{rapm}} = \sum_{i \in \text{Home XI}} r_i - \sum_{j \in \text{Away XI}} r_j$$
   with shrinkage fallback for unrated/sparse players.
4. `PxGRapmCovariate <: AbstractCovariateConfig` for direct composition in `CountModelBuilder`.

### C. ReverseDiff AD Safety & Type Stability
1. Covariates must be concrete, immutable parametric structs (`struct Foo{T<:Real} <: AbstractCovariateConfig`).
2. Tapes are compiled once in $O(1)$; no mutating operations inside Turing model blocks during AD passes.
3. Zero allocations during likelihood evaluation.

---

## 3. Deliverables & Testing Ladder

1. **Feature Implementation**:
   - `src/features/pxg.jl`: Pure point-in-time `PxGFeature` engine.
   - `src/features/pxg_rapm.jl`: Pure point-in-time `PxGRapmFeature` engine.
2. **Builder Integration**:
   - `src/models/pregame/builder/components.jl`: Export `PxGCovariate`, `PxGRapmCovariate`.
   - `src/models/pregame/builder/engine.jl`: Covariate site dispatches and ReverseDiff compilation.
3. **Unit Test Suite**:
   - `test/test_pxg_rapm_features.jl`:
     - Test PIT integrity (`stamp < kickoff` assertion).
     - Test ReverseDiff gradient compilation and tape execution.
     - Test Out-of-sample feature extraction and fallback behavior.
   - Run `julia --project -t 8 test/runtests.jl` to guarantee 100% pass rate across the full repository.
4. **Empirical Benchmark Runner**:
   - `current_development/scottish_lower/r40_train_pxg_rapm_models.jl`:
     - Compare baseline `m00`, `m05_production_wealth`, `m06_pxg`, and `m07_pxg_rapm` on Fold 1 and season walk-forward grids.
