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

---

## 4. Implementation Record (delivered)

| Deliverable | Location |
|---|---|
| `PxGFeature` engine | `src/features/pxg.jl` |
| `PxGRapmFeature` engine | `src/features/pxg_rapm.jl` |
| `PxGCovariate`, `PxGRapmCovariate` | `src/models/pregame/builder/components.jl` |
| Unit tests | `test/test_pxg_rapm_features.jl` (registered in `test/runtests.jl`) |
| Benchmark runner | `current_development/scottish_lower/r40_train_pxg_rapm_models.jl` + `l40_pxg_rapm_bench.jl` |

### 4.1 Deviation from §2.A — the supremacy sign

The spec writes the supremacy role as

$$(pxG_{\text{h,att}} - pxG_{\text{a,def}}) - (pxG_{\text{a,att}} - pxG_{\text{h,def}})$$

Taken literally this *subtracts* the opponent's concession rate from your own creation rate, so a
home side facing a leaky defence is scored **down** for it, and the two effects cancel exactly when
they should reinforce. The implementation uses the sign that makes the term an expected-pxG
difference:

$$x_{\text{sup}} = (att_h + def_a) - (att_a + def_h), \qquad
  x_{\text{lev}} = (att_h + def_a) + (att_a + def_h)$$

where `att_x` is x's pxG-scored deviation from the running league mean and `def_x` its
pxG-**conceded** deviation. Read `att_h + def_a` as "home's expected pxG in this fixture, relative
to a league-average pairing". Both quantities are deviations, so a cold-start fixture is exactly
`0.0`. This is the only departure from the written spec.

### 4.2 `engine.jl` was not modified

§3.2 anticipated covariate site dispatches in `engine.jl`. None were needed: the engine's covariate
block is already generic over the typed covariate tuple and unrolls at compile time, so both new
covariates reach the tape through the existing `_cov_block` walk. The tape-shape test in
`test_pxg_rapm_features.jl` confirms the instruction count is invariant to design length.

### 4.3 Measured coverage on `ScottishLower` (2009 matches, 2026-08-30 snapshot)

| pxG measurement rung | matches |
|---|---|
| BBC live-text commentary | 1109 (55.2%) |
| BBC match-page shot counts | 899 |
| goals fallback | 1 |

Fold-2 design columns: `pxg` sd 0.518, 2.8% neutral; `pxg_rapm` sd 0.942, 9.3% neutral, 537 rated
players, auto-scale 0.259. No fitted-feature match id falls inside any target block.
