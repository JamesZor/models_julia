# GRADUATION SPECIFICATION: 05-09 PROTOTYPES INTO `src/`

<agent_context>
You are tasked with graduating the verified prototypes (05 through 09) in `current_development/` into the production `src/` module of `BayesianFootball.jl`.
You are working on git branch `feat/graduate-unified-v2-pipeline`.

CRITICAL INSTRUCTIONS & GUARDRAILS:
1. NEVER BREAK EXISTING TESTS. The repository currently has 463 passing unit tests in `test/runtests.jl`. Every graduation step must maintain 100% test pass rate.
2. PRESERVE 100% BACKWARD COMPATIBILITY. All legacy functions, types, and constructor signatures must continue to work.
3. CONCRETE PARAMETRIC TYPING: Always write parametric types (`struct Foo{T<:Real}`) - never use untyped fields or `Vector{Any}`.
4. ZERO ALLOCATION SCORING: Use `compute_score_grid!` and `price_market!` in-place kernels with `GridWorkspace`.
5. AD SAFETY: Never mutate arrays inside Turing `@model` blocks during ReverseDiff AD passes.
6. STEP-BY-STEP VERIFICATION: After editing files for a phase, run `julia --project test/runtests.jl` to verify that all tests pass before moving to the next phase.
</agent_context>

---

## GRADUATION PHASES (In Strict Dependency Order)

### Phase 1: Graduate Typed Posterior Latents (`06_typed_posterior_latents`)
* **Source**: `current_development/06_typed_posterior_latents/`
* **Target Locations**:
  1. `src/Models/latents/`:
     - `types.jl`: Define `AbstractPosteriorLatents`, `CountLatents{T,D}`, `RecombLatents{T,D}`, `SmileLatents{T,D}`.
     - `extract.jl`: Extractors from Turing `Chains` / `DynamicPxGRecombModel` / `DynamicGoalsModel`.
  2. `src/predictions/score_grids/`:
     - `types.jl`: Define `GridWorkspace`, `alloc_score_grid`, `alloc_market_book`.
     - `kernels.jl`: Zero-allocation `compute_score_grid!` and `price_market!` (1X2, BTTS, Over/Under).
  3. `src/predictions/score_computation/`:
     - Wire `model_inference` to accept `AbstractPosteriorLatents`.
  4. `src/BayesianFootball.jl`: Export new types and methods.
* **Test Verification**:
  - Add `test/latents_tests.jl` (lifting tests from `06/r01_demo.jl`).
  - Run `julia --project test/runtests.jl` and ensure 100% pass.

---

### Phase 2: Graduate Composable Model Builder (`05_composable_count_builder`)
* **Source**: `current_development/scottish_lower/05_composable_count_builder/`
* **Target Locations**:
  1. `src/Models/PreGame/builder/`:
     - `types.jl`, `components.jl`, `engine.jl`, `builder.jl`.
  2. `src/BayesianFootball.jl`: Export `CountModelBuilder`, component blocks.
* **Test Verification**:
  - Add `test/builder_tests.jl`.
  - Run `julia --project test/runtests.jl` and ensure 100% pass.

---

### Phase 3: Graduate Unified Inference & Fit Lifecycle (`07_unified_inference_framework`)
* **Source**: `current_development/07_unified_inference_framework/`
* **Target Locations**:
  1. `src/Training/inference/`:
     - `types.jl` (`Fit`, `FitConfig`, `FoldFit`, `ConvergenceSummary`).
     - `convergence.jl` (MCMC diagnostics audit: R-hat, ESS, divergences).
     - `engine.jl` (`fit_model`).
     - `io.jl` (`save_fit`, `load_fit`).
     - `ingame.jl` (zero-alloc live rate solver).
* **Test Verification**:
  - Add `test/inference_tests.jl`.
  - Run `julia --project test/runtests.jl`.

---

### Phase 4: Graduate Unified Evaluation (`08_unified_evaluation_framework`)
* **Source**: `current_development/08_unified_evaluation_framework/`
* **Target Locations**:
  1. `src/evaluation/`:
     - Modernize `src/evaluation/` scoring rules with `OddsView`, `MatchOutcomes`, `MarketProbabilities`.
     - Apply upstream bug fixes in `src/evaluation/translator.jl` (`unroll(::String, ::Missing)`) and `src/predictions/score_computation/poisson.jl` (`get_latent_column_symbols`).
     - `batch_runner.jl` (`evaluate_fits` with convergence gating).
* **Test Verification**:
  - Run `test/runtests.jl`.

---

### Phase 5: Graduate Zero-Allocation Portfolio & Staking (`09_unified_portfolio_framework`)
* **Source**: `current_development/09_unified_portfolio_framework/`
* **Target Locations**:
  1. `src/Portfolio/`:
     - `book.jl`: Add `OddsIndex`, `BookWorkspace`, `price_fixture!`, `build_books` fast path, `BuildReport`.
     - `simulate.jl`: Add `simulate_portfolio`, `DailyState`, `PortfolioResult`.
* **Test Verification**:
  - Run `test/portfolio_tests.jl` and `test/runtests.jl`.
