# TASK: Implement Age-Adjusted Production Wealth Feature & Covariate

## Context & Objectives
You are tasked with implementing the **Age-Adjusted Production Wealth** feature and composable model component on branch `feat/production-wealth-covariate`.

### Core References & Empirical Findings:
- `docs/models/squad_age_wealth_findings.md`
- `current_development/scottish_lower/AGE_WEALTH_FINDINGS.md`
- Reference feature implementation: `LogSumWealthFeature` and `WealthCovariate` in `src/models/pregame/builder/components.jl`
- Lineup fetcher: `src/Data/fetchers/sql/lineups.jl`

### Constraints:
- **NO SUBAGENTS:** Do all implementation, testing, and execution directly in this session.
- **BRANCH:** Work on `feat/production-wealth-covariate`.
- **TYPE STABILITY:** Zero runtime allocations in inner valuation loops, ReverseDiff tape compilable.

---

## Required Deliverables:

### 1. Data Layer (`src/Data/fetchers/sql/lineups.jl`):
Ensure `sofascore.match_player_lineups` extracts `(l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint AS date_of_birth_timestamp` (and `NULL::bigint` for BBC fallback).

### 2. Feature & Model Layer (`src/models/pregame/builder/components.jl`):
- Implement candidate age-weighting curves:
  * `RichardsSigmoid(x0 = 23.0, k = 0.80, nu = 2.0)` -> $\phi(A) = (1 + e^{-k(A - x_0)})^{-1/\nu}$
  * `ShiftedGamma(a0 = 16.0, peak = 27.5, alpha = 3.5)` -> mode-normalized
  * `GaussianPrime(mu = 26.5, sigma = 4.5)`
- Implement `ProductionWealthFeature <: CB_Features.AbstractFeatureConfig`:
  * Computes player age at match kickoff: $(t_{\text{kickoff}} - t_{\text{dob}}) / (365.25 \times 86400)$
  * Fallback age for unmapped players = 26.5y (prime neutral)
  * Multiplies player valuation $V_{\text{prod}, i} = V_i \cdot \phi(\text{Age}_i)$
  * Starting-XI log-sum differential: $\Delta z_{\text{prod}} = (\log \sum_{h} V_{\text{prod}} - \log \sum_{a} V_{\text{prod}}) / \text{log\_scale}$
  * PIT safeguards: ignores future valuations or invalid DOBs.
- Implement `ProductionWealthCovariate <: AbstractCovariateConfig`:
  * Exports for `CountModelBuilder` and re-exports in `PreGame`, `Models`, and `BayesianFootball`.

### 3. Unit Test Suite (`test/test_production_wealth_feature.jl`):
- Exact numerical tests for curve formulations at ages 18, 23, 27.5, 34, 38.
- Lineup substitution filtering (only starting XI counted).
- Neutral fallback test when lineups are empty or invalid.
- Integration test with `CountModelBuilder` verifying ReverseDiff tape generation.

### 4. Visual Inspection & Diagnostic Script (`current_development/scottish_lower/r13_verify_production_wealth_feature.jl`):
- ASCII plot / print of candidate curves $\phi(\text{Age})$.
- Side-by-side inspection table of real Scottish Lower matches comparing $\Delta W_{\text{raw}}$ vs $\Delta W_{\text{prod}}$.
- Signal diagnostics: Pearson $r$, Spearman $\rho$ with match goal difference, and Poisson GLM deviance drop vs raw wealth.

---

## Instructions:
1. Implement the code changes.
2. Run the test suite: `julia --project -t 8 -e 'using Test; include("test/test_production_wealth_feature.jl")'`
3. Run the visual inspection script: `julia --project -t 8 current_development/scottish_lower/r13_verify_production_wealth_feature.jl`
4. Confirm all tests pass and report findings.
