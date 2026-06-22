# Cleared-session prompt — Ireland First Division (718) tournament EDA

Copy everything below the line into a fresh session.

---

## Objective

We have a **new competition in the SQL DB: Ireland First Division (tournament ID 718, slug
`first-division`)**, alongside the existing **Ireland Premier (ID 79)**. Characterise First
Division as a *data-generating process* before it's used for modelling/betting: extract it into
the Julia `DataStore`, audit all the data and per-season feature coverage, fit the discrete-count
ladder to goals (Poisson, **NB1, NB2**, Weibull, **Dixon-Coles**), and **contrast 718 vs 79
side-by-side** to settle the open question: *is First Division a distinct regime that needs its
own stratum / league-varying dispersion, or is it poolable with the Premier?* Produce a
maths-and-results report.

**Scope = EDA + report only.** A later **Stage B (separate session)** runs a model grid scored on
CLV / GLM-edge / LogLoss / backtest ROI / growth — this session only sets that up and records which
seasons are model-trainable (esp. which carry xG — the hypothesis is xG only lands from **2024**).

## Reuse — do NOT rebuild

Almost every fitter already exists. Include them; don't duplicate.

- `eda/basic_goals/l00_basic_goals_loader.jl`:
  - `fit_goal_distributions` / `analyze_goal_models` — Poisson, RobustNB, WeibullCount (AIC + χ²).
  - `analyze_heavyweight_models` / `fit_bivariate_models` — the **Dixon-Coles bivariate ladder**:
    Indep-Poisson, **DC-Poisson**, Indep-NB, **DC-NB**, Indep-Weibull, **DC-Weibull** (ρ + AIC).
- `eda/ireland_validation/l01_bigchance_logic.jl`:
  - `summarise_count` (mean/var/V-M/%zeros vs Poisson-implied/skew),
  - `compare_count_models` (adds ZIP/ZINB/COM-Poisson to Poisson/NB/Weibull),
  - `compare_nb1_nb2`, `rootogram_data`, `chi_square_gof`.
  - **NB1 vs NB2:** already established — the project's `RobustNegativeBinomial(r,μ)` IS **NB2**
    (Var = μ + μ²/r); **NB1** (Var = (1+α)μ) is the same family with r = μ/α. `compare_nb1_nb2`
    handles both; expect them near-identical on a single marginal mean (they only diverge once μ
    varies across observations, i.e. inside a model).
- `eda/ireland_validation/l00_validation_logic.jl`:
  `test_overdispersion`, `test_home_advantage_mean`, `test_home_advantage_variance`,
  `test_team_volatility`, `test_temporal_stability`.
- Report template to mirror: `eda/ireland_validation/bigchancecreated_eda.md`.

## Key data facts

- **718 is not in any existing segment** (`tournament_ids(::Ireland) = [79]`), so you must add one.
- **xG** = `ds.statistics.expectedGoals_home` / `expectedGoals_away`
  (see `src/features/extractors/stats_extractors.jl`). **bigChance** = `bigChanceCreated_home/away`
  with `period == "ALL"`. **shots** likewise live in `ds.statistics` (`period == "ALL"`).
- **Season** is a string column (e.g. `"2024"`, `"2025"`) — same format as `target_seasons`.
- The new betdb (`:5433`) is expected to carry **betfair for both 79 and 718** (memory
  `betdb-data-coverage`). 718 should therefore have `betfair_odds` — **verify, don't assume**
  (Stage-B CLV depends on it).

## Steps

### 1. Add the segment + sync (the only `src/` edit)
In `src/Data/fetchers/segments.jl` (per CLAUDE.md "Adding a New League/Segment"):
```julia
struct IrelandFirstDivision <: DataTournemantSegment end
tournament_ids(::IrelandFirstDivision) = [718]
```
This changes the `Data` module → **Revise will NOT pick it up**. Workflow:
1. local: `git add … && git commit … && git push`
2. server: `git pull --ff-only` in `/root/BayesianFootball`
3. **`manage_repl restart`** the kaimon session (required — new struct/method on a module include).
Run everything via the **kaimon MCP REPL on the server** (`ssh root@mcmc-beast`). `start_session`
spawns a process even on timeout — **never retry it**.

### 2. Load + full data audit (718, then 79)
- `ds718 = Data.load_datastore_sql(IrelandFirstDivision())` — **force a fresh SQL load**; don't
  trust a cache for a brand-new segment. `ds79 = Data.load_datastore_cached(Data.Ireland())`.
- For every `DataStore` field (`matches`, `statistics`, `odds`, `betfair_odds`, `lineups`,
  `incidents`): report row count, season span, and **per-season non-missing coverage** of
  **goals, xG, bigChance, shots**, plus presence/row-counts of odds/betfair/lineups/incidents.
- Emit a **trainable-season table** for 718: which seasons have xG (confirm the ~2024 jump),
  which have betfair, which have full stats — this is the Stage-B readiness map.
- Add only the missing helper to the new loader, e.g. `feature_coverage_by_season(ds)`; everything
  else is reused.

### 3. Marginal moments
`summarise_count` on home / away / total goals for **718 and 79** (mean, var, V/M, %zeros vs
Poisson-implied, skew). Goals vectors: `collect(skipmissing(ds.matches.home_score))` etc.

### 4. Discrete-count fits (goals), per league
For 718 and 79: `analyze_goal_models` (univariate Poisson/NB/Weibull), `compare_count_models`
(+ZIP/ZINB/COM), `compare_nb1_nb2`, and `analyze_heavyweight_models` (the **Dixon-Coles**
Poisson/NB/Weibull bivariate ladder with ρ + AIC). Then `rootogram_data` + `chi_square_gof` for the
winning family in each league.

### 5. League diagnostics
`test_overdispersion`, `test_home_advantage_mean` & `_variance`, `test_team_volatility`,
`test_temporal_stability` for each league.

### 6. 718-vs-79 contrast (HEADLINE)
Side-by-side table: best marginal family (by AIC and BIC), V/M, DC ρ (dependence), home advantage,
temporal stability. **Verdict:** is First Division a *distinct regime* (→ its own stratum /
league-varying dispersion drawn from a shared hyperprior) or *poolable* with the Premier (79)?
This directly informs the pool-vs-stratify decision flagged in prior work.

### 7. Report
Write `eda/first_division_validation/first_division_eda.md` mirroring the bigChance report
structure: motivation → data & coverage → marginal moments → candidate-distribution maths (PMFs +
AIC/BIC) → model-comparison results + rootogram/χ² → **718-vs-79 contrast** → recommendation. Add a
short **Stage-B readiness** note: trainable seasons, betfair availability, and recommended model
families to grid for 718 given its distributional signature.

## Deliverables

- `src/Data/fetchers/segments.jl` — `IrelandFirstDivision` struct + `tournament_ids` (graduated).
- `eda/first_division_validation/l01_first_division_logic.jl` — thin loader: `include`s the three
  existing eda loaders + only the new per-season coverage helper(s).
- `eda/first_division_validation/r01_first_division_runner.jl` — runner with captured `#= =#`
  result blocks (mirror `eda/ireland_validation/r02_bigchance_runner.jl`), looping 718 and 79.
- `eda/first_division_validation/first_division_eda.md` — the report.
- **Memory:** a `project` note on 718's distributional signature + the pool-vs-stratify verdict +
  trainable-xG-season window; update `MEMORY.md` index. Link `[[bigchancecreated-eda-findings]]`,
  `[[betdb-data-coverage]]`, `[[staking-research-conclusions]]`.

## Verification

- **Validate the fitters on a known target first:** run the goals fits on **79** and check the
  Poisson/NB AICs reproduce the existing `eda/basic_goals` / `ireland_validation` numbers before
  trusting them on 718.
- Sanity: 718 goal mean in a plausible football range (~1.2–1.6/side); V/M finite > 0; DC ρ ∈ (−1,1);
  χ² dof > 0; per-season xG coverage shows the expected jump (~2024).
- **Confirm 718 returns `betfair_odds` rows** (Stage-B CLV dependency) — flag loudly if it doesn't.
- `Pkg.test()` not required (prototype + one additive segment struct).

## Out of scope (Stage B — separate session)
- Training the model grid on 718 and scoring on **CLV / GLM-edge / LogLoss / backtest ROI / growth**.
- Any decision to pool 718+79 in a shared backtest (this EDA only produces the evidence for it).
