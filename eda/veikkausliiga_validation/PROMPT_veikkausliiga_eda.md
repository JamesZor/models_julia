# Veikkausliiga (31) — Stage-A EDA brief

## Task
Characterise **Veikkausliiga** (Finnish top flight, tournament id `31`, slug `veikkausliiga`) as a
data-generating process before it is used for modelling/betting — mirroring the completed
`eda/first_division_validation/` (718) study, plus three extras the 718 study did not cover.

## Goals
1. **Goals distribution / dispersion** — fix the likelihood family for the L1 Bayesian engines
   (Poisson vs Negative-Binomial vs COM/Weibull): marginal moments, index of dispersion, the
   univariate count ladder, NB1/NB2, the Dixon-Coles bivariate ladder, rootogram + χ² GoF, and the
   league diagnostics (home advantage, temporal stability, within-team dispersion).
2. **Per-team attack/defence (goals & xG)** — for each team fit `goals_for`/`goals_against` (and
   `xg_for`/`xg_against`) and compare to the league average to rank who is genuinely good.
3. **Player-rating coverage + per-team rating distributions** — confirm the SofaScore `rating` field is
   usable (per-season + per-position coverage), then aggregate to a minute-weighted team match rating
   and rank squad quality vs the league average.

## Locked design decisions (from grilling the user)
- **Team framing:** attack & defence split (for vs against), for **both** goals and xG.
- **Comparison rigor:** per-team distribution fit + **formal test vs the pooled league rate** +
  **empirical-Bayes shrinkage** ranking (Gamma–Poisson for counts; Normal–Normal for xG/ratings), with
  Benjamini–Hochberg FDR adjustment across the ~12 simultaneous team comparisons.
- **Ratings:** coverage audit **and** per-team rating distribution (no cross-league rating contrast).
- **Scope:** **standalone** characterisation of 31 (no 79/718 contrast league); EDA + report only
  (no model training). The only contrast used is a one-line **validation guard** that re-fits Ireland 79
  to confirm the fitters reproduce the published number before trusting 31's output.
- **Execution:** end-to-end via the kaimon REPL on the server; capture real numbers into the report.

## Verified DB signature (betdb probe, tournament 31)
- ~678 played matches, 2021–2026, spring–autumn calendar (~132/season; 2026 partial: 18 so far),
  ~12 teams.
- **xG: 100% from 2023** (2021–22 absent).
- **Player ratings: from 2023** (~29 rated players/match); lineups present all seasons, ratings null pre-2023.
- **No `bigChanceCreated`** column at all (same as 718) → that pillar is unavailable.
- **Full betfair coverage** (~674 matches; MATCH_ODDS, all OVER_UNDER lines, BTTS, etc.) → Stage-B CLV
  feasible. (Corrects the stale `betdb-data-coverage` memory that claimed betfair was Ireland-only.)

## Files
- `l01_veikkausliiga_logic.jl` — loader; includes the 718/ireland fitter libraries and adds the
  per-team attack/defence, xG, rating-coverage and per-team rating functions.
- `r01_veikkausliiga_runner.jl` — sectioned execution + captured output.
- `veikkausliiga_eda.md` — the report (filled with captured numbers).
- The only `src/` change is the `Veikkausliiga` segment in `src/Data/fetchers/segments.jl`
  (+ export in `src/Data/data-module.jl`).
