# Staking-Strategy Simulation Lab — Experiments Tracking

## Overview
Monte Carlo race of staking strategies on simulated double-Poisson league seasons with
drifting team strengths (Ireland Premier-calibrated). We know the TRUE match probabilities;
a "market" observes them with noise and quotes odds with per-family vig; "our model"
observes them with less noise but carries **planted per-line bias** (totals compression +
BTTS boost, the li_smile50 signature). All strategies bet identical books with sequential
bankroll compounding — the statistical power the 275-match real backtest
(`unified_staking/r02`) can't provide.

Theory this tests: `docs/bets_multi/unified_kelly_postgrad_notes.md` (P)/(U-MC),
`docs/bets_multi/trust_blend_notes.md` (per-line trust blend). Report deliverable:
`docs/bets_multi/staking_sim_report.{md,tex,pdf}`.

## Files
- `l01_sim_market_model.jl` — `SimConfig`, truth/market/model generators, 11-selection book
  builder (`SimMatch`). Includes `unified_staking/l01_structural_kelly.jl` (solver, masks).
- `l02_strategies.jl` — strategy registry (12 strategies), trust EB fit (port of
  `_verify_trust.py` T4), closed-form IPF coherent tilt, `run_season`, metrics, oracle w,
  `sanity_checks()`.
- `r01_calibrate_ireland.jl` — one-off dial calibration vs cached Ireland datastore.
- `r02_mc_runner.jl` — `smoke()` → `run_mc()` (chunked/checkpointed/threaded) →
  `make_outputs()` (summary.csv + 4 plots).

## Server run steps (kaimon, james@192.168.1.88)
```julia
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"   # broken LanguageServer dep; skip env precompile
using BayesianFootball
include("current_development/staking_sim/r01_calibrate_ireland.jl")  # once; bake numbers
include("current_development/staking_sim/r02_mc_runner.jl")
smoke()          # sanity_checks() + timed 1 season
run_mc()         # N=300, chunks of 25, resumable from results/results_partial.jls
make_outputs()   # results/summary.csv + plots/p1..p4
```
Then locally: `scp 'james@192.168.1.88:~/bet_project/BayesianFootball/current_development/staking_sim/plots/*.png' ~/bet_project/docs/bets_multi/figs_sim/`

## Calibration (r01) — status: PENDING
Ireland Premier (tournament 79), from `.cache/datastore_Ireland.jls`:

| dial | value | source |
|---|---|---|
| μ (log base away rate) | TBD | log(mean_away) − σ0² (Jensen) |
| ha (home advantage) | TBD | log(mean_home/mean_away) |
| O_1x2 / O_ou / O_btts | TBD | mean(overround_close) per family |

Sanity targets: sim-vs-empirical mean goals, home-win %, draw %, over-2.5 % within ~5%.

## Conventions (stated once, uniform across strategies)
- Ruin: cumulative W < 0.01 ⇒ betting frozen for the rest of the season (logw = 0 after).
- Per-bet strategies: 3% min-edge filter vs offered price; total-exposure guard rescales
  to Σa ≤ 0.98.
- Trust: 7 units (home/draw/away/O1.5/O2.5/O3.5/btts), complements share w; warm start on
  270 no-betting pre-history matches, EB refit every 30 matches; 1X2 blended targets
  renormalised to a proper partition before the coherent tilt.
- (U-MC): S_dec=50 deterministic-stride draws, k grid 0.05:0.05:1.0.
- No commission variable (vig lives in the quoted odds); no favourite–longshot bias in the
  vig (multiplicative) — documented limitation, v2 dial.

## Experiment 1: main race — status: PENDING
Config: `SimConfig()` (r01-calibrated), N=300 seasons × 330 matches, base_seed 20260704.
Strategies: FLAT_1pct, K_full/half/quarter, BM_ana, BM_num, U_cap02/05/100, U_UMC,
TRUST_U_cap02, TRUST_UMC.

Expected headline reads (to confirm/refute):
1. K_full ruin frequency ≫ 0 despite genuine model edge (bias adversely selected).
2. BM_* (variance-only shrinkage) does NOT fix the planted bias lines; TRUST_* does.
3. Cap is the dominant unified risk lever (cap02 vs cap100), k* secondary when cap binds.
4. p4: learned w recovers the planted structure (1X2 high, totals/BTTS low) with pooling
   noise consistent with trust_blend_notes §3.1 (~600 obs/line ⇒ wide w posteriors).

Results: (paste summary.csv table + reads here after the run)

## Experiment 2+ (sensitivity, later)
- Cold-start trust (n_prehist=0) vs warm.
- σ_mod sweep (model quality) and γ sweep (bias size): where does TRUST_* stop paying?
- Vig sweep; FLB power-law vig (v2).
