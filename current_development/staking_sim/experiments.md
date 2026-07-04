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

## Calibration (r01) — status: DONE (2026-07-04, server, n=1013 matches)
Ireland Premier (tournament 79), from `.cache/datastore_Ireland.jls`:

| dial | value | source |
|---|---|---|
| μ (log base away rate) | **0.0532** | log(mean_away=1.098) − σ0² (Jensen) |
| ha (home advantage) | **0.2459** | log(1.404/1.098) |
| O_1x2 / O_ou / O_btts | **1.0841 / 1.0491 / 1.0696** | mean(overround_close) per family |

Sanity (15×330-match campaigns vs empirical): home goals 1.387 vs 1.404, away 1.098 vs
1.098, HW 44.7% vs 42.8%, draw 24.8% vs 27.5%, over2.5 44.4% vs 47.0%. Draw gap = pure
double Poisson lacks low-score correlation — accepted for the toy. NOTE: a single long
continuous run is the wrong sanity check (zero-sum RW spread grows unboundedly and
inflates goals 5–15% over ~1000 rounds); campaigns are fresh per replication.

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

## Experiment 1: main race — status: DONE
Config: `SimConfig()` (r01-calibrated), N=300 seasons × 330 matches, base_seed 20260704.
Strategies: FLAT_1pct, K_full/half/quarter, BM_ana, BM_num, U_cap02/05/100, U_UMC,
TRUST_U_cap02, TRUST_UMC.

Expected headline reads (to confirm/refute):
1. K_full ruin frequency ≫ 0 despite genuine model edge (bias adversely selected).
2. BM_* (variance-only shrinkage) does NOT fix the planted bias lines; TRUST_* does.
3. Cap is the dominant unified risk lever (cap02 vs cap100), k* secondary when cap binds.
4. p4: learned w recovers the planted structure (1X2 high, totals/BTTS low) with pooling
   noise consistent with trust_blend_notes §3.1 (~600 obs/line ⇒ wide w posteriors).

### Results — status: DONE (2026-07-04, kaimon server, 8 threads, 75.4 min, 300/300 seasons)

From `results/summary.csv` (medW = median terminal wealth; q05/q95 = terminal wealth
quantiles; meanG = mean log-growth per match; medDD = median max drawdown; ruin% = seasons
frozen below W=0.01; bets/m = bets per match):

| strategy | medW | q05W | q95W | meanG | medDD | ruin% | bets/m |
|---|---|---|---|---|---|---|---|
| FLAT_1pct | 1.136 | 0.739 | 1.772 | +0.00043 | 22.6% | 0 | 0.75 |
| K_full | 0.148 | 0.006 | 5.664 | −0.00580 | 97.5% | 23.0 | 0.69 |
| K_half | 0.991 | 0.125 | 5.467 | −0.00016 | 73.7% | 0 | 0.75 |
| K_quarter | 1.227 | 0.447 | 2.992 | +0.00056 | 44.3% | 0 | 0.75 |
| BM_ana | 0.252 | 0.008 | 6.713 | −0.00446 | 95.5% | 16.3 | 0.71 |
| BM_num | 0.246 | 0.008 | 6.651 | −0.00447 | 95.5% | 16.3 | 0.71 |
| U_cap02 | 0.933 | 0.128 | 7.185 | −0.00018 | 78.9% | 0 | 2.06 |
| U_cap05 | 0.843 | 0.110 | 7.788 | −0.00039 | 81.8% | 0 | 2.08 |
| U_cap100 | 0.843 | 0.110 | 7.788 | −0.00039 | 81.8% | 0 | 2.08 |
| U_UMC | 1.118 | 0.186 | 6.994 | +0.00024 | 72.3% | 0 | 2.08 |
| TRUST_U_cap02 | **1.205** | 0.568 | 3.037 | **+0.00065** | 41.0% | 0 | 0.95 |
| TRUST_UMC | 1.097 | **0.680** | 2.198 | +0.00042 | **27.3%** | 0 | 0.95 |

Growth-oracle trust (40k-match sample, `worac.jls`), order home/draw/away/O1.5/O2.5/O3.5/btts:
**w_oracle = [0.72, 0.48, 0.72, 0.90, 0.96, 1.00, 0.64]**.

Reads vs the expected headlines:
1. **CONFIRMED** — K_full ruins 23% of seasons and has the worst mean growth (−0.0058/match)
   *despite a genuine information edge on every line*: stakes scale with divergence, and the
   biased lines make the biggest divergences (optimizer's curse in vivo).
2. **CONFIRMED** — BM_ana ≈ BM_num ≈ a slightly tamed K_full (medW 0.25, ruin 16.3%). The
   S=100 posteriors are tight (σ_post=0.05) and bias sits in the centre, so variance-only
   shrinkage shaves almost nothing. TRUST_* fixes exactly what BM can't see: TRUST_U_cap02
   turns U_cap02's −0.00018 into the best growth in the race (+0.00065) at half the drawdown.
3. **PARTIALLY CONFIRMED / REFINED** — the cap matters (cap0.2 beats cap0.5/1.0; cap>0.5
   *never binds*: U_cap05 ≡ U_cap100 to 6 decimals) but k\* is the stronger lever here:
   U_UMC (+0.00024) beats all fixed-cap variants without trust. With trust the ordering
   flips to preference: TRUST_U_cap02 = growth pick, TRUST_UMC = risk-adjusted pick
   (Kelly-level growth at flat-stake drawdown 27.3%, tightest terminal spread q05=0.68).
4. **REFUTED (informatively)** — learned w does NOT recover the planted per-line structure
   at ~600 obs/line: end-of-season EB w medians sit ≈0.53 on every unit (p4), exactly the
   §3.1 identifiability arithmetic. The pooling keeps the blend *safe* (near half-trust)
   rather than *sharp* — and that is already enough for read 2's growth win.
5. **NOVEL** — the oracle wants HIGH trust on totals (0.90–1.00) despite the planted
   compression tilt: gridding noisy λ draws over-disperses totals (Jensen), and the −0.05
   tilt partially *corrects* it. The sim independently reproduces the repo's
   "totals-compression-is-denoising" finding. BTTS (0.64) shows the pure planted bias;
   draw (0.48) is where model probabilities help least — matches real-data experience.
6. Benchmarks: FLAT_1pct is embarrassingly competitive (+0.00043, DD 22.6%, zero ruin);
   K_quarter has the best medW (1.227) but at double the drawdown of TRUST_UMC.

## Experiment 2+ (sensitivity, later)
- Cold-start trust (n_prehist=0) vs warm.
- σ_mod sweep (model quality) and γ sweep (bias size): where does TRUST_* stop paying?
- Vig sweep; FLB power-law vig (v2).
