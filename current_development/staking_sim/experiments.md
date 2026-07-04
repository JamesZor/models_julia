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

## Experiment 2: trust identifiability + time-decayed fit — status: DONE (2026-07-04, r03, 1.2 min)

Question (user): p4 shows learned w ≈ flat across lines (only draw slightly different) —
sensitivity issue? Would time-decay weighting (recent matches matter more) capture regime
shifts? Files: `r03_trust_experiments.jl`; `fit_trust_eb` gained a `halflife` kwarg (l02).

**E2a — identifiability ladder** (EB fit on 300→20k obs/line, 30 reps) + per-line Fisher
info Î = mean(δ²/(p̃(1−p̃))) on 40k matches (`results/e2_summary.txt`, plot p5):

| unit | Î/obs | n for sd(ŵ)=0.1 | med ŵ @600 | med ŵ @20k | w growth-oracle |
|---|---|---|---|---|---|
| home | 0.0075 | 13.3k | 0.55 | 0.67 | 0.72 |
| draw | 0.0023 | **43.9k** | 0.54 | 0.45 | 0.48 |
| away | 0.0064 | 15.6k | 0.58 | 0.72 | 0.72 |
| over_15 | 0.0067 | 15.0k | 0.52 | 0.61 | 0.90 |
| over_25 | 0.0090 | 11.1k | 0.57 | 0.61 | 0.96 |
| over_35 | 0.0090 | 11.1k | 0.57 | 0.57 | 1.00 |
| btts_yes | 0.0044 | 22.8k | 0.52 | 0.70 | 0.64 |

Reads:
1. **Flat w is information, not a fit bug.** Lines separate with n exactly as the Fisher
   arithmetic prices: sharp w needs 11k–44k obs/line, a campaign gives ~600, so the EB
   pooled prior (≈0.53) dominates everything. 1X2 + BTTS converge cleanly to their oracles
   by 10–20k (home 0.67, away 0.72, draw DOWN to 0.45, btts 0.70).
2. **Why the draw looks special in p4**: it carries 3–4× less information per observation
   than any other line (draw prob is nearly flat in strength differences ⇒ model and market
   rarely disagree) AND its optimum (0.45–0.48) is the only 1X2 one *below* the pooled
   mean — its weak likelihood nudges consistently down while home/away nudge up.
3. **Trust-for-calibration ≠ trust-for-growth on totals.** Even at 20k obs the Bernoulli
   fit puts totals at 0.57–0.61 vs growth-oracle 0.90–1.00: per-bet totals edges are tiny,
   so the growth objective is nearly flat in w (argmax rides the boundary) while the
   log-score optimum is interior. v3 idea: fit w against realized growth on curated bets.

**E2b — time decay** (half-life H ∈ {∞,1000,400,150} obs in the trust log-lik; 2 worlds ×
30 reps × 2000 matches, walk-forward refit/30, score = blend−market Bernoulli log-score
×1000 on the NEXT 30 matches; DRIFT world flips bias signs at m1000: γ_tot −0.05→+0.05,
γ_btts +0.10→−0.10; same seeds both worlds ⇒ paired; plots p6/p7):

| world | window | static | H=1000 | H=400 | H=150 |
|---|---|---|---|---|---|
| static | m300–1000 | 1.111 | 1.124 | 1.136 | 1.146 |
| static | m1000–2000 | 1.415 | 1.419 | 1.414 | 1.405 |
| drift | m300–1000 | 1.113 | 1.126 | 1.139 | 1.149 |
| drift | m1000–2000 | 1.200 | 1.206 | 1.206 | 1.208 |

Reads:
4. **Decay is a no-op at these information rates** — spreads of ±0.01 on an advantage of
   1.2–1.4. Even the hard sign-flip costs ALL fits equally (post-flip 1.42→1.20) and
   H=150 recovers just +0.008 over static. p6 shows why: every fit, decayed or not, hugs
   the pooled prior near 0.5–0.6 and barely reacts to the flip — there is no sharpness to
   protect, so there is nothing for decay to save.
5. **The blend is already regime-robust.** Half-trust is near-optimal in BOTH bias regimes
   (drift-world oracle [0.64, 1.0, 0.64, 0.58, 0.52, 0.50, 0.74] — still mid-range), so the
   EB prior's conservatism doubles as regime insurance. Verdict: don't add decay in v1;
   revisit only if units get coarser (fewer units ⇒ more obs each) or model–market
   divergence δ grows (bigger per-obs information).

## Experiment 3: fixed w=0.5 vs EB-learned trust — status: DONE (2026-07-04, r04, ~30 min)

Question (user): if the learned w is flat anyway, why not hard-code w=0.5? Race
(identical books, same seeds as Experiment 1): FLAT_1pct · U_cap02 (w=1, raw model) ·
TRUST05_U_cap02 (hard-coded 0.5) · TRUST_U_cap02 (EB), two worlds × 300 seasons:
GOOD = base cfg (real edge + bias); BAD = σ_mod=0.12 > σ_mkt (junk model, same bias).

| world | strategy | medW | q05W | q95W | meanG | medDD | ruin% |
|---|---|---|---|---|---|---|---|
| good | FLAT_1pct | 1.136 | 0.739 | 1.772 | +0.00043 | 22.6% | 0 |
| good | U_cap02 (w=1) | 0.933 | 0.128 | 7.185 | −0.00018 | 78.9% | 0 |
| good | **TRUST05 (fixed .5)** | **1.269** | 0.589 | 2.695 | **+0.00073** | 35.3% | 0 |
| good | TRUST_EB | 1.205 | 0.568 | 3.037 | +0.00065 | 41.0% | 0 |
| bad | FLAT_1pct | 0.986 | 0.536 | 1.919 | +0.00002 | 34.5% | 0 |
| bad | U_cap02 (w=1) | 0.110 | 0.009 | 4.111 | −0.00626 | 97.1% | 18.3 |
| bad | TRUST05 (fixed .5) | 0.843 | 0.214 | 4.037 | −0.00034 | 65.0% | 0 |
| bad | **TRUST_EB** | **0.877** | 0.280 | 3.557 | −0.00031 | **59.0%** | 0 |

End-of-season EB w medians: GOOD [0.535 0.499 0.532 0.544 0.531 0.528 0.538] (hugs 0.5);
BAD [0.381 0.431 0.386 0.409 0.374 0.391 0.413] — **pulled to ≈0.39 within one season**.

Reads:
1. **User intuition confirmed for the good-model case**: fixed 0.5 beats EB (medW 1.269
   vs 1.205, +0.00073 vs +0.00065, DD 35% vs 41%). The EB fit pays an estimation-noise
   tax (~0.0001/match) to learn a number that happens to be ≈0.5 anyway.
2. **The machinery's real job is insurance.** In the junk-model world the raw model is
   catastrophic (ruin 18%); half-trust rescues most of it (−0.0063 → −0.0003, ruin 0);
   and EB then beats fixed 0.5 on BOTH growth and drawdown by learning w↓0.39 in one
   season. Global level (pooled, ~4.2k obs/season) is learnable fast even when per-line
   w isn't. Direction right, speed modest: one season gets 0.5→0.39, not to the truth —
   over seasons it keeps sliding toward abstention while fixed 0.5 donates forever.
3. **Practical recipe**: STAKE with w=0.5 (or an EB fit with a strong prior at 0.5);
   RUN the EB fit alongside as a monitor — alarm / cut exposure when the pooled
   posterior w0 drops below ~0.4. Decouples the good-world tax from the bad-world
   insurance. Per-line sharpness arrives later with multi-season/league data (E2a).

## Experiment 4+ (sensitivity, later)
- Cold-start trust (n_prehist=0) vs warm.
- σ_mod sweep (model quality) and γ sweep (bias size): where does TRUST_* stop paying?
- Vig sweep; FLB power-law vig (v2).
- Fit w against realized growth on curated bets (closes the calibration-vs-growth gap, E2a read 3).
