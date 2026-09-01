# Experiment 05 — Player-Level Lineup and pxG Fusion

## Background and motivation

Team-state models smooth strength through time, but a pre-match teamsheet can contain a shock that
is not yet present in that state. This experiment tests whether point-in-time player RAPM aggregates
capture those shocks, whether a small substitute-bench contribution represents squad depth, and
whether player information complements the macro signal in age-adjusted production wealth.

The comparison holds the two-arm observation fixed. Every model fits goals with a Poisson arm and
available match proxy-xG with a Gamma arm. Player ratings are fitted only on each fold's frozen
history block. The candidate grid therefore asks whether lineup shocks, bench depth, and club wealth
provide complementary information rather than merely whether any one signal works in isolation.

## Candidate models

1. **`m05_joint_production_wealth`** — control two-arm joint model with Richards-sigmoid
   age-adjusted production wealth.
2. **`m09_player_shots_rapm_outfield`** — two-arm joint model whose structural dynamics are the
   starting outfield XI's shots RAPM ratings.
3. **`m10_player_shots_rapm_bench`** — the shots-RAPM starter model plus substitute-bench ratings
   at a fixed weight of `0.10`.
4. **`m11_player_pxg_rapm_bench`** — the same starter-and-bench architecture using pxG RAPM, with
   bench weight `0.10`.
5. **`m12_hybrid_production_wealth_player_rapm`** — the master synergy arm: shots-RAPM starters,
   bench weight `0.10`, and Richards-sigmoid production wealth in one two-arm joint model.

All five models use `GlobalInterception`, `GlobalHomeAdvantage`, a bounded rate guard, and the same
`JointGammaPoissonObservation`. Player arms replace latent team time dynamics with
`PlayerLineupDynamics`; the control retains 180-day time-decay team dynamics.

## Files

- `l50_loader.jl` — shared datastore, model recipes, 40-fold splitter, production sampler,
  portfolio specifications, PostgreSQL setup, and canonical registry writes.
- `r50_smoke_test_player_models.jl` — one-scored-fold, five-model integration test.
- `r51_train_player_models_40fold.jl` — production queued NUTS grid and PostgreSQL persistence.
- `r52_compare_player_models.jl` — proper scoring and model/market calibration curves.
- `r53_portfolio_backtest.jl` — multi-market fractional-Kelly backtest and persistence.

## Verification gates

The smoke runner must pass every gate for every candidate before the production grid is launched:

1. NUTS sampling completes without a failed chain or fold.
2. The six-part audit passes: split R-hat `< 1.05`, bulk and tail ESS `> 100`, exactly zero
   divergences, BFMI `> 0.30`, and tree-depth saturation `< 5%`.
3. Chain parameter extraction and held-out `CountLatents` extraction return finite positive rates.
4. A `SmileScoreGrid` compatibility fixture can be generated and priced. The candidate models are
   count models, not learned-smile engines; this gate wraps their count latents with a neutral
   `phi(K)=1` curve solely to exercise the shared smile-grid plumbing. It does not claim that the
   models estimate a market smile.
5. PostgreSQL `save_fit`/`load_fit` reproduces every chain value, latent match ID, and latent matrix
   exactly.
6. 1X2, Over/Under 2.5, and BTTS portfolio construction, simulation, and PostgreSQL artefact
   persistence complete successfully.

## Evaluation criteria

Only fully converged production fits are comparable. `r52_compare_player_models.jl` reports
out-of-sample LogLoss, Brier score, count CRPS, 1X2 RPS, ECE/MCE, and ten-bin reliability curves
for both the model and Betfair closing probabilities. Lower proper scores are better, but no model
graduates on a single metric: improvements must be consistent, calibration must remain credible,
and the comparison must retain all eligible held-out fixtures.

`r53_portfolio_backtest.jl` is a separate economic check. It uses 30% fractional Kelly across 1X2,
Over/Under 2.5, and BTTS, with a daily slate drawdown budget and exposure cap. ROI is not evidence
of predictive quality by itself; it is read alongside bet count, maximum drawdown, market
attribution, and the proper-score report.

## Execution

On `archpc`:

```bash
julia --project -t 8 experiments/scottish_lower/05_player_lineup_and_pxg_fusion/r50_smoke_test_player_models.jl
julia --project -t 8 test/runtests.jl
```

Before either command, the runner pins Julia threads to physical cores and sets BLAS threads to
one. The 40-fold runner is prepared for `mcmc-beast`; do not launch it while the `r46` grid is
active.
