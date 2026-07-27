# Overnight grid: double-Poisson market-pillar comparison (Ireland)

`r05_grid_search_double_poisson.jl` — modelled on `ab_test_dixon_coles/r07_grid_search_market_weight.jl`.

## Objective

Decide, on OOS backtest + CLV + LogLoss, whether the **split (level/supremacy) market pillar**
buys anything over (a) **no market** and (b) the **old isotropic market** pillar — and find the
weight regime where it does. Directly targets the open question from the κ-spread discussion:
does market-informed supremacy (which flows into the score grid via κ) improve 1X2, especially
top-vs-bottom mismatches, without wrecking totals?

## Lean 7-cell grid (all double-Poisson, goals + xG + outfield ratings)

| family | engine | cells |
|---|---|---|
| **NoMarket** (baseline) | `DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel` | `dp_nomarket` |
| **Old isotropic market** | `DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel` | `market_weight ∈ {0.5, 1.0}` |
| **New split market** | `SplitMarketDoublePoissonModel` (l02) | `supremacy_weight=1.0` FIXED, `level_weight ∈ {0, 0.25, 0.5, 1.0}` |

Total = **1 + 2 + 4 = 7 experiments**. `supremacy_weight` is held at the natural 1.0 (one market
observation per match, directly comparable to old `market_weight=1`); sw>1 just forces the model
onto a supremacy it genuinely disagrees with and pumps the κ distortion, so it is NOT swept. The
distinctive split knob is `level_weight`, which is what we sweep. `dp_split_lw100` (sw=1,lw=1) is
the anisotropic twin of `dp_old_mw100` (isolates "does anisotropy help"). All output Poisson
λ_h/λ_a → Poisson score grid (src Union for the src engines, loader overrides for the split model);
κ is in λ, so team differentiation reaches the grid/1X2.

## Fixed settings (match r07 for comparability)

- Data: `Data.Ireland()`. `half_life = 60`, `dynamics_col = :match_biweek`.
- Splitter: `target_seasons = ["2025","2026"]`, `history_seasons = 2`, `warmup_period = 0`.
- Sampler: `samples = 800`, `warmup = 300`, `chains = 4`, `use_queue = true`, `max_depth = 10`.
- Components identical across all cells (HierMonthlyInterception, HomeAwayDispersion,
  HierTeamHomeAdvantage, HierTeamKappa, BayesianTracker(6.5,1,0.5,0.01)).
- **Training market pillar source = plain `ds` (SofaScore-derived odds), as in r07.** Eval is on
  the Betfair-swapped `ds1`. This keeps train-pillar ≠ eval-line, avoiding CLV-vs-Betfair leakage.
  (Toggle: train on Betfair-swapped ds instead — better prior per `betfair-vs-bet365-market-anchor`,
  but then CLV vs Betfair-close partially leaks; not the default.)

## Execution (overnight, robust)

- **Phase 1** — loop the 7 specs; `run_experiment` + `save_experiment` each (persisted
  immediately), wrapped in try/catch so one failure doesn't lose the batch. Experiments run
  sequentially; each full-CV experiment's splits×chains queue already saturates the 16 pinned cores.
- **Phase 2** — build `ds1` (Betfair), then r07's eval trio: `GLMEdge`, `LogLoss`,
  Kelly `run_backtest` → `generate_tearsheet` (hurdle_G etc. per market selection).

## Runtime

~25–35 min per full-CV experiment (per-chain tape compile dominates, doesn't amortize — see NOTES).
7 experiments ≈ **3–4 h**. Trim/expand the grid vectors at the top of r05 to change.

## What we're reading in the morning

- **LogLoss diff (model−market)** per family: does split beat no-market / old on 1X2 vs totals?
- **GLMEdge** `spread_fair_coef` (CLV signal) per cell.
- **hurdle_G** (Kelly growth) per market selection (over/under, 1X2, BTTS).
- Expectation from the κ analysis: split should help **1X2** as supremacy weight rises, but risk
  **totals** as κ-encoded supremacy bleeds into the rate. No-market should still own totals.

## Phase 3 (next session, build against saved experiments)

The decisive test for the user's hypothesis — **1X2 LogLoss/CLV stratified by supremacy magnitude
(mismatch vs even)** — is NOT in the overnight runner (keep it robust). Build it against the saved
`./data/double_poisson_market_grid/` experiments once they exist: bucket matches by |market supremacy|,
compare split vs no-market 1X2 LogLoss per bucket. If the hypothesis holds, split wins specifically
in the high-supremacy (top-vs-bottom) bucket.

## Caveats

- Screening-quality: per NOTES the base player-rating engine is weakly identified; treat rankings as
  directional, not final. Convergence is acceptable (sampled-σ) but ESS is modest.
