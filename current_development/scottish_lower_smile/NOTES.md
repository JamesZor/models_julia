# Scottish Lower — team-level goals-only time-decay sup+smile

Research stream for **team-level, goals-only** versions of the sup+smile market-pillar engines,
targeting `Data.ScottishLower()` (tournaments **56 = League One, 57 = League Two**). These leagues
have **no xG, no stats, no player ratings, no Betfair** — only goals + full Bet365 (SofaScore) odds
(1X2, BTTS, O/U ladder 0.5–7.5, DC/DNB/AH on ~99% of matches; 5 full seasons 21/22→25/26, 20/21
~50% odds coverage). Season kicks off ~early Aug 2026.

Parent stream: `current_development/split_market_pillar/` (player-level, xG; keeper `li_smile50`
graduated to src as `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`). The smile machinery
(`MarketSmileFeature`, `_smile_intensity`, `SmileScoreMatrix`, per-line O/U pricing) is
rate-source-agnostic and reused as-is.

## Canonical naming (inherited from split_market_pillar/NOTES.md — single source of truth)

A model is two independent axes; reference name = `<pillar>-<disp>`; cell name =
`<pillar>_<disp>_<knob><val>`.

**Axis A — market pillar:** `none` (structural only) · `iso` (isotropic shared σ + `mw`) ·
`smile` (supremacy + per-strike Λ(K), knobs `sup`/`sw`). (`split` is NOT built here — nulled at
the latent-rate layer in the parent stream.)

**Axis B — dispersion:** `pois` (double Poisson) · `nb` (NegBin, reference row via the existing
src team engines).

Grid-cell suffixes: `hl<days>` (days_half_life), `hs<n>` (history_seasons), `sup<pct>`, `sw<pct>`,
`mw<pct>`. Examples: `none_pois_hl120_hs2`, `smile_pois_sup70_sw50`, `iso_nb_mw100`.

## Fixed design decisions (2026-07-11 planning session)

- **Targeted ladder:** 3 new loader engines `TeamDPGoalsModel` (`none_pois`),
  `TeamIsoDPGoalsModel` (`iso_pois`), `TeamSmileDPGoalsModel` (`smile_pois`); existing
  `DynamicGoalsTimeDecayModel`/`DynamicMarketGoalsTimeDecayModel` = `nb` reference row.
- **Pooled + league intercept:** one model over both divisions (shared team map survives
  promotion/relegation) + zero-sum per-league offset δ_league on log-λ (goal levels differ:
  2.87 vs 2.71 goals/match). New loader-local `LeagueFeature`.
- **Benchmark:** de-vigged Bet365 close (`prob_fair_close`). NO Betfair swap (none exists for
  56/57; Betfair historical may be loaded later as a secondary check only — thin liquidity, so per
  [[betfair-vs-bet365-market-anchor]] Bet365 stays the anchor regardless).
- **Grids:** Stage A `hl{60,120,180,365} × hs{1,2,3}` on `none_pois`; Stage B
  `sup{40,70,100} × sw{0,40,50}` on `smile_pois` at the Stage-A winner. Priors / Kmax=4 /
  sampled-σ at Ireland-validated values. CV target seasons 23/24→25/26, `dynamics_col=:match_biweek`.
- **Eval:** per-line LogLoss diff vs close + GLMEdge + per-bet Kelly hurdle_G per market family
  (r06 lesson: judge per line, not grouped). Convergence gate (R-hat<1.01, ≥95% splits) before
  reading ANY grid table.
- σ's are **SAMPLED, never fixed** (parent-stream convergence saga); **no σ hierarchy**
  ([[hierarchical-smile-sigma-null]]); market inversion stays **Poisson-referenced**
  ([[no-pregame-intensity-smile]]).

## Known gotchas carried in

- `Features.MarketLambdaFeature` is a **phantom** (exported, never defined) — every src `*Market*`
  engine's `required_features` throws. Loader overrides the NB market engine's `required_features`
  with `DoublePoissonMarketFeature()`; the src fix lands at graduation.
- New structs are not in any score-computation Union → loader ships `extract_params` +
  `compute_score_matrix` overrides ([[dixoncoles-prediction-dispatch-union]]).
- Market λ plausibility mask `0.02 < λ < 20` on the pillar (degenerate inversions on thin closes,
  [[outfield-xg-engine-gotchas]]).
- Silent split drop: check `training_results.items` length per cell before eval
  ([[xg-pillar-nan-and-sampler-gotchas]]).

## Files

- `r00_data_qa.jl` — Stage 0 data QA (seasons, odds-ladder coverage, league diff, team churn,
  fold counts, inversion sanity, phantom-feature confirmation).
- `l01_team_dp_league.jl` — loader: `LeagueFeature` + the 3 engines + prediction overrides +
  NB `required_features` fix.
- `r01_smoke.jl` — single-split smoke: convergence, δ_league read, PPD end-to-end, smile≠grid O/U.
- (planned) `r02_grid_decay_history.jl` / `r03_eval_decay.jl` — Stage A.
- (planned) `r04_grid_smile.jl` / `r05_eval_smile.jl` — Stage B.
- (planned) `RESULTS_scottish_grid.md`, graduation + `r06_smoke_src.jl` — Stage 4.

Live match_day_inference wiring for these leagues = follow-up session (NOT this stream).

## Findings log

(append dated entries here as runners complete)
