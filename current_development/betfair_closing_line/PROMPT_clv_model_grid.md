# Cleared-session prompt — Closing-Line-Value of the M1/M2/M3 model grid

Copy everything below the line into a fresh session.

---

## Objective

Measure the **Closing Line Value (CLV)** of our three-model A/B grid against the Betfair
closing line, now that the Betfair odds database has been expanded to many more markets.
CLV is the cleanest forward test of model edge at our small sample size (~281 Ireland
matches): instead of asking "did we profit?" (high-variance P&L), it asks **"does the
model's probability anticipate the direction the market moves to at close?"** A model with
positive CLV alpha (β>0, directional hit-rate >0.5, mean log-CLV >0) is finding genuine
information ahead of a sharp market, even when realised P&L is noisy. This is the metric
to judge the grid on.

## The grid (3 saved experiments)

From `current_development/ab_test_dixon_coles/r03_bigchance_ab_ireland.jl`, saved in
`./data/bigchance_ab/`:

- **M1** `DP_Goals_Market_XG`            — `{goals, market, xG}` (baseline)
- **M2** `DP_Goals_Market_BigChance`     — `{goals, market, bigChance}`
- **M3** `DP_Goals_Market_BigChance_XG`  — `{goals, market, bigChance, xG}`

Prior verdict (LogLoss + 1X2 Kelly P&L, 281 matches): **M2 > M3 ≳ M1**, but all three
*lose to the market* on LogLoss and gaps are within noise. CLV is the tie-breaker: a model
that consistently leads the close is the one to keep, regardless of the noisy P&L ordering.

## Reuse — do NOT rebuild

The full CLV engine already exists and is **model-agnostic** (takes a `ppd` + `ds`):

- `current_development/betfair_closing_line/l01_clv_eval.jl` — loader with all stages:
  - `build_clv_panel(ppd, ds; targets, horizons, window_width)` — horizon-resolved,
    vig-removed, LOCF'd panel joined to model scalar probs + graded `is_winner`.
  - `coverage_table(panel)` — matches / LOCF-fraction / mean-ticks per selection×horizon.
  - `edge_by_horizon(panel; group)` — **Stage 1**: model vs market log-loss/Brier, bootstrap CI.
  - `clv_alpha(panel; group)` — **Stage 2 (headline)**: OLS `realized_move ~ model_signal`
    (β>0 ⇒ model predicts the line move), directional hit-rate + binomial test, mean log-CLV.
  - `entry_timing_pnl(panel; edge_threshold, group)` — **Stage 3**: filtered-bet ROI / log-growth.
  - `roll_spread`, `pit_calibration` — **Stage 4**: microstructure + posterior calibration.
- `current_development/betfair_closing_line/r01_clv_eval.jl` — the single-model runner to mirror.

The machinery groups generically by `(match_id, market_name, market_line, selection)`, so it
already handles any market the panel contains. **No code changes to `l01` are needed** beyond
possibly widening `targets`.

## What's actually new — and the one real gotcha

1. **Richer Betfair panel.** The CLV engine reads `ds.betfair_odds` (the raw last-traded TICK
   long table), NOT `ds.odds` and NOT the `summarize_betfair_market` `ds1`. The new markets must
   be present in `ds.betfair_odds`. **STEP 1 is a coverage triage** before trusting anything.

2. **Model-emittable ∩ Betfair-liquid ∩ gradeable.** The model can only emit probabilities for
   markets that have a `compute_market_probs` implementation: **1X2, BTTS, Double-Chance,
   OverUnder lines** (`DEFAULT_MARKET_CONFIG` already emits all of these — O/U 0.5…10.5).
   The PPD therefore already covers the full emittable set; **no need to widen the market_config**.
   Any *new* Betfair market that is NOT one of those (e.g. Asian Handicap, Correct Score) has no
   model probability and no `grade_selection` rule → it will simply not join and is out of scope
   for this study. So expand `targets` only to selections that are (a) in the new Betfair panel
   with real liquidity, (b) model-emittable, (c) gradeable by `Data.grade_selection`. Good
   candidates beyond the existing five: `home`, `draw`, `away` (1X2), `btts_no`, more O/U lines
   (`over_15`, `over_35`, `under_45`), and the Double-Chance legs.

## Steps

1. **Load the updated datastore** (new betdb, Ireland on `:5433`):
   `ds = Data.load_datastore_cached(Data.Ireland())` — or the SQL loader / a cache refresh if the
   cache predates the betfair DB update (force a fresh `load_datastore_sql` if unsure).

2. **Coverage triage** of `ds.betfair_odds`: unique `market_name × market_line`, number of distinct
   matches and total ticks each, and the `minutes_to_kickoff` span. Decide the final `targets`
   list = selections with enough matches AND a non-trivial tick count near the close (so CLV is
   real, not LOCF noise). Print it; this list is the scope of the whole study.

3. **Per model**, load and infer:
   ```julia
   files = Experiments.list_experiments("./data/bigchance_ab/"; data_dir="")
   res   = Experiments.load_experiment(files, k)            # k for each of M1/M2/M3
   ppd   = BayesianFootball.Predictions.model_inference(ds, res)   # DEFAULT_MARKET_CONFIG covers all emittable markets
   panel = build_clv_panel(ppd, ds; targets = TARGETS, window_width = 60.0)
   panel.model_name .= res.name                              # tag for the cross-model compare
   ```
   Concatenate the three tagged panels into one `grid_panel`.

4. **Cross-model comparison** (group everything by `[:model_name, …]`):
   - Stage 1: `edge_by_horizon(grid_panel; group=[:model_name, :horizon])` — does any model beat
     the market log-loss (`diff_ll < 0`) and at which horizon?
   - **Stage 2 (headline):** `clv_alpha(grid_panel; group=[:model_name, :horizon])` and a pooled
     `[:model_name]` version — compare β, hit-rate (+binomial p), and **mean log-CLV** across the
     three models. The winner is the model with the largest, most-significant positive CLV that
     persists closest to the close.
   - Stage 3: `entry_timing_pnl(grid_panel; edge_threshold ∈ {0.0,0.02,0.05}, group=[:model_name,:horizon])`
     — sanity P&L, but treat Stage 2 as the verdict.
   - Stage 4: `pit_calibration` per model (posterior vs closing line) + one pooled `roll_spread`.
   - Optionally also break Stage 2 down by `[:model_name, :selection]` to see *which markets* carry
     each model's CLV (e.g. does M2's bigChance edge live in totals/BTTS vs 1X2?).

5. **Headline:** which model leads the close best overall; specifically **does M2 (bigChance) beat
   M1 (xG) on CLV** — i.e. does the LogLoss/P&L ordering reproduce on the more robust forward
   metric, or flip? Save plots (`plot_edge_decay`, `plot_clv`, `plot_pnl`, `plot_pit`) per model.

## Deliverables

- A new runner `current_development/betfair_closing_line/r02_clv_model_grid.jl` (mirror `r01`,
  loop over the 3 experiments, captured `#= =#` result blocks).
- A results write-up `current_development/betfair_closing_line/CLV_MODEL_GRID_RESULTS.md` with the
  per-model Stage-1/2/3 tables and the headline.
- If the CLV verdict on **M2 vs M1** differs from the LogLoss/P&L verdict, update memory
  `bigchancecreated-eda-findings` with the CLV result (CLV is the stronger evidence).

## Operational

- Run **only via the kaimon MCP REPL on the server** (`ssh root@mcmc-beast`, repo
  `/root/BayesianFootball`). Local edits don't auto-appear there: `git add/commit/push` locally,
  then `git pull --ff-only` on the server before `include`. `start_session` spawns a process even
  on timeout — never retry it.
- These are **PPD-only** runs (no MCMC) — fast, fine to run inline in the REPL; **no nohup needed**.
  Only adding/changing module `include`s/exports needs `manage_repl restart`; pure
  runner/loader edits are picked up by Revise.
- First confirm the three experiments actually exist on the **server** disk under
  `./data/bigchance_ab/` (they were saved during the r03 run). If missing, re-pull / re-locate.

## Caveats

- CLV is only meaningful where Betfair last-traded liquidity is real near the close — watch
  `locf_frac` in `coverage_table`; drop selection×horizon cells that are mostly LOCF.
- Vig is removed per market instance via overround inside `snapshot_at` (limits 0.90–1.10).
- n is still modest; report bootstrap CIs and binomial p-values, don't over-read point estimates.
- A model can have **positive CLV alpha even with noisy/negative P&L** — that is exactly the
  signal we want; it means the edge is real but staking/variance is washing it out in P&L.
