# Kickoff brief — graduate the APM player rating into an L1 engine (fresh session)

*Paste this as the first message of a fresh Claude Code session. It hands off an approved plan;
the previous session's context grew too large to continue there.*

---

## Your mission

Wire the validated **plus-minus (RAPM) player rating** for the Scottish lower leagues (tournaments
**56/57**) into the L1 modelling stack as a **first-class feature + a goals engine**, then run
experiments on `ScottishLower` to measure whether an APM-informed model beats the current
team-level winner. This mirrors how the Ireland engines consume the SofaScore player rating.

The research (`current_development/plus_minus_ratings/`, WP0–WP7) is **done and green-lit** —
this is the *graduation to `src/`*, not new research.

## Read these first (authoritative — do not re-derive)

1. **The approved plan** (the spec you are executing, WP-A → WP-D + verification):
   `/home/james/.claude/plans/home-james-bet-project-docs-modern-port-parallel-pebble.md`
2. **The research log / status + WP7 verdict**:
   `current_development/plus_minus_ratings/NOTES.md`
3. **The validated math to port** (keep faithful — the WP7 verdict rests on it):
   `l01_segments.jl`, `l02_shot_parser.jl`, `l03_targets.jl`, `l04_ridge_apm.jl`,
   and the SQL loaders `l00_pm_data.jl` (esp. `fetch_pm_livetext`'s `-fc`-normalised 3-way
   `is_home_event` CASE, and `fetch_pm_incidents`' jsonb player-ID extraction).
4. **The contract to emit** (mirror exactly): `src/features/extractors/player_extractors.jl`.
5. **The baseline engine to imitate structurally / beat**:
   `src/models/pregame/engines/team_level/time_decay/goals_funnel_league.jl`.

## Branch & workflow (already set up)

- Work on branch **`feat/apm-player-rating-l1`** (already created off `feat/graduate-funnel-engine`,
  which carries the `ds.bbc` DataStore domain the APM work depends on — do **not** rebase onto main).
- **Commit per work package** with clear messages. Do not merge; leave the branch for review.
- **Execution loop**: edit locally → `git push` → on the server (`archpc`, `/root/BayesianFootball`)
  `git pull` → run via the **kaimon MCP REPL**. Set `JULIA_PKG_PRECOMPILE_AUTO=0` before
  `using BayesianFootball`; reuse warm sessions; for long `include`s redirect stdout to a file and
  re-check with a trivial `ex` (the 10-min kaimon gate is cosmetic — Julia keeps running).
  The betdb MCP is only reachable on the home network; run DB queries through the server REPL
  (`LibPQ.Connection(ENV["BF_DB_URL"])`), which is local to the DB.

## The four work packages (summary — full detail in the plan file)

- **WP-A — Data**: add a 9th DataStore field `bbc_events::DataFrame` (raw per-event commentary) and
  player IDs on `ds.incidents`; new SQL in `bbc.jl` / `incidents.jl`; wire through
  `datastore.jl`/`schemas.jl`/`preprocessing.jl` and the backward-compat constructor chain.
  **Leave `ds.bbc` (per-match funnel totals) untouched** — the funnel winner depends on it.
- **WP-B — Feature family**: `abstract type AbstractPlusMinusFeature <: AbstractFeatureConfig` with
  one concrete config struct per target (`ShotsPlusMinusFeature` [green-lit], `XGPlusMinusFeature`,
  `GoalsPlusMinusFeature`, `ShotsOnTargetPlusMinusFeature`) each holding `(w_sim, λ, half_life_days)`;
  a `pm_target` trait → response column; **one shared** `add_feature!(::AbstractPlusMinusFeature,…)`
  that ports the ridge fit into `src/features/plus_minus/` and emits the identical 8-vector +
  `player_ratings_map` contract. Add a `Features.rating_base` accessor (`0.0` for the PM family).
- **WP-C — Model**: `DynamicGoalsPlusMinusLeagueTimeDecayModel` (goals double-Poisson + APM pillar),
  field `player_ratings_feature::P where {P<:AbstractPlusMinusFeature}`. **Register it in
  `src/predictions/score_computation/poisson.jl`'s dispatch Union** and include in
  `pregame-module.jl`. No-APM twin baseline = `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel`.
- **WP-D — Experiments**: stash `F_data[:history_match_ids]` in `builder.jl`; runner
  `r10_src_experiment.jl` sweeps the APM variants vs the goals baseline and the funnel winner via
  `evaluate_experiments([LogLoss(),LPD(),CRPS()], …, ds)` (primary gate) **and** a Kelly
  `run_backtest → summarize_models` (growth/CLV). Paste tables + a dated verdict into `NOTES.md`.

## Load-bearing guardrails (easy to get wrong)

- **Leak-safe fit**: the APM ridge must be fit on **history matches only** (`F_data[:history_match_ids]`),
  one rating vector applied to the whole fold. Never let target-fold matches into the fit.
- **Minutes are dead on 56/57**: `lineups.minutes_played` is 0 pre-23/24 and NULL for much of 25/26.
  Weight the positional aggregation by **on-pitch/starter status**, not `minutes_played` (coalescing
  to 0 would zero out real ratings).
- **Score Union registration**: skip it and PPD errors on a missing NegBin `r` column.
- **Backward-compat DataStore constructor**: many call sites rebuild the store positionally — the new
  `bbc_events` field must default to empty in the compat constructors, and every extractor must
  tolerate an empty `ds.bbc_events` (return zero ratings for non-Scottish segments).
- **Cross-check WP-B** against the prototype's `r08_reliability.jl` `fit_ratings` on the same window —
  the src ratings must reproduce the validated numbers.

## Definition of done

All four WP verification gates in the plan pass (see its Verification section), and `NOTES.md` has a
dated verdict on **apm_shots vs goals_baseline** (its no-APM twin) and **apm_shots vs funnel_winner**
(the current best team-level model) on both scoring and growth. A clean negative — APM does not beat
team-level — is a valid, publishable outcome; record it the same way.

## Memory pointers (context this session already holds)

`plus-minus-rapm-stream`, `funnel-src-graduation`, `funnel-cascade-stream`, `kaimon-repl-on-server`,
`server-file-sync-workflow`, `kaimon-server-precompile-broken-dep`, `turing-suffstat-and-init-gotchas`,
`research-stream-handoff-workflow`. Read `NOTES.md` for the WP7 verdict details.
