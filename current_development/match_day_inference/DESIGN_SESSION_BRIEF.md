# Design-session brief — `MatchDay`: from fixture list to executable stake sheet

**Paste this whole file as the opening message of a fresh Claude Code session.**
Working directory: `/home/james/bet_project/BayesianFootball`. Branch: `feat/portfolio-module`.

---

## 0. What this session is, and what it is not

You are running a **design session**, not a build session.

**Deliverable**: a system-design document for a `MatchDay` layer, at the same depth and in the
same house style as `current_development/portfolio_runbook/ARCHITECTURE.md` — ASCII flow
diagrams, the domain objects, the abstract-type seams with their one-method contracts, the
staged pipeline, and the honest list of what is unresolved.

**Explicitly NOT in scope for this session**: writing `src/MatchDay/`, refactoring the
prototype, or running the pipeline against live fixtures. The user wants to go back and forth
on scope and shape first, and build in a later session. If you find yourself writing
implementation code, you have drifted.

**Working style the user has asked for**: iterative. Produce a strawman, expect it to be
argued with, revise. Do not present a single take as settled. Where a decision is genuinely
the user's (what to bet, what risk to take, what to cut), surface it as a question with a
recommendation rather than quietly picking.

One thing that will make this session good rather than mediocre: **the hard part of this
system is not the maths, it is identity resolution and data freshness.** The staking maths is
already built and audited (see §2). If your design document spends most of its length on Kelly
and a paragraph on "join Betfair to SofaScore", you have designed the wrong system. See §4.

---

## 1. Orientation — read these, in this order

| # | Path | Why |
|---|---|---|
| 1 | `current_development/portfolio_runbook/ARCHITECTURE.md` | **The template.** 365 lines. This is the shape and depth of the deliverable. |
| 2 | `src/Portfolio/types.jl` | The nine abstract seams + domain objects. The design vocabulary to reuse. |
| 3 | `src/Portfolio/matchday.jl` | 89 lines. The existing seam between Portfolio and match day — `stake_sheet`. |
| 4 | `src/Portfolio/book.jl` (`build_book`, `extract_selections`) | What the staking layer *demands* as input. Your design must supply exactly this. |
| 5 | `current_development/match_day_inference/src/*.jl` | The prototype being redesigned. 1740 lines across 6 files. |
| 6 | `current_development/match_day_inference/loader.jl` | Shows the intended assembly order of the prototype. |
| 7 | `CLAUDE.md` | Repo conventions — the `lXX`/`rXX` prototyping contract, AD-safety rules, module layout idiom. |

Skim `r05_smile_staking_compare_03_07_26.jl` (72k) only if you need to see how the panels were
actually driven on a live match day. It is a runner, not a design artefact.

There are two `paper_tracks/*.md` files (`ireland_12_06_26.md`, `ireland_19_06_26.md`) — these
are the user's own notes from paper-trading sessions. They are the closest thing to a
requirements document that exists. Read them.

---

## 2. What changed underneath this prototype

The match-day prototype was written **before** `src/Portfolio/` existed. That module has since
absorbed roughly half of what the prototype does, and did it more carefully.

`src/Portfolio/` (19 files, ~1760 lines, on branch `feat/portfolio-module`) is a graduated,
tested, audited multi-market Kelly staking manager. It provides:

- `Selection` / `MatchBook` / `Slate` / `SlateContext` / `Trajectory` domain objects
- a single `payoff(sel, h, a, commission)` morphism through which **both** the payoff matrix
  and settlement factor — so win-mask and settlement cannot drift apart
- `DeArb` pricing, `PerBetCommission`, `KellyLogUtility` (Jacot & Mochkovitch joint allocator
  with KKT audit), `BakerMcHale` posterior-uncertainty shrinkage, `SlateDrawdown` (Busseti),
  `FixedCap`, filters, daily slate grouping
- **`stake_sheet(sys, latents_df, expr, odds_df, ds; bankroll)`** — the match-day entry point,
  already written, which builds books with `require_result = false` so unplayed fixtures get
  `settle = nothing`
- 74 property tests in `test/portfolio_tests.jl`

**Therefore**: the staking half of `src/unified_staking.jl` and `calculate_betting_signals` in
`src/live_betting.jl` are now duplicated logic, and the prototype's versions are the weaker
ones (per-match cap only, no slate-level drawdown budget, no de-arb, hand-rolled `solve_P`).
A large part of your design job is deciding **what the MatchDay layer stops doing** because
Portfolio already does it.

The seam Portfolio defines, and which MatchDay must satisfy:

```
stake_sheet(sys, latents_df, expr, odds_df, ds)
                    │           │      │       │
                    │           │      │       └── ds.matches needs kick-off date for the fixture
                    │           │      └────────── DataFrame with EXACTLY:
                    │           │                  :match_id, :market_name, :market_line,
                    │           │                  :selection, :odds_close
                    │           └───────────────── trained L1 experiment (for the score matrix)
                    └───────────────────────────── one row per fixture, posterior draw vectors
```

That `odds_df` schema is the contract. **A live price feed that produces that DataFrame plugs
straight into an audited staking engine.** Most of the design work is upstream of it.

⚠️ One real problem to resolve, not paper over: `stake_sheet` is documented as taking
`latents_df` from the match-day inference pipeline, but `build_book` calls
`fixture_table(ds)` for the kick-off date, and `ds.matches` (the curated DataStore) contains
**finished matches only**. Today's fixtures live in `sofascore.events`. Check whether
`stake_sheet` can actually see an unplayed fixture at all, or whether `build_book` returns
`nothing` for every one of them. If it is broken, say so plainly — this is exactly the kind of
silently-empty result the user has already been bitten by twice this project.

---

## 3. The prototype, file by file — what it does and what state it is in

`current_development/match_day_inference/`. Last touched 2026-07-27. It is *deconstructed*: a
`loader.jl` + `src/` mini-module plus a scatter of dated runners, scratch files, and two dead
files. Nothing here is under test.

### `src/fixtures.jl` (68 lines) — **keep the idea, formalise it**
`fetch_todays_matches(segment)` → raw SQL against `sofascore.events` for
`status_type='notstarted'` in today's epoch window, filtered by `Data.tournament_ids(segment)`.
Opens its own `LibPQ` connection off `ENV["BF_DB_URL"]`.

Notes for the design:
- It stamps `match_week .= 999` and `match_date .= today()`. The 999 is a sentinel that the
  feature builders may or may not tolerate — **verify what depends on `match_week`**.
- "Today" is `CURRENT_DATE` in the DB's timezone; kick-offs are `start_timestamp` epoch ints.
  A 19:45 Irish kick-off and a UTC-midnight boundary is a real bug surface.
- No notion of "how long until kick-off", which the whole system needs (see §4.3).

### `src/lineups.jl` (243 lines) — **the interesting one**
Three-tier lineup resolution with an explicit priority: (1) local JSON override
`data/lineups/<match_id>.json` (manual pin), (2) `sofascore.lineup_provisional` (the scraper's
announced/predicted XI, carries a `confirmed` boolean), (3) fallback to each team's most recent
historical XI from `ds.lineups`.

Plus `fetch_lineup_from_sofascore(match_id)` — a direct scrape of
`api.sofascore.com/api/v1/event/<id>/lineups` with a browser User-Agent, bypassing the DB.

This tiering is a genuine design pattern and the obvious first candidate for a dispatch seam
(`AbstractLineupSource`). The fallback quality difference is measurable and there is already a
diagnostic for it — see `compare_matchday_lineups` in `ratings.jl`, which reports the
**positional-sum delta between provisional and fallback XI**, i.e. exactly how much the model's
inputs move. Whether that delta is big enough to matter is an open empirical question the design
should name.

### `src/ratings.jl` (320 lines) — **mostly belongs in `src/features/`, not MatchDay**
`calculate_latest_player_rating` dispatches on the tracker type (`BayesianTracker` runs a
Kalman filter, `EWMATracker`, `LastValueTracker`, `WindowAverageTracker`, generic mean
fallback) to roll every player's history forward to a current rating. Then
`build_matchday_ratings_map` aggregates starters into per-side positional sums (G/D/M/F),
which is the model's actual input.

Design question: this re-derives a "latest state" that the training-time feature extractors
already compute for historical matches. **Is `calculate_latest_player_rating` a duplicate of
logic in `src/features/extractors/`, and if so is it a *consistent* duplicate?** A silent
divergence between training-time and inference-time rating construction would be
train/serve skew — the classic and most expensive failure in this class of system. Check it.

### `src/inference.jl` (112 lines) — **the load-bearing hack**
`compute_todays_matches_latents` is the core trick: rebuild the feature collection from the
trained experiment's splitter, take the **last** split's chain and `FeatureSet`, *mutate* that
FeatureSet's `player_ratings_map` in place with today's ratings
(`inject_matchday_features!`), then call `Models.PreGame.extract_parameters(model,
todays_matches, feature_set, chain)`.

This works and is clever, but note what it assumes:
- the last training split's posterior is the right one to condition on (no re-fit, no decay
  past the training cut-off — **how stale is the chain?**)
- every feature a model needs on match day is reachable by injecting into
  `:player_ratings_map`. Any model whose `required_features` includes something else
  (form, market pillar, APM, funnel shots) has **no injection path at all**.
- it mutates a `FeatureSet` that came out of a cache

That last point is the one to design out. The generic version of this is
"**given a trained engine and a fixture that is not in any fold, materialise its features**",
and it should dispatch per feature rather than special-case ratings.

⚠️ Note the interaction with the engine the user now favours: the Ireland champion is
`src_sup40_sw40`, a **smile double-Poisson with a market pillar** — it takes market odds as a
*model feature*. On match day, that pillar has to come from the live book. So inference depends
on odds, and staking depends on inference. That is not a straight line; the design must show
where the market price enters twice and make sure it is the same price both times.

### `src/live_betting.jl` (783 lines) — **the biggest file, the most replaceable**
Redis-backed live odds + a PrettyTables dashboard.

- `ppd_to_betfair_type` / `betfair_to_ppd_type` — string mapping `"OverUnder", 2.5` ↔
  `"OVER_UNDER_25"`. Handles 1X2 / OverUnder / BTTS only; everything else returns `nothing`.
- `get_live_market_mappings(redis_conn)` — reads the `live_market_meta` Redis hash into
  `Dict{(home_slug, away_slug, bf_type) => market_id}` plus `market_id => {selection_id => runner_name}`.
- `parse_runners_for_market` / `_resolve_runner_role` — maps Betfair runner names back to PPD
  selection symbols.
- `fetch_live_odds_for_market` — reads the `live_markets` Redis hash, returns
  `(back, lay, back_size, lay_size)` per selection.
- `calculate_betting_signals` — per-bet EV + `Signals.BayesianKelly`. **Superseded by Portfolio.**
- `print_live_betting_dashboard` / `..._compare` — terminal panels.

The team-name-slug tuple as a join key is the weak point (§4.1). The dashboard is presentation
and should be a thin layer over a DataFrame, not 400 lines of `Matrix{Any}` construction.

### `src/unified_staking.jl` (214 lines) — **now redundant, but read the header**
Structural Kelly on the live book, `include`ing `../../unified_staking/l01_structural_kelly.jl`
for `solve_P` / `state_draws` / `mask_for` / `G_growth`. Restricted to Over/Under + BTTS with
`cap=0.10`, `commission=0.02`, back-only.

Superseded by `src/Portfolio`. **But its header comment records hard-won curation findings**
(full-book Kelly bankrupts; O/U + BTTS only; 1X2 display-only) that the design must not lose.
Cross-check against `current_development/staking_layer/` and `unified_staking/NOTES.md`.

### Dead weight to call out
- `deprecated_r01_matchday_runner.jl` — **0 bytes**
- `deprecated_runner.jl`, `l00_matchday_utils.jl`, `l00_matchday_utils_restored.jl` — superseded
- `scratch_redis_*.jl` (3 files) — one of them redefines a fake `module Predictions` to stub a test
- `debug_script.jl`, `test_db.jl`, `clear_cache.jl` — one-liners
- `fetch_lineups.py` / `.sh` / `run_fetch_all.jl` — a Python side-channel for lineups
- `r00`–`r05`, eight dated runners, 240k total, one per match day

The runners are a **log**, not a library. Part of the design's value is replacing "copy last
week's runner and edit the date" with one parameterised entry point.

---

## 4. The data layer — measured, not assumed

Verified against betdb (`:5433`) on 2026-08-06 via the `betdb-postgres` MCP. Re-run these if
anything looks off; do not trust the numbers without checking, they age.

### 4.1 The join problem — **this is the hardest unsolved thing in the system**

```sql
SELECT COUNT(DISTINCT m.event_id) AS bf_events, COUNT(DISTINCT e.match_id) AS matched
FROM betfair_live.market_metadata m
LEFT JOIN sofascore.events e
  ON lower(e.home_team)=lower(m.home_team) AND lower(e.away_team)=lower(m.away_team);
--  bf_events = 93 ,  matched_sofascore = 25
```

**Only 25 of 93 Betfair events resolve to a SofaScore fixture by exact team-name match — 27%.**

`betfair_live.market_metadata` carries `market_id, event_id, event_name, competition,
market_type, home_team, away_team, open_date, competition_id`. There is **no `match_id` column
and no mapping table.** Betfair says `"Waterford v Shelbourne"` / `"St Patricks"`; SofaScore
says something else. The Redis path papers over this with hand-maintained `home_slug`/`away_slug`
strings, which is why `scratch_redis_types.jl` exists — it is a script for debugging why
`("bohemian", "dundalk-fc", "MATCH_ODDS")` was not found.

Everything downstream is gated on this. A fixture that does not resolve is not merely
mispriced, it is **invisible** — and invisible failures are this project's recurring theme.
Treat identity resolution as a first-class subsystem with its own types, its own confidence
level, and its own audit output. Candidate approaches to weigh: a persisted alias/crosswalk
table, normalised-name + kick-off-time + competition matching, fuzzy match with a manual
review queue, or resolving via `betfair.match_meta` (check whether that older table already
solves this for historical data — if it does, reuse rather than reinvent).

### 4.2 `betfair_live.order_book_1m` — full depth, and it is *archived*

```
428,411 rows | 941 markets | 2026-05-29 → 2026-08-02 14:57 UTC
columns: market_id, symbol, ts, bid_prices[], bid_volumes[], ask_prices[], ask_volumes[],
         total_matched, market_matched, last_price_traded
```

3 levels each side; prices and volumes are integers scaled **×10000**. Sample
(`Waterford v Shelbourne`, OVER_UNDER_25, 2026-08-02):

```
Under 2.5   bid [12500,12400,12300]   ask [12600,12700,12800]   →  back 1.25 / lay 1.26
Over  2.5   bid [48000,47000,46000]   ask [50000,51000,52000]   →  back 4.80 / lay 5.00
```

The bid/ask convention checks out arithmetically: back side `1/1.25 + 1/4.80 = 1.0083`
(0.83% overround, correct sign), lay side `1/1.26 + 1/5.00 = 0.9937` (< 1, correct sign).
So **`bid` = available to back, `ask` = available to lay.** Worth re-deriving rather than
trusting this line.

Two consequences the design should exploit:

1. **`odds_close` in the historical backtest is `last_price_traded`, but the executable price
   is the bid.** Portfolio's backtest settles at a price you may not have been able to get. A
   1% haircut test (`r05_extending.jl`) costs ~24% of cumulative gain. A MatchDay layer that
   reads the *actual* book can size on the price it can actually take — and can measure the
   gap. This is a real improvement over the backtest, not just parity.

2. **This is a historical archive of a live feed.** Which means match-day behaviour can be
   *backtested*: replay the 1-minute bars, run the pipeline as of T-minus-N minutes, compare
   to what closed. The current prototype can only be exercised on a live Saturday, which is
   why nothing about it has ever been validated. **Designing for replay from day one is
   probably the single highest-value structural decision available here.** Check whether the
   Redis feed and `order_book_1m` are written by the same collector (inferred, not proven) —
   if they are, one adapter can serve both live and replay.

Coverage by competition (distinct markets in `market_metadata`):

```
Irish Division 1        38 events × 9 market types (+DOUBLE_CHANCE 19, ASIAN_HANDICAP 19)
Irish Premier Division  33 events × 9 market types (+DOUBLE_CHANCE 12, ASIAN_HANDICAP 12)
Scottish Premiership     6 events
market types present: MATCH_ODDS, BOTH_TEAMS_TO_SCORE, CORRECT_SCORE,
                      OVER_UNDER_05/15/25/35/45/55, DOUBLE_CHANCE, ASIAN_HANDICAP
```

O/U **5.5** and CORRECT_SCORE are in the live feed but not in the runbook's `MARKETS`.
Conversely the modelled leagues are ScottishLower (56/57) and Ireland (79/718) — note
**Scottish Premiership** appears in the live feed and is a *different* segment from
ScottishLower. Reconcile what is modelled, what is quoted live, and what is worth betting.

⚠️ Last order-book timestamp is **2026-08-02**, four days before this brief was written.
Either the collector is down, or the leagues are between fixtures. Establish which before
designing around feed availability — and the design should surface staleness rather than
silently price off a four-day-old book.

### 4.3 Freshness and the clock

`sofascore.events.start_timestamp` is an epoch integer. `lineup_provisional` carries
`scraped_at` and `confirmed`. `order_book_1m.ts` is minute-resolution.

The system currently has **no concept of time-to-kick-off**, yet almost every decision depends
on it: lineups firm up ~1h before, liquidity builds toward the off, and the whole "closing
line" framing of the backtest presupposes you bet near the close. A `MatchDay` design that
does not carry an as-of timestamp through the pipeline cannot be replayed, cannot be audited,
and cannot tell a confirmed XI from a guess.

Recommend making **as-of time an explicit parameter of every stage**, not an implicit `now()`.

### 4.4 Connection surfaces

`ENV["BF_DB_URL"]` → `postgresql://…@100.124.38.117:5433/betdb`.
Redis → `RedisConnection(host="100.124.38.117", port=6379)`, hashes `live_market_meta` and
`live_markets`. Both are hardcoded in scratch files; neither is configured anywhere central.

---

## 5. Open design questions — ranked

Bring a recommendation for each, not just the question.

1. **Identity resolution.** Betfair event ↔ SofaScore match_id at 27%. Own subsystem, or a
   field on a fixture object? Persisted crosswalk or computed each run? What happens to an
   unresolved fixture — dropped, or surfaced loudly?
2. **Replay or live-only.** Does `MatchDay` take an as-of timestamp and a pluggable book
   source (live Redis / archived `order_book_1m`), or is it live-only like the prototype? This
   determines whether the system can ever be validated.
3. **Where MatchDay ends and Portfolio begins.** The clean answer is "MatchDay produces
   `(latents_df, odds_df)` and hands them to `stake_sheet`". Does that survive contact with
   the market-pillar circularity in §3 (`inference.jl`)?
4. **Execution price.** Back-only at the bid, or model back/lay properly? The user has already
   observed that backing Over ≈ laying Under, so the O/U ladder plus lays is a redundant
   over-complete book — potentially a real edge (take whichever side is cheaper), potentially
   a footgun (double-counted exposure). Portfolio has an `AbstractPricePolicy` seam that could
   host this; `BestExecution` was scoped and never built.
5. **Feature materialisation for unplayed fixtures.** Generalise `inject_matchday_features!`
   into a per-feature dispatch, or keep the ratings special case and accept that only
   ratings-based engines can be run on match day?
6. **Train/serve skew.** Is match-day rating construction provably the same as training-time?
   How would you *test* that, given the design is meant to be testable?
7. **Staleness and gating.** What are the refuse-to-bet conditions — no confirmed XI, book
   older than N minutes, spread wider than X, market matched below £Y? The `MinLiquidity`
   filter was scoped for Portfolio and never built; it may belong here instead.
8. **What gets bet.** `unified_staking.jl` says O/U + BTTS only, 1X2 display-only. The
   staking-layer work found CorrectScore a −20% ROI drag. Does the live book's market set
   (which includes CS, DC, AH, O/U 5.5) change that? Should the answer be a config, or baked in?
9. **Output surface.** Terminal dashboard, persisted stake sheet, or both? If bets are to be
   reconciled against results later, something must be written down at bet time — the
   prototype writes nothing, so no paper-trading record exists beyond the two `paper_tracks`
   notes.
10. **Module home.** `src/MatchDay/` alongside `src/Portfolio/`, or does part of it belong in
    `src/Data/` (fixtures, live odds) and `src/features/` (materialisation)? The repo idiom is
    `<name>-module.jl` → `types.jl` → `interfaces.jl` → `implementations/`.

---

## 6. Strawman to argue with

Deliberately incomplete and probably wrong in places. Improve it; do not defend it.

```
                    ┌──────────────────────────────────────────────┐
   as_of::DateTime  │              MatchDaySpec                     │
   (explicit, never │  fixtures  :: AbstractFixtureSource           │
    implicit now()) │  identity  :: AbstractIdentityResolver        │
                    │  lineups   :: AbstractLineupSource            │
                    │  book      :: AbstractBookSource              │
                    │  gate      :: AbstractReadinessGate           │
                    └──────────────────────────────────────────────┘
                                       │
   STAGE 1  fixtures ──────────────────┼──────────────────────────────────────────
   sofascore.events                    ▼
   status='notstarted'          Vector{Fixture}
                                (m_id, home, away, kickoff::DateTime, tournament)
                                       │
   STAGE 2  identity ──────────────────┼──────────────────────────────────────────
   market_metadata            resolve(r, fixture) -> MarketRef | Unresolved
   + crosswalk                         │            (bf_event_id, market_ids, confidence)
   ⚠ 27% exact-match today             │
                                       ▼
   STAGE 3  lineups ───────────────────┼──────────────────────────────────────────
   JSON pin > provisional      lineup(src, fixture, as_of) -> Lineup
   > last historical XI                │   (home, away, confirmed::Bool, source::Symbol)
                                       │
   STAGE 4  features ──────────────────┼──────────────────────────────────────────
   tracker rolls history       materialise(::Val{:player_ratings}, ...)
   forward to as_of                    │   ← the generic replacement for
   ⚠ train/serve skew risk             │     inject_matchday_features!
                                       ▼
   STAGE 5  book ──────────────────────┼──────────────────────────────────────────
   Redis (live) OR                quotes(src, ref, as_of) -> odds_df
   order_book_1m (replay)              │   :match_id :market_name :market_line
   ⚠ bid=back, ask=lay, ×10000         │   :selection :odds_close   ← PORTFOLIO'S SCHEMA
                                       │
   STAGE 6  inference ─────────────────┼──────────────────────────────────────────
   last split's chain          latents_df (λ draws per fixture)
   ⚠ market-pillar engines             │   ⚠ needs odds_df from STAGE 5 → not a straight line
     consume STAGE 5 output            │
                                       ▼
   STAGE 7  gate ──────────────────────┼──────────────────────────────────────────
   refuse to bet on:           ready(gate, ctx) -> Bool + reason
   stale book / no XI /                │   loud, never silent
   unresolved identity                 │
                                       ▼
   ═══════════════════════ HANDOFF ════╪═══════════════════════════════════════════
                                       ▼
                    PF.stake_sheet(sys, latents_df, expr, odds_df, ds; bankroll)
                                       │
                                       ▼
                              stake sheet + slate_summary
                              (persisted, with as_of + book snapshot,
                               so it can be reconciled later)
```

Candidate seams, in the `src/Portfolio` idiom (abstract type + exactly one contract method):

| Abstract type | Contract | Implementations to consider |
|---|---|---|
| `AbstractFixtureSource` | `fixtures(src, segment, as_of) -> Vector{Fixture}` | `SofaScoreEvents`, `ExplicitFixtures` (for replay/tests) |
| `AbstractIdentityResolver` | `resolve(r, fixture) -> MarketRef \| Unresolved` | `ExactName`, `Crosswalk`, `FuzzyName`, `ResolverChain` |
| `AbstractLineupSource` | `lineup(src, fixture, as_of) -> Lineup` | `JsonPin`, `ProvisionalDB`, `LastHistorical`, `SofaScoreAPI`, `SourceChain` |
| `AbstractBookSource` | `quotes(src, ref, as_of) -> DataFrame` | `RedisLive`, `ArchivedOrderBook`, `HistoricalClose` |
| `AbstractReadinessGate` | `ready(gate, ctx) -> (Bool, String)` | `ConfirmedXI`, `MaxBookAge`, `MinMatched`, `GateChain` |

Note the `…Chain` pattern recurs — the prototype's lineup tiering and the identity fallbacks
are both "try these in order". `src/Portfolio/implementations/filters.jl` already has
`FilterChain`; steal its shape.

**Question the strawman explicitly**: is `AbstractBookSource` really one seam, or is
"where do prices come from" and "which price in the book do I take" two seams that want
separating — the latter arguably belonging to Portfolio's existing `AbstractPricePolicy`?

---

## 7. Deliverables for this session

1. `current_development/match_day_inference/ARCHITECTURE.md` — the design document. Match the
   depth of the Portfolio one: file map, seams table, staged pipeline, objects, call graph,
   "things that will confuse you", open questions.
2. ASCII diagrams as in §6 but correct and complete.
3. A types/dispatch sketch — abstract types, contracts, domain structs, config structs — as a
   fenced Julia block **inside the document**, not as `.jl` files.
4. A short honest section on what the prototype got right and should be preserved verbatim.
5. A migration note: which prototype files die, which move to `src/`, which stay as runners.
6. The ranked open-questions list, updated with your recommendations, for the user to decide on.

---

## 8. Ground rules

- **Verify before asserting.** Every number in §4 came from a query you can re-run. Do the
  same for anything you add. This project has been burned repeatedly by confident claims about
  data coverage that turned out to be wrong.
- **Silent failures are the enemy.** Two bugs shipped this month were guards that quietly did
  nothing (a league switch that no-opped; a cache key that collided across leagues). Design
  so failure is loud.
- **The kaimon MCP REPL** runs Julia on the user's server (`mcmc-beast`, repo
  `/root/BayesianFootball`) — better machine than the laptop. Local edits reach it only via
  git push then pull on the server. Use `betdb-postgres` MCP for SQL. Do not run long Julia
  jobs locally.
- **Do not build.** If the design feels ready, that is the signal to hand back to the user for
  the next session, not to start writing `src/MatchDay/`.
- The user is the domain expert and has strong, evidence-backed priors from prior streams
  (curated market sets beat full books; per-bet full Kelly bankrupts; calibrate the centre and
  find edge in the tails). Ask rather than assume when a design choice touches those.
