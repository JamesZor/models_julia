# MatchDay Live Execution & Paper Trading — architectural blueprint

**Branch:** `feat/matchday-live-architecture` (from `feat/pxg-rapm-unified-covariates`, `e24bde97`)
**Measured:** 2026-09-01, against `betdb` @ `192.168.1.88:5433` and `mcmc_experiments` @ `mcmc-beast:5432`
**Scope:** Scottish Premiership / Championship / League One / League Two (SofaScore 54, 55, 56, 57)
**Status:** research + design. Nothing here is implemented; §12 is the work plan.

---

## 0. How to read this

Every number in §4 was measured in this session against the live database, and the query that
produced it is described where it is used. §13 lists what is **not** verified, separately, so a
reader can tell an assertion from a measurement.

Three documents already exist and this one does not replace them:

| document | what it owns |
|---|---|
| `current_development/match_day_inference/ARCHITECTURE.md` | why `src/MatchDay` has the seams it has; the user's 2026-08-06 decisions |
| `current_development/matchday_runbook/ARCHITECTURE.md` | the system map of `src/MatchDay` as built |
| `current_development/orderbook_layer2/RESULTS.md` | the Layer-2 verdict on Ireland: entry timing, trust, curation |

This document owns the part none of them cover: **the live loop** — collector to price to order to
ledger to console — and the paper-trading substrate that has to exist before a single real order
is placed.

**§5 is the spine.** The organising commitment is that the unit of execution is the *slate* — the
simultaneous card, solved as one joint allocation and committed as one batch — not the individual
bet. §6 (ledger), §7 (concurrency), §8 (console) and §12 (roadmap) are all consequences of it.

---

## 1. Executive summary

### 1.1 The system already exists in three pieces that have never been joined

```
  mcmc_experiments (mcmc-beast:5432)      betdb (192.168.1.88:5433)        archpc
  ┌──────────────────────────┐            ┌──────────────────────┐      ┌──────────────┐
  │ runs / fold_results      │            │ sofascore.events     │      │ Textual TUI  │
  │ fit_artifacts  (868 MB)  │            │ sofascore.lineup_*   │      │ betdb_mcp    │
  │ match_latents  (343 MB)  │            │ betfair.match_meta   │      │ supervisor   │
  │ portfolio_runs / _bets   │            │ betfair_live.*       │      │ stream_worker│
  └──────────────────────────┘            │ core.matchday_action │      └──────────────┘
             ▲                            └──────────────────────┘              ▲
             │  load_fit / extend_fit                 ▲                          │
             │                                        │ direct SQL               │ arm/drain
        ┌────┴────────────────────────────────────────┴──────────┐               │
        │  BayesianFootball.jl :  MatchDay -> Portfolio           │───────────────┘
        │  fixtures→identity→lineups→BOOK→features→inference→gate │   (no link today)
        └─────────────────────────────────────────────────────────┘
                                 │
                            stake_sheet  ──────────►  ??? nothing consumes this
```

`src/MatchDay` (2,498 lines) produces an executable stake sheet — a **slate-wide stake vector**,
solved jointly under `SlateDrawdown` and `FixedCap` — with `side`, `venue_odds`, `venue_selection`,
`risk` and `venue_stake` columns, and `order_ticket(row)` renders one order.
**Nothing consumes it.** There is no order store, no fill model, no bankroll of record, no
settlement, no console. That gap is what this blueprint fills.

### 1.2 Six decisions

| # | Decision | Rests on |
|---|---|---|
| **D0** | **The slate is the atom.** The Saturday 15:00 card is priced, reserved and executed as **one joint allocation** — `SlateDrawdown(λ)` returns a single scalar `k` for all 21 fixtures and `FixedCap(c)` rescales the whole vector. A stake vector is only valid whole; partial execution produces a position the allocator never authorised. | §5 |
| **D1** | **Enter at the close, not at the lineup drop.** Target `T−12m`, with a hard floor at `T−4m`. | §4.6 + `orderbook_layer2/RESULTS.md` §4.1, which agree independently |
| **D2** | **Capacity, not edge, is the binding constraint below the Premiership.** Size legs from the measured fill curve, not from Kelly alone. | §4.5 — £25 fills 74–85% of the time on League Two 1X2; £100 fills 25–31% |
| **D3** | **Paper ledger in `betdb`, schema `paper`,** not in `mcmc_experiments`. The ledger has to be written while the model DB may be mid-training and unreachable. | §6 |
| **D4** | **One bankroll writer, one atomic reservation.** The whole slate's `Σ risk` moves in a single `SELECT … FOR UPDATE` transaction *before* any order is submitted — the `SlateBatch` `RESERVED` transition. Submitters afterwards touch no account row. Never per-fixture concurrency on the bankroll. | §5.3, §7.2 |
| **D5** | **Web dashboard, not a new TUI.** A Textual TUI already exists for the *collector*; the trading console must show the **whole slate at once** — 21 cards with model-vs-market bars, EV%, and one atomic **Execute Slate Batch** action. That is where a terminal grid stops paying. Julia + HTTP.jl + WebSockets + Alpine, ~600 lines. | §8 |

### 1.3 The blocker that outranks all of it

**Live capture has been decided-but-not-executed since 2026-08-28.** `core.matchday_action` holds
569 rows for 2026-08-29 alone, all `arm / not_armed / executed=false`, reason
*"nothing is capturing and kickoff in 89 min"*, `{"in_play": 0, "fixtures": 38}`. The supervisor
is running in **dry-run** mode: it decides correctly every 60 s and carries nothing out. The last
order-book row in the database is `2026-08-28 20:59`. A full Scottish round on 2026-08-29 (16
fixtures) has no book at all.

Nothing in §12 can be validated until that is flipped to `execute`. It is a one-line operational
change in the Python daemon, not an engineering project, and it is Phase 0.

---

## 2. Current state — what is actually built

### 2.1 Julia: `src/MatchDay` (this repo)

Nine swappable seams, `as_of` threaded through every stage and no stage reading the clock. The
pipeline is **not** a straight line:

```
fixtures ─► identity ─► lineups ─► BOOK ─► features ─► inference ─► gate ─► stake_sheet
                                     ▲                                          ▲
                        market-pillar engines consume            everything past here
                        odds as a MODEL feature                     is src/Portfolio
```

| seam | default | file |
|---|---|---|
| `AbstractFixtureSource` | `SofaScoreEvents(horizon=36h)` | `implementations/sources.jl` |
| `AbstractIdentityResolver` | `MatchMetaCrosswalk` (+ `LiveNameMatch` fallback) | `implementations/sources.jl` |
| `AbstractLineupSource` | `SourceChain(ProvisionalDB(), LastHistorical())` | `implementations/sources.jl` |
| `AbstractBookSource` | `ArchivedOrderBook(max_age=6h)` | `implementations/book.jl` |
| `AbstractQuoteRule` | `BestAvailable` | `implementations/book.jl` |
| `AbstractInstrumentRule` | `BestOfBackLay(max_leverage=20)` | `instruments.jl` |
| `AbstractStakeRounding` | `NoMinimum` | `implementations/book.jl` |
| `AbstractFeatureMaterialiser` | `MaterialiserChain(RatingsFromTracker, LeagueFromFixture)` | `inference.jl` |
| `AbstractReadinessGate` | `GateChain(IdentityResolved, MaxBookAge(30m))` | `implementations/gates.jl` |

The load-bearing idea is the **instrument morphism**: laying at `d` is backing at `d/(d−1)` with
backer stake `risk/(d−1)`, so `Portfolio` never learns that lays exist and `FixedCap` sums
liability by construction. Only `order_ticket` sees the difference — and `Instrument.venue_key`
exists because on a synthetic the runner the order touches is the *complement*.

### 2.2 Julia: `src/Portfolio`

`stake_sheet(sys, latents_df, expr, odds_df, fixtures::Dict{Int,FixtureInfo}; bankroll)` is the
live entry point, and it is the **same code path** the backtest was audited on — a match-day book
differs from a backtest book in exactly one respect, the absence of a settlement vector. Risk is
solved **per slate**, so every fixture settling together shares one drawdown budget and one
exposure cap. That is why the 15:00 Saturday card must be solved as one problem, not as 21.

### 2.3 Julia: model artefacts (`mcmc_experiments` @ `mcmc-beast:5432`)

```
runs (uuid, name, experiment_name, status, git_commit, git_branch)
 └─ fold_results (fold_id, fold_idx, r_hat_max, ess_*, divergences, converged, logloss, …)
     └─ match_latents (match_id, mean/std/p10/p50/p90 λ_h λ_a, draws_blob)   343 MB
fit_artifacts (868 MB)     config_registry     portfolio_runs / portfolio_bets / portfolio_artifacts
```

31 runs registered. The Scottish-relevant experiment families are
`scottish_lower_poisson_2426`, `scottish_lower_joint_2426`, `scottish_lower_player_grid_2426`
and `scottish_lower_joint_player_2426`; the production-shaped ones carry 40 folds
(`m00_baseline`, `m02_wealth`, `m04_joint`, `m05_production_wealth`, `m08_joint_composite`,
`m00_poisson_control`, `m05_joint_production_wealth`), the rest are 1-fold smokes.
Access is `PostgresStorage(name)` → `load_fit`, `load_model`, `preview_extension`, `extend_fit`.

`portfolio_runs` / `portfolio_bets` already exist as a **backtest** ledger. The paper ledger is
deliberately *not* an extension of them (§6.1).

### 2.4 Python: the operational stack (`/home/james/bet_project/database`)

A uv workspace of four packages, and it is more mature than anything on the Julia side:

| package | role |
|---|---|
| `core` (`betdb`) | SQLAlchemy read model, `matchday_log`, `locks`, `agent_requests` |
| `runtime` (`betdb_runtime`) | controllers: `matchday_supervisor`, `live`, `operations`, `state` |
| `tui` | Textual app — `screens/live/{dashboard,markets,matches,monitor,tournament_cards}`, `widgets/status_card.py` (527 lines) |
| `mcpd` (`betdb_mcp`) | MCP server exposing `betdb_matchday_status`, `betdb_crosswalk_rebuild`, `betdb_matchday_arm`, … over the same controllers |

`matchday_supervisor.decide()` is a **pure function of `(report, now, memo)`** returning
`NOTHING | ARM | REARM | START_DRAIN`, logged to `core.matchday_action`. This is the pattern the
trade-execution state machine in §6.3 should copy verbatim: pure decision, impure shell, every
decision written down whether or not it fired.

`core.tui_heartbeat` shows `archpc:904812` alive at 2026-09-01 20:10; `core.worker_lease` shows
that same pid holding the `live-drain` lease since 2026-08-28 20:38. The process is up. It is
just not executing.

---

## 3. Mid-week model pipeline → match-day in-flight pricing

### 3.1 The weekly cadence

```
Tue    extend_fit(db, :m05_joint_production_wealth, ds)   # new folds only; existing untouched
       └─ preview_extension first: prints folds needed, new matches, estimated compute
Wed    audit_convergence — r_hat_max, ess_bulk_min, divergences per fold  (fold_results)
Thu    Portfolio walk-forward on the extended fit; PolicySpec sweep reuses cached books
Fri    dry-run match_day() at as_of = Sat 15:00 on LAST week's book — proves the plumbing
Sat    the live loop (§9)
```

`extend_fit` is the right primitive because the fold indices are **global**: `_ExtensionSampler`
preserves the splitter's global positions while `run_folds` sees a dense delta, so a mid-season
extension does not renumber the folds that `select_split` will later name.

### 3.2 Picking the fold to condition on — the silent failure this already fixed

`MatchDay.select_split` has three rules, in order:

1. **Positive identification.** `Data.get_next_matches(ds, boundaries[i], config)` is the block
   fold `i` was built to predict; the right fold for a card is the one whose next block *is* this
   card. Same call the OOS path uses, so train and serve agree by construction.
2. **Negative fallback (`exclude`).** The most recent fold whose target window contains none of
   the ids being priced. This is the normal LIVE case — an unplayed fixture is not in `ds.matches`
   at all, so rule 1 cannot fire.
3. **Positional.** `min(n_trained, n_bounds)` — the original behaviour, and the one that silently
   conditioned on a fold that had already seen the card when a cache rebuild regrew the last fold.

**Operational consequence.** Because live fixtures always fall through to rule 2, the pricing job
must pass `exclude = [f.m_id for f in fixtures]`. `matchday_latents` already does. Any new caller
that does not is reproducing the bug the rule exists to prevent.

### 3.3 Lineup release → ratings → λ → grid

```
sofascore.lineup_provisional      ProvisionalDB      RatingsFromTracker
   (player_id, position,     ──►  Lineup(home,  ──►  latest_player_ratings(ds, tracker)
    substitute, is_home,          away, source,      = calculate_player_ratings(tracker,
    confirmed, scraped_at)        scraped_at)          vcat(history, missing)) |> last
                                                             │
                                                             ▼
                              player_ratings_map[m_id][(side, pos)] = Σ over STARTERS
                                                             │
   league_lookup[m_id] ◄── LeagueFromFixture                  │
                                                             ▼
                              extract_parameters(model, frame, fs, chain)
                                                             │
                                       λ_h, λ_a draws  ──►  SmileScoreGrid  ──►  1X2 / O-U / BTTS
```

Three things are load-bearing and easy to get wrong:

* **`INJECTABLE_KEYS = (:player_ratings_map, :league_lookup)`.** Both are read as
  `get(map, match_id, default)`. A fixture missing from `player_ratings_map` prices at **zero
  player strength**; one missing from `league_lookup` gets index 0, which **zeroes the zero-sum
  `δ_league` offset** — on a pooled `[56, 57]` engine that is exactly the goal-level gap between
  League One and League Two, so the fixture is priced at the mean of the two tiers. `check_coverage`
  refuses rather than allowing either.
* **`month_idx` must be supplied explicitly.** Engines read it as
  `hasproperty(row, :month_idx) ? Int(row.month_idx) : 1`, so omitting it applies **January's**
  seasonality to every fixture.
* **`season_idx` must NOT be supplied.** Its fallback is `n_seasons`, which is already the correct
  season for an upcoming fixture.

**Markets.** `MatchDaySpec.markets` defaults to `1X2 + BTTS + O/U {0.5, 1.5, 2.5, 3.5, 4.5}` —
17 selections per fixture. The smile pillar prices O/U through its own intensity `λ_tot·φ(K)`
while 1X2/BTTS/CS come off the goals grid; `smile_poisson.jl` owns that split.

### 3.4 Measured: the lineup feed is now good, and now dead

`sofascore.lineup_provisional`, all 1,533 rows / 40 matches:

| fact | value | what it retires |
|---|---|---|
| `confirmed = true` rows | **1,071 of 1,533 (70%)** | *"`confirmed` has never been true"* — `types.jl`, `gates.jl`, `sources.jl` |
| scrape lead on the 2026-08-08/09 round | **T−13 to T−42 min**, median ≈ T−29 | *"every scrape has run 4.4–5.8 h before kick-off"* |
| scrapes per match | 1–2 | — |
| **last scrape anywhere** | **2026-08-09 15:24** | the feed is dead, 3 weeks |

Two consequences.

1. `ConfirmedXI(blocking = true)` is now a **usable** gate; the docstring saying it would block
   100% of fixtures is three weeks out of date. `MaxLineupAge(Hour(2), blocking = true)` is
   satisfiable too.
2. **The XI lands at T−29, not T−60.** The premise "enter at the T−60 lineup drop" is not
   achievable against this feed regardless of what the market is doing. The earliest instant at
   which the model has today's XI is roughly **T−25**, which is inside the window D1 recommends
   anyway. This is a happy accident, not a design: it should be made deliberate by scheduling the
   scrape at T−35 and the pricing run at T−25.

---

## 4. `betdb` order-book audit

All figures below: `betfair_live.order_book_1m` ⋈ `betfair_live.market_metadata`, Betfair
competition ids `105` (Premiership), `107` (Championship), `109` (League One), `111` (League Two),
season 26/27, kick-offs 2026-07-31 … 2026-08-28.

### 4.0 Scaling — corrected

Prices **and** volumes are integers scaled **×10 000**. `market_matched` and `total_matched` are
too. This was verified two ways: per-runner `total_matched` sums exactly to `market_matched`, and
Kilmarnock v Celtic's MATCH_ODDS peak `market_matched = 2 499 888 300` → **£249,989**, which is the
right order of magnitude for a Scottish Premiership fixture (a ×100 reading gives £25 M, which is
not). `src/MatchDay/implementations/book.jl:_unscale` already divides by 10,000 — **it is correct**,
and any new consumer must use the same factor. Top-of-book sizes of `20000` are therefore **£2.00**,
the Betfair minimum, not £200.

### 4.1 Inventory

```
betfair_live.order_book_1m   TimescaleDB hypertable, 11 chunks, uncompressed
                             635,765 rows | 1,813 markets | 2026-05-29 13:11 → 2026-08-28 20:59
  market_id, symbol, ts, bid_prices[], bid_volumes[], ask_prices[], ask_volumes[],
  total_matched, market_matched, last_price_traded
betfair_live.market_metadata 1,725 rows: market_id, event_id, event_name, competition,
                             competition_id, market_type, home_team, away_team, open_date
```

`bid_*` is the **back** side and `ask_*` the **lay** side — the only assignment for which the back
side sums above 1 and the lay side below 1. Verified again here: League One O/U 2.5 at
2026-08-15 13:01 gave back 1/1.43 + 1/2.92 = **1.042**, lay 1/1.47 + 1/3.35 = **0.979**.

Ladder depth stored, all 635,765 rows:

| levels | rows |
|---|---|
| 3 | 537,428 (84.5%) |
| 2 | 29,658 |
| 1 | 30,638 |
| none | 38,041 (6.0%) |

**Only 3 levels are archived.** Every capacity number in §4.5 is therefore a *lower bound* on what
the live API would show — the live `listMarketBook` can return 10 levels. Replay-derived fill
estimates are conservative by construction, and that is the safe direction.

NULL rates: `market_matched` 42.0% overall / **14–15% on Scottish**; `last_price_traded` 75.4%
overall. But see 4.2 — that number is a historical average of two different regimes.

### 4.2 `last_price_traded` started working on 2026-08-07

| kick-off day | rows | % `last_price_traded` present |
|---|---|---|
| 2026-05-29 … 2026-08-02 | 428,411 | **0.0** |
| 2026-08-07 | 30,856 | 69.6 |
| 2026-08-08 | 43,298 | 68.5 |
| 2026-08-09 | 23,539 | 87.5 |
| 2026-08-15 | 21,113 | 55.6 |
| 2026-08-21 | 20,098 | 83.1 |
| 2026-08-22 | 44,687 | 84.3 |
| 2026-08-28 | 23,763 | 79.1 |

This **retires** the health warning in `src/MatchDay/matchday-module.jl` and
`implementations/book.jl` that says *"`last_price_traded` is NULL in 100% of `order_book_1m`, so
this module … cannot reproduce the backtest's `odds_close`"*. From 2026-08-07 it can, on ~80% of
rows. That matters for CLV: a traded price and a book price are different quantities, and now both
are available to compare rather than only one.

### 4.3 Collector coverage — the real constraint

**Per-fixture.** 93 Scottish fixtures were played 2026-08-01 … 2026-08-31. Only **69** reached
`betfair_live.market_metadata` at all. Of those 69, using the MATCH_ODDS market:

| had a book at… | events | % of the 69 | % of all 93 |
|---|---|---|---|
| T−75 … T−45 min | 54 | 78% | **58%** |
| T−20 … T−5 min | 53 | 77% | 57% |
| T−5 … T−0 min | 53 | 77% | 57% |

**Per-week — the collector's lead time collapsed.** First snapshot relative to kick-off:

| kick-off | first snapshot | markets/event | note |
|---|---|---|---|
| 2026-08-01 → 08-03 | T−1685 … T−1012 min | 11 | continuous collection |
| 2026-08-07 → 08-09 | T−270 … T−120 min | 11 | |
| 2026-08-15 | T−116 min | 11 | 9 events only; Championship & Premiership absent |
| 2026-08-21 / 22 | **T+6 min** | 8 | started *after* kick-off — 15 fixtures unpriceable |
| 2026-08-28 | T−103 / T−88 min | 8 | 2 events only |
| 2026-08-29 onward | **nothing** | — | supervisor dry-run, §1.3 |

**Market set narrowed on 2026-08-21**, from 11 markets/event to 8: `ASIAN_HANDICAP`,
`DOUBLE_CHANCE` and `OVER_UNDER_55` were dropped. This coincides with expansion from 4 to 7
competitions (272 markets / 34 events on 2026-08-22), which is consistent with staying under
Betfair's per-connection subscription cap — a cap that `betdb_matchday_status` documents as failing
*silently* when exceeded. **The 8 that survived are exactly `MatchDaySpec`'s default market set
plus `CORRECT_SCORE`**, so this costs the current pricing path nothing. It is a live constraint to
watch, not a defect.

**Identity crosswalk is dead again.** `betfair.match_meta` resolves 100% of Scottish fixtures
2026-08-01 → 2026-08-28 except 2026-08-15 (4/5 in tournament 56) and 2026-08-22 (1/6 in 54), and
**0 of 159 fixture-days from 2026-08-29 onward**. `MatchMetaCrosswalk` alone would refuse the
entire card; the fix is `ResolverChain(MatchMetaCrosswalk(), LiveNameMatch())` — which is opt-in
precisely so a dead crosswalk stays visible — plus running `betdb_crosswalk_rebuild` **before**
kick-off, since there is no retrospective fix.

### 4.4 Spreads and depth at the touch

Median across all snapshots in each bucket. `tick` = Betfair ladder steps between best back and
best lay. `depth ≤2 ticks` = size resting within two ticks of the touch, in £.

| competition | market | bucket | tick | spread % | back ≤2tk | lay ≤2tk | matched £ |
|---|---|---|---|---|---|---|---|
| Premiership | MATCH_ODDS | T−130…T−61 | 1 | 1.42 | 380 | 391 | 14,755 |
| | | T−60…T−31 | 1 | 1.42 | 522 | 706 | 19,901 |
| | | T−15…T−6 | 1 | 1.40 | 702 | 736 | 28,839 |
| | | T−5…T−0 | 1 | 1.39 | 672 | 630 | 32,951 |
| | OVER_UNDER_25 | T−60…T−31 | 2 | 1.61 | 186 | 158 | 1,924 |
| | | T−15…T−6 | 2 | 1.28 | 254 | 259 | 3,818 |
| Championship | MATCH_ODDS | T−60…T−31 | 2 | 3.54 | 87 | 85 | 1,345 |
| | | T−15…T−6 | 2 | 2.86 | 110 | 126 | 3,478 |
| | OVER_UNDER_25 | T−60…T−31 | 5 | 3.58 | 28 | 39 | 176 |
| | | T−15…T−6 | 4 | 2.95 | 61 | 40 | 219 |
| League One | MATCH_ODDS | T−60…T−31 | 4 | 4.44 | 53 | 42 | 448 |
| | | T−15…T−6 | 3 | 3.87 | 79 | 69 | 2,103 |
| | OVER_UNDER_25 | T−60…T−31 | 5 | 3.81 | 21 | 34 | 50 |
| | | T−15…T−6 | 5 | 3.59 | 33 | 54 | 137 |
| League Two | MATCH_ODDS | T−60…T−31 | 5 | 5.45 | 42 | 19 | 583 |
| | | T−15…T−6 | 3 | 4.07 | 49 | 39 | 1,330 |
| | OVER_UNDER_25 | T−60…T−31 | 6 | 4.21 | 12 | 32 | 17 |
| | | T−15…T−6 | 6 | 4.26 | 31 | 22 | 39 |
| League Two | BTTS | T−60…T−31 | 9 | 6.33 | 34 | 168 | 25 |

Three readings.

* **The book tightens toward the off, monotonically and everywhere.** League Two 1X2 goes 5 ticks
  → 3 ticks and 5.45% → 4.07%; League One 4 → 3. Nowhere does the spread widen into the close.
* **BTTS is a trap in the lower leagues.** 9 ticks wide with £168 resting on the lay side and £25
  ever matched. There *is* size — it is just parked far from anything resembling fair value.
  A depth-only filter passes it; a spread filter refuses it. Filter on both.
* **`MinMatched(500.0)` would block almost everything below the Premiership.** League Two O/U 2.5
  markets carry £17–46 of matched volume in total. The gate is non-blocking by default, which is
  correct, but its threshold was chosen for a different league.

### 4.5 Capacity — the fill curve

For each snapshot, sweep the archived back ladder and ask: can `£X` be filled in full, and at what
VWAP relative to the touch? Cells are **% of snapshots filling in full / median slippage**.

| competition | market | window | £10 | £25 | £50 | £100 | £250 |
|---|---|---|---|---|---|---|---|
| Premiership | MATCH_ODDS | T−75…T−45 | 100% / 0.00% | 100% / 0.00% | 100% / 0.00% | 99% / 0.00% | 72% / 0.46% |
| | | T−20…T−5 | 100% / 0.00% | 100% / 0.00% | 100% / 0.00% | 100% / 0.00% | 94% / 0.02% |
| | OVER_UNDER_25 | T−20…T−5 | 100% / 0.00% | 99% / 0.00% | 98% / 0.00% | 92% / 0.11% | 50% / 0.39% |
| Championship | MATCH_ODDS | T−75…T−45 | 100% / 0.00% | 92% / 0.27% | 78% / 0.69% | 41% / 0.79% | 10% / 0.91% |
| | | T−20…T−5 | 100% / 0.00% | 98% / 0.15% | 80% / 0.52% | 54% / 0.50% | 19% / 0.10% |
| | OVER_UNDER_25 | T−20…T−5 | 100% / 0.00% | 92% / 0.23% | 67% / 0.49% | 41% / 0.69% | 5% / 1.25% |
| League One | MATCH_ODDS | T−75…T−45 | 99% / 0.30% | 75% / 0.84% | 58% / 1.27% | 26% / 1.20% | 3% / 0.74% |
| | | T−20…T−5 | 98% / 0.00% | 85% / 0.00% | 65% / 0.00% | 40% / 0.00% | 15% / 0.18% |
| | OVER_UNDER_25 | T−20…T−5 | 99% / 0.00% | 65% / 0.28% | 44% / 0.75% | 30% / 0.81% | 5% / 0.27% |
| League Two | MATCH_ODDS | T−75…T−45 | 94% / 0.46% | 74% / 0.91% | 55% / 1.25% | 25% / 1.25% | 6% / 1.79% |
| | | T−20…T−5 | 99% / 0.00% | 85% / 0.08% | 59% / 0.90% | 31% / 1.18% | 5% / — |
| | OVER_UNDER_25 | T−20…T−5 | 96% / 0.37% | 65% / 0.57% | 54% / 1.07% | 20% / 2.08% | 6% / — |
| League Two | BTTS | T−20…T−5 | 98% / 0.62% | 87% / 1.36% | 82% / 1.63% | 79% / 1.72% | 44% / 3.01% |

**The per-leg capacity ceiling, at ≥80% fill and ≤1% slippage:**

| competition | 1X2 | O/U central | BTTS |
|---|---|---|---|
| Premiership | £250 | £100 | £50 |
| Championship | £50 | £25 | £100 † |
| League One | £25 | £10 | £100 † |
| League Two | £25 | £10 | £25 |

† Championship / League One BTTS fills at £100 but at a 5–9 tick spread — the fill is real and the
price is not. Cap it by spread, not by depth.

Because only 3 ladder levels are archived, treat these as **floors**. And note the morphism nearly
doubles them: `BestOfBackLay` can take the position on either side of a two-outcome market, so the
accessible size is `back ≤2tk` **plus** the complement's `lay ≤2tk`, not either alone.

### 4.6 Entry timing

For every (market, runner) with snapshots at T−60, T−15 and T−2 (±8 min), in probability points:

| market | n | median \|Δp\| T−60→close | median \|Δp\| T−15→close | half-spread @T−60 | half-spread @T−15 | p95 \|Δp\| T−60 |
|---|---|---|---|---|---|---|
| MATCH_ODDS | 156 | **0.0131** | 0.0027 | 0.0065 | 0.0051 | **0.0636** |
| OVER_UNDER_15 | 92 | 0.0052 | 0.0018 | 0.0083 | 0.0080 | 0.0153 |
| OVER_UNDER_25 | 100 | 0.0067 | 0.0022 | 0.0087 | 0.0086 | 0.0266 |
| OVER_UNDER_35 | 88 | 0.0053 | 0.0024 | 0.0095 | 0.0088 | 0.0203 |
| BOTH_TEAMS_TO_SCORE | 96 | 0.0065 | 0.0023 | 0.0136 | 0.0119 | 0.0343 |

By competition (all markets pooled):

| competition | n | \|Δp\| T−60→close | \|Δp\| T−15→close | half-spread @T−60 | @T−15 |
|---|---|---|---|---|---|
| Championship | 130 | 0.0078 | 0.0024 | 0.0087 | 0.0079 |
| League One | 128 | 0.0101 | 0.0026 | 0.0096 | 0.0084 |
| League Two | 153 | 0.0052 | 0.0019 | 0.0106 | 0.0101 |
| Premiership | 121 | 0.0057 | 0.0024 | 0.0048 | 0.0037 |

**Reading it.** A T−60 quote is on average **1.31 pp** (1X2) away from where the market ends up,
with a p95 of 6.4 pp. Waiting to T−15 cuts that to 0.27 pp — a 5× reduction in residual price
uncertainty — while the half-spread barely improves (0.65 → 0.51 pp on 1X2; essentially flat on
O/U). So the *cost* of waiting is ~0.1 pp of spread and the *benefit* is ~1.0 pp of avoided
disagreement with the closing price.

That only argues for waiting if the drift is **information rather than noise**. It is. Two
independent pieces of evidence:

1. `orderbook_layer2/RESULTS.md` §4.1 measured leg-weighted CLV monotone in entry time on Ireland,
   both leagues separately: AtClose −0.0051 / −0.0072, FixedLead(60m) −0.0118 / −0.0127,
   FixedLead(120m) −0.0139 / −0.0147. Entering two hours out costs ~0.8–0.9 pp of CLV per leg.
   *"There is no interior optimum."*
2. Capacity moves the same way (§4.5): League One 1X2 at £50 fills 58% at T−75 and 65% at T−20,
   with slippage 1.27% → 0.00%. League Two 1X2 at £25: 74% → 85%.

**Every axis points the same direction, so there is no trade-off to balance.** The T−60 story —
"highest model edge against a slow-reacting market" — is not supported by anything measured here,
and the market that is supposed to be slow-reacting has in fact moved 1.3 pp by the time we would
have wanted to bet against it.

**D1, stated operationally.** Price at **T−25** (first instant the XI exists, §3.4). Submit at
**T−12**. Hard floor **T−4** — after that, in-play suspension risk and the Betfair in-play delay
make a pre-match assumption unsafe. Do **not** implement a dynamic threshold-triggered entry that
can fire at T−60: with p95 |Δp| = 6.4 pp at T−60, a threshold of δ = 2 pp fires mostly on stale
quotes. A threshold rule is only meaningful *inside* the T−25…T−4 window, where it degenerates to
"submit as soon as edge and depth both clear", which is what §6.3's `TRIGGERED` transition already
is.

---

## 5. The slate is the unit of execution

This section is the spine of the design. Everything after it — the ledger, the state machine, the
console, the roadmap — is downstream of one commitment: **the atom of this system is a slate, not a
bet.**

### 5.1 What a slate is, and why the joint solve is not optional

A **slate** is the set of fixtures that settle simultaneously — the Saturday 15:00 UTC card, up to
**21 Scottish fixtures** on eleven separate Saturdays this season (§7.1). `Portfolio` groups books
into slates via `AbstractSlateGrouping` (`DailySlate`, `WeeklySlate`, `MatchSlate`) and then solves
`stake_slate(policy, slate, ctx)` **once for the whole group**:

```
                        per match                         SLATE-WIDE
  ┌───────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌────────┐   ┌────────┐
  │  a_kelly      │──►│  trust   │──►│  shrink   │──►│ risk_factor  │──►│  cap   │──►│ filter │
  │ (joint Kelly, │   │ w_j      │   │ k_shrink  │   │ ONE k for    │   │ Σ ≤ c  │   │ remove │
  │  Jacot &      │   │          │   │ (Baker &  │   │ every leg in │   │        │   │ only   │
  │  Mochkovitch) │   │          │   │  McHale)  │   │ the slate    │   │        │   │        │
  └───────────────┘   └──────────┘   └───────────┘   └──────────────┘   └────────┘   └────────┘
                                                            ▲                ▲
                                              SlateDrawdown(λ)        FixedCap(c)
```

The last three stages **couple every leg to every other leg**:

* `SlateDrawdown(λ)` solves `Σ_t log E[(1 + k R_t)^{-λ}] ≤ 0` across all `L` matches at once and
  returns **one scalar `k`** applied to every stake in the slate. Adding a 22nd fixture lowers `k`
  for all 21 others.
* `FixedCap(c)` rescales the whole vector by `c / Σ stakes` the moment total simultaneous exposure
  exceeds `c`. It is **mandatory by construction** — there is deliberately no `NoCap`, because the
  prototype without one lost 129.5% of bankroll on its worst slate and flipped the sign of every
  subsequent compounding step. `0 < c < 1` is enforced in the constructor, which is what makes
  `slate_pl > −1` a theorem rather than an assertion.

**Therefore a stake vector is only valid as a vector.** Take 15 of the 21 legs and the drawdown
budget the other 6 were funding is unspent; take them in sequence and `k` was solved for a portfolio
that never existed. This is not a performance argument about batching — it is a correctness
argument. *Partial execution of a slate produces a position the allocator never authorised.*

### 5.2 The parameters, stated precisely

The shorthand `SlateDrawdown(20.0)` / `FixedCap(0.25)` maps onto the code as:

| written | means | note |
|---|---|---|
| `SlateDrawdown(20.0)` | `λ = 20.0`, `mode = :sequential` | **λ is not a percentage.** `risk_lambda(D, β) = log β / log D`; the default `λ = 23` targets a *real* 20% drawdown, because measured realised drawdown overshoots the nominal by a stable ~1.15×. `λ = 20` is therefore slightly *more* aggressive than a 20% floor — deliberate is fine, accidental is not. |
| `FixedCap(0.25)` | at most **25% of bankroll** at risk across the whole simultaneous slate | On a £2,400 account: £600 of liability across up to 21 fixtures ≈ £29/leg at 21 legs — which lands squarely in the §4.5 capacity band for the lower leagues. The cap and the book agree by accident here; do not rely on that as the account grows. |

One property drives most of the system's behaviour and is worth restating because it is
counter-intuitive: **`risk_factor` is homogeneous of degree 0.** Hand it twice the stakes and it
returns half the factor. So once the drawdown constraint binds, trust and shrinkage can only
*reshape* the slate, never resize it — measured, at `λ = 20` a stake multiplier of 0.25, 1.0 or 4.0
all produce mean slate exposure 0.1088. **To move exposure, move λ, not trust.** A console control
that "scales up the slate" by multiplying stakes is a no-op and must not exist.

### 5.3 The consequence: a batch, not a queue

Because the vector is only valid whole, execution is organised as a **`SlateBatch`** — a
first-class object with its own lifecycle sitting *above* the per-order state machine of §6.3.

```
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                         SLATE BATCH  (the atom)                              │
   │  slate_id · window 2026-09-05 15:00Z · 21 fixtures · 34 legs · Σrisk £186    │
   │  k_risk 0.0412 · exposure 7.7% · capped false · λ 20.0 · cap 0.25            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  PRICED ──► REVIEWED ──► RESERVED ──► SUBMITTING ──► EXECUTED ──► SETTLED    │
   │     │           │            │             │             │                   │
   │     └───────────┴────────────┴─────────────┴─────► ABANDONED / KILLED        │
   └──────────────────────────────────────────────────────────────────────────────┘
              contains N paper_orders, each running its own TRIGGERED..SETTLED
```

| batch state | meaning | invariant |
|---|---|---|
| `PRICED` | `stake_sheet` returned a vector; orders written `TRIGGERED`; **no money moved** | `Σ risk ≤ bankroll × cap` asserted at write time |
| `REVIEWED` | an operator has seen `slate_summary` and the blocked report | human gate; skippable only in an explicitly `unattended` account |
| `RESERVED` | **one transaction** took `Σ risk` out of `balance` into `reserved` for every leg at once | `reserved` moved exactly once; `account_ledger` has exactly one `RESERVE` row for the batch |
| `SUBMITTING` | submitters are working, sharded by `bf_market_id` | no account row is touched |
| `EXECUTED` | every order is terminal (`MATCHED` / `PARTIALLY_MATCHED` / `CANCELLED` / `REJECTED`); unfilled remainders released | `reserved(batch) == Σ risk_filled` |
| `SETTLED` | every fixture graded, PnL booked | `reserved(batch) == 0` |
| `KILLED` | operator abort — every non-terminal order `CANCELLED`, full release | `reserved(batch) == 0`, `Σ risk_filled == 0` |
| `ABANDONED` | a pre-reservation gate failed (stale book, exposure assert, crosswalk gap) | never reserved; every order `CANCELLED` with a reason |

**The atomic action is `RESERVED`, and only that.** The reservation of the entire slate is one
`SELECT … FOR UPDATE` transaction (§7.2). Submission afterwards is *not* atomic and cannot be —
the venue accepts orders one market at a time — but by then the allocator's answer has already been
committed in full, so a submission failure is a **fill shortfall on an authorised position**, not an
unauthorised position. That distinction is the entire safety argument.

**Partial fills do not trigger a re-solve.** Releasing the unfilled fraction back to `balance` and
recording it is correct; re-running `stake_slate` on the remainder is not. A re-solve mid-batch
makes the final position depend on fill *order*, which is both wrong and untestable.

### 5.4 What this adds to the schema

Two additions to §6.2, and the reason `paper_slates` already carries `bankroll`, `book_spec_hash`
and `policy_spec_hash`:

```sql
ALTER TABLE paper.paper_slates
  ADD COLUMN batch_status  text NOT NULL DEFAULT 'PRICED'
      CHECK (batch_status IN ('PRICED','REVIEWED','RESERVED','SUBMITTING',
                              'EXECUTED','SETTLED','KILLED','ABANDONED')),
  ADD COLUMN k_risk        numeric(10,6) NOT NULL,   -- the ONE slate-wide drawdown factor
  ADD COLUMN slate_exposure numeric(9,6) NOT NULL,   -- Σ stake / bankroll, pre-cap
  ADD COLUMN capped        boolean       NOT NULL,   -- did FixedCap bind?
  ADD COLUMN risk_lambda   numeric(8,3)  NOT NULL,   -- λ actually used
  ADD COLUMN exposure_cap  numeric(6,4)  NOT NULL,   -- c actually used
  ADD COLUMN total_risk    numeric(14,2) NOT NULL,   -- Σ risk over all legs
  ADD COLUMN reviewed_by   text,
  ADD COLUMN reviewed_at   timestamptz,
  ADD COLUMN reserved_at   timestamptz,
  ADD COLUMN terminal_at   timestamptz;

-- One RESERVE ledger row per batch. This partial unique index makes double-reservation
-- unrepresentable rather than merely guarded against.
CREATE UNIQUE INDEX paper_ledger_one_reserve_per_slate
    ON paper.account_ledger (slate_id) WHERE kind = 'RESERVE';
```

`k_risk`, `slate_exposure` and `capped` come straight off `Portfolio.SlateAllocation` and are
recorded because they are **not recoverable after the fact**: `k` depends on the whole book, and a
later re-price with a different fixture list gives a different `k` for the same legs. Without them,
"why was this leg £26 and not £40?" is unanswerable.

### 5.5 Slate-level metrics

Reported per batch, not per bet:

| metric | definition |
|---|---|
| slate exposure | `Σ risk / bankroll` — asserted `≤ exposure_cap` before `RESERVED` |
| `k_risk` | the drawdown factor actually applied; the honest measure of how hard the budget bound |
| capped | whether `FixedCap` bound. If `true` on most slates, `λ` is set too loose and should be moved, not the cap |
| batch fill rate | `Σ risk_filled / Σ risk` — the fraction of the *authorised* position actually taken |
| slate PnL | `Σ net_pnl` over the batch; the only PnL series with a meaningful drawdown |
| slate CLV | leg-weighted `Σ (risk_filled × clv) / Σ risk_filled` |

**Slate PnL is the series that goes into `BackTesting`.** Per-bet PnL has no drawdown because bets
inside a slate settle together — a per-bet equity curve is an artefact of the order rows happen to
be written in.

---

## 6. Paper trading — schema and state machine

### 6.1 Where it lives, and why

**In `betdb`, schema `paper`.** Three reasons.

1. **Availability.** The ledger is written at T−12 on a Saturday. `mcmc-beast` may be saturated or
   restarting from a training run; `betdb` is the operational database and is already the one the
   supervisor, the TUI and the collector all depend on being up.
2. **Locality.** Mark-to-market and CLV both need `order_book_1m`, which is in `betdb`. Putting the
   ledger there makes settlement and CLV a join instead of a cross-database transfer.
3. **Separation of concerns.** `portfolio_runs` / `portfolio_bets` in `mcmc_experiments` are the
   *backtest* ledger: one row per simulated bet, no lifecycle, no time, no fills. Paper trading has
   a lifecycle. Overloading one table with both would make "what did the backtest say" and "what
   did we actually do" the same query, which is the one distinction the whole exercise exists to
   preserve.

The link between them is `paper_slates.model_run_id → mcmc_experiments.runs.run_id`, carried as an
opaque UUID. No foreign key across databases; a nightly reconciliation job asserts it resolves.

### 6.2 Schema

```sql
CREATE SCHEMA IF NOT EXISTS paper;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Accounts — the bankroll of record. ONE row per strategy. See §7.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.paper_accounts (
    account_id        text PRIMARY KEY,             -- 'scottish_smile_v1'
    currency          char(3)     NOT NULL DEFAULT 'GBP',
    opening_balance   numeric(14,2) NOT NULL,
    balance           numeric(14,2) NOT NULL,       -- settled cash
    reserved          numeric(14,2) NOT NULL DEFAULT 0,  -- liability of live orders
    commission_rate   numeric(6,4)  NOT NULL DEFAULT 0.02,
    max_slate_exposure numeric(6,4) NOT NULL DEFAULT 0.10,  -- fraction of equity
    is_live           boolean     NOT NULL DEFAULT false,   -- false = paper. NEVER default true.
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT balance_nonneg  CHECK (balance  >= 0),
    CONSTRAINT reserved_nonneg CHECK (reserved >= 0)
);
-- equity = balance + reserved ; free = balance - (reserved is already out of balance? no):
--   INVARIANT: balance is CASH NOT COMMITTED. reserved is CASH COMMITTED TO OPEN RISK.
--   equity := balance + reserved + unrealised_mtm

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Slates — one pricing run. The idempotency anchor for the whole system.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.paper_slates (
    slate_id       uuid PRIMARY KEY,
    account_id     text        NOT NULL REFERENCES paper.paper_accounts,
    slate_window   date        NOT NULL,            -- Portfolio's settlement window
    as_of          timestamptz NOT NULL,            -- the instant priced. NEVER now().
    model_run_id   uuid        NOT NULL,            -- mcmc_experiments.runs.run_id
    fold_idx       int         NOT NULL,            -- what select_split chose
    book_spec_hash text        NOT NULL,            -- Portfolio.portfolio_spec_hash
    policy_spec_hash text      NOT NULL,
    bankroll       numeric(14,2) NOT NULL,          -- equity at pricing time
    git_commit     text        NOT NULL,
    n_fixtures     int         NOT NULL,
    n_blocked      int         NOT NULL,
    blocked_report jsonb       NOT NULL DEFAULT '[]'::jsonb,  -- MatchDay.blocked_report
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (account_id, slate_window, as_of)        -- ← re-running the same instant is a no-op
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Orders — one row per leg. The state machine lives in `status`.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TYPE paper.order_status AS ENUM (
    'TRIGGERED', 'PENDING_SUBMISSION', 'SUBMITTED',
    'PARTIALLY_MATCHED', 'MATCHED', 'CANCELLED', 'REJECTED',
    'VOIDED', 'SETTLED'
);

CREATE TABLE paper.paper_orders (
    order_id        uuid PRIMARY KEY,
    slate_id        uuid NOT NULL REFERENCES paper.paper_slates ON DELETE CASCADE,
    account_id      text NOT NULL REFERENCES paper.paper_accounts,

    -- identity
    match_id        int  NOT NULL,                  -- sofascore.events.match_id
    bf_event_id     text NOT NULL,
    bf_market_id    text NOT NULL,
    kickoff         timestamptz NOT NULL,

    -- the MODEL's position (SelectionKey) — what we grade against
    market_group    text NOT NULL,                  -- '1X2' | 'OverUnder' | 'BTTS'
    market_line     numeric(4,1) NOT NULL,
    selection       text NOT NULL,                  -- 'home' | 'over_25' | 'btts_yes' | …

    -- the VENUE's instrument — what the order touches. NOT the same on a synthetic.
    venue_selection text NOT NULL,
    side            text NOT NULL CHECK (side IN ('back','lay')),
    venue_odds      numeric(8,3) NOT NULL,
    leverage        numeric(10,4) NOT NULL,         -- 1.0 back; 1/(d-1) lay

    -- sizing, all in RISK units (Portfolio's denomination)
    effective_odds  numeric(10,4) NOT NULL,         -- Instrument.odds  (= d/(d-1) for a lay)
    p_model         numeric(9,6) NOT NULL,
    p_market        numeric(9,6) NOT NULL,
    edge            numeric(9,6) NOT NULL,
    stake_fraction  numeric(9,6) NOT NULL,
    risk            numeric(12,2) NOT NULL,         -- liability. THIS is what `reserved` holds.
    venue_stake     numeric(12,2) NOT NULL,         -- risk * leverage — what is placed

    -- book at decision time (the CLV baseline and the fill model's input)
    quote_ts        timestamptz NOT NULL,
    book_snapshot   jsonb NOT NULL,                 -- 3 levels both sides, unscaled

    status          paper.order_status NOT NULL DEFAULT 'TRIGGERED',
    status_reason   text,
    submitted_at    timestamptz,
    terminal_at     timestamptz,

    -- idempotency: one leg per (slate, match, market, selection). Re-running cannot double up.
    UNIQUE (slate_id, match_id, market_group, market_line, selection)
);
CREATE INDEX ON paper.paper_orders (account_id, status);
CREATE INDEX ON paper.paper_orders (kickoff) WHERE status <> 'SETTLED';

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Fills — append-only. A partial fill is N rows, never an UPDATE.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.paper_fills (
    fill_id      bigserial PRIMARY KEY,
    order_id     uuid NOT NULL REFERENCES paper.paper_orders ON DELETE CASCADE,
    filled_at    timestamptz NOT NULL,
    price        numeric(8,3) NOT NULL,             -- the price actually taken at the venue
    size         numeric(12,2) NOT NULL,            -- venue stake filled
    risk_filled  numeric(12,2) NOT NULL,            -- size / leverage
    fill_model   text NOT NULL,                     -- 'ladder_sweep_v1' | 'touch_only' | 'live'
    level_depth  int,                               -- how many ladder levels were consumed
    CONSTRAINT size_pos CHECK (size > 0)
);
CREATE INDEX ON paper.paper_fills (order_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Market snapshots — the mark-to-market and CLV series, one row per (order, ts).
--    Deliberately denormalised from order_book_1m so a ledger read is one table.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.market_snapshots (
    order_id     uuid NOT NULL REFERENCES paper.paper_orders ON DELETE CASCADE,
    ts           timestamptz NOT NULL,
    best_back    numeric(8,3),
    best_lay     numeric(8,3),
    back_size    numeric(12,2),
    lay_size     numeric(12,2),
    mid_prob     numeric(9,6),                      -- (1/back + 1/lay)/2, the fair mark
    market_matched numeric(14,2),
    mtm_pnl      numeric(12,2),                     -- see §6.4
    PRIMARY KEY (order_id, ts)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. CLV audit — one row per order, written once at settlement.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.clv_audit (
    order_id        uuid PRIMARY KEY REFERENCES paper.paper_orders ON DELETE CASCADE,
    entry_prob      numeric(9,6) NOT NULL,          -- 1 / effective_odds at fill
    close_prob      numeric(9,6) NOT NULL,          -- de-vigged mid at the last pre-off snapshot
    close_ts        timestamptz NOT NULL,
    close_source    text NOT NULL,                  -- 'order_book_1m_mid' | 'last_price_traded'
    clv             numeric(9,6) NOT NULL,          -- close_prob - entry_prob, in prob points
    clv_pct         numeric(9,6) NOT NULL,          -- price-relative
    beat_close      boolean NOT NULL,
    entry_lead_min  int NOT NULL                    -- minutes before kick-off the fill happened
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 7. Settlements — the grading, and the only writer of paper_accounts.balance.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.paper_settlements (
    order_id      uuid PRIMARY KEY REFERENCES paper.paper_orders ON DELETE CASCADE,
    settled_at    timestamptz NOT NULL DEFAULT now(),
    result_source text NOT NULL,                    -- 'sofascore.events' | 'manual'
    home_goals    int,
    away_goals    int,
    outcome       text NOT NULL CHECK (outcome IN ('WIN','LOSE','VOID','HALF_WIN','HALF_LOSE')),
    gross_return  numeric(12,2) NOT NULL,
    commission    numeric(12,2) NOT NULL,
    net_pnl       numeric(12,2) NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 8. Ledger — append-only double entry against the account. Audit trail of §7.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE paper.account_ledger (
    entry_id   bigserial PRIMARY KEY,
    account_id text NOT NULL REFERENCES paper.paper_accounts,
    at         timestamptz NOT NULL DEFAULT now(),
    kind       text NOT NULL,   -- 'RESERVE'|'RELEASE'|'SETTLE'|'COMMISSION'|'DEPOSIT'|'ADJUST'
    order_id   uuid REFERENCES paper.paper_orders,
    slate_id   uuid REFERENCES paper.paper_slates,
    delta_balance  numeric(12,2) NOT NULL,
    delta_reserved numeric(12,2) NOT NULL,
    balance_after  numeric(14,2) NOT NULL,
    reserved_after numeric(14,2) NOT NULL,
    note       text
);
CREATE INDEX ON paper.account_ledger (account_id, at DESC);
```

**Idempotency.** Three mechanisms, and they are the reason this can be re-run after a crash:

* `paper_slates (account_id, slate_window, as_of)` is UNIQUE. Re-pricing the same instant returns
  the existing slate.
* `paper_orders (slate_id, match_id, market_group, market_line, selection)` is UNIQUE. Re-inserting
  a leg is `ON CONFLICT DO NOTHING`.
* `paper_fills` and `account_ledger` are append-only, and `paper_accounts` is only ever moved by an
  `account_ledger` insert in the same transaction. So the balance is reconstructible:
  `SELECT sum(delta_balance) FROM account_ledger WHERE account_id = ?` must equal
  `balance - opening_balance`. Assert that in a nightly job; a mismatch is a bug, not a rounding
  error.

### 6.3 The state machine

```
                    ┌──────────────┐
   stake_sheet row  │  TRIGGERED   │  a leg the allocator sized > 0
                    └──────┬───────┘
                           │  gate: readiness OK, edge ≥ δ, depth ≥ venue_stake,
                           │        spread ≤ σ, account has free equity
                           │
                  ┌────────┴────────┐
                  │                 │  refused: no reservation taken
                  ▼                 ▼
        ┌───────────────────┐   ┌───────────┐
        │PENDING_SUBMISSION │   │ CANCELLED │ (terminal; status_reason says which gate)
        └────────┬──────────┘   └───────────┘
                 │  RESERVE risk against paper_accounts (§7) — the only bankroll write
                 ▼
          ┌─────────────┐
          │  SUBMITTED  │  venue accepted (paper: always). submitted_at stamped.
          └──┬───┬───┬──┘
             │   │   └──────────────► REJECTED   (venue refused; RELEASE full reservation)
             │   │
             │   └────► ┌────────────────────┐  fill_size < venue_stake at expiry
             │          │ PARTIALLY_MATCHED  │──► RELEASE the unfilled remainder
             │          └─────────┬──────────┘
             │                    │
             ▼                    ▼
          ┌─────────┐        (both are "has exposure")
          │ MATCHED │◄────────────┘
          └────┬────┘
               │  kick-off passes; result lands in sofascore.events
               ▼
          ┌─────────┐        ┌────────┐
          │ SETTLED │        │ VOIDED │  market voided / fixture abandoned
          └─────────┘        └────────┘   RELEASE reservation, no PnL
```

Transitions are the **only** thing that moves money, and each is one transaction:

| transition | account effect |
|---|---|
| `PENDING_SUBMISSION` | `RESERVE`: `balance −= risk`, `reserved += risk` |
| `→ CANCELLED` / `REJECTED` | `RELEASE`: `balance += risk`, `reserved −= risk` |
| `→ PARTIALLY_MATCHED` | `RELEASE` the unfilled fraction only |
| `→ SETTLED` | `SETTLE`: `reserved −= risk_filled`, `balance += gross_return − commission` |
| `→ VOIDED` | `RELEASE` `risk_filled` |

**Copy `matchday_supervisor.decide()`.** Make the transition function pure —
`decide(order, book, account, now) -> (new_status, reason, ledger_delta)` — with all I/O in a shell
around it. That is what makes the 15:00 slate replayable as a table of numbers rather than only on
a real Saturday, and it is exactly the split the Python supervisor already proved out.

**Every refusal is a row.** `CANCELLED` with `status_reason` is written even though no money moved.
`MatchDay` already treats a refusal as a value (`MatchDayResult.blocked`, `blocked_report`); the
ledger must too, or "no bets today" and "the pipeline is broken" become the same empty table again.

### 6.4 The fill model

Paper trading is only worth anything if the fill model is pessimistic in the right places. Three,
selectable per slate and recorded in `paper_fills.fill_model`:

| model | rule | use |
|---|---|---|
| `touch_only` | fill `min(venue_stake, size_at_best)` at the best price; the rest expires | the honest default |
| `ladder_sweep_v1` | sweep the archived 3 levels, VWAP; unfilled remainder expires at T−0 | matches §4.5, conservative because only 3 levels are stored |
| `optimistic` | fill in full at the touch | **research only.** Never for a paper track that will be compared to live. |

Default `touch_only`. Rationale: `ladder_sweep_v1` assumes we cross three levels instantly, which
is true of a market order and not of what we would actually do (rest at the touch and wait). The
difference is exactly the £25→£100 column of §4.5, which is the whole capacity question — so it
should be a decision, not a default.

**Mark to market.** `mtm_pnl` at snapshot `ts` = `risk_filled × (p_close_now / p_entry − 1)` where
`p_* = 1 / effective_odds`, i.e. the position is marked against the price at which it could be
closed out **on the opposite side** — for a back, the current best lay; for a lay, the current best
back. Marking against the same side you entered flatters every position by the full spread.

---

## 7. Concurrent order placement without bankroll races

### 7.1 The size of the problem, measured

`sofascore.events`, 2026-09-05: **36 fixtures kick off simultaneously at 14:00 UTC** across six
tournaments (Scottish 54/55/56/57 + English 3/84). Scottish alone peaks at **21 fixtures at
15:00 UTC** on eleven separate Saturdays between October and January.

21 fixtures × 17 default selections = 357 candidate legs. The runbook's Ireland example produced 23
bets from 5 fixtures, so a full Scottish card is plausibly **80–120 orders inside a 10-minute
window**, all against one bankroll.

### 7.2 The rule: one writer, one reservation, one transaction

**Do not parallelise the bankroll.** Portfolio already solves the whole slate jointly —
`SlateDrawdown` and `FixedCap` are slate-level constraints, so the allocation is a single convex
problem whose answer is only valid if every leg is taken. Splitting it across concurrent workers
that each reserve independently does not just risk a race; it **changes the answer**.

```
                    ┌─────────────────────────────────────────┐
   T−25  price ────►│  ONE pricing run over the whole slate    │
                    │  Portfolio.stake_sheet(sys, …, bankroll) │
                    └────────────────┬────────────────────────┘
                                     │  sheet: 80–120 rows, one slate
                    ┌────────────────▼────────────────────────┐
   T−12  reserve ──►│  ONE transaction:                        │
                    │    SELECT … FROM paper_accounts          │
                    │      WHERE account_id = $1 FOR UPDATE;   │
                    │    assert Σ risk ≤ balance × max_slate_  │
                    │           exposure;                      │
                    │    INSERT paper_orders (all legs);       │
                    │    INSERT account_ledger (one RESERVE);  │
                    │    UPDATE paper_accounts;                │
                    │  COMMIT                                  │
                    └────────────────┬────────────────────────┘
                                     │  reservation is now atomic and total
                    ┌────────────────▼────────────────────────┐
   T−12  submit ───►│  N concurrent submitters (per MARKET)    │
                    │  — touch NO account row                  │
                    │  — write paper_orders.status + fills     │
                    │  — bounded by a semaphore (rate limit)   │
                    └─────────────────────────────────────────┘
```

The bankroll is committed **once, before any order is submitted**, for the whole slate. After that
the submitters are embarrassingly parallel because they only ever touch rows they own. There is no
lock contention because there is nothing left to contend over.

Partial fills release back into `balance` and **do not** trigger a re-solve within the same slate.
Re-solving on each partial fill turns one convex problem into a path-dependent sequence whose
outcome depends on fill order — which is both wrong and untestable. Release, record, move on.

### 7.3 Submitter concurrency and rate limits

Shard by `bf_market_id`, not by fixture: Betfair's `placeOrders` takes **one market per request**
with multiple instructions, so a market is the natural batch and two orders in the same market must
never race. A bounded worker pool (start at 4) with per-request retry, plus a global token bucket,
covers it.

The exact Betfair limits (instructions per request, transactions per hour, market subscription cap
per connection) are **not verified here** — see §13. Make them config, not constants, and have the
adapter surface a `429`/`TOO_MANY_REQUESTS` as a first-class `REJECTED` reason rather than a retry
loop. The subscription cap in particular is documented by `betdb_matchday_status` as failing
*silently*, which is precisely the class of failure this system exists to make loud.

### 7.4 Crash safety

The recovery query is:

```sql
SELECT * FROM paper.paper_orders
 WHERE account_id = $1 AND status IN ('PENDING_SUBMISSION','SUBMITTED','PARTIALLY_MATCHED')
   AND kickoff > now();
```

Anything in `PENDING_SUBMISSION` at startup was reserved but never submitted: submit it if we are
still inside the window, `CANCELLED` + `RELEASE` if not. Anything `SUBMITTED` needs a venue
reconciliation (in paper mode: apply the fill model against the book at the recorded
`submitted_at`). The reservation is already durable, so no money is ever lost or double-counted
across a crash — that is the entire reason `RESERVE` precedes submission.

---

## 8. Operator console — TUI vs web

### 8.1 What the console has to do

| # | job | when |
|---|---|---|
| 1 | Is capture armed, is the crosswalk populated, is the XI in? | T−90 … T−30 |
| 2 | Show **the whole slate as one object**: 21 fixtures, ~100 legs, `k_risk`, exposure against cap, `capped` | T−25 |
| 3 | Per leg, show **model vs market** — fair probability and fair odds side by side — plus EV% | T−25 |
| 4 | Show *why* a fixture was refused (`blocked_report`) | T−25 |
| 5 | **Execute Slate Batch** — one atomic action reserving the entire vector; plus kill-leg and kill-all | T−12 |
| 6 | Watch fills land, exposure build, reservation drain | T−12 … T−0 |
| 7 | Watch mark-to-market and CLV during and after the match | T−0 … T+120 |

Jobs 1 and 4 are text. Jobs 2, 3 and 5 are **the reason this needs a screen**: the operator has to
judge a 21-card vector *as a vector* and then commit it in one action, which means seeing every card
at once with a comparable model-vs-market read on each. Jobs 6 and 7 want a **ladder** and a **time
series**. That split is the whole decision.

### 8.2 The comparison

| | **A. Terminal UI** | **B. Lightweight web** |
|---|---|---|
| Stack | Julia `Term.jl`, or extend the existing Python Textual app | HTTP.jl + WebSockets.jl + HTML/Tailwind/Alpine |
| Reuse | **Textual app already exists** — `screens/live/*`, `widgets/status_card.py` (527 lines), card-grid dashboard, `LiveTournamentCard` | nothing exists |
| Latency | keystroke-local | one WS hop on a LAN — sub-ms, not a real difference |
| Deployment | tmux over SSH; already how the collector is run | a browser tab; still one process on `archpc` |
| 21-card grid | works, but a terminal is ~50 rows: 21 cards × 4 lines = 84. **Requires scrolling exactly when you need to see everything at once** | 21 cards in a responsive grid on one 1440p screen, no scroll |
| Price ladder | ASCII, ~3 levels legible | 3–10 levels, colour-graded by size, trivially |
| Sparkline / MTM | `widgets/sparkline.py` exists — coarse | inline SVG, exact |
| Sort by EV, live | re-render the whole table | Alpine `x-for` over a sorted array; one line |
| Manual override | keybinding — fast, and unforgiving | click — slower, and confirmable |
| Cost to build | **~0 for jobs 1/3/4** (extend Textual); high for 2/5/6 | ~600 lines total, all six jobs |
| Language boundary | the Textual app is Python; the stake sheet is Julia | Julia serves its own state — no boundary |

### 8.3 Recommendation: **both, split by job — and the split is already natural**

* **Jobs 1, 3, 4 (collector health, refusals, arm/disarm) stay in the existing Python Textual TUI.**
  It already renders exactly this — `betdb_matchday_status`'s verdict ladder
  (`no_fixtures → not_armed → stale_subscription → drain_off → not_landing → unlinked → ready`) is
  the single best thing in the whole stack and it would be strictly worse re-implemented in Julia.
  Add one screen: `screens/paper/slate.py` reading `paper.paper_slates` + `paper_orders`.
* **Jobs 2, 5, 6 (the trading console) become a Julia-served web dashboard.** This is where a
  terminal stops paying: a 21-card grid that must be visible at once, ladders, and a live MTM
  series.

The two never fight, because they own different tables and different decisions. And critically the
web dashboard needs **no new state store** — it renders `paper.*`, which the Julia process is
already writing.

### 8.4 The web console — Gödel-Terminal-style modular cards, cheaply

The target aesthetic is Gödel Terminal's: a dense, dark, monospaced **grid of self-contained
tiles**, each a small window with a title bar, terminal-flat borders, and numbers that update in
place. What makes that look work is not the framework — it is (a) one card component repeated, (b)
a strict type scale with tabular numerals, (c) a two-colour semantic palette on a near-black
ground, and (d) *nothing moving except the numbers*. None of that requires React.

**Stack — four dependencies, no build step:**

```
HTTP.jl            serve one static HTML file + one JSON endpoint
HTTP.WebSockets    push a delta every 1 s (the book is 1-minute data; 1 Hz is generous)
Alpine.js  (15 kB) x-data / x-for / x-text — reactivity for a page with one list
Tailwind CDN       utility classes; no PostCSS, no bundler, no node_modules
```

HTMX is deliberately **not** used. HTMX swaps server-rendered fragments, which is excellent for
forms and wrong for a 1 Hz stream of 100 numbers: it would re-render DOM subtrees every tick and
lose focus, scroll and selection state. Alpine mutates a JS array in place and lets the browser
diff the text nodes. One WebSocket, one array, `x-for` — the entire client is a single `<script>`
block.

**Server, ~150 lines of Julia:**

```julia
# The dashboard is a VIEW. It owns no state; it renders `paper.*`.
struct ConsoleState
    account_id::String
    slate_id::UUID
    clients::Vector{HTTP.WebSocket}
    lock::ReentrantLock
end

# one query, one struct, one JSON payload — no ORM, no serialisation layer
snapshot(st) = (
    account = account_row(st.account_id),          # balance, reserved, equity
    batch   = batch_row(st.slate_id),              # batch_status, k_risk, slate_exposure,
                                                   # capped, risk_lambda, exposure_cap,
                                                   # total_risk, n_fixtures, n_legs, n_blocked
    cards   = [fixture_card(r) for r in fixture_rows(st.slate_id)],
                                                   # per leg: p_model, p_market, edge,
                                                   # fair_odds = 1/p_model, venue_odds, side,
                                                   # risk, fill_confidence, status
    blocked = blocked_rows(st.slate_id),
    at      = now(),
)

# 1 Hz push. Delta, not full state, once cards exceed ~30.
@async while true
    payload = JSON3.write(snapshot(st))
    lock(st.lock) do
        foreach(ws -> trysend(ws, payload), st.clients)
    end
    sleep(1)
end
```

**Wireframe — the slate batch console (jobs 2, 3, 5):**

The header is not decoration. It is the **batch**, and it carries the four numbers that decide
whether the vector is safe to commit: `k_risk`, exposure against cap, whether `FixedCap` bound, and
how many legs the allocator authorised. `Execute Slate Batch` is one button because §5.3 says the
reservation is one transaction.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ SLATE 2026-09-05 15:00Z · SCOTTISH 54/55/56/57       as_of 14:35:00Z   run m05_joint · fold 38│
│───────────────────────────────────────────────────────────────────────────────────────────────│
│ equity £2,410.55   Σrisk £186.00   exposure ▉▉▉▉▉▉▉░░░░░░░░░░░░░░░░░░░  7.7% / 25.0%  cap ok  │
│ k_risk 0.0412   λ 20.0   capped ✗   fixtures 21   legs 34   blocked 0   PRICED                │
│ book 41s · XI 21/21 · crosswalk 21/21                                                         │
│                                                                                               │
│           ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                    │
│           ┃  ⏎  EXECUTE SLATE BATCH      ┃   [R]eview  [K]ill slate  [P]ause auto-sort        │
│           ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   reserves £186.00 across 34 legs in ONE tx        │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│  sort ▸ ●EV%  ○edge  ○risk  ○capacity  ○kickoff    filter ▸ ☑1X2 ☑O/U ☑BTTS  ☐blocked         │
├───────────────────────────┬───────────────────────────┬───────────────────────────────────────┤
│ ┌───────────────────────┐ │ ┌───────────────────────┐ │ ┌───────────────────────┐             │
│ │ ALLOA v MONTROSE   L1 │ │ │ FORFAR v STIRLING  L2 │ │ │ CLYDE v ANNAN      L2 │             │
│ │ 15:00Z · T−25         │ │ │ 15:00Z · T−25         │ │ │ 15:00Z · T−25         │             │
│ ├───────────────────────┤ │ ├───────────────────────┤ │ ├───────────────────────┤             │
│ │ EV +4.81%      ▲      │ │ │ EV +3.90%      ▲      │ │ │ EV +2.11%             │             │
│ │ risk £26.00  2 legs   │ │ │ risk £18.00  2 legs   │ │ │ risk £12.00  1 leg    │             │
│ ├───────────────────────┤ │ ├───────────────────────┤ │ ├───────────────────────┤             │
│ │ O2.5   back  £14 ●●○  │ │ │ home   back  £10 ●●●  │ │ │ btts   lay   £12 ●○○  │             │
│ │  model ▇▇▇▇▇▇▇▇▇░ .509│ │ │  model ▇▇▇▇▇▇▇░░░ .435│ │ │  model ▇▇▇▇▇▇▇▇░░ .531│             │
│ │  mkt   ▇▇▇▇▇▇▇▇░░ .485│ │ │  mkt   ▇▇▇▇▇▇░░░░ .427│ │ │  mkt   ▇▇▇▇▇▇▇▇░░ .521│             │
│ │  fair 1.96  bk 2.06 ▲ │ │ │  fair 2.30  bk 2.34 ▲ │ │ │  fair 1.88  ly 1.92   │             │
│ │  edge +2.4pp  +4.4%   │ │ │  edge +0.8pp  +5.1%   │ │ │  edge +1.0pp  +2.1%   │             │
│ ├───────────────────────┤ │ ├───────────────────────┤ │ ├───────────────────────┤             │
│ │ home   lay   £12 ●●○  │ │ │ O3.5   back   £8 ●●○  │ │ │                       │             │
│ │  model ▇▇▇▇░░░░░░ .375│ │ │  model ▇▇▇░░░░░░░ .303│ │ │                       │             │
│ │  mkt   ▇▇▇░░░░░░░ .347│ │ │  mkt   ▇▇▇░░░░░░░ .294│ │ │                       │             │
│ │  fair 2.67  ly 2.88 ▲ │ │ │  fair 3.30  bk 3.40 ▲ │ │ │                       │             │
│ │  edge +2.8pp  +5.2%   │ │ │  edge +0.9pp  +2.8%   │ │ │                       │             │
│ ├───────────────────────┤ │ ├───────────────────────┤ │ ├───────────────────────┤             │
│ │ XI ✓29m   book 38s    │ │ │ XI ✓31m   book 38s    │ │ │ XI ✓26m   book 41s    │             │
│ └───────────────────────┘ │ └───────────────────────┘ │ └───────────────────────┘             │
└───────────────────────────┴───────────────────────────┴───────────────────────────────────────┘
   cards auto-sort by EV% · ●●○ = fill confidence at this size (§4.5) · ▲ = still above threshold
   model/mkt bars share ONE scale so the gap between them IS the edge, read directly off the page
```

**The model-vs-market bar pair is the centrepiece of the card**, and the reason it works is that
both bars are drawn on **one shared 0–1 probability scale**, stacked and left-aligned. The visible
overhang of the model bar past the market bar *is* the edge — no arithmetic, no colour-coding
convention to learn, and 21 cards' worth of it comparable at a glance. Underneath, the same
comparison in odds: `fair` = `1 / p_model` (the model's fair price) against the actual `bk`/`ly`
price we would take. Two representations of one quantity, because the probability form is what the
model thinks and the odds form is what gets typed into an exchange.

Card anatomy, five fixed zones — this is what makes 21 of them read as one instrument rather than
21 unrelated widgets:

```
  ① title      fixture, league, kickoff, T−minus
  ② headline   EV% (the sort key) + total risk + leg count   ← the only large type on the card
  ③ legs       per leg: selection · side · stake · fill dots
                         model bar   (p_model,  shared scale)
                         market bar  (p_market, shared scale)
                         fair odds → venue odds, edge in pp and %
  ④ health     XI age, book age                          ← greys the card when either goes stale
  ⑤ border     batch-derived: PRICED / RESERVED / SUBMITTING / filled / partial / blocked
```

**Fill-confidence dots (`●●○`) are the one novel element and they earn their place.** They render
§4.5's fill curve for *this* leg at *this* size against *this* book: three dots = ≥90% expected
fill, two = 60–90%, one = <60%. This is the number the operator cannot compute in their head and
the one that decides whether a League Two O/U leg is real. It is a direct read of
`book_snapshot` + `venue_stake`, needs no extra state, and is the piece a generic dashboard would
never have.

**Interaction — deliberately minimal, and slate-first:**

* **`Execute Slate Batch` is the only path to a reservation.** It posts one intent; the Julia
  process runs the §7.2 transaction and the batch moves `REVIEWED → RESERVED` or fails whole. The
  button is disabled while any card shows a stale book or an unresolved identity, because those are
  the gates that make the vector invalid rather than merely unattractive.
* **Killing a leg re-prices the slate, and the console says so.** Removing a leg changes `k_risk`
  for every other leg, so a per-leg `CANCEL` before `RESERVED` triggers a re-solve and a visible
  header update. After `RESERVED` it does not: the vector is committed, and a cancel is a release,
  not a re-allocation. The console must make that distinction obvious — it is the single most
  likely place for an operator to form a wrong mental model.
* Cards auto-sort by EV% each tick. A `data-order` transition of ~150 ms makes reordering legible;
  faster and it flickers, slower and it lags the numbers.
* Click a card → a slide-over with the full ladder both sides, the MTM sparkline, and per-leg detail.
* Keyboard, because the operator came from tmux: `⏎` execute batch, `k` kill slate, `/` filter,
  `1..9` jump to card.
* **Nothing in the browser writes to the venue.** The web client posts an *intent* to the Julia
  process, which validates and performs the same transaction §7.2 describes. The browser is not in
  the trust path.

**Theme.** One CSS custom-property block, near-black ground, a single accent for "above threshold"
and a single warning for "stale/blocked". The model bar takes the accent, the market bar a neutral
grey — never two competing hues, or the overhang stops reading as a magnitude. Tabular numerals
(`font-variant-numeric: tabular-nums`) throughout — without it a 1 Hz price update makes every card
jitter horizontally, which is the single most common way a dashboard like this ends up feeling
cheap.

**Why this stays small.** No build step, no bundler, no component library, no client-side router,
no state manager. The server sends one JSON object; the client has one array and one sort
comparator. Total: ~150 lines Julia, ~250 lines HTML/CSS, ~100 lines Alpine. If it ever needs to
grow past that, the thing to add is a second *page*, not a framework.

---

## 9. End-to-end Saturday

```
Tue 09:00  extend_fit(db, :m05_joint_production_wealth, ds)   ──► mcmc_experiments
Tue 11:00  audit_convergence — refuse the week if any fold diverged
Thu        Portfolio walk-forward; PolicySpec sweep; freeze book_spec_hash / policy_spec_hash
Fri 18:00  betdb_crosswalk_rebuild(dry_run=true)   ──► must resolve 100% of Saturday's card
Fri 18:05  dry-run match_day(as_of = last Saturday 14:35) — proves the plumbing end to end
```

```
 SATURDAY, 15:00 UTC card
 ─────────────────────────────────────────────────────────────────────────────────────────
 12:00  supervisor    ARM. stream_worker subscribes; order_book_1m starts landing.
        (execute mode)  ── §1.3: this is the step that has not run since 2026-08-28
 12:05  crosswalk     betdb_crosswalk_rebuild — MUST run while subscribed, before kick-off.
                      There is no retrospective fix.
 13:00  guard         betdb_matchday_status → verdict must be `ready`. Anything else: STOP.
 14:25  lineups       sofascore XI scrape (T−35). confirmed=true lands ~T−29 (§3.4).
 ─────────────────────────────────────────────────────────────────────────────────────────
 14:35  PRICE         MatchDay.match_day(spec, sys, ScottishLower(), expr, ds;
        (T−25)          as_of = 14:35:00, bankroll = equity)
                      ├ fixtures    SofaScoreEvents(36h)         21 fixtures
                      ├ identity    ResolverChain(Crosswalk, LiveNameMatch)
                      ├ lineups     SourceChain(ProvisionalDB, LastHistorical)
                      ├ BOOK        ArchivedOrderBook(max_age = 5min)   ← tightened from 6h
                      ├ features    RatingsFromTracker + LeagueFromFixture + check_coverage
                      ├ inference   select_split(exclude = ids) → extract_parameters → grids
                      ├ gate        IdentityResolved + MaxBookAge(5min)
                      │             + ConfirmedXI(blocking=true)     ← now usable (§3.4)
                      │             + MaxLineupAge(90min, blocking=true)
                      └ stake_sheet Portfolio, ONE slate, ~100 legs
                      ══► INSERT paper_slates  batch_status = PRICED
                            (as_of, k_risk, slate_exposure, capped, λ, cap, total_risk,
                             blocked_report)                                    ← §5.4
                      ══► INSERT paper_orders  status = TRIGGERED  (the whole vector)
 14:36  review        Operator reads the console header FIRST — k_risk, exposure vs cap,
        (batch:       capped — then the cards, then blocked. Portfolio.slate_summary before
        →REVIEWED)    the sheet, always.
 ─────────────────────────────────────────────────────────────────────────────────────────
 14:48  EXECUTE       "Execute Slate Batch" — ONE transaction (§7.2), the atom of the system:
        SLATE BATCH     SELECT … FOR UPDATE; assert Σrisk ≤ bankroll × cap;
        (T−12)          INSERT all orders → PENDING_SUBMISSION; ONE RESERVE ledger row
        (batch:       Fails whole or succeeds whole. After this the allocator's answer is
        →RESERVED)    committed; anything later is a fill shortfall, not a new position.
 14:48  SUBMIT        N submitters sharded by bf_market_id → SUBMITTED
        (→SUBMITTING)   they touch NO account row
 14:49  FILL          fill model against the T−12 book → paper_fills → MATCHED / PARTIAL
        …14:56        RELEASE unfilled remainders. Anything still open at T−4 is CANCELLED.
        (→EXECUTED)   invariant: reserved(batch) == Σ risk_filled
 ─────────────────────────────────────────────────────────────────────────────────────────
 15:00  CLOSE         last pre-off snapshot → clv_audit (close_prob, clv, entry_lead_min)
 15:00  mark          market_snapshots every 60 s while in-play → mtm_pnl
 16:50  RESULT        sofascore.events status_type = 'finished'
 16:55  SETTLE        grade → paper_settlements → SETTLE ledger entry → paper_accounts
        (→SETTLED)    invariant: reserved(batch) == 0
 17:00  reconcile     assert Σ account_ledger.delta_balance == balance − opening_balance
                      assert Σ reserved over open orders == paper_accounts.reserved
```

**Note what moved.** `MaxBookAge` drops from 30 min (default) to **5 min**: at T−25 with a live
1-minute drain, a 5-minute-old book means the drain has stalled, and §4.3 shows that is the single
most likely thing to be wrong. And `ConfirmedXI(blocking = true)` is switched **on**, which §3.4
now permits and which is the only thing standing between a live price and a lineup guess.

---

## 10. Metrics the console and the nightly job compute

| metric | definition | where |
|---|---|---|
| equity | `balance + reserved + Σ mtm_pnl(open)` | live |
| slate exposure | `Σ risk(open) / equity` — must respect `max_slate_exposure` | live, pre-arm assert |
| fill rate | `Σ risk_filled / Σ risk` per slate, per competition, per market | nightly |
| realised slippage | `venue_odds(intended) − price(filled)`, in ticks and % | per fill |
| CLV | `close_prob − entry_prob` in probability points, leg-weighted | at close |
| CLV hit rate | `mean(beat_close)` | nightly |
| slate drawdown | peak-to-trough of `equity` across the slate window | `BackTesting` metrics |
| hurdle ROI | `AbstractDistributionalMetric` — already implemented | nightly |

**CLV is the primary metric for the first N slates, not ROI.** §4.6 and
`orderbook_layer2/RESULTS.md` both say so: with ~100 legs a slate, ROI intervals span ±40 pp and
the ordering of two policies is a coin toss, while CLV has the power to separate them. Judge the
execution layer on CLV. Judge the model on log score. Do not mix them.

---

## 11. Threats to validity

| # | threat | mitigation |
|---|---|---|
| T1 | **Paper fills are optimistic.** No paper model reproduces queue position or the fact that a resting order moves the book. | `touch_only` default; report the `optimistic` delta alongside so the assumption is visible |
| T2 | **Only 3 ladder levels archived.** Capacity beyond ~£50/leg in the lower leagues is extrapolation. | Capture 10 levels going forward; until then, cap per-leg size at the 3-level number |
| T3 | **Small n.** §4.6's timing table rests on 88–156 runner-series from ~5 match rounds. | Re-run monthly; the corpus is a query, not a fixed list |
| T4 | **Selection effect.** Only 54 of 93 Scottish fixtures had a T−60 book; the collector may have covered the *liquid* ones. | Compare `market_matched` of covered vs uncovered once coverage improves |
| T5 | **The model has no significant edge yet.** `Portfolio`'s own health warning: default policy ROI's bootstrap interval includes zero on 628 ScottishLower matches. | Paper trading is being built to *measure* this, not because it is settled |
| T6 | **`last_price_traded` regime change on 2026-08-07** means pre- and post-August CLV baselines are different quantities. | Record `clv_audit.close_source`; never pool the two |

---

## 12. Phased roadmap

### Phase 0 — unblock (hours, not days). **Nothing else can start.**

| | task | done when |
|---|---|---|
| 0.1 | Flip `matchday_supervisor` from `dry_run` to `execute` on `archpc` | `core.matchday_action` shows `executed = true` |
| 0.2 | Restore the SofaScore XI scrape (dead since 2026-08-09) | `lineup_provisional` has rows for the next round at T−35 |
| 0.3 | Schedule `betdb_crosswalk_rebuild` T−180, T−60, T−20 | `betfair.match_meta` resolves 100% of the Saturday card |
| 0.4 | Raise the archived ladder from 3 to 10 levels in `stream_worker.py` | `array_length(bid_prices,1) = 10` on new rows |
| 0.5 | Verify against the collector's own guard: `betdb_matchday_status` = `ready` at T−60 | one clean Saturday |

### Phase 1 — data & pricing engine (1–2 weeks)

| | task | file |
|---|---|---|
| 1.1 | Correct the three stale health warnings in `src/MatchDay` (`confirmed`, scrape lead, `last_price_traded`) — they are now **wrong**, and a wrong measured claim is worse than none | `matchday-module.jl`, `types.jl`, `gates.jl`, `sources.jl`, `book.jl` |
| 1.2 | `LiveOrderBook` book source reading 10 levels with `max_age = 5min` | `implementations/book.jl` |
| 1.3 | `SizedBestOfBackLay` — the instrument rule that consults `bid_volumes`/`ask_volumes` and refuses a leg whose `venue_stake` exceeds available depth. §4.5 shows Decision 8 ("size checking out of scope") no longer holds on a 21-fixture Scottish card. | `instruments.jl` |
| 1.4 | `MaxSpread(σ)` gate — the missing half of `MinMatched`; §4.4's BTTS trap passes every existing gate | `implementations/gates.jl` |
| 1.5 | `MinMatched` per-competition thresholds (£500 is a Premiership number) | `implementations/gates.jl` |
| 1.6 | Replay harness: run every Saturday since 2026-08-01 through `match_day` at T−25 and dump the sheet | `current_development/match_day_inference/r06_*` |

### Phase 2 — paper ledger & state machine (2–3 weeks)

| | task |
|---|---|
| 2.1 | `paper` schema migration incl. the §5.4 batch columns and the one-RESERVE-per-slate partial unique index; nightly ledger-reconciliation assert |
| 2.2 | `src/PaperTrading/` — `types.jl`, `db.jl`, `batch.jl` (**the `SlateBatch` lifecycle**), `state_machine.jl` (**pure `decide`**), `fills.jl`, `settle.jl` |
| 2.3 | **`execute_slate_batch(account, slate)`** — the atom: one `SELECT … FOR UPDATE`, the `Σ risk ≤ bankroll × cap` assert, all orders + one `RESERVE` row, `PRICED/REVIEWED → RESERVED`. Fails whole. (§5.3, §7.2) |
| 2.3b | Persist `k_risk`, `slate_exposure`, `capped`, `risk_lambda`, `exposure_cap` off `Portfolio.SlateAllocation` — they are not recoverable after the fact (§5.4) |
| 2.3c | Pre-`RESERVED` re-solve on leg removal; post-`RESERVED` release-only. A test asserts a cancel after reservation does **not** re-run `stake_slate` |
| 2.4 | Fill models: `touch_only`, `ladder_sweep_v1`, `optimistic` |
| 2.5 | Settlement job off `sofascore.events`; CLV job off the last pre-off `order_book_1m` snapshot |
| 2.6 | Crash-recovery path (§7.4) with a test that kills the process mid-slate |
| 2.7 | Backfill: replay every Saturday since 2026-08-01 into a paper account **as batches**. **This is the deliverable** — it produces a real slate-level CLV and drawdown distribution before a single live order |

### Phase 3 — operator console (1–2 weeks)

| | task |
|---|---|
| 3.1 | `screens/paper/slate.py` in the existing Textual TUI — read-only batch header + blocked report |
| 3.2 | Julia `src/Console/` — HTTP.jl static serve + `/api/snapshot` (account + **batch** + cards) + WebSocket 1 Hz push |
| 3.3 | The batch header (§8.4): `k_risk`, exposure-vs-cap bar, `capped`, batch_status, leg count |
| 3.4 | The card grid: Alpine `x-for`, EV sort, **shared-scale model/market probability bars**, fair-odds vs venue-odds line, fill-confidence dots, tabular numerals |
| 3.5 | **`Execute Slate Batch`** — one button, one intent, one server-side transaction; disabled while any gate is failing |
| 3.6 | Slide-over: 10-level ladder, MTM sparkline, per-leg detail and cancel (with the pre/post-`RESERVED` semantics of §8.4) |
| 3.7 | Intent endpoint: execute-batch / kill-leg / kill-slate, validated server-side, ledger-audited |

### Phase 4 — live (gated, not scheduled)

Do not start until, over **≥ 10 paper slates**: leg-weighted CLV ≥ 0, fill rate ≥ 70% at the sized
stakes, zero ledger reconciliation failures, and zero slates priced off a book older than 5 minutes.
Then build the Betfair `placeOrders` adapter behind the *same* state machine, with
`paper_accounts.is_live = true` as the only switch — and note there is **no exchange order-placement
code anywhere in either repo today**, so that adapter is genuinely new work, not a refactor.

---

## 13. What is verified, and what is not

**Verified in this session, against the live databases:**

* Every number in §4 (inventory, scaling, NULL rates, coverage, spreads, depth, fill curves,
  timing), §3.4 (lineup timing), §7.1 (kick-off concurrency), §2.3 (`mcmc_experiments` contents),
  §1.3 (the supervisor dry-run blocker).
* The ×10 000 volume scaling, cross-checked two independent ways.
* That `src/MatchDay/implementations/book.jl:_unscale` is correct.

**Asserted from reading code, not executed:**

* The behaviour of `select_split`, `extend_fit`, `stake_sheet` and the instrument morphism. Nothing
  in `src/` was run — `kaimon`, the Julia REPL gateway, refused connection for this session, and
  the project's convention is not to invoke `julia -e` from a shell.
* That the Textual TUI's live screens do what their docstrings say.

**Not verified at all — treat as design assumptions:**

* Betfair `placeOrders` limits: instructions per request, transactions per hour, per-connection
  market subscription cap. §7.3 makes these config for exactly this reason.
* That 10 ladder levels are available from the collector's subscription tier.
* Whether the 2026-08-21 drop from 11 to 8 markets/event was a deliberate config change or a
  silent subscription-cap truncation. The evidence (§4.3) is consistent with both; `core.tournament_config.live_markets`
  is NULL for every tournament, which suggests the group default changed rather than a per-tournament override.
* Commission treatment: `PerBetCommission` is assumed; Betfair's actual market-base-rate netting
  within a market is not modelled. §0's note that the optimum never covers a full market group
  (r04's "market groups fully covered: 0") is what makes this safe, and it should be re-asserted on
  Scottish data rather than inherited from Ireland.
