# matchday_2026_08_08 — the match-day workflow

Five files. Three phases: **train**, **serve**, **grade**.

This describes the workflow *as it actually runs*. The two modules it drives have their own maps
and this file does not repeat them:

* `current_development/matchday_runbook/ARCHITECTURE.md` — `src/MatchDay` (9 seams, 22 components)
* `current_development/portfolio_runbook/ARCHITECTURE.md` — `src/Portfolio` (9 seams, 24 components)

Read this one for **the order things happen in and which file to edit**. Read those two for what
any individual component does.

---

## 1. The 30-second version

```
                          ┌──────────────────────────────────┐
                          │             betdb                │
                          └──────────────────────────────────┘
                    history │                        │ live + archived
                            v                        v
   ┌──────────────────────────────────┐   ┌────────────────────────────────────┐
   │  PHASE 1 · TRAIN                 │   │  sofascore.events                  │
   │  r01_train_weekend.jl            │   │  betfair.match_meta                │
   │  l01_weekend_training.jl         │   │  betfair_live.market_metadata      │
   │                                  │   │  betfair_live.order_book_1m        │
   │  weekend_task → run_experiment   │   │  sofascore.lineup_provisional      │
   │  → data/matchday_wknd_0808/…     │   │  sofascore.matches   (results)     │
   └──────────────────────────────────┘   └────────────────────────────────────┘
                            │                        │
                  experiment │  (posterior chains)   │ prices, identities, XIs
                            └───────────┬────────────┘
                                        v
                        ╔═══════════════════════════════════╗
                        ║   MatchDay.match_day(spec, sys,   ║
                        ║     segment, expr, ds; as_of)     ║
                        ╚═══════════════════════════════════╝
                              │                      │
             as_of = now(UTC) │                      │ as_of = a past instant
                              v                      v
        ┌─────────────────────────────┐   ┌──────────────────────────────────┐
        │ PHASE 2 · SERVE             │   │ PHASE 3 · GRADE                  │
        │ r02_price_tonight.jl        │   │ r03_replay_scot_lower.jl         │
        │                             │   │ l02_slate_replay.jl              │
        │ stake sheet + order tickets │   │ N× as_of, then grade vs result   │
        │ → data/paper_trades/        │   │ → data/replays/…                 │
        └─────────────────────────────┘   └──────────────────────────────────┘
```

**The one idea.** Phases 2 and 3 call *the same function with the same arguments*. The only
difference is the value of `as_of` and where the fixture list comes from. Nothing in the pipeline
reads the clock internally — `as_of` is a call-site argument at every stage. That is what makes a
past match day replayable and a live decision auditable, and it is the single most important
property of the whole design.

---

## 2. File map

The house convention is `lXX` = **loader** (definitions, no side effects) and `rXX` = **runner**
(execution, prints, writes files). You run `r`, you edit `l`.

```
matchday_2026_08_08/
│
├── l01_weekend_training.jl     TRAIN helpers
│     floor_warmup(ds, season)          largest safe warmup_period  ← the point of the file
│     assert_splits(ds, cfg)            refuse a run whose last fold is empty
│     weekend_task(ds, model, …)        create_experiment_task with warmup DERIVED not typed
│     poisson_outfield_model()          engine for 79/718/54/55 (needs xG + player ratings)
│     funnel_model()                    engine for 56/57 (BBC shots only)
│
├── r01_train_weekend.jl        TRAIN runner        ← overnight, 16 cores, ~hours
│     loads 3 DataStores, builds 3 tasks, runs them sequentially
│     writes data/matchday_wknd_0808/<name>_<stamp>/
│
├── r02_price_tonight.jl        SERVE runner        ← live, ~1 min
│     spec_for(ds) + SYS, loops RUNS (one per segment), writes sheet + tickets CSV
│
├── l02_slate_replay.jl         GRADE helpers
│     slate_from_db(t_ids, day)         fixtures + results out of SQL
│     book_coverage(fixtures)           how deep the replayable window really is  ← run FIRST
│     snapshot_grid(fx; lookback, step) the as_of instants
│     replay_spec(fixtures; …)          MatchDaySpec for a PLAYED day
│     latents_invariant(…)              does the model move with as_of? (assert, don't assume)
│     replay(spec, sys, …, snaps)       N× match_day, keeping everything
│     grade!(sheet, results, sys)       :graded :unit_payoff :pnl
│     venue_leg(row)                    independent check on order_ticket
│     slate_trace / family_trace        view 1  exposure & P&L vs time-to-KO
│     churn                             view 2  book stability
│     clv_vs_close                      view 3  did the market come to us
│     fill_report                       view 4  could it have been filled
│     divergence_vs_experience          view 5  cold start
│     policy_sweep / family_trust        view 6  policy A/B at a fixed instant
│
└── r03_replay_scot_lower.jl    GRADE runner        ← ~4 min for 21 snapshots
      provenance → replay → 6 views → CSVs in data/replays/scot_lower_20260808/
```

---

## 3. PHASE 1 · TRAIN

```
  r01_train_weekend.jl
      │
      ├─ Data.load_datastore_cached(segment)      ⚠ 48h TTL, rebuilds silently. See trap T1.
      │
      ├─ weekend_task(ds, model, name, dir, season)         [l01]
      │     │
      │     ├─ w = floor_warmup(ds, season)
      │     │     min over pooled tournaments of each one's last match_week
      │     │     (min, not max — the two divisions are rarely level)
      │     │
      │     ├─ GroupedCVConfig(warmup_period = w, dynamics_col = :match_week, …)
      │     │
      │     ├─ assert_splits(ds, cfg)   ← ERRORS if the last fold has 0 target matches
      │     │
      │     └─ Experiments.create_experiment_task(…)
      │
      └─ Experiments.run_experiment(task) → save_experiment(res)
                 │
                 └─→ data/matchday_wknd_0808/scot_lower_funnel_20260807_012812/
                         └─ training_results :: Vector  (chain per fold)
                            config { model, splitter, … }
```

**Why `warmup_period` is derived, never typed.** `_process_tournament_group_ids` keeps folds at
`dynamics_step >= warmup_period`, plus one injected baseline fold with *zero* target matches. Set
the warmup past the season's last step and `valid_steps` is empty — no error, you just get the
baseline fold alone: a model trained on history only, that has never seen the target season, and
that looks perfectly healthy from the outside. That is what `r05_ireland_03_07_26.jl` shipped.
`floor_warmup` + `assert_splits` make it unrepresentable.

**The number to check in the morning:** `folds` must be `> 1` for every segment.

---

## 4. PHASE 2 · SERVE (live)

```
  r02_price_tonight.jl
      │
      ├─ AS_OF = now(UTC)          ⚠ UTC, never now(). A BST clock puts every gate an hour out.
      │
      ├─ spec_for(ds) :: MatchDaySpec        ← WHERE THE NUMBERS COME FROM
      ├─ SYS         :: PortfolioSystem      ← HOW MUCH TO STAKE
      │
      └─ for r in RUNS                        one entry per segment
            ds   = load_datastore_cached(r.seg)
            expr = load_experiment(r.path)    ← the phase-1 artefact
            res  = MD.match_day(spec, SYS, r.seg, expr, ds; as_of = AS_OF, bankroll)
            │
            ├─ blocked_report(res)     READ THIS FIRST. "no bets" ≠ "broken".
            ├─ res.sheet               → data/paper_trades/sheet_<stamp>.csv
            └─ order_ticket.(rows)     → data/paper_trades/tickets_<stamp>.csv
```

⚠ **One `match_day` call per segment means one `FixedCap` per segment.** Three segments at
`FixedCap(0.25)` can put 75% of the bankroll live at once. See trap T5.

---

## 5. PHASE 3 · GRADE (replay a played day)

```
  r03_replay_scot_lower.jl
      │
      ├─ 0 PROVENANCE ─ what the replay can and cannot see
      │     slate_from_db(t_ids, day)   → fixtures + results        [SQL, not ds.matches]
      │     book_coverage(fixtures)     → first/last tick, n_snaps  ← RUN THIS FIRST
      │     split-pairing / leak check  → assert 0 slate fixtures in the conditioning fold
      │     latents_invariant(…)        → does the model move with as_of?
      │
      ├─ snaps = snapshot_grid(fx; lookback = Minute(60), step = Minute(3))
      │            anchored on the EARLIEST kickoff, so no snapshot sits after a KO
      │
      ├─ out = replay(spec, sys, segment, expr, ds, snaps; bankroll, results)
      │     │
      │     └─ for t in snaps:
      │           res = MD.match_day(…; as_of = t)      ← the SAME call as phase 2
      │           grade!(res.sheet, results, sys)
      │           capture depth (back/lay sizes) for fill analysis
      │        keeps: legs, quotes, depth, blocked, slate, close
      │
      └─ six views → stdout + data/replays/scot_lower_20260808/*.csv
```

**Why the fixture source must change.** `SofaScoreEvents` filters `status_type = 'notstarted'`.
The moment a match kicks off it becomes invisible to that query, so a played day **cannot** be
replayed with the live source. `replay_spec` uses `ExplicitFixtures(fixtures)`, which filters
`kickoff >= as_of` — and that filter is also what keeps in-play ticks out of a replay, because a
fixture drops off the list as soon as `as_of` passes its kickoff.

---

## 6. Inside one `match_day` call

This is the part worth understanding. Stage numbers are **deliberately out of order**: the book
is built before the features.

```
match_day(spec, sys, segment, expr, ds; as_of, bankroll)
│
├─ build_cards(spec, segment, as_of) ─────────────────────────────── STAGES 1-3
│   │
│   ├─ 1  fixtures(spec.fixtures, segment, as_of) -> Vector{Fixture}
│   │       SofaScoreEvents  → sofascore.events, status='notstarted', KO ∈ [as_of, as_of+horizon)
│   │       ExplicitFixtures → your list, filtered kickoff >= as_of        ← REPLAY
│   │
│   ├─ 2  resolve(spec.identity, f) -> Resolved | Unresolved
│   │       MatchMetaCrosswalk → betfair.match_meta ⨝ betfair_live.market_metadata
│   │       LiveNameMatch      → fuzzy fallback, verified = false
│   │       ⚠ BOTH flow downstream. Filtering Unresolved out here is how a fixture
│   │         becomes invisible instead of reported.
│   │
│   └─ 3  lineup(spec.lineups, f, as_of) -> Lineup | nothing
│           SourceChain is FIRST-SUCCESS:  JsonPin → ProvisionalDB → LastHistorical
│           ProvisionalDB   → sofascore.lineup_provisional WHERE scraped_at <= as_of
│           LastHistorical  → ds.lineups, each team's last completed XI
│           ⚠ LastHistorical() with no ds returns nothing unconditionally
│
│   ═> Vector{FixtureCard}   fixture + identity + lineup + as_of
│
├─ price_cards(spec, cards, as_of) ──────────────────────────────────── STAGE 5
│   │  for every RESOLVED card:
│   │
│   ├─ quotes(spec.book, identity, as_of) -> Dict{SelectionKey, BookLevels}
│   │     ArchivedOrderBook → betfair_live.order_book_1m
│   │       SELECT DISTINCT ON (market_id, symbol) … WHERE ts <= as_of ORDER BY ts DESC
│   │                                                      └─ POINT IN TIME. Never sees the future.
│   │     BookLevels = back[] back_size[] lay[] lay_size[] matched ts
│   │                  └─ bid = available to BACK,  ask = available to LAY
│   │
│   ├─ complement_of(key, keys)          two-outcome groups only; 1X2 has none
│   │
│   ├─ instrument(spec.instrument, key, comp, book, quote_rule) -> Instrument
│   │     BestOfBackLay: back it directly, OR lay its complement, whichever prices better
│   │        lay at d  ≡  back at d/(d-1)  once measured in RISK
│   │     Instrument(key, eff_odds, side, venue_odds, leverage, venue_key)
│   │                 └ position                                  └ runner the ORDER touches
│   │
│   └─ stamps :book_age and :max_matched onto the card, for the gates
│
│   ═> odds_df  (match_id, market_name, market_line, selection, odds_close)
│      insts    Dict{(match_id, SelectionKey), Instrument}
│
├─ ready(spec.gate, card)  for EVERY card ───────────────────────────── STAGE 7
│     GateChain is CONJUNCTIVE — runs all gates, concatenates every reason.
│     (SourceChain and ResolverChain are FIRST-SUCCESS. Two different combinators, on purpose.)
│     ═> passed / blocked        blocked_report(res) says why
│
├─ matchday_latents(spec, expr, ds, passed, odds_df, as_of) ───────── STAGES 4+6
│   │
│   ├─ boundaries = Data.create_id_boundaries(ds, expr.config.splitter)     ← rebuilt TODAY
│   ├─ sel = select_split(expr, boundaries)     idx = min(n_trained, n_rebuilt)   ⚠ trap T2
│   ├─ fs = deepcopy(create_features(boundaries, ds, model)[sel.idx][1])
│   │        └─ deepcopy because the prototype mutated a CACHED FeatureSet
│   │
│   ├─ for k in INJECTABLE_KEYS = (:player_ratings_map, :league_lookup)
│   │     materialise!(spec.features, Val(k), fs, fixtures, ctx)  || error
│   │       RatingsFromTracker → :player_ratings_map   (rolls ratings fwd, sums the XI by position)
│   │       LeagueFromFixture  → :league_lookup        (tournament_id → training league index)
│   │     ⚠ these maps are read as get(map, match_id, <default>), so an unmaterialised
│   │       fixture is priced SILENTLY off the fallback. Hence:
│   │
│   ├─ check_coverage(fs, fixtures, model)   ← PER FIXTURE, not per feature
│   │     ⚠ GLOBAL abort: one uncovered fixture kills the whole segment
│   │
│   └─ extract_parameters(model, frame, fs, sel.chain)
│         frame carries match_id, home_team, away_team, match_date, month_idx, match_week
│                                                       └─ omitting this applies JANUARY to everything
│
│   ═> latents_df  (match_id, λ_h, λ_a, …)   posterior DRAWS, not point estimates
│
├─ Portfolio.stake_sheet(sys, latents, expr, odds_df, fixture_info(passed); bankroll)
│      │  see portfolio_runbook/ARCHITECTURE.md for the inside of this
│      ├─ build_books   score matrix → market probs → selections → payoff matrix
│      │                → Kelly allocate → BakerMcHale shrink      ← EXPENSIVE (~40ms/match)
│      ├─ group into slates (DailySlate: everything settling the same day)
│      └─ stake_slate    a_kelly × trust × shrink × scale → risk → cap → filter   ← CHEAP
│
│   ═> sheet, one row per bet
│
└─ _attach_instruments!(sheet, insts, spec.rounding)
      adds  side, venue_odds, venue_selection, risk, venue_stake
      applies the exchange minimum to the VENUE STAKE (not to risk)
      drops rows with risk <= 0

   ═> MatchDayResult(sheet, cards, blocked, odds, instruments, as_of)
```

---

## 7. The two objects you actually turn

```
┌── MatchDaySpec ─────────────────────── WHERE THE NUMBERS COME FROM ──────────────────────┐
│ field        question it answers      implementations                                    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ fixtures     what is on               SofaScoreEvents · ExplicitFixtures                 │
│ identity     who is who               MatchMetaCrosswalk · LiveNameMatch · ResolverChain  │
│ lineups      which XI                 JsonPin · ProvisionalDB · LastHistorical · SourceChain│
│ book         where prices come from   ArchivedOrderBook · RedisLive*                     │
│ quote_rule   which number in the book BestAvailable · MidPrice(not executable)           │
│ instrument   back it, or lay the comp DirectBackOnly · BestOfBackLay                     │
│ rounding     the exchange minimum     NoMinimum · FloorOrDrop · FloorOrRoundUp           │
│ features     materialise for unseen   RatingsFromTracker · LeagueFromFixture · Chain     │
│ gate         bet at all?              IdentityResolved · MaxBookAge · MinMatched ·       │
│                                       MaxLineupAge · ConfirmedXI · GateChain             │
│ markets      which markets            Data.MarketConfig                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                                                        * declared, errors if used

┌── PortfolioSystem ─────────────────────── HOW MUCH TO STAKE ─────────────────────────────┐
│                                                                                          │
│  .book :: BookSpec        ═══ EXPENSIVE — this IS the cache key, hash it ═══             │
│      markets      which markets to price                                                 │
│      price        DeArb · Normalise · RawPrice                                           │
│      allocator    KellyLogUtility                                                        │
│      shrink       BakerMcHale · FractionalKelly · NoShrinkage                            │
│      exec         commission, max/min stake, budget, require_complete_markets            │
│  ────────────────────────────────────────────────────────────────────────────────────    │
│  .policy :: PolicySpec    ═══ CHEAP — pure multipliers, free to sweep ═══                │
│      trust        FlatTrust · SelectionTrust · ScheduledTrust                            │
│      risk         SlateDrawdown(λ; mode) · IsolatedDrawdown · NoRisk                     │
│      cap          FixedCap  (no NoCap, by design)                                        │
│      filter       KeepAll · MinEdge · MarketWhitelist · MinOdds · FilterChain            │
│      grouping     DailySlate · SingleMatchSlate                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why the split matters.** Changing anything in `BookSpec` rebuilds every `MatchBook` (~40ms per
match). Changing anything in `PolicySpec` is a multiplication on books you already have. That is
why `policy_sweep` runs ten policies in milliseconds and why a model A/B costs a full rebuild.

---

## 8. Which table feeds which stage

```
 STAGE            TABLE                              FILTERED BY
 ───────────────  ─────────────────────────────────  ──────────────────────────────────────
 1 fixtures       sofascore.events                   status_type, start_timestamp ∈ window
 2 identity       betfair.match_meta                 match_id
                  betfair_live.market_metadata       event_id
 3 lineups        sofascore.lineup_provisional       match_id, scraped_at <= as_of
                  ds.lineups  (LastHistorical)       team's last match <= as_of
 5 book           betfair_live.order_book_1m         market_id, ts <= as_of  (DISTINCT ON)
 4/6 features     ds  (the cached DataStore)         boundary match-id pointers
 grading          sofascore.matches                  match_id → home_score, away_score

 All SQL lives in ONE file: src/MatchDay/db.jl. A schema change has exactly one blast radius.
```

---

## 9. How to A/B two models

This is a **clean controlled experiment**, and the reason is structural: `as_of` fixes the book,
so two models see *byte-identical prices*. The only thing that varies is `latents_df`.

```
                        ┌────────────────────────────┐
                        │  snaps  (fixed)            │
                        │  spec   (fixed)            │
                        │  sys    (fixed)            │
                        │  ds     (fixed)            │
                        └────────────┬───────────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             v                       v                       v
     expr_A = load(...)      expr_B = load(...)      expr_C = load(...)
     funnel                  funnel_apm_xg           smile
             │                       │                       │
             └───────────► replay(spec, sys, seg, expr, ds, snaps) ◄──┘
                                     │
                         SAME odds_df at every t
                         DIFFERENT latents_df
                                     v
                    ┌────────────────────────────────────┐
                    │  compare on:                        │
                    │   · log loss vs market (per family) │  ← sharpness
                    │   · sd(p_model)/sd(p_market)        │  ← dispersion
                    │   · λ_model − λ_market              │  ← level bias
                    │   · CLV                             │  ← higher-powered than P&L
                    │   · growth = log(1 + Σpnl/B)        │  ← judge on THIS, not ROI
                    └────────────────────────────────────┘
```

Concretely, on top of what already exists:

```julia
ARMS = ["funnel"       => "./data/matchday_wknd_0808/scot_lower_funnel_20260807_012812",
        "funnel_apm_xg"=> "./data/matchday_wknd_0808/scot_lower_apm_xg_<stamp>"]

results_by_arm = Dict{String,Any}()
for (name, path) in ARMS
    expr = EXPR.load_experiment(path)
    results_by_arm[name] = replay(spec, sys, SEGMENT, expr, ds, snaps;
                                  bankroll = BANKROLL, results = results)
end

for (name, out) in results_by_arm
    println(name); show(out.slate)                 # exposure & P&L vs time-to-KO
end
```

**Three rules for an A/B through this path.**

1. **Both arms must be trained on the same target season**, or the loser is just the one whose
   `team_map` is missing a promoted club — `check_coverage` aborts the segment and you get an
   empty sheet that looks like "no edge". Train both in the same `r01` run.
2. **Hold `BookSpec` fixed.** It is the cache key and it changes what a `MatchBook` *is*. Sweeping
   `PolicySpec` across arms is fine and free; sweeping `BookSpec` means you are no longer
   comparing models.
3. **Rank on growth and CLV, not ROI or log loss alone.** ROI is blind to flat trust (a uniform
   scaling cancels). Log loss ranked the APM and funnel engines as indistinguishable while Betfair
   growth separated them.

---

## 10. Traps

```
T1  THE DATASTORE CACHE EXPIRES AT 48h AND REBUILDS ITSELF ON LOAD.
    Between training and replay it will silently acquire the matches you are pricing, and
    create_id_boundaries then returns MORE folds than the experiment has chains.
    → always print (n_trained, n_rebuilt) and assert the slate is not in the conditioning fold.

T2  select_split USED TO PAIR BY INDEX, NOT BY CONTENT.        (FIXED)
    idx = min(n_trained, n_rebuilt). It bit for real on ScottishUpper 2026-08-09 after a
    force-rebuild of the cache:

        fold   targets   last target date   slate fixtures inside
          2        10        2026-08-02       0    <- correct
          3        22        2026-08-09       6    <- what the positional rule chose

    Both counts were 3, so the count-mismatch warning NEVER FIRED. Silent.
    → `select_split(...; exclude = <ids being priced>)` now picks the most recent fold whose
      target window is clear of the card, and errors when no such fold exists. `matchday_latents`
      always passes it. Pinned by test M6c.
    → this does NOT excuse a stale pairing: stepping back to fold 2 means the two most recent
      windows are unused, so the model is a week behind. Retrain rather than lean on it.

T3  check_coverage IS A GLOBAL ABORT.
    One fixture with an unknown team kills the whole segment, not just that fixture.
    Newly promoted/relegated clubs are the usual cause in August.

T4  SofaScoreEvents CANNOT REPLAY.
    status_type = 'notstarted'. Use ExplicitFixtures for anything already played.

T5  FixedCap BINDS PER match_day CALL, NOT PER DAY.
    r02 makes three calls, so three × 0.25 = up to 75% of bankroll live simultaneously.

T6  TRUST IS ABSORBED BY THE DRAWDOWN SOLVER.
    risk_factor is homogeneous of degree 0. Measured on this slate: α ∈ {0.25, 0.35, 0.5, 1.0}
    give BIT-IDENTICAL books, with α × k_risk constant at 0.1316.
    → to move exposure move λ. α only does work when it DIFFERS between selections.

T7  FILTERS RUN AFTER THE CAP, SO THEY TRUNCATE.
    MarketWhitelist zeroes stakes that were sized for a book still containing what you removed;
    the freed capacity is not reused. Per-family SelectionTrust runs before the allocator and
    lets the drawdown budget re-expand. Prefer trust to filter for curation.

T8  NOTHING READS DEPTH.
    BestAvailable takes the price and discards back_size/lay_size; max_leverage filters on price
    alone. Measured: BestOfBackLay took +0.22% of price and gave up 93% of capacity on one leg.
    Risk-weighted fill at the close was 69%.

T9  now(UTC), NEVER now().
    Kickoffs come from unix2datetime and open_date is a timestamptz. A BST clock puts every gate
    an hour out and silently drops the late kickoffs.

T10 A REFUSAL IS A VALUE.
    An empty sheet because the gate refused everything and an empty sheet because the model found
    no edge are the same DataFrame. Always read blocked_report(res) first.
```

---

## 11. Running it

```bash
# PHASE 1 — overnight, hours
julia --project -t 16 current_development/matchday_2026_08_08/r01_train_weekend.jl

# PHASE 2 — live, ~1 min. Re-run per match day; the collector only carries TODAY.
julia --project -t 16 current_development/matchday_2026_08_08/r02_price_tonight.jl

# PHASE 3 — post mortem, ~4 min for 21 snapshots
julia --project -t 16 current_development/matchday_2026_08_08/r03_replay_scot_lower.jl
```

In a warm REPL, include into a fresh module so the `const`s do not collide:

```julia
module R03; end
Base.include(R03, "current_development/matchday_2026_08_08/r03_replay_scot_lower.jl")
R03.out.slate        # the trace
R03.out.legs         # every staked leg at every snapshot, graded
```

**To point phase 3 at a different day or league**, edit four constants at the top of `r03`:
`MATCH_DAY`, `SEGMENT`, `EXP_PATH`, `OUT_DIR`. Everything else derives from those — the fixture
list, the results, the snapshot grid and the coverage report are all queried, not hard-coded.
