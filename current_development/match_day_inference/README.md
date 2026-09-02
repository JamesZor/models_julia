# MatchDay live execution

This directory is the active operational suite for MatchDay live execution. It prices an
entire simultaneous fixture slate, records the planned portfolio in the paper ledger, and
presents the slate through the operator console. The slate is the execution atom: reservation
is performed for the whole stake vector in one transaction before individual orders are
submitted.

## Two consoles, two ports, two schemas

They run side by side and neither can reach the other's rows.

| | port | schema | clock | what it is for |
|---|---|---|---|---|
| **live** — `r07_serve_console.jl` | **8085** | `paper_runbook` | `now()` | committing a slate on a Saturday |
| **replay** — `r08_replay_console.jl` | **8086** | `paper_replay` | a scrubber | backtesting a past Saturday |

The replay process refuses `paper_runbook` at every ledger call site (`assert_replay_schema`) and
refuses to bind 8085 (`serve_replay`). Both refusals are asserted in
[`test/test_matchday_replay.jl`](../../test/test_matchday_replay.jl) (R1, R2, R18) rather than
left to convention.

## Active files

- [`QUICKSTART_LIVE.md`](QUICKSTART_LIVE.md) — operator-oriented quickstart for pricing,
  reserving, submitting, settling, and serving a paper slate.
- [`r06_slate_ledger_console.jl`](r06_slate_ledger_console.jl) — replayable end-to-end Scottish
  slate runner: canonical fit, slate pricing, paper ledger, settlement, and console snapshot.
- [`r07_serve_console.jl`](r07_serve_console.jl) — the **live** operator console on port 8085.
- [`r08_replay_console.jl`](r08_replay_console.jl) — the **replay** console on port 8086 (below).
- [`replay_state.jl`](replay_state.jl) — replay loader: the clock, the point-in-time sources, the
  model registry, execution and settlement.
- [`replay_server.jl`](replay_server.jl) — replay read model and HTTP/WebSocket surface.
- [`replay_console.html`](replay_console.html) — the replay single-page dashboard.
- [`RESEARCH_MATCHDAY_ARCHITECTURE.md`](RESEARCH_MATCHDAY_ARCHITECTURE.md) — live execution
  design, dataflow, controls, and validation rationale.
- [`AI_AGENT_HANDOVER.md`](AI_AGENT_HANDOVER.md) — system state and operational context for
  follow-on work.

Run the worked example from the repository root:

```bash
julia --project -t 8 current_development/match_day_inference/r06_slate_ledger_console.jl
```

The historical exploratory inference and single-fixture runbook prototypes are retained under
[`current_development/archived/matchday/`](../archived/matchday/):
`legacy_inference/` and `legacy_runbook/`, respectively. They are archival material, not the
active live execution path.

---

# The replay console (`r08_replay_console.jl`, port 8086)

## What it answers

> *"What would this model have said, at this minute, against the book that actually existed
> then — and what would it have won?"*

It drives the **same** pipeline the live console drives —

```
fixtures → identity → lineups → BOOK → features → inference → gate → stake_sheet
```

— and replaces only the sources that read a clock or a network. The gates, the instrument rule,
the stake rounding, the market set and the portfolio policy are the live ones, unchanged. A
replay that relaxed any of them would prove nothing about a Saturday.

Nothing here samples: every posterior is loaded from a completed run in `mcmc_experiments` via
`MD.canonical_fit`.

## Launching it

```bash
# from the repository root
julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl

# then open
#   http://localhost:8086          (LAN: http://192.168.1.88:8086)
```

Environment:

| variable | for |
|---|---|
| `BF_DB_URL` | `betdb` — fixtures, the order book, lineups, and the `paper_replay` ledger |
| `BF_EXPERIMENTS_DB_URL` *or* `~/.pgpass` | `mcmc_experiments` — the canonical fits |

Overrides, so the file does not have to be edited:

```bash
R08_DAY=2026-08-15  julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl
R08_MODEL=m12       julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl
```

Boot takes roughly 60–90 s: DataStore (cached), the match day read into memory (~3 s), then one
canonical fit plus its feature collection. The other two models load lazily on first selection.

## Which Saturdays can be replayed

A day is replayable when it has fixtures **and** an archived order book. The console's
`matchday` dropdown reports both counts and disables the days that have no book; the same table
is printed at boot and is available at `GET /api/replay/matchdays`. Measured on this database:

| day | fixtures | book rows | book span | scraped XI |
|---|---|---|---|---|
| 2026-08-01 | 10 | 123,796 | 00:02 → 23:57 | 0 |
| **2026-08-08** | 10 | 43,298 | 12:00 → 16:03 | **9** |
| 2026-08-15 | 10 | 21,113 | 12:04 → 16:04 | 0 |
| 2026-08-22 | 10 | 44,687 | 14:06 → 20:26 | 0 |

**2026-08-08 is the default and the only one that shows the lineup shock.** It is the single
Saturday carrying both a book and a provisional-XI scrape — nine fixtures, published T−13 to
T−40 with a median near T−29. On the other three the player pillar contributes zero throughout,
which is correct and is also not very interesting.

## Using it

### The VCR

| control | route | keyboard |
|---|---|---|
| ▶ / ⏸ | `POST /api/replay/{play,pause}` | `space` |
| ⏮ −1m / +1m ⏭ | `POST /api/replay/step` `{"minutes": ±1}` | `←` `→` |
| speed 1x / 5x / 30x / 60x | `POST /api/replay/speed` `{"speed": 60}` | |
| scrubber | `POST /api/replay/seek` `{"t": -15}` | |
| jump | `POST /api/replay/jump` `{"target": "lineups"｜"exec"｜"kickoff"｜"settlement"}` | `x` `e` `k` `f` |

The clock is **minutes relative to kick-off**, spanning T−60m to T+105m; the absolute `as_of`
handed to `MatchDay` is derived from it and never stored separately. 60x means one simulated
minute per wall second *including* the re-pricing that minute costs — the tick time is measured
and subtracted, and the header shows `ms/tick`.

### The suggested pass

1. **T−60m** — the opening book. Thin, wide, no XI. Player models contribute zero.
2. **`x` → T−30m** — watch the XI land. On m12 the green (model) bars move and the slate
   (market) bars do not: that difference *is* the lineup shock. On m00 and m05 nothing moves,
   which is the control.
3. **`e` → T−15m** — the entry window. Press **EXECUTE**: the whole stake vector is reserved in
   one transaction and filled against the ladder that minute actually had, with `LadderSweep`.
4. **`f` → settlement** — grade every filled leg against the real score, book the P&L and the
   2% commission, and measure CLV against the de-vigged close at T−0.
5. Press **Results** for the leg-by-leg table, the aggregate ROI and beat-close, and the equity
   before/after bars.

### The two views

The header carries two tabs over the **same** state — switching is client-side and posts nothing,
so playback, the scrubber, the model and the match day stay exactly where they were.

**▦ Slate Cards (Radar)** is the card grid the live console shows, plus two depth facts per leg:

- a **WOM pill** (`████████░░ 68% WOM`) — the share of the three archived levels of resting size
  that sits on the BACK side. Green above 60% (money queued to back, price shortening), pink
  below 40% (queued to lay, drifting), slate in between.
- `depth £X (3 lvls)` — the whole archived ladder on the side the order would consume, against
  `depth_touch`, which is only the first level.
- `↗ Ladder Desk` — opens that fixture on the desk.

**☱ Multi-Ladder Desk** is the exchange screen: one fixture, one market, every runner side by
side as a classic Bet Angel vertical ladder.

| part | what it shows |
|---|---|
| runner header | fair odds `1/p_model`, the market mid, EV%, and the model/market overhang bars |
| WOM bar | back share against lay share, three levels deep |
| ask levels | the three lay prices, worst-first, so the touch sits against the spread row |
| spread row | `SPREAD 0.15 (3 ticks) 4.26%` — currency, **ticks**, and relative |
| bid levels | the three back prices, touch-first |
| order marker | amber, on the runner the order actually **touches**, with the £ consumed per level |
| footer | matched volume and the **book** VWAP (see below) |
| `📈 chart` | opens that runner's trajectory window |

Markets: `[ Match Odds ] [ Over/Under 2.5 ] [ BTTS ]`. Fixtures come from a dropdown, and
`?desk=<match_id>[&market=…][&chart=home]` is a deep link to any of it.

Three things the desk does **not** pretend to know:

1. **A traded VWAP.** `betfair_live.order_book_1m` archives resting depth and a running
   `market_matched` total, never a traded price series. What is shown is `book vwap` — the
   probability-space volume-weighted average of the visible ladder — and it is labelled as such.
2. **Levels beyond the third.** The archive carries at most three, verified over 635,765 rows, so
   every depth and WOM figure says `(3 lvls)`.
3. **A model opinion on a gated fixture.** The model column is priced by the same pipeline the
   card grid uses, on the same gate-passed set; a refused fixture shows its book and an empty
   model column rather than a number derived from inputs the pipeline declined to use.

### The trajectory chart

`📈 chart` opens a draggable, minimisable window over the desk:

- **top pane** — market best back (blue) and best lay (pink) against the model's fair odds
  (green, dashed, **stepped**). On m12 the green line steps at the minute the XI became visible;
  on m00 and m05 it is flat, which is the control. A shaded band marks the T−25…T−12 execution
  window, a dashed amber vertical marks the lineup drop, and a solid blue needle tracks the
  replay clock.
- **bottom pane** — matched volume, i.e. the liquidity S-curve into kick-off.

`GET /api/replay/history` is the same data without the browser. The model is evaluated on a
coarse grid with the drop minute pinned, not every minute: the posterior is memoised on the
lineup signature and moves only when an XI lands, so a 165-minute chart costs two extractions
rather than 165.

### Switching models mid-replay

`POST /api/replay/set_model {"model": "m00"|"m05"|"m12"}` swaps the posterior **in the running
process** and re-prices at the current instant. The clock does not move — the question is what
*this* model says at *this* minute.

| key | run | experiment |
|---|---|---|
| `m00` | `m00_poisson_control` | `scottish_lower_joint_2426` |
| `m05` | `m05_joint_production_wealth` | `scottish_lower_joint_2426` |
| `m12` | `m12_hybrid_production_wealth_player_rapm` | `scottish_lower_player_grid_2426` |

The first selection of a model costs one `Features.create_features` (≈10 s for the team-level
pillars, ≈80 s for the hybrid player pillar) and is then held for the life of the process;
switching back is instant. Switching the *match day* rebinds the loaded models — a fold
re-selection, about a second — rather than rebuilding their features.

A fold that cannot represent a fixture refuses it **by name** in the `NOT COVERED BY …` panel
rather than pricing it at the league mean. On 2026-08-08 two teams (`ross-county`,
`airdrieonians`) are absent from these folds' `team_map`, so 8 of 10 fixtures are priced.

### Execution and settlement

Everything lands in `betdb.paper_replay`, under account `replay_scottish`:

```
paper_slates · paper_orders · paper_fills · paper_settlements · clv_audit · account_ledger
```

`↺ Reset ledger` (`POST /api/replay/reset`) deletes that one account's rows and restores the
bankroll, so a replay can be re-run without dropping a schema.

## What is honest about it, and what is not

Three leaks are possible in a replay and all three are closed **structurally**, not by care:

1. **The book.** `PreloadedBook` holds each runner's ladder sorted by `ts` and reads it with
   `searchsortedlast(stamps, as_of)`. A tick from after the replayed instant is unreachable.
2. **The XI.** `PreloadedLineups` filters `scraped_at <= as_of` and has **no historical fallback
   behind it**, so before the scrape lands a player model prices with no lineup and contributes
   exactly zero. The live spec chains `LastHistorical` there; here that would hide the event the
   console exists to show.
3. **The player ratings.** `:player_lineup_ratings_map` is emitted by the feature extractor over
   *every* match in the store, so for a finished fixture it already holds the teamsheet that took
   the field. `PointInTimeLineupRatings` overwrites it each tick from the visible XI. Without
   that materialiser a T−60m decision would be priced off the teamsheet.

The fold is chosen by `MD.select_split`, which identifies it positively — the fold whose next
observed round *is* this card — and steps back from any fold whose target window contains the
fixtures being priced.

Two things it does **not** claim:

- **`LadderSweep` is the optimistic fill model.** It assumes we cross up to three archived
  levels instantly, which is what a market order does and not what the live system does (it rests
  at the touch). A replay P&L built on it is an **upper bound** on the resting-order path.
  `fill_model` is recorded per fill row, so a `ladder_sweep_v1` track and a `touch_only` one are
  never pooled by accident.
- **After kick-off the model is pre-game and the book is in-play.** The book has seen goals the
  posterior has not, so post-T−0 "edges" reach four figures and are a measurement of that gap
  rather than a signal. The console says so in red and disables Execute; the API refuses too
  unless `{"allow_in_play": true}` is passed deliberately.

## Verification

```bash
julia --project -t 8 test/test_matchday_replay.jl
```

803 assertions in four tiers — pure (clock, filtration contract; no database), the ladder desk
(ticks, weight of money, the three-level book, the order marker, one runner's history), ledger (`paper_replay` execution and settlement, plus a direct assertion that `paper_runbook`
row counts are unchanged), and models (a real Saturday, real canonical fits, hot-swapping and the
lineup shock). The ledger and model tiers skip **with a message** when the database or the
DataStore cache is out of reach, never silently.

## API reference

```
GET  /                          the page
GET  /api/snapshot              the whole payload (replay · account · batch · cards · settlement)
GET  /api/health                liveness, client count, port, schema, current minute
GET  /api/replay/matchdays      which days are replayable, and how well
GET  /api/replay/ladder         ?match_id=…&market=MATCH_ODDS|OVER_UNDER_25|BOTH_TEAMS_TO_SCORE
GET  /api/replay/history        ?match_id=…&symbol=home&market=…[&from=-60&to=105]
POST /api/replay/play
POST /api/replay/pause
POST /api/replay/speed          {"speed": 1|5|30|60}
POST /api/replay/step           {"minutes": 1|-1}
POST /api/replay/jump           {"target": "start|lineups|exec|kickoff|settlement"}
POST /api/replay/seek           {"t": -60 … 105}
POST /api/replay/set_model      {"model": "m00|m05|m12"}
POST /api/replay/set_matchday   {"day": "2026-08-08"}
POST /api/replay/execute        [{"allow_in_play": true}]
POST /api/replay/settle
POST /api/replay/reset
```

Every control also accepts a query string (`POST /api/replay/seek?t=-15`), so the whole console
is drivable from `curl` without a browser.
