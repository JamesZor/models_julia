# matchday_runbook

Worked examples for `src/MatchDay` — turning a fixture list into an executable stake sheet.
**Runners only**: the loader is the `MatchDay` module itself, so there are no `lXX` files.

Read `ARCHITECTURE.md` first if you want the system map; read these in order if you want to see
it work.

| file | what it teaches |
|---|---|
| `_setup.jl` | shared loading — **Ireland Premier (79)** / `src_sup40_sw40`. Every runner includes it. |
| `r01_quickstart.jl` | the whole pipeline in one call, on a real match day. Ends with the £1-minimum surprise. |
| `r02_replay.jl` | **the capability the prototype never had** — re-run any past instant, measure CLV. |
| `r03_instruments.jl` | back vs lay, why they are the same bet, and why Portfolio never learned the difference. |
| `r04_diagnostics.jl` | what to check before placing anything. Start here when you get no bets. |
| `r05_extending.jl` | adding your own gate, quote rule and rounding rule without touching `src/`. |
| `r06_slate_ledger_console.jl` | **the live loop** — canonical fit → slate → paper ledger → console, on a real Scottish card. |

**New here? Read [`QUICKSTART_LIVE.md`](QUICKSTART_LIVE.md).** `r01`–`r05` document the
single-call `match_day` path on Ireland; the live system is `price_slate` plus the `paper`
ledger and the console, and the quickstart is task-shaped rather than tutorial-shaped.

## The one idea

MatchDay manufactures Portfolio's two inputs — a `latents_df` of posterior draws and an
`odds_df` of prices — for fixtures that have **not been played**, and refuses loudly when it
cannot. It does no staking maths.

```
fixtures -> identity -> lineups -> BOOK -> features -> inference -> gate -> stake_sheet
                                                                            ^
                                                            everything after this arrow
                                                                is src/Portfolio
```

```julia
include("_setup.jl")
res = MD.match_day(replay_spec(Date(2026,6,19)), SYS, DD.Ireland(), expr, ds;
                   as_of = DateTime(2026,6,19,17,15), bankroll = 1000.0)

res                              # MatchDayResult(23 bets, 5 priced, 0 blocked, ...)
PF.slate_summary(res.sheet)      # read EXPOSURE before you read the bets
MD.blocked_report(res)           # read THIS before you conclude "no bets today"
```

## `as_of` is a parameter, never `now()`

No stage reads the clock. That is what makes a past match day replayable and a live decision
auditable, and it is the only reason any of this can be validated — the prototype could only be
exercised on a live Saturday, which is why none of it ever was.

Replay corpus today: **35 matches, 2026-05-29 .. 2026-06-26**, the intersection of order-book
coverage and a resolvable `match_id`. It grows every match week, so `r02` computes it as a query.

## Back and lay are the same bet

Laying at `d` is backing the complement at `d/(d-1)`, with backer stake `risk/(d-1)`. Because
`AbstractInstrumentRule` hands Portfolio the **effective** odds, the payoff matrix, the allocator
and `FixedCap` are already denominated in risk and need no knowledge of lays — `FixedCap` sums
liability by construction. Only `order_ticket` sees the difference.

Measured on `order_book_1m`: worth ~0.3% on Ireland Premier's central lines and **3.5–6.4% on
Scottish League One/Two's O/U 3.5**. It tracks book width, so it is worth most exactly where the
book is worst.

## Read the refusals

The recurring failure on this project is *silent emptiness*. A blocked fixture is a **value**:

```
0 bets, 1 blocked
  :identity  unresolved (absent_from_crosswalk) -- no betfair.match_meta row
  :book      no quotes retrieved
```

Two reasons, not one, because `GateChain` is conjunctive. The second is usually the informative
one: "unresolved" alone is a dead resolver; "unresolved" *and* "no quotes" is a dead collector
too.

## Health warnings, re-measured 2026-09-01

**All three upstream jobs are dead again**, and none of them raises anything when it stops:

```
order-book collector   last output 2026-08-28   -> supervisor is in DRY-RUN: 569 rows of
                                                   `arm / not_armed / executed=false` on 08-29
identity crosswalk     last output 2026-08-28   -> 0 of 159 fixture-days since
provisional lineups    last output 2026-08-09   -> but see below: it now works when it runs
```

`r04` prints them. Nothing can price a **live** card until the supervisor is flipped to
`execute`; replay is unaffected.

**Two of this file's older warnings were wrong and are withdrawn.** `confirmed` *is* now true —
1,071 of 1,533 rows in `sofascore.lineup_provisional` — and the XI lands at **T−13..T−42 min**,
not 4.4–5.8 h out, so `ConfirmedXI(blocking = true)` and `MaxLineupAge` are both usable gates.
And `last_price_traded` began populating on **2026-08-07** (56–88% of rows since), so replay
*can* now be compared with a traded price — but pre- and post-August baselines are different
quantities and must not be pooled.

**The split-index defect is fixed.** `select_split` now identifies the fold positively via
`Data.get_next_matches` (rule 1) and falls back to excluding any fold whose target window
contains the card (rule 2). The positional rule survives only as rule 3.

**£15 is below the operating threshold.** With `FloorOrDrop(£1)` a 5-fixture slate yields **zero**
placeable bets at £15 and 17 at £1000. A 25% cap over 5 matches leaves ~£3.75 across 23
positions. Not a tuning problem.

**Replay is not the backtest.** Replay prices off the **book**; the Portfolio backtest settles at
**traded** prices. Different quantities — say which one you used. (`last_price_traded` is no
longer universally NULL, so both are now available; that makes it easier to conflate them, not
harder.)

**Volumes and prices are both scaled ×10000.** A top-of-book size of `20000` is **£2.00**, the
Betfair minimum — not £200. Verified two ways: per-runner `total_matched` sums exactly to
`market_matched`, and Kilmarnock v Celtic's MATCH_ODDS peaked at £249,989.
