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

## Health warnings

**All three upstream jobs are currently dead**, and none of them raises anything when it stops:

```
identity resolver      last output 2026-06-22   -> resolution 100% before, 0% after
provisional lineups    last output 2026-06-26   -> `confirmed` has NEVER been true
order-book drain       last output 2026-08-02   -> 99 markets opened after it
```

`r04` prints all three. Restarting the resolver is what unlocks everything since June.

**The split index is a known unresolved defect.** The chain is chosen positionally
(`training_results[29]`) while the boundary list is rebuilt at inference time (31 today), so the
two most recent windows go unused and the pairing is only correct if the splitter appends rather
than recomputes. `select_split` warns on every run. Worth fixing before this prices anything you
actually bet — and it may affect `Experiments.extract_oos_predictions` too.

**£15 is below the operating threshold.** With `FloorOrDrop(£1)` a 5-fixture slate yields **zero**
placeable bets at £15 and 17 at £1000. A 25% cap over 5 matches leaves ~£3.75 across 23
positions. Not a tuning problem.

**Replay is not the backtest.** `last_price_traded` is NULL in 100% of `order_book_1m`, so replay
prices off the book while the Portfolio backtest settles at traded prices. Different quantities.
