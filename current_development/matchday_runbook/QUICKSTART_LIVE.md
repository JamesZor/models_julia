# Quickstart — the live slate loop

Five minutes from a cold REPL to a priced slate, a committed paper batch and a console.

For the *older* single-fixture path (`match_day`, Ireland) read `README.md` and `r01`–`r05`.
This file covers the **live loop** added in `feat/matchday-live-architecture`: `price_slate` →
`paper` ledger → console. Design rationale is in
`current_development/match_day_inference/RESEARCH_MATCHDAY_ARCHITECTURE.md`; you do not need it
to use this.

---

## 0. The one idea

**The slate is the atom.** `Portfolio` solves one joint problem for every fixture that settles
together — `SlateDrawdown` returns a single `k` for all of them and `FixedCap` rescales the whole
vector — so a stake vector is only valid *as a vector*. Everything below follows from that:

```
canonical_fit ─► price_slate ─► insert_slate! ─► execute_slate_batch! ─► submit_slate! ─► settle_slate!
                     │              (rows,          (ONE tx, the atom:      (parallel,       (PnL,
                     │              no money)        whole vector or none)   no account)      reserved→0)
                     └──────────────────────► serve_console ◄─────────────────────────────────────┘
```

---

## 1. Prerequisites

```bash
export BF_DB_URL="postgresql://user:pass@192.168.1.88:5433/betdb"   # operational DB
# mcmc_experiments (mcmc-beast:5432) is reached via ~/.pgpass, or BF_EXPERIMENTS_DB_URL
ls .cache/datastore_ScottishLower.jls          # or whichever segment you are pricing
```

Check all three before anything else:

```julia
using BayesianFootball
const MD = BayesianFootball.MatchDay
const TT = BayesianFootball.Training

MD.paper_connection()                                    # betdb reachable?
MD.canonical_fit(TT.PostgresStorage("scottish_lower_joint_2426"), "m00_poisson_control")
```

---

## 2. Run the worked example first

```bash
julia --project -t 8 current_development/matchday_runbook/r06_slate_ledger_console.jl
```

It replays the 2026-08-08 Scottish League One/Two card at T−25 and prints every stage. Expected
output, verified 2026-09-01:

```
REFUSED, correctly: teams absent from team_map: ross-county, airdrieonians
8 fixtures → 22 legs · k_risk 0.1173 · exposure 11.13% of 25% cap · not capped
RESERVATION   21 admitted, 1 refused (negative edge), £255.80
EXECUTION     10 matched, 11 partial, £132.78 filled of £263.85
SETTLEMENT    +£11.39, reserved → 0, reconciles
```

If that runs, your environment is good. Copy it and change the constants at the top.

---

## 3. The five calls you actually use

```julia
# ── price ──────────────────────────────────────────────────────────────────────
cf    = MD.canonical_fit(TT.PostgresStorage("scottish_lower_joint_2426"), "m00_poisson_control")
slate = MD.price_slate(spec, sys, DD.ScottishLower(), cf, ds;
                       as_of = DateTime(2026,9,5,14,35),   # T−25. NEVER now().
                       bankroll = 2400.0, account_id = "live")

MD.slate_batch_summary(slate)     # ← READ THIS FIRST: exposure, k_risk, capped
MD.blocked_report(...)            # ← READ THIS SECOND, before "no bets today"

# ── commit ─────────────────────────────────────────────────────────────────────
conn = MD.paper_connection()
MD.migrate_paper_schema!(conn)                             # idempotent, run every start-up
MD.ensure_account!(conn, MD.PaperAccount(account_id = "live", opening_balance = 2400.0,
                                         balance = 2400.0, max_slate_exposure = 0.25))
sid = MD.insert_slate!(conn, slate)                        # returns the id the DB HOLDS
MD.insert_orders!(conn, MD.orders_to_paper(slate; slate_id = sid))

MD.execute_slate_batch!(conn, "live", sid)                 # THE ATOM

# ── execute, settle ────────────────────────────────────────────────────────────
MD.submit_slate!(conn, sid, slate.books, MD.TouchOnly())
MD.settle_slate!(conn, sid, Dict(16362409 => (2, 1), ...))
MD.reconcile_account(conn, "live")                         # must be .ok
```

### `as_of` is a parameter, never `now()`

No stage reads the clock. That is the only reason a past match day is replayable and a live
decision is auditable. Pass the instant explicitly, everywhere.

---

## 4. The console

```julia
state = MD.ConsoleState(
    () -> MD.slate_snapshot(slate, MD.account_row(conn, "live")),
    on_execute = () -> begin
        r = MD.execute_slate_batch!(conn, "live", sid)
        (ok = r.status === MD.RESERVED, note = "reserved $(r.n_admitted) legs", error = r.reason)
    end,
    on_kill = () -> (MD.kill_slate!(conn, sid); (ok = true, note = "killed")))

MD.serve_console(state; port = 8080)      # loopback only, by design
MD.stop_console!(state)
```

Reach it over a tunnel — this page can commit a slate:

```bash
ssh -N -L 8080:127.0.0.1:8080 archpc      # then http://127.0.0.1:8080
```

**Reading the page.** Header first (`k_risk`, exposure vs cap, `capped`), then cards. Each leg
draws model and market probability as two bars on **one shared scale** — the overhang *is* the
edge. `●●●` = fills at the touch, `●●○` = fills within the ladder at ≤1% slippage, `●○○` = it
does not fill. `⏎` executes the batch, `k` kills it.

The browser is never in the trust path: it POSTs an intent, the server runs the same transaction
a script would.

---

## 5. Recipes

**Replay a past match day.** `SofaScoreEvents` filters `status_type = 'notstarted'`, which is
false for a played match — so on a replay it returns nothing and a wrong query looks like a quiet
Saturday. Use `ExplicitFixtures`; see r06 §3.

**Try a different fill model.** `TouchOnly()` is the default and the honest one.
`LadderSweep()` assumes you cross three levels instantly — that is a market order, not a resting
one. `Optimistic()` is research only; a paper track built on it cannot be compared with a live
one.

**Tighten the gates.** As of 2026-09-01 the lineup feed sets `confirmed = true` and lands the XI
at T−13..T−42, so both of these are now usable where they previously blocked everything:

```julia
gate = MD.GateChain(MD.IdentityResolved(),
                    MD.MaxBookAge(Minute(10)),                       # 30 is too slack at T−25
                    MD.MaxSpread(0.08),                              # catches League Two BTTS
                    MD.ConfirmedXI(blocking = true),
                    MD.MaxLineupAge(max_age = Hour(2), blocking = true))
```

**Recover after a crash.** The reservation is durable, so nothing is lost or double-counted:

```julia
MD.recover_open_orders(conn, "live")      # PENDING_SUBMISSION → submit or cancel; SUBMITTED → reconcile
```

**Move exposure.** Change **λ**, not trust and not the stake multiplier. `risk_factor` is
homogeneous of degree 0, so once the drawdown constraint binds, trust and shrinkage can only
*reshape* the slate — measured, `FlatTrust` at 0.25, 0.5 and 1.0 give identical exposure. A
control that "scales up the slate" is a no-op, which is why none exists.

---

## 6. When it does not work

| you see | it means | do |
|---|---|---|
| `teams absent from team_map` | promoted/relegated club outside the chosen fold's training window | correct refusal. Retrain, or price the covered subset |
| `index inter.μ not found` | the run stores a **synthetic** 2-parameter smoke chain | check `canonical_fit(...).fit.folds[1].chain \|> names` before trusting a run name. All of `scottish_lower_poisson_2426` is synthetic |
| 0 bets, N blocked | read `blocked_report` — two reasons, not one | `:identity` alone = dead resolver; `:identity` **and** `:book` = dead collector too |
| `unresolved (absent_from_crosswalk)` | `betfair.match_meta` has no row | run `betdb_crosswalk_rebuild` **before kick-off** — there is no retrospective fix. Meanwhile use `ResolverChain(MatchMetaCrosswalk(), LiveNameMatch())` |
| `no quotes retrieved` / stale book | the collector is not armed | see §7 |
| slate `ABANDONED`, "above the cap" | `Σ risk` exceeds `max_slate_exposure × equity` | **do not scale the stakes.** Re-price with a lower λ — `FixedCap` already had the rescaling job with the whole book in hand |
| `reservation is a no-op` | the slate is already past `PRICED`/`REVIEWED` | intended. A retry cannot double-reserve |
| `reconcile_account(...).ok == false` | a write bypassed `post_ledger!`, or a transaction half-committed | a defect, not rounding. Stop and look |
| every leg dropped at small bankroll | `FloorOrDrop(£1)` against a 25% cap | not a tuning problem. £15 yields zero placeable bets on a 5-fixture slate |

---

## 7. Health, as of 2026-09-01

**The collector has been decided-but-not-executed since 2026-08-28.** `core.matchday_action`
holds 569 `arm / not_armed / executed=false` rows for 2026-08-29 alone. The supervisor is in
**dry-run**: it decides correctly every 60 s and carries nothing out. The last `order_book_1m`
row is `2026-08-28 20:59`. **Nothing here can price a live card until that is flipped to
`execute`** — replay still works.

Also currently dead: the identity crosswalk (0 of 159 fixture-days from 2026-08-29) and the XI
scrape (last output 2026-08-09).

Retired claims — the README's health section still carries these and they are **wrong**:

* `confirmed` *is* now true, in 1,071 of 1,533 lineup rows, and the XI lands at T−13..T−42, not
  4.4–5.8 h out.
* `last_price_traded` began populating on 2026-08-07 and is present in 56–88% of rows since.
  Pre- and post-August CLV baselines are different quantities; do not pool them.
* The `select_split` positional defect is **fixed** — rule 1 identifies the fold positively via
  `get_next_matches`, rule 2 excludes any fold whose target window contains the card.

**Volumes and prices are both ×10000.** A top-of-book size of `20000` is **£2.00**, the Betfair
minimum — not £200. `_unscale` in `implementations/book.jl` is correct; match it.

---

## 8. Capacity, in one table

Per-leg ceiling at ≥80% fill and ≤1% slippage, measured on the 26/27 Scottish book. Only 3 ladder
levels are archived, so these are **floors**.

| | 1X2 | O/U central | BTTS |
|---|---|---|---|
| Premiership | £250 | £100 | £50 |
| Championship | £50 | £25 | £100 † |
| League One | £25 | £10 | £100 † |
| League Two | £25 | £10 | £25 |

† fills, but at a 5–9 tick spread — the fill is real and the price is not. Cap by spread, not by
depth.

**Enter at T−12, not at the lineup drop.** A T−60 quote sits 1.31 pp (median, 1X2) from the close
with a p95 of 6.4 pp; T−15 cuts that to 0.27 pp while the half-spread barely moves and depth
doubles. Independently reproduced by `orderbook_layer2/RESULTS.md` §4.1 on Ireland.

---

## 9. Where things are

| | |
|---|---|
| `src/MatchDay/slate.jl` | `PricedSlate`, `price_slate`, capacity annotation |
| `src/MatchDay/fits.jl` | `canonical_fit` — loading and auditing a run from `mcmc_experiments` |
| `src/MatchDay/ledger/state_machine.jl` | `decide_order`, `reserve_plan` — **pure**, no DB |
| `src/MatchDay/ledger/reservation.jl` | `execute_slate_batch!` — the atom |
| `src/MatchDay/ledger/fills.jl` | `TouchOnly`, `LadderSweep`, `Optimistic` |
| `src/MatchDay/ledger/schema.jl` | the `paper` DDL |
| `src/MatchDay/console/` | server, read model, the page |
| `test/test_matchday_live_pipeline.jl` | 221 tests; every one pins a claim or a defect |
| `r06_slate_ledger_console.jl` | the worked example this file describes |
