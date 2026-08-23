# A framework for testing models and portfolios

Cross-cutting process doc. Not a stream — this is how the streams are meant to be run.

---

## 0. The problem, stated honestly

```
   leagues        7   ScottishLower/Upper, IrelandAll, Ireland, First Div, Veikkausliiga, Norway
   engines       12   goals, xg, copula, smile, iso, funnel, funnel+apm, apm, dixon-coles, …
   engine knobs  ~5   market_weight, half-life, dispersion, interception, HA, kappa …
   price source   3   historical traded / order book / live
   book spec     ~9   markets × price policy × shrinkage
   policy       ~∞   alpha × lambda × cap × filter × grouping
```

Multiplied out that is millions of cells and several MCMC-years. That is the thing that has been
eating you.

**It is not a product.** The layers are separable and they cost different amounts by five orders
of magnitude. Exploit both facts and it collapses to something you can run in an evening.

---

## 1. The one idea: stratify by cost, never sweep upward

```
                          COST PER CELL        CELLS/WEEK      SWEEP IT?
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ L0   DATA      segment, league          hours (fetch)         1–5   │ NO       │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ L1   ENGINE    model type, components   HOURS  (MCMC)        ~10    │ RARELY   │
 │                priors, half-life                                     │          │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ INF  FOLD      which chain + features   seconds              derived│ NEVER    │
 │                                                                      │ (it is   │
 │                                                                      │  implied)│
 ├──────────────────────────────────────────────────────────────────────────┤
 │ PX   PRICE     historical / book /      seconds–minutes       10s   │ YES      │
 │                which as_of instant                                   │          │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ L2   BOOK      markets, price policy,   ~30 s per rebuild     dozens│ YES      │
 │                allocator, shrinkage      (~40 ms × n matches)        │          │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ L3   POLICY    alpha, lambda, cap,      MILLISECONDS          1000s │ ALWAYS   │
 │                filter, grouping                                      │          │
 └──────────────────────────────────────────────────────────────────────────┘

   RULE:  cheap loops go INSIDE expensive ones. Never the reverse.
          one L1 fit  →  many L2 books  →  thousands of L3 cells
```

`src/Portfolio` already encodes the bottom half of this: `BookSpec` is the cache key, `PolicySpec`
is free. `book_cache_key(spec)` hashes only the expensive half. The framework is that idea
extended up the stack.

---

## 2. The trap, and the fix

```
   WHAT YOU HAVE BEEN DOING                 WHAT TO DO INSTEAD
   ────────────────────────────             ──────────────────────────────────────
   for engine in 8:                         STAGE A — score 8 engines vs the market
     for policy in 24:                        no portfolio involved AT ALL
       full backtest()                        8 runs · keep the best 2
                                                    │
   = 192 full backtests                             v
   = weeks of compute                       STAGE B — sweep 24 policies on those 2
   = no idea which layer caused what          books built ONCE per engine
                                              48 policy cells in ~1 second

                                            = 8 + 2 runs, one evening
                                            = and you know which layer moved the number
```

**The decoupling that makes it work:** an engine can be scored *without any portfolio*, and a
policy can be scored *without refitting any engine*. So `N engines × M policies` becomes
`N + M`.

---

## 3. What each layer is judged on — and what it must not be judged on

```
 LAYER       JUDGE ON                             NEVER JUDGE ON
 ──────────  ───────────────────────────────────  ─────────────────────────────────────
 L1 ENGINE   log loss / Brier vs the DE-VIGGED     ROI. P&L. Growth. An engine has no
             market, PER MARKET GROUP              stakes — anything with money in it is
             dispersion ratio vs market            measuring the policy, not the engine.
             calibration by probability decile

 PRICE       CLV vs the close                      P&L. At the n an order book gives you
             risk-weighted fill at top of book     it is pure noise for months.
             spread paid                           mid prices — not executable.

 L3 POLICY   growth per slate                      ROI — blind to flat trust; every flat
             max drawdown, mean exposure           alpha gives the SAME ROI and very
             ruin probability, capped fraction     different wealth.
                                                   log loss — a policy cannot change the
                                                   model's probabilities.
```

Two rules that follow, both learned the hard way in this project:

* **Score per market GROUP, with probabilities renormalised inside the group.** Averaging log
  loss across selections counts a 3-way event three times, and on Double Chance it reversed a
  headline result outright.
* **Rank on growth, not ROI.** `risk_factor` is homogeneous of degree 0, so once the drawdown
  constraint binds, alpha 0.25 and alpha 0.5 give bit-identical books with identical ROI. Measured
  on 2026-08-08: `alpha × k_risk` constant at 0.1316 across alpha ∈ {0.25, 0.35, 0.5, 1.0}.

---

## 4. The three test beds

They answer different questions and are **not comparable to each other**. Putting bed 1 and bed 2
numbers in the same table without saying so is the single easiest way to fool yourself here.

```
 ┌─ BED 1 ── WALK-FORWARD BACKTEST ─────────────────────────────────── the workhorse ─┐
 │ prices    TRADED  (betfair.odds_history / ds.odds)                                 │
 │ n         703 ScottishLower matches over 2 seasons. Similar for Ireland.           │
 │ path      extract_oos_predictions → build_books → group → simulate                 │
 │ answers   does the engine know anything?  does the policy compound?                │
 │ CANNOT    whether you could have been filled. What execution costs.                │
 │ bias      OPTIMISTIC. Settles at traded price; the executable book price is        │
 │           ~1.2% worse, which costs ~24% of cumulative gain.                        │
 └────────────────────────────────────────────────────────────────────────────────────┘

 ┌─ BED 2 ── ORDER-BOOK REPLAY ──────────────────────────── the new, under-used one ──┐
 │ prices    BOOK, with SIZES  (betfair_live.order_book_1m). `as_of` moves freely.    │
 │ n         Ireland ~2 weeks; Scotland from 2026-08-08 onward. Small, growing.       │
 │ path      matchday() / replay() at N instants → grade                              │
 │ answers   executable price · depth · CLV · how the book moves into kick-off ·      │
 │           whether the sheet was fillable at all                                    │
 │ CANNOT    edge. n is far too small and will stay so for months.                    │
 │ NOTE      DIFFERENT QUANTITY FROM BED 1. Never one table without a note.           │
 └────────────────────────────────────────────────────────────────────────────────────┘

 ┌─ BED 3 ── LIVE MATCH DAY ─────────────────────────────────── the plumbing test ────┐
 │ prices    same feeds, as_of = now(UTC)                                             │
 │ n         one slate per week                                                       │
 │ answers   are the feeds alive? does the pipeline hold? are tickets correct?        │
 │ CANNOT    anything statistical. Ever. Not with 6 fixtures.                          │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

**Bed 2 is the one you have barely used, and it is the one that answers the question the backtest
structurally cannot: could this book have been traded.** On 2026-08-08 it said 69% risk-weighted
fill and £1–2 available on the legs the model wanted most. No amount of bed-1 work finds that.

---

## 5. The gates

An idea should die as early and as cheaply as possible.

```
   many engine ideas
        │
        │  GATE 1  — CHEAP, no portfolio
        │  beats the de-vigged market on ≥1 market group,
        │  on ≥300 walk-forward matches, in ≥1 league
        v
   engines worth pricing                    ← most ideas die here, in minutes
        │
        │  GATE 2  — bed 1
        │  positive growth on settled books under ≥2 DIFFERENT policies
        │  (if only one policy works, you fitted the policy, not the model)
        v
   candidate systems
        │
        │  GATE 3  — bed 2
        │  ≥60% risk-weighted fill at top of book
        │  CLV not negative vs the close
        v
   tradeable systems
        │
        │  GATE 4  — bed 3
        │  4+ weeks live paper, plumbing green, tickets verified
        v
   micro-stake live
```

Note gate 1 needs **no odds beyond the de-vigged market and no staking code at all**. That is the
gate that saves you weeks, and it is the one that has been skipped.

---

## 6. The two loops

```
  WEEKLY — automatic, ~5 min              EXPERIMENT — when you have an idea
  ────────────────────────────            ──────────────────────────────────────────
  for each league:                        1. WRITE THE PREDICTION DOWN FIRST
      ctx = matchday(seg, saturday)          "gap shrinks through the season" is a
      pregame(ctx)                            test; "let's see what happens" is not
      → 5 numbers                          2. pick the bed that can answer it
      append to registry.csv               3. run STAGE A or STAGE B — not both
                                           4. append to registry.csv
  3 rows/week, accumulating.               5. one line: KEPT or KILLED, and why
  In 6 weeks you have a trend
  instead of one slate of noise.           An experiment that cannot kill anything
                                           is not an experiment.
```

---

## 7. The registry — the thing that fixes "I lost the overview"

One CSV. Every run appends. Nothing else changes.

```
  results/registry.csv
  ────────────────────────────────────────────────────────────────────────────────
  run_date   league   engine        bed   n     ← what was run
  LL_1x2_gap  LL_ou_gap  LL_btts_gap        ← L1 verdict (negative = beats market)
  dispersion  goal_gap                       ← L1 diagnostics
  policy      growth   roi   exposure  maxdd ← L3 verdict
  fill_wtd    clv_med                        ← bed-2 verdict
  verdict     note                           ← KEPT / KILLED / PENDING + one line
```

The reason nothing feels settled is that results have been evaporating into terminal scrollback.
A file with 40 rows in it is worth more than another clever run.

---

## 8. The scoreboard as of 2026-08-09

Real numbers, so the first rows are already filled in.

```
  ENGINE QUALITY (gate 1)                                        bed 1 / bed 3
  ──────────────────────────────────────────────────────────────────────────────
  goal-level calibration    model 2.74  market 2.78  ACTUAL 2.71   n=703  ✅ FINE
                            gap −0.035 (t −4.3) — no upward bias. Cross it off.
  BTTS         ScottishUpper   LL model 0.699 vs market 0.768      n=6    ✅ BEATS
  O/U 1.5      ScottishUpper   LL model 0.627 vs market 0.672      n=6    ✅ BEATS
  1X2          ScottishLower   LL gap +0.028                       n=10   ❌
  1X2          ScottishUpper   LL gap +0.120  ← worst group        n=6    ❌❌
  dispersion   ScottishLower   0.50 (funnel)  ← cannot rank teams  n=10   ❌
  dispersion   ScottishUpper   1.18 (player)  ← fine               n=6    ✅

  EXECUTION (gate 3)                                                     bed 2
  ──────────────────────────────────────────────────────────────────────────────
  risk-weighted fill        69% at top of book, ScottishLower       ⚠
  book depth (BTTS)         £1–2 available on the wanted side       ❌
  book depth (Upper)        median back size £41.83                 ✅
  BestOfBackLay             +0.22% price for −93% capacity          ⚠ needs a size rule
```

**"Nothing works" is not what this says.** It says: totals level is fine, BTTS and O/U 1.5 beat
the closing line on the Upper, 1X2 is broken everywhere, and League One/Two is too thin to trade
whatever the model says. Four specific statements, three of which are actionable.

---

## 9. Worked example — "is funnel_apm_xg better than funnel?"

The wrong way is to backtest both under six policies on three leagues. Here is the cheap way.

```
  STAGE A                                             cost
  ─────────────────────────────────────────────       ────────────────────────────
  1. train BOTH on the SAME target season             overnight, once
     (else the loser is just the one whose
      team_map lacks a promoted club)
  2. extract_oos_predictions for each                 ~2 min each
  3. build_books against the SAME odds                ~30 s each
  4. score per market group vs de-vigged market       seconds
     + dispersion ratio + calibration deciles
  → GATE 1. If neither beats the market anywhere,     STOP. No portfolio work.
    the portfolio cannot rescue it.

  STAGE B  (only for whatever survived)
  ─────────────────────────────────────────────
  5. reuse the books from step 3                      free — already built
  6. sweep 24 policies                                ~1 s
  7. rank on GROWTH, with two controls:
       · alpha 0.25 vs 0.5  → must be identical (homogeneity)
       · cap 0.50           → must be identical if it never binds
     A sweep where the controls differ has a bug in it, not a finding.

  STAGE C  (only for what survived B)
  ─────────────────────────────────────────────
  8. order-book replay on the weeks you have          ~4 min per slate
  9. fill ratio + CLV
```

Three stages, each gated. Most ideas never reach stage B.

---

## 10. Where the code already is

| stage | what to call | lives in |
|---|---|---|
| load a segment-day | `matchday(segment, day)` | `matchday_2026_08_08/l05_simple.jl` |
| L1 scoring vs market | `pregame(ctx)` | same |
| walk-forward latents | `extract_oos_predictions(ds, expr)` | `src/experiments/post_processing.jl` |
| build books once | `build_books(spec, lat, expr, odds, ds)` | `src/Portfolio/book.jl` |
| sweep policies | `returns(ctx, policies; times)` | `l05_simple.jl` |
| order-book replay | `replay(spec, sys, …, snaps)` | `matchday_2026_08_08/l02_slate_replay.jl` |
| fill / CLV | `fill_report`, `clv_vs_close` | same |
| level-bias style test | `r06_level_bias.jl` | template for a gate-1 run |

**Nothing here needs building. It needs a registry and an order of operations, which is what this
document is.**

---

## 11. What is genuinely missing

Three gaps, in priority order:

1. **A gate-1 harness that scores an engine against the market on the walk-forward, per league,
   with no portfolio involved.** `r06_level_bias.jl` is 80% of it — generalise it from "goal
   level" to the full scoring table and it becomes the cheapest filter you own.
2. **The registry.** One CSV, one append function.
3. **A size-aware quote rule.** Nothing in `src` reads `back_size`/`lay_size`, so gate 3 currently
   has to be done by eye.

Everything else — in-play, meta models, alpha buckets, `funnel_apm_xg` — sits behind those and
cannot be evaluated without them.
