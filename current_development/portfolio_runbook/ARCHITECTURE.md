# src/Portfolio — system map

19 files, ~1760 lines, 9 swappable seams, 24 concrete components.

---

## 1. The 30-second version

```
                    THE MODEL SAYS              THE MARKET SAYS
                    (posterior score grid)      (quoted prices)
                              \                  /
                               \                /
                                v              v
                          +----------------------------+
                          |        MatchBook           |   <-- EXPENSIVE (~40ms/match)
                          |  payoff matrix + full      |       cache it
                          |  Kelly stakes, per match   |
                          +----------------------------+
                                       |
                          +----------------------------+
                          |          Slate             |   matches that settle together
                          +----------------------------+
                                       |
                     apply POLICY: trust -> shrink -> risk -> cap -> filter   <-- CHEAP (~ms)
                                       |
                        +--------------+--------------+
                        v                             v
                   simulate()                   stake_sheet()
                  (has results)                (no results yet)
                        |                             |
                   Trajectory                    one row per bet
                        |
                     report()
```

**The one idea:** a `MatchBook` is expensive and knows nothing about your betting policy.
A policy is a set of cheap multipliers applied to it. Everything else follows from that split.

---

## 2. File map (this is the include order = the dependency order)

```
src/Portfolio/
|
|  FOUNDATION -- no dependencies on each other
+- types.jl          241  9 abstract types, 6 domain objects, 3 config structs
+- interfaces.jl     141  one contract per seam + error stubs + show methods
|
|  COMPONENTS -- each implements exactly one contract
+- implementations/
|  +- commission.jl   40  PerBetCommission  NetMarketCommission*  NoCommission
|  +- pricing.jl      36  DeArb  Normalise  RawPrice
|  +- allocators.jl   99  KellyLogUtility            (+ kkt_residual, growth)
|  +- shrinkage.jl    82  NoShrinkage  FractionalKelly  BakerMcHale
|  +- trust.jl        70  FlatTrust  SelectionTrust  ScheduledTrust
|  +- risk.jl        113  NoRisk  IsolatedDrawdown  SlateDrawdown
|  +- caps.jl         58  FixedCap  VolTargetCap*
|  +- filters.jl      59  KeepAll  MinEdge  MarketWhitelist  MinOdds  FilterChain
|
|  PIPELINE -- in order of execution
+- payoff.jl          67  THE keystone. payoff / payoff_matrix / settle_vector
+- book.jl           196  extract_selections -> build_book -> build_books + cache key
+- slates.jl          49  group() into settlement windows
+- stake.jl           78  stake_slate: the 5 multipliers
+- simulate.jl        74  walk slates, settle, compound -> Trajectory
+- matchday.jl        89  stake_sheet / slate_summary  (live, no results)
+- metrics.jl        102  path_metrics / bootstrap_roi / report / attribution
+- calibrate.jl       72  calibrate_lambda (correct dial) / calibrate_scale (trap)
+- portfolio-module.jl 95 module + include order + exports

* = declared seam, not implemented (errors if selected)
```

---

## 3. The nine seams

Every swappable component is an abstract type with **one** required method.
Adding a new one = add a struct + one method. No existing file changes.

```
SEAM                      CONTRACT                                    LIVES IN
========================  ==========================================  ==========
AbstractPricePolicy       settlement_odds(p, d, overround) -> Float64  BookSpec
                          DeArb | Normalise | RawPrice

AbstractCommissionModel   net_return(c, d) -> Float64                  BookSpec.exec
                          PerBetCommission | NoCommission | NetMarket*

AbstractAllocator         allocate(a, p, R, exec) -> (a, kkt, conv)    BookSpec
                          KellyLogUtility                               <- MPT goes here

AbstractShrinkage         shrink_factor(s, sm, R, p, alloc, exec)      BookSpec
                          NoShrinkage | FractionalKelly | BakerMcHale
--------------------------------------------------------------------  ----------
                                     ^ above = CACHE KEY.  below = FREE TO SWEEP
--------------------------------------------------------------------  ----------
AbstractTrustModel        trust_for(t, sel, ctx) -> Float64            PolicySpec
                          FlatTrust | SelectionTrust | ScheduledTrust

AbstractRiskModel         risk_factor(r, probs, rets) -> Float64|Vec   PolicySpec
                          NoRisk | IsolatedDrawdown | SlateDrawdown

AbstractExposureCap       apply_cap(c, stakes) -> (stakes, capped)     PolicySpec
                          FixedCap | VolTargetCap*      (no NoCap, by design)

AbstractSelectionFilter   keep(f, sel, stake, ctx) -> Bool             PolicySpec
                          KeepAll | MinEdge | MarketWhitelist | MinOdds | FilterChain

AbstractSlateGrouping     group(g, books) -> Vector{Slate}             PolicySpec
                          DailySlate | SingleMatchSlate
```

**Why the line matters:** the four above it change what a `MatchBook` *is*, so changing one
means rebuilding 628 books (~26s). The five below are pure multipliers on an existing book,
so a 24-cell policy sweep runs in 0.9s. `book_cache_key(spec)` hashes only the top half.

---

## 4. Objects, and what each one carries

```
   LatentRow                one row of L1 posterior summaries for one match
        |
        | Predictions.extract_params + compute_score_matrix
        v
   ScoreMatrix              12 x 12 x 4000   (home goals, away goals, posterior draws)
        |
        +--- mean over draws, normalise ------> p_grid    144-vector, sums to 1
        +--- compute_market_probs ------------> model probability per selection
        |
        v
   Selection[]              n legs (mean 6.5, max 15). Each carries:
                              family        "1X2_home"      <- the trust key
                              group/line    "1X2" / 0.0     <- the grading key
                              odds_quoted   as traded
                              odds_used     after price policy   <- what you settle at
                              p_model       what we think
                              p_market      vig-removed market   <- BENCHMARK ONLY, never a price
        |
        v
   MatchBook                p_grid   144
                            R        144 x n    payoff matrix. wealth = 1 .+ R*a
                            settle   n | nothing    <- nothing => unplayed fixture
                            a_kelly  n          full Kelly on the posterior mean
                            k_shrink scalar     Baker-McHale factor
                            kkt      scalar     solver quality, want ~1e-6
        |
        v
   Slate                    window::Date + the books that settle in it
        |
        v
   SlateAllocation          stakes (per book, per selection) + k_risk, exposure, capped
        |
        v
   Trajectory               bankroll path, per-slate P/L, k_risk, exposure, bets::DataFrame
```

---

## 5. STAGE A — `build_books`  (the expensive half)

```
build_books(spec, latents_df, expr, odds_df, ds; require_result=true)
  |
  +- fixture_table(ds)                      date + score for every match, built once
  |
  +- Threads.@threads over ~710 matches
  |    |
  |    +- build_book(spec, row, expr, odds_df, fixtures)
  |         |
  |         +- fixture known?  odds exist?        <- fast-fail BEFORE the expensive bit
  |         |
  |         +- compute_score_matrix               <-- LOOP 1: the MCMC grid
  |         +- compute_market_probs per market    <-- LOOP 2: 7 markets
  |         |
  |         +- extract_selections
  |         |     reject group unless EVERY leg quoted   (70% of O/U 0.5 fail this)
  |         |     overround = sum(1/d)
  |         |     odds_used = settlement_odds(price, d, overround)
  |         |
  |         +- payoff_matrix(sels, 12, 12, commission)   <-- LOOP 3: 144 x n
  |         |     each cell = payoff(sel, h, a, comm) -> Data.grade_selection
  |         |       win  -> (1-c)(odds_used - 1)
  |         |       push -> 0.0        <- stake returned, NOT a loss
  |         |       lose -> -1.0
  |         |
  |         +- allocate(KellyLogUtility, p_grid, R, exec)
  |         |     max sum_w p_w log(1 + R_w' a)   s.t. 0<=a<=0.5, sum(a)<=0.99
  |         |     Fminbox(LBFGS) + log-barrier in BOTH objective and gradient
  |         |     -> a_kelly, kkt
  |         |
  |         +- shrink_factor(BakerMcHale, ...)    <-- DOMINANT COST
  |         |     LOOP 4: 128 posterior draws, re-solve allocate() on each
  |         |     LOOP 5: grid over k, maximise mean_j <p, log(1 + k R a*(q_j))>
  |         |     -> k_shrink
  |         |
  |         +- settle_vector(sels, h, a, comm)    or `nothing` if unplayed
  |
  +- sort by (date, m_id)      <-- chronology established ONCE, here
```

Result on the reference data: **628 books** from 710 matches, ~26 seconds on 16 threads.

---

## 6. STAGE B — `stake_slate`  (the cheap half)

Five multipliers, in this order. This is the heart of the module.

```
       a_kelly            full Kelly, per selection      e.g. median 16% of bankroll per match
          |
    (1)   x  trust_for(sel)           per selection      FlatTrust(0.25) -> x0.25
          |                                              exact blend at fair odds (see note A)
    (2)   x  k_shrink                 per match          BakerMcHale, median 0.64
          |
    (3)   x  global_scale             global             usually 1.0; NEARLY A NO-OP (note B)
          |
          v
       rets[t] = R_t * stakes[t]      portfolio return in each of 144 states, per match
          |
    (4)   x  risk_factor(probs, rets) PER SLATE          Busseti drawdown budget
          |                                              bisect k s.t. sum_t log E[(1+kR)^-lam] <= 0
          |                                              SlateDrawdown -> one k for the whole day
          |                                              IsolatedDrawdown -> one k per match
          v
    (5)   apply_cap                   PER SLATE          FixedCap(0.25): if sum > 0.25, scale down
          |                                              THIS is what makes ruin impossible
          v
          filter                      per selection      keep() -> false zeroes a stake
          |                                              runs LAST: can only REMOVE exposure
          v
       SlateAllocation
```

**Note A — why trust is a multiplier, not a re-solve.** Market probabilities are vig-removed,
so `p_market * odds == 1`. Therefore blending `w*p_model + (1-w)*p_market` scales the marginal
Kelly edge by exactly `w`. Exact for a single selection; first-order for the portfolio.

**Note B — why `global_scale` does almost nothing.** See invariant 1 below.

---

## 7. The two exits

```
  BACKTEST                                   MATCH DAY
  ========                                   =========
  build_books(...; require_result=true)      build_books(...; require_result=false)
        |                                          |
   every book has `settle`                    `settle` may be nothing
        |                                          |
   group -> slates                            group -> slates
        |                                          |
   simulate(policy, slates)                   stake_sheet(sys, ...; bankroll=1000)
     for each slate:                            for each slate:
       stake_slate                                stake_slate          <-- SAME CALL
       settle: pl += stake * settle[j]            emit one row per bet
       ASSERT pl > -1
       bank *= (1 + pl)                         slate_summary(sheet)
        |                                        -> fixtures, bets, exposure, k_risk, capped
   Trajectory
        |
   report(traj, [SharpeRatio(), ...])
   path_metrics / bootstrap_roi / attribution
```

`simulate` **asserts every book is settled**. A missing result raises rather than being
scored as a loss. That is the only difference between the two paths.

---

## 8. Call graph (who calls whom)

```
build_books ---> build_book ---> extract_selections ---> settlement_odds   [PricePolicy]
                     |                                        |
                     |           payoff_matrix -----> payoff -+-> net_return [Commission]
                     |           settle_vector ------^   |
                     |                                   +-> Data.grade_selection  (ONE grader)
                     |
                     +-------> allocate  ---> kkt_residual                  [Allocator]
                     |
                     +-------> shrink_factor ---> allocate (x128 draws)     [Shrinkage]

group  --------------------------------------------------------------------[SlateGrouping]

simulate ------> stake_slate ---> trust_vector ---> trust_for               [TrustModel]
   |                  |
stake_sheet ----------+---------> risk_factor                               [RiskModel]
                      |
                      +---------> apply_cap                                 [ExposureCap]
                      |
                      +---------> keep                                      [Filter]

calibrate_lambda ---> simulate (repeatedly, bisecting lambda)
report -----------> path_metrics + bootstrap_roi + BackTesting.compute_metric
```

---

## 9. Five things that will confuse you when reading output

```
1. LAMBDA SUBSUMES TRUST.
   Once the drawdown constraint binds, trust 0.25 / 0.5 / 1.0 give IDENTICAL results.
   risk_factor is homogeneous of degree 0: hand it 2x the stakes, it returns half the
   factor. Measured: stake multiplier 0.25 / 1.0 / 4.0 all -> mean exposure 0.1088.
   => trust only RESHAPES the book. To move exposure, move lambda (calibrate_lambda).

2. ROI IS BLIND TO FLAT TRUST.
   ROI = P/L / stake, so a uniform scaling cancels. All flat trust levels give the same
   ROI and very different growth. Judge on growth_per_slate, never ROI.

3. NEGATIVE-EDGE BETS ARE HEDGES, NOT BUGS.
   One portfolio solve over all 144 states, not a list of value bets. A small negative-edge
   draw position often hedges a big home/away one in the same match.

4. `kkt` IS NOT A CONVERGENCE FLAG.
   It is the actual worst first-order-condition violation. Optim can report success at a
   non-KKT point. Want median ~1e-6, p99 < 1e-4.

5. THERE IS NO `NoCap`, DELIBERATELY.
   FixedCap validates 0 < cap < 1 in its constructor, which makes a non-positive bankroll
   unrepresentable rather than merely asserted against. The prototype had no cap and its
   simulated bankroll reached -0.697.
```

---

## 10. Where to look for what

| I want to... | Look at |
|---|---|
| see it run | `r01_quickstart.jl` |
| understand the cache split | `r02_policy_sweep.jl`, `book_cache_key` in `book.jl` |
| produce a stake sheet | `matchday.jl`, `r03_matchday_stakes.jl` |
| check a result is trustworthy | `r04_diagnostics.jl` |
| add my own component | `r05_extending.jl`, then `interfaces.jl` |
| know how a bet is scored | `payoff.jl` (60 lines, the keystone) |
| know how stakes are sized | `stake.jl` (the 5 multipliers) |
| know what is guaranteed | `test/portfolio_tests.jl` (74 property tests) |
| verify against the prototype | `portfolio_explore/r18_src_parity.jl` (bit-exact) |

---

## 11. Reference numbers (ScottishLower, funnel_apm_xg, 2024-08 → 2026-04)

```
books / slates              628 / 99          median 8 matches per slate
build time                  ~26s on 16 threads
policy sweep                24 cells in 0.9s
KKT residual                median 1.2e-6, p99 3.3e-6
de-arb                      41.5% of quotes shrunk, mean 0.216%
Baker-McHale k*             median 0.640, mean 0.584
full-Kelly stake per match  median 16.2%, max 97.1% of bankroll
default policy result       ROI 9.31%, 1.915x, MDD -23.5%, mean exposure 8.3%
ROI 95% CI (by match)       [-1.47%, +20.89%]     <- includes zero
```

Treat the backtest as an **upper bound**: it settles at traded prices, and the order book
shows the executable back price is ~1.2% worse, which costs roughly 24% of cumulative gain.
