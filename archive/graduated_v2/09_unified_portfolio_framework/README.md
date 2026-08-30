# 09 — Unified Portfolio & Staking Framework

The `src/Portfolio/` staking pipeline rebuilt on `06`'s zero-allocation score-grid
kernels and `07`'s `Fit`, with convergence as a gate in front of the bankroll, and
every legacy call site kept working — mostly by not moving it.

```
julia --project current_development/09_unified_portfolio_framework/r01_demo.jl
```

Runs in seconds, needs no database, no cache and no MCMC, and exits non-zero if any
gate fails. **114 gates, all passing.**

---

## The problem

`src/Portfolio/build_book` (`book.jl:78-114`) does this, once per fixture:

```julia
score_matrix = Predictions.compute_score_matrix(model, extract_params(model, latents_row))
model_probs  = Dict(string(m) => compute_market_probs(score_matrix, m) for m in markets)
rows         = view(odds_df, odds_df.match_id .== match_id, :)
```

| | |
|---|---|
| **A fresh tensor per fixture** | `compute_score_matrix` allocates `(12 × 12 × n_draws)` `Float64` — 1.4 MB at 1,200 draws, 3.7 MB at 3,200. A 500-fixture fold is ~1.8 GB of churn, all of it live for microseconds. GC is the dominant cost of a backtest and none of that memory does any work. |
| **Unboxing a `Vector{Any}`** | `latents_row` is a row of the `DataFrame` whose every cell holds a posterior sample vector inside an `Any` column — the fragmentation `06` exists to remove, re-entered at the top of the staking loop. |
| **A full-frame scan per fixture** | `odds_df.match_id .== match_id` allocates a `BitVector` over the *whole* odds frame to find ~11 rows, then two more per market. On a 50,000-row frame that is 25 million comparisons a fold. |
| **No convergence gate at all** | Nothing between `fit.diagnostics` and a stake. The pipeline will size a Kelly bet off a chain that did not mix. |
| **Five silent failure modes** | `build_book` returns `nothing` for an unknown fixture, an unplayed one, one with no quotes, one with no complete market group, and one that raised — and `build_books` filters all five out identically (`book.jl:151`). A data outage and a clean fold look the same from outside. |

The last one is not cosmetic. An unconverged posterior is not merely noisier: it is too
**narrow** and biased toward wherever the sampler stuck, so every model probability
looks more confident than the evidence supports, every `p_model − p_market` edge looks
larger than it is, and **Kelly stake size is monotone in that edge**. The failure is not
a worse backtest; it is a larger bet on a number that is not a number.

## The replacement

One workspace per **fold**, not per fixture:

```julia
books, report = build_books_reported(spec, fit, ds.odds, ds)   # gated by default
result        = simulate_portfolio(policy, books, report; initial_bankroll = 1000.0)
display_portfolio(result)
```

```
  ALLOCATION — one fixture, 1200 draws, 5 markets
  ----------------------------------------------------------------------
  measurement                                 bytes     budget  verdict
  ----------------------------------------------------------------------
  baseline (empty closure)                        0          0  pass
  compute_score_grid!  Poisson                    0          0  pass
  compute_score_grid!  NegBin                     0          0  pass
  price_market!  1X2                              0          0  pass
  price_market!  BTTS                             0          0  pass
  price_market!  O/U 2.5                          0          0  pass
  price_fixture!  Poisson, 5 markets              0          0  pass
  price_fixture!  NegBin, 5 markets               0          0  pass
  price_fixture!  every fixture, worst            0          0  pass
  ----------------------------------------------------------------------
```

| index | is |
|---|---|
| `OddsIndex` | the five odds columns unboxed once into concrete vectors, plus `match_id → UnitRange` — replacing the per-fixture full-frame scan |
| `BookWorkspace` | one score grid, one `GridWorkspace`, and one destination vector per market outcome, reused by every fixture in the fold |
| `BuildReport` | what the builder skipped and why, plus the MCMC verdict |
| `DailyState` / `PortfolioResult` | a settlement window as a row rather than five parallel vectors, and the metric set `path_metrics` has no fields for |

## Files

| file | contents |
|---|---|
| `l01_types.jl` | the type layer — what is aliased from `src`, what is new, and why the line is where it is |
| `l02_book_builder.jl` | `OddsIndex`, `BookWorkspace`, `price_fixture!`, `extract_selections`, `build_book(s)`, and the convergence gate |
| `l03_stake_and_simulate.jl` | `simulate_portfolio`, `portfolio_summary`, `bootstrap_portfolio`, `stake_sheet`, `run_portfolio_simulation`, `display_portfolio` |
| `l04_compat_bridge.jl` | `module UnifiedPortfolio` + every legacy name |
| `l05_parity.jl` | the parity harness against the live `src` builder, the allocation audit, the cost measurement, and the runner's deterministic fixtures |
| `r01_demo.jl` | the runner: 13 sections, 114 gates |

`l04` includes `l03` includes … `l01`, and `l01` pulls in
[`08_unified_evaluation_framework`](../08_unified_evaluation_framework/) and, through
it, [`07`](../07_unified_inference_framework/),
[`06`](../06_typed_posterior_latents/) and
[`05`](../scottish_lower/05_composable_count_builder/). One `include` loads everything.

`l05` is included from inside `l04`'s module body rather than wrapping it: it holds both
builders at once and neither is reachable from the other's namespace. The briefing's
file *numbering* is kept; the include *order* follows the dependency, as in `08`.

---

## The one design decision that shapes everything else

**This framework declares no domain or configuration type of its own.**

```julia
const MatchBook = BayesianFootball.Portfolio.MatchBook     # not a look-alike
const BookSpec  = BayesianFootball.Portfolio.BookSpec
const PolicySpec = BayesianFootball.Portfolio.PolicySpec
```

The briefing asks for `Selection(family, group, line, sel, odds_close, odds_settle,
prob_model, prob_market)` and `MatchBook(match_id, date, selections, p_grid,
payoff_matrix, settle_vector, raw_alloc, shrink_k, kkt, converged)`. `src` already has
both, field for field, in the same positional order, under different names — and adds
nothing the briefing's version has. So the types are aliased and the briefing's names
are provided as accessors (`book_payoff`, `sel_odds_settle`, …).

Three reasons, in descending order of how much they cost to get wrong:

1. **Two types answering to one name in one session is the failure mode this whole
   prototype line exists to remove.** `06` renamed its smile container `SmileScoreGrid`
   for exactly this reason. A parity harness has to hold both implementations at once;
   if `Portfolio.MatchBook` and `UnifiedPortfolio.MatchBook` were different structs with
   one name, every `MethodError` in this directory would be ambiguous to read.
2. **Backward compatibility becomes identity rather than emulation.** See below.
3. **The parity claim gets sharper.** `l05` compares two *builders* over one set of
   types. If the types differed, every comparison would be field-by-field transcription
   and a field this framework forgot to copy would read as a pass.

The same logic extends to the *seams*. `allocate` (Jacot & Mochkovitch's
non-mutually-exclusive Kelly solve), `shrink_factor` (Baker & McHale), `risk_factor`
(Busseti-Ryu-Boyd), `stake_slate`, `group` and `simulate` are aliased, not
reimplemented. They are correct, they are covered by `test/portfolio_tests.jl`, and none
of them is on the path this framework speeds up — the cost is the tensor churn *in front
of* the convex solve, not the solve. A second Kelly solver whose only job is to agree
with the first one to the last bit is a liability.

**What that leaves as genuinely new**, and it is the whole of the briefing's substance:
the zero-allocation book builder, the `Fit` integration, the convergence gate, the
`BuildReport`, and the richer simulation result.

## Backward compatibility

Stronger than in `07` or `08`, and for a structural reason. All four of these hold:

1. A book built here can be handed to any legacy function, including ones this framework
   has never heard of.
2. A book **unserialised from an existing `.jls` cache** can be handed to any function
   here. Neither direction needs a version check.
3. `book_cache_key(spec)` returns the same `UInt` as before, so an existing book cache
   still **hits** rather than silently rebuilding under a new key.
4. `simulate`, `stake_slate`, `group`, `path_metrics`, `bootstrap_roi`, `report`,
   `attribution`, `slate_summary`, `calibrate_lambda` and `calibrate_scale` are not
   reimplemented at all. There is no number they could disagree on.

A legacy call site's **body** is unchanged. Its import line changes:

```julia
# before
using BayesianFootball
books = Portfolio.build_books(spec, latents.df, expr, odds_df, ds)
traj  = Portfolio.simulate(policy, Portfolio.group(policy.grouping, books))

# after — the same lines, one different import
import BayesianFootball
using .UnifiedPortfolio.Legacy      # binds `Portfolio`
…identical body…
```

The second line is needed because `BayesianFootball` *exports* the name `Portfolio`, and
Julia refuses to rebind an imported name. Nothing can make two modules answer to one
name; the honest claim is that everything *after* the import is untouched. `r01_demo.jl`
§10 proves it with a `LegacyCallSite` module whose body is copied verbatim from
`current_development/scottish_lower/02_poisson_wealth/r03_growth_clv.jl:76-94`.

| legacy expression | still works |
|---|---|
| `build_books(spec, latents_df, expr, odds_df, ds)` | routed onto the typed-container fast path when the family has a legacy-frame reader; delegated to `src` when it does not |
| `build_book(spec, latents_row, expr, odds_df, fixtures)` | delegated verbatim — a single boxed row has no fast path to route onto |
| `extract_selections(odds_df, match_id, spec, model_probs)` | delegated verbatim |
| `fixture_table(ds)` | `==` to `src`'s, plus methods for a bare `matches` frame and any `AbstractDict` |
| `simulate(policy, slates)` / `group` / `stake_slate` | **are** the `src` functions |
| `stake_sheet(sys, latents_df, expr, odds_df, fixtures)` | same columns, same order, same rows |
| `path_metrics` / `bootstrap_roi` / `report` / `attribution` / `slate_summary` | **are** the `src` functions |
| `book_cache_key` / `component_hash` | unchanged, including the `BakerMcHale` case a naive `hash` breaks |
| every component (`DeArb`, `BakerMcHale`, `FlatTrust`, `SlateDrawdown`, `FixedCap`, …) | re-exported, same objects |

### Two deliberate behaviour changes

1. **`build_books` over a `Fit` refuses an unconverged posterior by default.** Every
   other entry point is ungated, because none of them is handed anything to gate *on*.
   A legacy caller therefore sees no change; a caller who upgraded to a `Fit` opted in
   by doing so.
2. **A declined fixture is counted and named** rather than silently dropped. The books
   produced are identical — this adds a second return value, it does not change the
   first.

## Convergence gating

```julia
build_books(spec, fit, odds, ds)                            # refuses, and says why
build_books(spec, fit, odds, ds; require_converged = false) # builds, and flags
stake_sheet(sys, fit, odds, fixtures)                       # gated the same way
```

The verdict comes off `fit.diagnostics` — a **field**, audited by the run that produced
it (`07/l02_convergence.jl`) — so gating two hundred fits loaded from disk needs no
chains, no `DataStore` and no re-audit. `BuildReport.converged` and
`PortfolioResult.converged` carry it forward, so a result pulled off disk six months
from now answers "should this be believed" without recomputing anything.

**`require_converged` defaults to `true` here and to `false` nowhere.** That asymmetry
is the point: `08` gates *evaluation*, where an unconverged run inflates a leaderboard
row; this gates *staking*, where it inflates a bet.

**An unaudited container counts as not converged.** A `Fit` carrying something other
than a `ConvergenceSummary` returns `(false, ["no audit"])`, for the same reason `07`
abstains on an unmeasured gate: letting a container earn a clean bill of health by
recording nothing is precisely backwards.

The gate **refuses; it does not change arithmetic.** §9 builds the same unconverged fit
with the gate lifted and checks the books are bit-identical to the ungated route.

## What `r01_demo.jl` establishes

**Parity, at 0 ULP, against the live `src` builder.** Not a transcription: the legacy
side runs the real `extract_params` → `compute_score_matrix` → `compute_market_probs` →
`Portfolio.extract_selections` → `Portfolio.allocate` path, fed a `DataFrame` built from
the same typed container the new side reads.

```
  PARITY — Poisson container, BakerMcHale
  --------------------------------------------------------------------------------------
  check                                compared      max |Δ|   max ULP  ULP bgt  verdict
  --------------------------------------------------------------------------------------
  p_grid (posterior-mean score grid)       3456    0.000e+00         0        0  pass
  odds_quoted                               264    0.000e+00         0        0  pass
  odds_used (post price policy)             264    0.000e+00         0        0  pass
  p_model                                   264    0.000e+00         0        0  pass
  p_market (vig-removed)                    264    0.000e+00         0        0  pass
  R (payoff matrix)                       38016    0.000e+00         0        0  pass
  settle vector                             264    0.000e+00         0        0  pass
  a_kelly (Kelly allocation)                264    0.000e+00         0        0  pass
  k_shrink (Baker-McHale)                    24    0.000e+00         0        0  pass
  kkt residual                               24    0.000e+00         0        0  pass
  --------------------------------------------------------------------------------------
```

**The gate is 0 ULP, not the briefing's `1e-12`.** `06` established why: a one-ULP
perturbation of a single λ propagates as ~1e-19 absolute and 2 ULP — comfortably inside
a `1e-12` tolerance and unmistakable in ULP. §6d is the negative control that proves
this: it perturbs one λ by one ULP, requires the parity table to fail, and then requires
the same difference to be *invisible* to `1e-12`.

Even the Kelly allocation and the KKT residual come out bit-identical, which is worth
saying explicitly because a reader expects the optimiser to be the loose row. It is not:
`p` and `R` are bit-identical, LBFGS is deterministic, so the iterate sequence is
bit-identical too.

**Zero allocations.** `price_fixture!` — one score grid *and* every market book in the
spec — measures 0 bytes under `@allocated`, for a Poisson container and a NegBin one,
for every fixture in the fold, alongside an empty-closure baseline that must also read 0.

**The simulation is `src`'s, to the last bit.** `simulate_portfolio` runs its own
forward walk (if it called `simulate` and decorated the result, §8 would be checking
that a function agrees with itself) and is then required to reproduce
`Portfolio.simulate`'s bankroll series, slate P&L, `k_risk`, exposure and bet frame at 0
ULP, every field of `path_metrics`, and `bootstrap_roi`'s interval exactly.

**Reproducibility, and RNG hygiene.** The same inputs give the same trajectory and the
same bootstrap interval twice, and simulating never perturbs the caller's global RNG —
which matters because `BakerMcHale` and the bootstrap both sample.

**The four skip causes, separated.**

```
BuildReport
  built            : 20 of 24 fixtures  (0.05 s)
  skipped, no fixture row  : 1 [9000004]
  skipped, unplayed        : 1 [9000005]
  skipped, no quotes       : 1 [9000002]
  skipped, no selections   : 1 [9000003]
```

`src` returns a bare `nothing` for all four. The books are still bit-identical on the
same damaged inputs — §11 checks that too, because a builder that dropped a *different*
fixture would also produce a shorter list.

**Cost, measured.** 24 fixtures × 1,200 draws, five markets:

```
  path                               legacy s       new s  speedup   legacy KiB      new KiB   shrink
  pricing only (grid + 5 markets)      0.0683      0.0334    2.05×      35011.1       1454.4    24.1×
  full build, FractionalKelly          0.0823      0.0565    1.46×     105032.0      71102.1     1.5×
  full build, BakerMcHale(16)          0.4698      0.4297    1.09×    1271761.5    1237830.8     1.0×
```

Read the first row. It isolates what changed: the legacy side allocates a fresh
`(12 × 12 × 1200)` tensor and five dictionaries per fixture; the new side allocates one
workspace for the whole fold and then nothing, so the `24.1×` is a *fold-size* ratio and
grows with the fold. Rows 2 and 3 add the convex solve, which both sides pay identically
— `BakerMcHale` re-solves the allocator 16 times per fixture here and 128 in production,
and it dominates everything. Row 3's `1.09×` is not a disappointing result; it is a
correct one, and it says where the next optimisation belongs.

No gate is attached to the cost section: a timing on a 24-fixture synthetic fold is an
indication, not a measurement, and gating CI on it would make the run flaky for a reason
unrelated to correctness.

### Not established

That any model fits anything, that any strategy makes money, or that any number in the
transcript is good. Posteriors are prior draws with a fixed seed (`06/l04_parity.jl` §9)
and the synthetic market is the model's own prices, perturbed and vigged — so the model
beats it **by construction** and every positive ROI in the output is an artefact of the
fixture.

Three numbers in §7 are artefacts of a three-slate fixture specifically, and the runner
says so where it prints them: CAGR annualises a 14-day span; MDD is `0.00%` and Sortino
is `∞` because no window lost; `mean k_risk` is `1.0000` because the drawdown budget
never bound on a book this small.

The **smile** container family is not exercised. `06`'s kernels price it and
`BookWorkspace` routes it (a `SmileScoreGrid` is built once, holding the shared grid and
the φ buffers by reference, so the Over/Under pricer reaches the smile method with no
per-fixture allocation) — but this runner builds no smile engine, so that route is
reached structurally and not measured.

No live `DataStore` is touched. The store is assembled from `matches` and `odds` alone,
which is every domain the portfolio pipeline reads.

**The standing health warning is `src`'s and is unchanged.** On the only real
out-of-sample evaluation this repository has — ScottishLower, 628 matches — the default
policy returns a flat ROI whose match-clustered 95% interval **includes zero**, and
every attempt to *learn* a per-selection trust weight lost money out of sample. Nothing
here changes that; this framework makes the same numbers cheaper to compute and harder
to compute on a chain that did not mix.

## Using it

```julia
include("current_development/09_unified_portfolio_framework/l04_compat_bridge.jl")
using .UnifiedPortfolio

using BayesianFootball.Data: MarketConfig, Market1X2, MarketBTTS, MarketOverUnder

# Name the markets you want. `Data.DEFAULT_MARKET_CONFIG` also works and is ~40 markets,
# most of which take the fallback path below — pricing all of them to stake three is the
# same waste `08` removed from evaluation.
spec = BookSpec(markets   = MarketConfig(Market1X2(), MarketBTTS(), MarketOverUnder(2.5)),
                price     = DeArb(),
                allocator = KellyLogUtility(),
                shrink    = BakerMcHale(),
                exec      = ExecutionConfig(commission = PerBetCommission(0.02)))

policy = PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(23.0),
                    cap = FixedCap(0.25), grouping = DailySlate())

# from a Fit (07) — the posterior is READ, not re-extracted, and it is GATED
books, build = build_books_reported(spec, fit, ds.odds, ds)
show(stdout, MIME"text/plain"(), build)          # what was skipped, and the verdict

result = simulate_portfolio(policy, books, build; initial_bankroll = 1000.0)
display_portfolio(result)

result.summary.cagr          # the six metrics `path_metrics` has no field for
result.bootstrap_ci          # match-clustered ROI, slate-blocked growth
states_frame(result)         # one row per settlement window
result.trajectory            # the legacy Trajectory, for generate_tearsheet
```

Or in one call:

```julia
result, books, build = run_portfolio_simulation(spec, policy, fit, ds.odds, ds)
```

Match day — note the fixture table must come from a **fixture list**, not a `DataStore`:

```julia
fixtures = fixture_table(upcoming_matches_df)     # ds.matches holds only FINISHED matches
sheet    = stake_sheet(PortfolioSystem(spec, policy), fit, live_odds, fixtures;
                       bankroll = 5_000.0)
slate_summary(sheet)                               # check exposure BEFORE the sheet
```

### Where the fast path applies, and where it does not

`06` has `price_market!` kernels for 1X2, BTTS and Over/Under. Anything else in the spec
— the Asian-handicap ladder, correct score, double chance, draw-no-bet — is priced
through `Predictions.compute_market_probs` against a `ScoreMatrix` **view** of the same
shared grid. Those markets still avoid the per-fixture tensor, but they allocate a
`Dict` and its vectors per fixture as before. `BookWorkspace` warns once when the spec
contains any, and `BuildReport.fallback_markets` names them.

Moving one onto the fast path is "add a `price_market!` method in
`06/l03_score_grids.jl` §7" and nothing else.

## Where this deviates from the briefing, and why

1. **The type hierarchy is aliased from `src`, not redeclared.** The briefing's
   `Selection` and `MatchBook` have `src`'s fields under different names and add
   nothing. Reasons above; the briefing's names are accessors.

2. **`PolicySpec` keeps `src`'s five fields.** The briefing's
   `PolicySpec(allocator, caps, commission, risk, shrinkage, trust)` moves `allocator`,
   `commission` and `shrinkage` out of `BookSpec`. Those three change the `MatchBook`,
   and the `BookSpec`/`PolicySpec` split is exactly the line between "invalidates the
   book cache" and "is a pure post-multiplier". Moving them would turn every policy
   sweep back into a full rebuild — the thing that makes walk-forward evaluation
   affordable.

3. **The parity gate is 0 ULP, not `|Δ| < 1e-12`.** Stronger than asked. §6d shows the
   briefing's tolerance passing a change that has really happened.

4. **`simulate_portfolio` is new; `simulate` is `src`'s.** The briefing names one
   simulator. Two exist because the legacy return type must stay reachable *and* be a
   non-vacuous parity target.

5. **`initial_bankroll` is a reporting scale.** `SlateContext.bankroll` is handed the
   *fraction*, exactly as `Portfolio.simulate` hands it, so a bankroll-dependent trust
   or filter sees the same number under both simulators and the two cannot diverge.
   §7b checks the trajectory is bit-identical at `1.0` and at `1000.0`.

6. **`run_portfolio_simulation` did not previously exist.** The briefing lists it under
   "legacy names"; nothing in the repository defines it. It is provided as a
   build → group → simulate convenience returning `(result, books, build_report)`.

7. **`BuildReport` is a second return value, not a change to the first.**
   `build_books` still returns a bare `Vector{MatchBook}`; `build_books_reported`
   returns both.

8. **`l05_parity.jl` is included from `l04`, and also holds the runner's fixtures.**
   The harness must see both builders, so it lives inside the module. The deterministic
   store lives there too, because the prototype's rule is definitions in the loaders and
   execution in the runner, and `08`'s `synthetic_datastore` puts one fixture per date —
   which would make every slate a single match and quietly delete the simultaneous-
   settlement story this framework is about.

## Graduating this to `src/`

In dependency order, and each step is independently useful:

1. **`OddsIndex` + `extract_selections`** → `src/Portfolio/book.jl`. Self-contained, no
   dependency on `06` or `07`, and it removes the per-fixture full-frame scan on its
   own. Parity-checked by `l05` against the function it replaces.
2. **`fixture_table`'s `DataFrame` and `AbstractDict` methods** → the same file. Three
   lines, and it makes match-day fixture tables constructible by comprehension without a
   `MethodError`.
3. **`BuildReport`** → the same file, as a second return value from a new
   `build_books_reported`. Changes no existing signature.
4. **`BookWorkspace` + `price_fixture!` + the typed `build_books`** → `src/Portfolio/`,
   once `06`'s containers have graduated. This is the zero-allocation change and it
   needs `06` underneath it.
5. **The `Fit` method and its gate** → last, once `07`'s `Fit` exists in `src` and
   `fit.diagnostics` is a field. Until then the gate has nothing to read.
6. **`DailyState` / `PortfolioSummary` / `simulate_portfolio`** can go in at any point
   after 3; they add fields and add no constraint on anything existing.

## Notes carried forward

The `src` defects `06` and `07` documented are unchanged and none is fixed here:
`DynamicPxGRecombModel`'s OOS extractor cannot be called; three extractors mis-size
their output under multiple chains; nine call sites use a `haskey(::Chains, ::Symbol)`
that MCMCChains 7.7 removed. See [`06/README.md`](../06_typed_posterior_latents/README.md),
[`07/README.md`](../07_unified_inference_framework/README.md) and
[`08/README.md`](../08_unified_evaluation_framework/README.md).

Two `src/Portfolio/` behaviours are reproduced verbatim because changing them would make
new numbers incomparable with every number this repository has recorded:

* **An Over/Under push is dropped, not voided.** On an integer line, cells whose total
  equals the line count into neither side, so `over + under < 1`
  (`over_under.jl:6-52`, mirrored in `06/l03_score_grids.jl` §7).
* **`NetMarketCommission` is declared and raises.** Charging Betfair's actual net-market
  commission requires settlement-side netting across a market group, which does not
  exist. The seam is there; selecting it errors rather than over-stating returns.
