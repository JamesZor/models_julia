# Layer-2 order-book stream — running log

Convention follows `split_market_pillar/NOTES.md`: dated entries, findings stated with the
measurement that produced them, corrections left visible rather than edited away.

The question this stream exists to answer is **not** "does the model work" — that was settled in
`split_market_pillar` (`src_sup40_sw40`). It is **"given the model, when and how do you touch the
book?"** That is Layer 2, and `src/Portfolio` cannot currently express it: it carries one price
per bet (`:odds_close`) and no `DateTime` anywhere.

---

## Design axes

| axis | values | seam |
|---|---|---|
| entry time | `FixedLead`, `AtClose`, `FirstQualifying`, `BestPrice` (oracle) | `AbstractEntryRule` (new, `l01`) |
| latent arm | `:frozen`, `:live` | `L2Config.arm` |
| trust | flat / curated / EB / walk-forward | `Portfolio.AbstractTrustModel` (exists) |
| curation | odds band, spread, top-of-book size, market whitelist | `Portfolio.AbstractSelectionFilter` (exists) |
| risk regime | binding (`SlateDrawdown(23)`) / slack (`NoRisk`) | `Portfolio.AbstractRiskModel` (exists) |

Cost model — this is the whole architecture:

```
TIER 1  SNAPSHOTS  expensive  DB + latent extraction, once per (slate, as_of)
TIER 2  STAKING    cheap      build_books + stake_slate for a given PortfolioSystem
TIER 3  ENTRY      free       select rows from the resulting ledger
```

---

## 2026-08-11 — WP0 corpus built. Gates G1–G3 PASS.

`build_corpus("ireland", [79, 718]; from = 2026-05-20, to = 2026-08-10)`, 92s
(connection-bound: `MatchDay._query` opens a fresh Postgres connection per call, ~210 calls).

```
L2Corpus "ireland"  [79, 718]
├─ fixtures   81 kept, 25 excluded
├─ window     2026-05-29 .. 2026-08-09
├─ cadence    median 3.0 min (p10 3.0, p90 3.0) -> step 3 minutes
├─ depth      median first tick T-225 min, 66 snapshots pre-KO
├─ markets    9 types
└─ slates     12 settlement windows
```

* **G1 corpus size 81/81 PASS.** Every Ireland fixture with a book entered the corpus — the
  loader's count agrees exactly with a direct SQL count, which is the check that the resolver
  and the database agree about what exists.
* **G2 market breadth 9.0 median PASS.**
* **G3 cadence PASS, emphatically.** Median 3.0 min with p10 = p90 = 3.0. The collector is a
  metronome. The table is named `order_book_1m`; **the true cadence is 3 minutes**, matching the
  ScottishLower slate. Any grid finer than 3 min re-reads the same tick.
* Excluded 25 = 14 `no_markets` + 10 `absent_from_crosswalk` + 1 `not_verified` — these are the
  fixtures played before collection started (2026-05-28) plus stragglers. Expected, not a defect.
* Crosswalk resolved 81/81, all `is_verified`. **`MatchMetaCrosswalk` is not dead for this
  corpus** — the module docstring's "0% after the job stopped" was measured on a ScottishUpper
  slate on 2026-08-07, after the job's last write (2026-08-04). Everything here predates that.
  `LiveNameMatch` is deliberately absent from the replay resolver chain.

### CORRECTION 1 — pre-kickoff depth. The mean lied.

First reading was "avg first tick T-379 min (79) / T-591 min (718)". Those are **means**, dragged
by a 48-hour outlier. Measured distribution over the 81:

```
min 78 | p10 133 | p25 199 | median 225 | p75 334 | p90 545 | max 2878   (minutes before KO)

  >= T-60   min : 81 / 81       >= T-240  min : 37 / 81
  >= T-120  min : 73 / 81       >= T-360  min : 15 / 81
  >= T-180  min : 66 / 81       >= T-720  min :  8 / 81
```

**Consequence: the planned T-360 grid is wrong.** It would be honoured by 15 of 81 fixtures and
would spend 4x the snapshots producing `no quotes retrieved` blocks for the other 66.

**Grid is now T-180** (66/81 = 81% honoured), adaptive: 15-min steps T-180 → T-60, then the
measured 3-min cadence T-60 → KO. ~28 snapshots per slate x 12 slates = ~336 `match_day` calls
per arm. `recommend_grid(corpus; coverage = 0.80)` derives this from the quantile rather than
from intuition, so it re-derives correctly if the corpus grows.

### CORRECTION 2 — `market_matched` is a property of the collector, not the exchange.

The matched-volume-by-time-to-kickoff profile first reported (£8 at T-24h rising to £10.6k at
the off) was computed with `avg()`, which skips NULLs. Measured properly:

| | rows |
|---|---|
| pre-KO rows (3 main market types) | 33,954 |
| with both bid and ask | 33,790 (**99.5%**) |
| with top-of-book size | 33,940 (**100%** of rows with a bid) |
| with `market_matched` | 8,085 (**23.8%**) |

`market_matched` and `total_matched` are **NULL on every Ireland fixture before 2026-08-02** and
populated after. That profile therefore described only the 18 August fixtures.

**Consequences:**
1. A `MinMatched` liquidity gate would silently apply to 18 of 81 fixtures. **Do not build the
   WP6 liquidity filter on `market_matched`** — build it on top-of-book size, which is 100%
   populated.
2. Price, spread and drift work is unaffected and runs on the full corpus. The high-power
   estimator (`PriceDrift`, ~100k observations) is safe.

### CORRECTION 3 — the timing tradeoff is about SIZE, not SPREAD.

Re-measured on all 81 fixtures, medians (not means), two-sided quotes only:

| mins to KO | 0–5 | 5–15 | 15–30 | 30–60 | 60–120 | 120–180 | 180–240 | >240 |
|---|---|---|---|---|---|---|---|---|
| **MATCH_ODDS** spread | 2.27% | 2.38% | 2.60% | 2.86% | 2.80% | 2.86% | 2.70% | 4.23% |
| **MATCH_ODDS** top size | £7,641 | £7,980 | £6,544 | £3,517 | £2,661 | £2,229 | £1,906 | £2,071 |
| **O/U 2.5** spread | 1.91% | 1.90% | 2.08% | 1.94% | 2.11% | 1.86% | 1.98% | 3.32% |
| **O/U 2.5** top size | £5,841 | £4,614 | £4,933 | £3,823 | £2,406 | £2,000 | £2,195 | £2,000 |
| **BTTS** spread | 2.94% | 2.83% | 2.97% | 2.97% | 3.59% | 3.81% | 4.13% | 5.88% |

This overturns the shape of the original hypothesis:

* **Spread barely moves inside T-240.** MATCH_ODDS improves 2.86% → 2.27% from T-60 to the off —
  0.6 percentage points, not the 4.6% → 2.6% first reported (that 4.6% was the 8–24h bucket,
  dominated by the August subset and by garbage far-out quotes).
* **O/U 2.5 spread is FLAT** at ~1.9–2.1% across the entire T-240 → KO window. For the tightest
  and most-tradeable totals market there is essentially **no execution-cost argument for
  waiting**.
* **Top-of-book size is where the real gradient is**: MATCH_ODDS roughly quadruples, £1,906 at
  T-240 → £7,641 at the off.

**Revised pre-registration for WP4.** The cost of entering early is *capacity*, not *price*. So:

> H1. At small stake sizes (top-of-book absorbs the whole order), entry time should have
>     **little effect on realised price**, and the optimum is driven by model-edge decay, not by
>     execution cost — pushing the optimum EARLIER than the T-120..T-30 band first predicted.
> H2. The size gradient should bind only for stakes above ~£2,000 per leg, which a paper-trading
>     bankroll never reaches. Test by re-running the fill-cost estimator at several notional
>     stake sizes; expect the curves to separate only at the top.
> H3. BTTS is the exception — its spread genuinely does widen with lead time (2.9% → 4.1%), so
>     if any market rewards waiting it is that one.

### CORRECTION 4 — 12 slates, not ~20. Path metrics are out.

The corpus is 12 settlement windows, two of which carry 2–3 fixtures:

```
05-29:10  06-12: 9  06-19:10  06-22: 2  06-26: 9  07-03: 9
07-10: 7  07-31: 7  08-02: 3  08-03: 5  08-07: 8  08-09: 2
```

Collection has gaps at 06-05, 07-17 and 07-24. `Portfolio.simulate` compounds once per slate, so
this corpus supports **12 compounding steps**.

**Consequence: drawdown-path metrics are not usable here.** Max drawdown, Calmar, Ulcer, Burke
and Sterling over 12 points are noise. They will still be computed (they are free) but must not
be reported as evidence. The usable metrics on this corpus are the ledger-level ones —
`PriceDrift`, `ClosingLineValue`, `FillCost`, `BernoulliGammaHurdle` ROI/growth — plus a
match-clustered ROI interval. This reinforces the estimator hierarchy the stream was designed
around and is the single strongest argument for having built Layer-2-specific metrics rather
than reusing the wealth metrics alone.

### G4 PASS — and it inverts which league is better instrumented

Ran against freshly loaded DataStores. **718 passes**, and the surprise is that the *second* tier
is the better-instrumented one.

xG presence, from `ds.statistics` filtered to `period == "ALL"` (matches with a stats row):

| season | 79 (Premier) | 718 (First Div) |
|---|---|---|
| 2021 | 0 / 180 | 0 / 135 |
| 2022 | 0 / 180 | 0 / 144 |
| 2023 | **0 / 180** | 179 / 180 |
| 2024 | **0 / 180** | 178 / 180 |
| 2025 | 176 / 180 | 177 / 180 |
| 2026 | 129 / 134 | 124 / 135 |

Player ratings (`ds.lineups.rating`) run ~72–75% for **both** leagues from 2023, and 0% before.

* **718 G4: PASS.** xG 92–99% across 2023+, ratings ~68–72%, and the single present-but-zero xG
  (`wexford-fc v kerry-fc`, 2024-10-11, away xG 0.0 in a 4–0) is already neutralised in src —
  `outfield_xg_smile_double_poisson.jl:196` floors xG at `1e-3` before the Gamma pillar. The
  -Inf initialisation trap is closed.
* **79 carries no statistics at all before 2025.** With the validated
  `target_seasons = ["2025","2026"], history_seasons = 2`, tournament 79's history window
  (2023–24) contributes **goals only** — the xG pillar is silent there — while 718's history
  window contributes full xG.

**Consequence.** The two leagues are trained on materially different information under the same
config. This is not new breakage — it is exactly the config r21 validated and won with — but it
means:

1. Cross-league pooling of Layer-2 results must state the asymmetry rather than average over it.
2. **718 is the stronger corpus on both counts**: 43 order-book fixtures to 79's 38, *and* two
   extra seasons of xG history. Where the two disagree, 718 is the better-evidenced answer. This
   is the opposite of the usual assumption that the top tier has the better data.
3. Do **not** re-tune `history_seasons` to chase 79's xG. That is a Layer-1 decision and
   re-opening it inside a Layer-2 stream is how a clean result gets confounded.

**Trap found:** `ds.matches.has_xg` is `false` for **every season of 79, including 2025–26**,
where xG demonstrably exists in `ds.statistics`. The flag is stale — gate on the statistics rows,
never on `has_xg`.

**Trap found:** loading `IrelandFirstDivision` silently rebuilt its DataStore (cache 110h old
against a 48h TTL). This is trap T1 from the matchday ARCHITECTURE and the documented root cause
of the fold-selection defect: a store that grows between training and post-processing makes
`extract_oos_predictions` mis-pair folds. The store must be pinned before WP2 training and the
`length(boundaries) == length(training_results)` assertion must hold at every use.

### WP0 verdict

**G1 PASS** (81/81) · **G2 PASS** (9.0 median markets) · **G3 PASS** (3.0 min, p10 = p90) ·
**G4 PASS** (718 carries the engine).

Carried into WP3: lookback **T-180**, coarse step 15 min to T-60, fine step **3 min** to KO,
~28 snapshots/slate × 12 slates ≈ 336 `match_day` calls per arm.

### Still open after WP0

* `MatchDay._query` opens a connection per call; the corpus build is 92s of mostly connection
  setup. Worth a batched read if the corpus is rebuilt often.
* DataStore pinning for WP2 (see the T1 trap above).

---

## 2026-08-11 — WP1 apparatus built and gated. 96/96.

`r01_apparatus_smoke.jl`, 2.9s, **no DB, no cache, no trained experiment** — it drives the real
`Portfolio.simulate` over `MatchBook`s built from the hand-rolled score grid in
`test/portfolio_tests.jl`. That is what makes a WP1 gate possible before WP2 and WP3 exist.

Files: `l01_l2_experiment.jl` (run side), `l02_l2_ledger.jl` (judge side),
`l03_l2_metrics.jl` (metrics), `r01_apparatus_smoke.jl` (gate).

### The gate that matters: A2

The apparatus makes exactly one claim that can be wrong *silently* — that a tearsheet row means
the same thing as a `Portfolio.report`. A2 asserts it against the real system, not a
reimplementation:

* `l2_curve(ledger) ≈ Trajectory.bankroll` — exact, elementwise
* `l2_path_metrics` reproduces `Portfolio.path_metrics` on all seven shared fields
* `BackTesting.compute_metric(m, l2_curve(...))` reproduces `Portfolio.report(traj, ms)`

**This is where the compounding bug would have surfaced.** `BackTesting._compute_wealth_metrics`
builds its curve as `cumsum(pnl)` — arithmetic accumulation, correct for flat staking and wrong
for a fractional-Kelly system that compounds once per settlement window. A Sharpe or Calmar off
`cumsum` measures a strategy nobody ran, and nothing complains. `l2_curve` compounds the way
`simulate` does; A2 is the proof, and A3 additionally shows a shuffled ledger still yields the
chronological curve (final wealth is order-invariant, every drawdown statistic is not).

### Design decisions worth remembering

**Three-tier cost model.** Snapshots are expensive (DB + latent extraction), staking is cheap,
entry selection is free. Entry rules are therefore pure selections over an already-replayed
ledger, not separate replays. Without this a trust sweep would re-run ~800 `match_day` calls for
an answer a `groupby` away — the same insight that lets `r02_policy_sweep` sweep 24 policies in
0.9s.

**The entry-assembly trap, and its repair.** `FixedLead` and `AtClose` fire a whole slate at one
instant, so the Kelly solve, drawdown factor and exposure cap all still hold. `FirstQualifying`
and `BestPrice` assemble legs from *different* instants, where those constraints were solved
per-snapshot — so the assembled book can breach the cap that made each part legal. Every
individual number is correct, which is why it is invisible. `recap_slates!` scales the slate
back, preserving relative Kelly weights (A13 asserts a single scale factor across legs and that
single-instant rules never trip it). The weights stay only *locally* optimal in those two rules;
that is inherent to picking across time, and is why `BestPrice` is labelled an oracle.

**Units.** `:stake` is a bankroll FRACTION (matching `Trajectory.bets.stake` and `stake_sheet`'s
`:frac`), `:payoff` is unit payoff, `:pnl = stake * payoff`. Currency lives in `:stake_cash` /
`:pnl_cash` and never enters a metric. A1 asserts the fractions are in `[0,1]`.

### Metrics

| metric | obs on this corpus | reads |
|---|---|---|
| `PriceDrift` | ~100k | nothing but the book — `log(odds_close/odds_entry)`, backer sign |
| `ClosingLineValue` | ~1k | the model's own picks vs the de-vigged close |
| `FillCost` | ~100k | VWAP down the ladder at several notional sizes |
| `BernoulliGammaHurdle` | ~500 | the outcome (Layer 1's, unchanged) |

`FillCost` takes a *vector* of stake sizes deliberately: WP0 found the timing gradient is in
capacity, not price, so the pre-registered prediction is that these curves are flat in
`:entry_bucket` at small stakes and separate only at large ones. One stake size would make that
untestable — the shape across sizes is the result.

Every interval resamples **matches**, matching `Portfolio.bootstrap_roi`'s scheme, B and seed so
a CLV interval and an ROI interval from the same slice are comparable rather than merely
similar. Nine markets on a fixture share a scoreline and ~28 snapshots share a book; unclustered
intervals would shrink by roughly `sqrt(28)` and manufacture significance.

### Two bugs found by the gate

1. `merge(::NamedTuple, ::AbstractDict)` is a **two-argument** method — the varargs form is
   NamedTuple-to-NamedTuple. Merging stats + path + ci + wealth + dist in one call is a
   `MethodError`. Chained pairwise, which is why `BackTesting` does the same.
2. A7's tolerance was tighter than `ClosingLineValue`'s own 5-dp rounding, so it failed on the
   rounding rather than the arithmetic. Test bug, not code bug.

### Health warning wired in, not just documented (WP1)

`l2_path_metrics` returns `path_reliable = n_slates >= 25`, and `path_warning(tearsheet)` returns
the sentence that must accompany any drawdown column. On this corpus (12 slates) it always
fires. A11 asserts it does.

---

## 2026-08-11 — WP2 setup. A forced asymmetry between the two leagues.

### The blocker: 718 has no SofaScore O/U ladder

r21 trained the market pillar on SofaScore `ds.odds` and evaluated CLV against Betfair, keeping
the feeds cleanly separated. Measured, that is impossible for 718:

| matches with market, by season | 79 SofaScore | 718 SofaScore | 718 Betfair |
|---|---|---|---|
| 2023 O/U | 174/180 | **0** | 144/180 |
| 2024 O/U | 180/180 | **0** | 144/180 |
| 2025 O/U | 180/180 | **0** | 144/180 |
| 2025 1X2 | 180/180 | **27/180** | 144/180 |
| 2026 O/U | 134/134 | **0** | 80/135 |

718's SofaScore feed carries **1X2 only, from 2022 onward**. `MarketSmileFeature(Kmax=4)` inverts
the O/U ladder per strike, so on that feed it has nothing to invert and `smile_weight = 0.4`
anchors to an empty feature. That is a **silently mis-specified** model, not a degraded one —
it would train, converge, and be wrong.

Decision (user's, with the alternatives priced): **79 stays r21-exact on SofaScore; 718 trains on
Betfair.** The alternative of moving both to Betfair would have made them comparable but broken
bit-identity with r21/b21 for 79; dropping 718 would have halved the corpus.

### The window is not the close, and that is load-bearing

`market_extractors.jl:71` reads `prob_fair_close`. So whatever is passed as `close_window` **is**
the training pillar, whatever it is named. Training on the actual close would be circular: the
Layer-2 evaluation measures CLV *against* the close, so the model would be scored against its own
input.

Chosen: `close_window = (-360, -180)`, which ends exactly where the WP4 decision grid begins.
Measured coverage cost of that principle:

| close_window | 718 matches with an O/U ladder (2023–26) |
|---|---|
| (−1440, −360) | 71 — too thin |
| (−720, −360) | 232 |
| **(−360, −180)** | **276** ← chosen |
| (−180, −60) | 312 — but overlaps the decision window |

### The caveat this leaves

79's pillar is SofaScore at ~100%; 718's is Betfair at ~54%. **WP4–WP6 must report per league,
never pooled** — a pooled difference would be confounded by the feed, not by the league. Combined
with the WP0 finding that 79 has no xG before 2025 while 718 does, the two leagues now differ on
*both* pillars, in *opposite* directions:

| | 79 | 718 |
|---|---|---|
| xG history (2023–24) | absent | present |
| market pillar | SofaScore, dense | Betfair, ~54% |
| order-book fixtures | 38 | 43 |

Neither is uniformly better instrumented. That is a reason to read them as two independent
replications rather than as one pooled corpus — which, at this sample size, is arguably the more
honest design anyway.

### Job shape (dry-run, before training)

* **31 folds per league** (2025 runs to biweek 17, 2026 to biweek 12, plus a baseline fold each).
* Order-book window needs 2026 biweeks **8–12**; folds reach **0–12** → G-C will pass.
* 124 chains per league (31 × 4) through the queued NUTS global queue.
* Betfair pillar for 718: 2,534 rows over 518 matches.

### DataStores are now pinned

`load_datastore_cached` has a 48h TTL and rebuilt 718 silently during the WP0 preflight (trap
T1). A store that grows between training and `extract_oos_predictions` makes the latter rebuild
boundaries from the NEW store and zip them positionally against the OLD `training_results` —
folds mis-pair with no error. WP2 serialises the exact stores it trained on, including the
pillar-swapped 718, to `data/l2_ireland_engines/ds_*.jls`. **WP3 must load those, not the cache.**

---

## WP2 results — 2026-08-12

Both engines trained: `l2_ire79_sup40_sw40_20260811_203336`, `l2_ire718_sup40_sw40_20260812_003455`.

| gate | ire79 | ire718 |
|---|---|---|
| G-A splits kept / built | **31 / 31** | **31 / 31** |
| G-B max R-hat, all folds | 1.6160 (`ha.σ_γ`) ✗ | **1.0099** (`kappa`) ✓ |
| G-B max R-hat, corpus window | **1.0097** ✓ | **1.0060** ✓ |
| G-C folds reach 2026 biweeks 8–12 | ✓ (reach 0–12) | ✓ |

### G-B: the global failure does not touch this study

79's 1.616 looks alarming and is not. Median R-hat is 1.0012 and **37 of 1,059 parameters (3.5%)
exceed 1.01 — all inside three folds**: (2025, biweek 0), (2025, biweek 5), (2026, biweek 6).
The order-book corpus lives in **2026 biweeks 8–12**, and `MatchDay.select_split` picks exactly
one fold per fixture by date, so no non-converged fold is ever asked for a latent. Restricted to
the 165 window parameters, **max R-hat is 1.0097 on 79 and 1.0060 on 718, with zero exceedances
in either league**.

`gate_experiment` now reports both figures. The window one is the gate; the global one is kept
because "converges only inside its window" is a fact worth carrying, not hiding.

The failing parameters are the usual hierarchical-funnel suspects — `ha.σ_γ`, `kap.σ_κ`,
`σ_sup`, `σ_smile`, `kappa` — in the folds with the least data. On a re-run they are the ones a
non-centred parameterisation would fix; nothing here needs fixing for WP3.

### Structural finding: this engine has NO dispersion parameter

`Diagnostics.extract_chains` warns `index disp.log_r not found` once per fold, for both leagues.
The cause is not a naming drift — inspecting the chain directly, its 69 parameters are:

```
ha.γ_base, ha.γ_team_raw[], ha.σ_γ, inter.μ_base[], inter.raw_month[], inter.σ_month,
kap.κ_base, kap.κ_team_raw[], kap.σ_κ, log_φ[], p_dyn.w_{G,Outfield}_{att,def},
ν_xg, σ_smile, σ_sup    (+ NUTS internals)
```

There is **no dispersion variable at all**. `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecay`
is a pure double-Poisson on the goals side: variance is pinned to the mean, and the
`dispersion_config = HomeAwayDispersion()` we pass — copied from the r21 winning cell — is inert.
It is harmless but misleading, and it is worth being explicit that r21's grid never varied it.

**Why this matters to Layer 2.** The 2026-08-07 live serving run recorded "model 1X2 dispersion
half the market's", and the 2026-08-08 ScottishLower replay measured the same 0.5× ratio. That is
now explained rather than merely observed: the engine *cannot* widen its scoreline distribution
beyond Poisson, because it has no parameter with which to do so. The under-dispersion is
structural, not a fitting failure — so it is not something a Layer-2 staking policy can calibrate
away, and it is a standing reason to expect the trust/shrinkage machinery to be doing real work
rather than correcting noise. Consistent with [[no-pregame-intensity-smile]]: pregame FT totals
really are ~Poisson, so the constraint is closer to right than wrong on totals, and the 1X2
under-dispersion is the place it bites.

Registered as an observation, not a change: swapping the dispersion config is an L1 question and
belongs in the split-market-pillar stream, not here.

---

## WP3 results — 2026-08-12

G1 passed on the first run and that is the one that mattered: **`stake_snapshots` reproduces
`match_day`'s sheet row for row** (30 rows, every key column and every numeric column) at the
same instant with the same spec. The Tier-1/Tier-2 decomposition is licensed; WP5 and WP6 can
re-stake without touching the database.

Everything else on the first run was wrong, in three separate ways, and all three were wrong in
the direction of *looking fine*.

### 1. The frozen/live arms are the same arm

The plan, and l04's original header, argued that `src_sup40_sw40` must have latents that move
with `as_of` because it is player-level and `RatingsFromTracker` reacts to the announced XI.

Measured, T−120 vs kick-off, 2026-05-29 slate:

| | |
|---|---|
| columns compared | `true_xg_h/a, θ_1..3, λ_h, λ_a, λ_tot, ρ, φ` |
| fixtures | 5 |
| posterior draws per cell | 3,200 |
| **max abs Δ** | **0.0 — bit-identical** |

Serving latents are a pure function of `(fixture, split)`. The market pillar is a *training*-time
regulariser and never re-enters at serve time, and `replay_spec` wires `lineups = SourceChain()`
with no sources, so no XI is ever fetched. **Player-level in the posterior does not imply
clock-dependent at serve time.**

Consequences:
* `:live` is pure cost. WP4 runs `:frozen` only — halves its Tier-1 bill.
* **100% of movement in a replay is the book** — the clean reading l04's header claimed we had
  forfeited. H1–H3 are cleaner than expected.
* **H4 (value of team news) is retired, not tested.** `live − frozen` is identically zero by
  construction. Testing it needs a point-in-time lineup source this archive does not contain.

### 2. `latents_invariant` cannot fail

`matchday_2026_08_08/l02_slate_replay.jl:229` selects columns with `eltype(col) <: Number`.
Latent columns hold posterior draws — `Vector{Float64}` cells inside `Any`-eltype columns — so
the filter matches nothing, the comparison loop never runs, and it returns `(true, 0.0, :none)`
regardless of what the latents did. It reported a pass here on frames it never compared.

It is replaced by `latent_delta` (l04), which compares cell contents and reports `n_compared` so
a vacuous pass is visible. G2 now asserts `n_compared > 0` before asserting invariance, and
additionally pins the old helper's vacuity so a future reader does not trust it.

**This is worth carrying beyond this stream.** The 2026-08-08 ScottishLower replay's claim that
its funnel engine's latents were clock-invariant rested on this helper. The claim may well be
true — the funnel engine is team-level — but it was never actually verified.

### 3. First tick ≠ tradeable book

WP0 reported first-tick leads of T−334 for the 2026-05-29 slate and a median cadence of 3.0 min,
and `recommend_grid` set the lookback from that. Both are true and together they are misleading.
The readiness gate, at every instant of the grid:

| lead | cards | quotes | passing | reason |
|---|---|---|---|---|
| T−180 | 5 | 85 | **0** | stale book: newest tick 2h21m before as_of (limit 10m) |
| T−150 | 5 | 85 | **0** | stale book: newest tick 2h51m before as_of |
| T−120 | 5 | 85 | 5 | — |
| T−90 … KO | 5 | 85 | 5 | — |

The feed produces an early isolated tick and then goes **silent for over two hours**. Median
cadence of 3 min is a property of the dense tail, not of the window. So the tradeable book on
this slate begins around **T−120**, not T−334.

This is the failure mode that would have been hardest to catch downstream: an entry rule aimed at
T−180 returns zero legs, and zero legs in a tearsheet reads as *"the model finds no edge early"*
rather than *"there was no book"*. Worse, `FixedLead` snaps to the nearest snapshot, so it would
not even have gone empty — it would have quietly reported T−120's numbers under a T−180 label.

Fixed in three places:
* `_fixture_coverage` now reports `live_lead_min` (walk back from the close, stop at the first
  gap wider than `MAX_BOOK_AGE_MIN = 10`) and `max_gap_min` alongside `first_lead_min`.
* `recommend_grid` derives `lookback` from `live_lead_min`, and reports what the first-tick
  quantile *would* have said so the two can be compared.
* r04 prints a coverage-by-lead table **before** any wealth table, and prints `med_lead` beside
  every entry rule so two rungs that resolved to the same instant are visible as such.

### 4. A widened struct field routed staking to the wrong method

`L2Snapshot.fixtures` was declared `Dict{Int,Any}`. `Portfolio.stake_sheet` dispatches on
`Dict{Int,FixtureInfo}` to reach its live-fixture method; any other type reaches the DataStore
method, whose own docstring says it "returns an empty sheet for any fixture that has not been
played". Widening the field silently selected the method that cannot price an unplayed fixture —
which is the only kind of fixture this stream cares about.

It threw only because a `Dict` has no `.matches`. Had the fallback been merely wrong rather than
invalid, the replay would have returned empty sheets for exactly the fixtures under study. The
field is now concrete and `stake_snapshots` asserts the value type before its first call.

### Standing risk this leaves

Three of the four defects above were *silent-by-construction*: a vacuous test, a mis-labelled
lead, and a wrong-method dispatch. Only one threw. The pattern is that this pipeline fails by
returning plausible empty or unchanged results rather than by erroring, so every WP from here
reports a **precondition table** (what was priceable, how many cells were compared, which method
was dispatched) next to every result table.

### 5. Every lead was measured from midnight

Found only because G5 printed medians and they were **negative**: `AtClose −1125 min`.

`Portfolio.FixtureInfo` is `@NamedTuple{date::Date, score}` — it has no kick-off time. The lead
calculation used `hasproperty(fi, :kickoff) ? fi.kickoff : DateTime(slate_day)`, so the fallback
fired on every row and the origin was **midnight** rather than the actual 18:45 kick-off.

Nothing downstream complained. Negative leads simply put every leg in one entry bucket and made
`FixedLead` snap against a target measured from the wrong origin — the entire WP4 entry-time axis
would have been wrong, through a gate that passed 583/583 around it.

`L2Snapshots` now carries `kickoffs` from the corpus's own `Fixture` objects, `kickoff_of` throws
rather than defaulting, and `stake_snapshots` refuses a ledger containing a negative lead. After
the fix: leads span **T−375 … T−0** across 7 buckets, with `AtClose` at 0, `FixedLead(90m)` at 90
and `FirstQualifying(0.02)` at 135 — correctly ordered.

### 6. Deep entry buckets are a biased subsample

`adaptive_grid` anchors on the **earliest** kick-off in a slate, deliberately: anchoring on the
latest would let `ExplicitFixtures` shrink the slate mid-trace. But 79's slates are staggered by
up to **4 hours**, so a late fixture sees leads of `lookback + 240` while the earliest tops out
at `lookback` — hence a deepest lead of 375 against a 136-minute lookback.

So the deep buckets contain only late-kick-off fixtures. **A per-bucket ROI difference between
"120–180m" and "0–5m" is partly a difference between two sets of matches.** `reading_5_coverage`
prints `fixtures_priceable` per bucket so this is visible where it bites.

### WP3 gate: 591/591

| gate | result |
|---|---|
| G1 decomposition reproduces `match_day` row for row | **28/28 rows** |
| G2 latent arms, `n_compared > 0` asserted first | invariant, 50 cells compared |
| G3 grid spacing is exactly {15, 3} min | pass |
| G3b grid does not reach past the tradeable book | T−135 start vs T−143 deepest live |
| G4 closing coherence, 304 complete groups | all overround > 1 |
| G5 entry rules on real data | pass, correctly ordered |

Carried forward for WP4: the oracle beats the close on **144/267 legs, mean gain 2.04%**. That is
the number `RandomEntry` has to be subtracted from.

### Threading

Tier 1 was a serial loop and took ~23 min per league. Slates are independent, so `build_snapshots`
now runs them under `Threads.@threads` (16 threads on the server) with one buffer per slate,
concatenated in slate order so the result is deterministic. **23 min → 5m50s** for the full gate;
the remaining serial part is the corpus build and G1/G2 prologue.

Two pieces of shared state, both checked rather than assumed:
* `MatchDay._query` opens and closes its own connection per call (`db.jl:26`) — no shared handle,
  which is why this parallelises at all.
* `_CARD_META` is a module-level `IdDict` mutated through `get!`, so concurrent access can corrupt
  it. The gate loop is held under a `ReentrantLock`; `price_cards` and `matchday_latents` — the
  actual cost — stay outside it. `clear_card_meta!` moved to once *before* the parallel region.

`threaded = false` restores the serial path, which is the cheapest way to rule threading out if a
result ever looks odd.

---

## WP4 results — 2026-08-12

**Headline: enter as late as possible. The execution decision is an order of magnitude larger
than the edge being executed.**

### The two estimators disagreed, and the low-power one was read first

| entry rule | 79 ROI | 79 CLV | 718 ROI | 718 CLV |
|---|---|---|---|---|
| AtClose | 9.29% | **−0.0051** | −2.98% | **−0.0072** |
| FixedLead(5m) | 10.68% | −0.0057 | −2.56% | −0.0084 |
| FixedLead(15m) | 11.11% | −0.0051 | −3.34% | −0.0102 |
| FixedLead(30m) | 12.29% | −0.0082 | +1.27% | −0.0124 |
| FixedLead(60m) | 12.45% | −0.0118 | +3.01% | −0.0127 |
| FixedLead(90m) | 8.03% | −0.0143 | +1.46% | −0.0131 |
| FixedLead(120m) | 7.94% | −0.0139 | +1.02% | −0.0147 |
| RandomEntry (3 seeds) | ~11.6% | ~−0.0112 | ~+1.0% | ~−0.0099 |
| BestPrice (oracle) | 13.10% | +0.0152 | +0.56% | +0.0119 |

ROI says *AtClose is the worst rule in both leagues*. CLV says *AtClose is the best rule in both
leagues*, monotonically. **CLV wins**, and the pre-registration said so before either ran: the ROI
confidence intervals here are roughly **±40 percentage points** (79 AtClose: [−20.9, +43.2]), so
the ROI ordering is a coin toss. CLV is measured on ~250 legs with a consistent monotone gradient
reproduced in two leagues that share no fixtures and were fitted on different market pillars.

This is also the answer to the H3 oracle trap. On ROI, `RandomEntry` beat `AtClose` in both
leagues and `reading_4_oracle` reported "a real drift toward kickoff (H3 fails)". On CLV the
control is **worse** than the close (−0.0112 vs −0.0051 on 79; −0.0099 vs −0.0072 on 718), so the
oracle's +2.04% price gain is hindsight, exactly as pre-registered. The verdict function now runs
on CLV; the ROI figures are carried alongside so the disagreement stays visible.

### H1 rejected — waiting is not free, it is the whole game

The revised H1 predicted flat drift because WP0 measured spread as nearly flat pre-kickoff. Wrong.
Entering 120 min out costs **0.9 pp of CLV per leg on 79** (−0.0051 → −0.0139) and **0.75 pp on
718** (−0.0072 → −0.0147). Against a `hurdle_G_emp` of ~0.0003 at the close, the timing decision
is roughly **30× the size of the modelled edge**.

`PriceDrift` corroborates from ~19k quotes with no model in the loop: mean drift is positive and
CI-excludes-zero in the 30–120 min buckets of both leagues (79: +0.0020 [0.0005, 0.0037] and
+0.0034 [0.0012, 0.0056]; 718: +0.0038 [0.0002, 0.0088] and +0.0057 [0.0026, 0.0095]), and ~0
with a CI spanning zero inside 15 min. Positive drift means quoted odds LENGTHEN toward the
close — which is what overround compression does to every selection at once as liquidity arrives.

### The apparent contradiction, resolved

Average quotes lengthen toward the close (`PriceDrift` > 0), yet entering at the close gives the
best CLV. Both are true and they are the same fact: the overround compresses, so a *later* entry
buys the same opinion at a *smaller* margin. `PriceDrift` measures the nominal price of an
arbitrary selection; CLV measures what the model's chosen legs pay after de-vigging. The margin
is the bridge.

### The model has no positive CLV anywhere

Every executable rule is negative on every family in both leagues. `clv_pos` at the close is
**8.8% (79) / 10.2% (718)** — the model's picks are essentially never on the right side of the
closing line. `AtClose` CLV of −0.005 to −0.007 is about what paying the half-spread costs.

**This is the finding to carry into WP5 and WP6.** There is no demonstrated edge against the
closing line at any entry time, so the staking layer is not amplifying a measured edge — it is
allocating against a model whose advantage, if any, is smaller than the spread it pays. It is
consistent with the standing note that the model loses narrowly to the close on 1X2.

### Sampling caveats that bound all of the above

* Deep entry buckets are populated only by late-kick-off fixtures in staggered slates (§WP3.6).
* 718's `120-180m` drift bucket has **34 quotes** — ignore it.
* 718's `FirstQualifying(0.020)` returned NaN CLV: one family with `clv_n = 0` propagating through
  a mean of means. Fixed in `clv_by_rule`, which now weights by leg count and drops NaN families.

### H2 never ran

`FillCost` is not in `_default_dist_metrics()` (that is `BernoulliGammaHurdle` alone), so the
capacity axis — the one thing the clock was supposed to buy — was never measured. `reading_6_fill`
now requests it explicitly and the ledger is saved so it can be re-asked without rebuilding
Tier 1. **Until it runs, "enter late" rests on price alone**, and the counter-argument that late
books are deeper is untested.

### H2 — capacity, and the finding that matters most

`FillCost` run per entry bucket (79 / 718), on the exact rows being staked:

| bucket | 79 slip@£100 | 79 unfillable@£100 | 718 slip@£100 | 718 unfillable@£100 |
|---|---|---|---|---|
| 0–5m | 0.80% | **54.1%** | 1.32% | 72.9% |
| 5–15m | 1.01% | 61.8% | 1.65% | 78.1% |
| 15–30m | 1.17% | 68.6% | 2.41% | 76.6% |
| 30–60m | 1.42% | 80.9% | 2.44% | 91.0% |
| 60–120m | 1.41% | 83.6% | 2.24% | 91.0% |
| 120–180m | 1.60% | 85.1% | 2.19% | 90.0% |

**H2 holds as stated** — capacity is monotone in entry time, and later is better. Slippage on a
£100 stake roughly **doubles** from the close to T−120 in both leagues.

But H2 also predicted `FillCost` would be *the only* estimator with a monotone gradient, on the
theory that price and capacity would trade off. They do not. **Price and capacity both say enter
at the close.** There is no interior optimum, and the plan's expected "T−120 to T−30" is wrong on
both axes independently.

#### The Irish book cannot absorb meaningful size

This is the practical headline and it was not on the plan's radar at all:

* **£1,000 per leg is unfillable ~100% of the time, in every bucket, in both leagues.**
* **£100 per leg is unfillable on 54% of legs at the best moment** (79) and 73% (718).
* **£10 per leg** still fails on 13% (79) / 21% (718) of legs at the close.

Measured directly on the staked rows: top-of-book back size has a median of **£41**, a 25th
percentile of **£11.70**, and a 5th percentile of **£1.04**. Summed across the whole visible
ladder the median is **£193**.

So the binding constraint on this stream is not the model, the trust weight, or the entry clock —
it is that the venue does not hold enough money. WP5's staking questions are being asked about
sizes the book cannot take.

#### Correction 4 to WP0: top-of-book size

WP0 (Correction 3) reported top-of-book size as "£1,906 at T−60 rising to £7,641". That is wrong
by roughly 50×. Measured directly against `betfair_live.order_book_1m`, raw `bid_volumes[1]` has
a median of 130,000 units, and the archive stores volumes **×10000** — i.e. **£13.00** across all
rows and markets, and £41 median on the more liquid selections the model actually stakes. The
earlier figure most likely summed across ladder levels and runners, or read `market_matched`
(cumulative traded volume, a different quantity — the same column that caused Correction 2).

#### Caveat that bounds the shortfall numbers

`betfair_live.order_book_1m` stores **at most 3 ladder levels** (`max(array_length(bid_prices,1))
= 3`, mean 2.87 over 526k rows). Depth beyond level 3 is not recorded, so every `short_*` figure
above is an **upper bound** on true shortfall and every `slip_*` is an upper bound on true cost.
The direction and the ordering across buckets are unaffected — all buckets are truncated the same
way — but the absolute claim "£100 is unfillable on 54% of legs" should be read as "…given the
top three levels the collector kept".

---

## WP5 results — 2026-08-12: market curation, per-line trust, match avoidance

Estimator changed, because WP4 disqualified the other two. **CLV is degenerate at the close** —
`odds_entry == odds_close` there, so `clv = log(odds_close · fair_close)` is exactly minus the
market's margin and carries no information about the model. WP4's per-family CLV table ranks
*spread*, not edge; curating on it would curate toward whichever markets are tightest. ROI cannot
separate 15 families at 267 legs. So:

    skill = logscore(p_model) − logscore(fair_close)     per leg, in nats, clustered by match

### C1 — per-market trust does NOT transfer. Pre-registered falsification, fired.

| | derive on 79 → test 718 | derive on 718 → test 79 |
|---|---|---|
| family skill correlation | **r = −0.647** over 12 families | same pair |
| sign agreement | **6 / 12** (chance) | — |
| held-out ROI, uncurated | −2.98% | +9.29% |
| held-out ROI, curated | −2.98% | **+6.82%** (worse) |

Not merely uncorrelated — *anti*-correlated. O/U 2.5 under: +0.053 (79) → −0.099 (718).
BTTS yes: +0.025 → −0.062. 1X2 away: −0.035 → +0.015. At market level 79 ranks
BTTS ≻ O/U ≻ 1X2 and 718 ranks 1X2 ≻ O/U ≻ BTTS.

The 79→718 direction assigned every family the default 0.25, i.e. a uniform rescale, and held-out
ROI was **identical to 4 significant figures** (−2.98% both). That is the homogeneity property of
`risk_factor` confirmed end-to-end on real data: a flat trust weight cannot change wealth once
`SlateDrawdown` binds.

The anti-correlation is probably not pure noise. 718 runs ~3.2 goals/match against 79's ~2.1, and
the sign flips cluster in the totals families — unders good in 79, overs good in 718. That reads
as a **per-league totals-level bias**, not per-market skill, and it is consistent with the standing
note that IrelandAll pooling mis-prices 718 by −0.47 goals. Level, not market, is the axis.

### C3 — the tail test. Replicated in both leagues.

| claimed disagreement (`p_model − fair_close`) | 79 skill | 79 beats mkt | 718 skill | 718 beats mkt |
|---|---|---|---|---|
| below −5pp | +0.024 (9) | 66.7% | −0.007 (7) | 57.1% |
| −5 .. −2pp | −0.012 (39) | 48.7% | −0.017 (29) | 37.9% |
| **within 2pp** | **+0.0061 (80)** | **53.8%** | **+0.0046 (84)** | **59.5%** |
| +2 .. +5pp | −0.003 (64) | 37.5% | −0.003 (54) | 38.9% |
| **above +5pp** | **−0.0125 (75)** | **37.3%** | **−0.0468 (89)** | **21.3%** |

`within 2pp` is the only positive band in either league, and it is positive in **both**, on the
largest bins, with the tightest intervals ([−0.0007, +0.0142] and [−0.0014, +0.0111]). `above
+5pp` is the worst band in both, beating the market on only 37.3% / **21.3%** of legs.

The asymmetry matters: large *downward* claims are fine, large *upward* claims are not. That is
the optimizer's-curse signature — the book is built from positive-edge selections, so it selects
precisely the legs where the model is most wrongly optimistic.

Filtering on it moves the beat rate from ~40–45% to ~53–55% **in both leagues**. But it does not
buy growth: ROI goes 9.29% → 10.78% on 79 and −2.98% → **−18.31%** on 718. Calibration improves,
money does not. Both are true — log score punishes confident errors that a long price can still
pay for — and per the standing instruction to judge on growth, this is a **calibration finding, not
a staking rule**.

### C2 — longshot: only the far tail replicates

Mid-range bands disagree between leagues. The one consistent fact is the top band: odds > 6.0
beats the market on **28.9% (79) / 14.0% (718)** of legs, negative skill in both. `odds < 6`
improves 79's ROI (9.29 → 11.45) and worsens 718's (−2.98 → −9.12), so again: not a growth rule.

### The number that settles it

Fit one parameter — the weight on the model in `w·p_model + (1−w)·fair_close` — by log loss:

| | n | **w\*** | LL at w\* | LL at w=0 (market) | LL at w=1 (model) |
|---|---|---|---|---|---|
| 79 | 267 | 0.30 | 0.52409 | 0.52491 | 0.52826 |
| 718 | 263 | **0.00** | 0.51729 | 0.51729 | 0.53429 |
| **pooled** | **530** | **0.00** | **0.52113** | **0.52113** | **0.53125** |

**The optimal weight on the model is zero.** Pooled, the log-loss curve is monotone increasing in
`w` — every ounce of model added to the de-vigged Betfair close makes the prediction worse. 79's
interior optimum at w = 0.30 is worth 0.00083 nats/leg, far inside its own skill CI of
[−0.018, +0.012].

One free parameter, 530 legs, replicated in the league it was not fitted on. This is the most
robust estimate in the entire stream and it is the one that matters:

> **On these two leagues, `src_sup40_sw40` adds no information to the closing Betfair price.**

### What this does and does not say

It does **not** say the apparatus is broken — WP1 passes 107/107, WP3 passes 591/591 including a
row-for-row reproduction of `match_day`, and the homogeneity property was confirmed on real data
as a by-product. The Layer-2 system measures correctly. What it measures is that there is nothing
for it to allocate.

It does **not** say the model is bad in absolute terms — it says it is dominated *by the Betfair
close*, which is the strongest available benchmark and one the standing notes already flagged it
loses to narrowly on 1X2.

It does mean **no Layer-2 intervention can help**: entry timing, per-market trust, skip rules and
staking all reallocate a signal, and `w* = 0` says the signal is weakly dominated before any
allocation happens. The binding constraint is Layer 1.

### The one lead worth following

The over-confidence is **directional and asymmetric** (fine below −5pp, bad above +5pp) and the
family ranking flips sign between a 2.1-goal league and a 3.2-goal league along the totals axis.
Both point at a **per-league level bias in λ_tot** rather than at per-market skill — which is an
L1 question (and one the split-market-pillar stream is equipped for), not an L2 one.

---

## WP8 results — 2026-08-12: the full-book test

WP5's `w* = 0` was measured on legs the staking layer chose, selected on `p_model > p_market` —
the same quantity under test. This scores **every quoted selection**, staked or not. The join was
exact: 267/267 (79) and 263/263 (718) staked legs found their counterpart, so the comparison is
clean.

### F1 — FAILED. It is not the selection rule.

Pooled, 1,140 quoted selections over 76 matches:

| set | n | skill | 95% CI | beats mkt | **w\*** |
|---|---|---|---|---|---|
| **ALL quoted** | 1140 | **−0.01017** | [−0.0220, +0.0026] | 43.8% | **0.00** |
| STAKED only | 530 | **−0.01012** | [−0.0225, +0.0036] | 42.5% | **0.00** |
| NOT staked | 610 | **−0.01022** | [−0.0225, +0.0029] | 44.9% | **0.00** |

The three are **identical to three decimal places**. Skill on legs the model never bet is the same
as skill on the legs it did. `w* = 0` on all three subsets, pooled and per league (79's full book
shows `w* = 0.26` worth 0.00061 nats — inside noise; 718 is flat zero throughout).

So the pre-registered world B is ruled out. **This is a Layer-1 problem.** The engine is
uninformative across the whole book, not merely where the staking rule selected it, and no
shrinkage rule, abstention threshold, trust weight, filter or entry clock can change that — they
all reallocate within a book whose every leg is equally uninformative.

### F2 — the curse curve is SYMMETRIC on the full book

| claim (`p_model − p_market`) | n | staked | skill | beats mkt |
|---|---|---|---|---|
| below −5pp | 191 | 16 | **−0.0269** | 41.9% |
| −5 .. −2pp | 220 | 68 | −0.0059 | 43.2% |
| **within 2pp** | 325 | 164 | **+0.0028** | **55.4%** |
| +2 .. +5pp | 212 | 118 | −0.0031 | 39.2% |
| above +5pp | 192 | 164 | **−0.0282** | 31.8% |

This corrects the WP5 reading. On the staked subset the pattern looked **asymmetric** — fine
below −5pp, bad above +5pp — and was written up as an optimizer's-curse signature. On the full
book it is **symmetric**: large disagreements in *either* direction score badly (−0.0269 and
−0.0282, near-identical). The apparent asymmetry was an artefact of staking only ever selecting
the positive side.

The honest statement is therefore stronger and simpler than "the model is over-confident upward":

> **The model's deviations from the de-vigged market are noise.** Its skill is concentrated
> exactly where it agrees with the market — and agreement is worth nothing, because the market is
> freely available.

Even the agreement band's +0.0028 carries a CI of [−0.0021, +0.0081], including zero.

### Why this is mechanically unsurprising

`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` has **four likelihood pillars, two of
which are the market**: C1 anchors `log λ_h − log λ_a` to the market's implied supremacy
(weight 0.4) and C2 anchors the per-strike total intensity to the market-inverted smile
(weight 0.4). With `market_on = true` the engine is a shrinkage-toward-market estimator by
construction.

Its only route to an edge is where pillars A (xG, Gamma) and B (goals, Poisson) pull it *off* the
anchors. The curse curve measures exactly that pull, and finds it is noise in both directions.
The market pillars are carrying the model's accuracy; the model's own contribution is not adding
to it.

### The test that follows directly

Run the `market_on = false` control on the same corpus. If the unanchored engine is much worse,
the anchoring is doing all the work and the L1 signal is weak — which is what everything above
predicts. If it is comparable, then the anchoring is *suppressing* a signal that exists, and the
weights (not the model) are the problem. r21's grid included such a control cell, but it was
judged on proper scoring against held-out matches, never against an executable de-vigged close.

That is an L1 question and belongs in the split-market-pillar stream. **The Layer-2 programme on
this corpus is complete and its answer is that there is nothing here for it to allocate.**
