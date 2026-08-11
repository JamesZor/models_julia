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

### Still open in WP0

* **G4 (718 xG + player ratings preflight) not yet run** — needs the DataStores loaded.
* `MatchDay._query` opens a connection per call; the corpus build is 92s of mostly connection
  setup. Worth a batched read if the corpus is rebuilt often.
