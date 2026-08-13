# Layer-2 on the Ireland order book — verdict

**Stream:** `current_development/orderbook_layer2/`
**Engine:** `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`, `supremacy_weight = 0.4`,
`smile_weight = 0.4` (the r21 winning cell, `src_sup40_sw40`)
**Leagues:** Ireland Premier (79) and First Division (718), trained separately
**Dated:** 2026-08-12
**Branch:** `design/matchday-layer`

---

## 1. The question

Given a validated Layer-1 engine, **when and how should you touch the book?** That is a Layer-2
question — entry time, per-market trust, skip rules, staking — and nothing in the repo answered
it, because `src/Portfolio` has no time axis at all.

## 2. Verdict

**The engine has a small positive edge, it survives allocation, and the one intervention that
improves it is the opposite of the one the calibration analysis pointed at.**

Reference policy, no curation, 293 matches over ~1.5 seasons:

| league | slates | bets | final bankroll | ROI | growth/slate |
|---|---|---|---|---|---|
| **79** | 100 | 1476 | **×1.52** | +7.32% [−8.16, +23.15] | +0.00418 |
| **718** | 81 | 833 | **×1.24** | +6.52% [−12.03, +26.71] | +0.00262 |
| **718 + `MinEdge(0.02)`** | 81 | 489 | **×1.47** | **+13.80%** [−10.62, +39.18] | **+0.00477** |

Positive in both leagues. **Neither is significant on its own** — every interval includes zero,
and that is the honest headline.

The blend weight on the model, fitted as one free parameter over 5,332 selections, is
**w\* = 0.28** (0.32 on 79 alone). Not zero: the model is individually *worse* than the de-vigged
market on log score (−0.0038) yet earns a real weight, which is the signature of a weaker
forecaster carrying independent information.

## 3. Three of this stream's own conclusions were withdrawn

This matters more than any single number, so it is stated before the findings rather than after.

**WP8 concluded "the engine is uninformative; this is a Layer-1 problem". That was wrong.** It
was measured on the 81-match order-book corpus, which turns out to be the only slice of the
out-of-sample set where the model loses. Using r21's own metric, r21's own benchmark, and the
experiment this stream trained:

| set | obs | matches | model LL | market LL | diff |
|---|---|---|---|---|---|
| ALL OOS | 5859 | 293 | 0.42454 | 0.44231 | **−0.01778** ← model wins |
| 2025, outside the OB window | 3580 | 180 | 0.41880 | 0.43360 | **−0.01479** ← model wins |
| 2026, outside the OB window | 1789 | 92 | 0.44402 | 0.47411 | **−0.03009** ← model wins |
| 2026, **inside** the OB window | 490 | 21 | 0.39528 | 0.38990 | **+0.00538** ← model loses |

There is no decay — 2026 *outside* the window is the model's best period anywhere. The
order-book matches are simply unusual: the market's own log loss there is 0.390 against 0.474 on
the rest of the same season, an unusually sharp market on unusually predictable fixtures.

Before blaming the window, the two benchmarks were put head to head on the same selections. The
de-vigged order book and `summarize_betfair_market`'s `prob_fair_close` agree to **0.003 in
probability and 0.001 in log loss**. It was the sample, not the yardstick.

| withdrawn | why |
|---|---|
| WP8: `w* = 0`, "Layer-1 problem" | true of 76 matches, false of the engine |
| WP5: "per-market trust does not transfer" | measured where there was no edge to allocate |
| WP5: "shrink the model's large claims" | **backwards** — see §4.3 |

Still standing: the WP1/WP3 apparatus (107/107 and 591/591, reproducing `match_day` row for row)
and WP4's execution results, which are market facts independent of model quality.

## 4. Findings

### 4.1 Execution: enter at the close, and the book will not take your money

Leg-weighted CLV is **monotone in entry time in both leagues independently**:

| entry rule | 79 CLV | 718 CLV |
|---|---|---|
| **AtClose** | **−0.0051** | **−0.0072** |
| FixedLead(30m) | −0.0082 | −0.0124 |
| FixedLead(60m) | −0.0118 | −0.0127 |
| FixedLead(120m) | −0.0139 | −0.0147 |
| BestPrice (oracle) | +0.0152 | +0.0119 |

Entering two hours out costs **~0.8–0.9 pp of CLV per leg**. Capacity says the same: slippage on
a £100 stake roughly doubles from the close to T−120. There is no interior optimum; the plan's
expected "T−120 to T−30" is wrong on both axes independently.

**The binding practical constraint is size, not timing.** Measured on the staked rows:

| stake per leg | unfillable at the *best* moment (79 / 718) |
|---|---|
| £1,000 | ~100% / ~100% |
| £100 | **54%** / **73%** |
| £10 | 13% / 21% |

Top-of-book back size: median **£41**, 5th percentile **£1.04**. Caveat: the archive stores at
most **3 ladder levels** (verified over 526k rows), so these are upper bounds on shortfall.

An oracle-versus-null control settled the timing question properly. On ROI, random entry beat the
close in both leagues and the verdict function reported "a real drift toward kickoff". On CLV —
which has the power — the random control is *worse* than the close, so the oracle's +2.04% price
gain is hindsight. With ~260 legs the ROI intervals span ±40 pp; the ROI ordering was a coin toss.

### 4.2 Trust: flat weights are a provable no-op, and it is now measured

| `FlatTrust` | binding (`SlateDrawdown`+cap) | slack (`NoRisk`+cap 0.05) |
|---|---|---|
| 0.10 | 1.4796 | 1.0869 |
| 0.25 | 1.5187 | 0.9940 |
| 0.50 | 1.5163 | 0.9347 |
| 1.00 | 1.5163 | 0.8928 |

Flat under the binding risk model — 0.5 and 1.0 identical to four decimals — and monotone
**decreasing** under the slack one. `risk_factor` is homogeneous of degree 0 exactly as
`src/Portfolio/stake.jl:5-18` states. Any trust study run under a binding drawdown constraint
must be *differential*, or it measures nothing.

### 4.3 The reversal: growth and log score disagree, and growth wins

The "curse curve" — skill against the size of the model's disagreement with the de-vigged market
— reproduces on 5,332 selections. It is **symmetric**: worst in both tails (−0.0139 below −5pp,
−0.0106 above +5pp), best near agreement. The shape is a real property of the engine.

**Acting on it destroys money, in both leagues:**

| policy | 79 final | 79 growth | 718 final | 718 growth |
|---|---|---|---|---|
| no filter | 1.5187 | 0.00418 | 1.2363 | 0.00262 |
| `MaxOdds(6.0)` | 1.0280 | 0.00028 | 1.0976 | 0.00115 |
| `MaxClaim(0.05)` | 0.9975 | −0.00003 | 0.8209 | −0.00244 |
| `MaxClaim(0.02)` | 1.0005 | 0.0 | 0.8366 | −0.00220 |
| **`MinEdge(0.02)`** — the opposite | **1.5188** | **0.00418** | **1.4713** | **0.00477** |

`MinEdge` keeps *precisely the legs `MaxClaim` discards* and is the best arm in both leagues,
lifting 718 from ×1.24 to ×1.47 (ROI 6.5% → 13.8%) while **cutting** drawdown 22.1% → 18.4%.

Both measurements are correct and they do not conflict. The model is **miscalibrated** —
over-confident, so log score punishes it in the tails — while being **directionally right often
enough that at the offered price those tail bets are +EV**. Log score asks whether the
probability is right; Kelly asks whether it is on the right side of the price. Only the second
pays.

This is why the house rule is to judge on growth rather than log loss, and it vindicates the
standing "calibrate the centre, edge in the tails" position: per-line expectation should sit on
the market, and the edge is the per-match deviation.

### 4.4 Curation is not derivable on this sample

Deriving per-selection trust on 2025 (180 books) and testing on 2026, **every family fell back to
the default weight** — nothing cleared a match-clustered sign test at ≥20 legs. This is not a
transfer failure; there simply is not enough per-family signal to act on. The intervention that
works is claim-based (`MinEdge`), not market-based.

### 4.5 The engine has no dispersion parameter

`src_sup40_sw40` samples 69 parameters and **none of them is a dispersion**. The source documents
this (`dispersion_config::D # config-compat; unused by the Poisson likelihood`), so the
`HomeAwayDispersion()` we pass — copied from the r21 winning cell — is inert.

Goals variance is pinned to the mean. The "model dispersion is half the market's" observation
from the 2026-08-07 live run and the 2026-08-08 replay is therefore **structural**, not a fitting
failure: the engine cannot widen its scoreline distribution. Consistent with the finding that
pregame totals really are ~Poisson, this is roughly right on totals and is where it bites on 1X2.

Two of the engine's four likelihood pillars **are the market** (supremacy and smile anchors, both
at 0.4, with `market_on = true`). It is a shrinkage-toward-market estimator by construction, and
its only route to an edge is where the xG and goals pillars pull it off those anchors.

## 5. What this cannot certify

- **Nothing here is statistically significant on its own.** Every ROI interval includes zero. The
  case rests on sign consistency across two leagues that share no fixtures and were fitted with
  different market pillars, not on any single interval.
- **Route 2 prices off summary closes, not executable prices.** Its ROI is an *allocation* number
  and is optimistic as an *execution* number. §4.1 is the authority on what a stake actually
  costs, and it says the book will not absorb £100 on half the legs.
- **The order-book corpus is 81 matches over 11 weeks**, and demonstrably an unusual 11 weeks.
  Every conclusion drawn on it is scoped to execution mechanics, where the model does not enter.
- **`MinEdge(0.02)` has not been tuned or validated out of sample.** It is one threshold that
  happened to be raced. Its replication across two leagues is the evidence; a threshold sweep
  would need its own out-of-sample design and has not been run.
- **Two leagues, one engine, one season and a half.** Nothing here generalises to other leagues
  or to the funnel/APM families without being re-measured.

## 6. Defects found

Ten, of which the majority were silent-by-construction — returning plausible wrong answers rather
than erroring.

| defect | consequence |
|---|---|
| **every lead measured from midnight** — `FixtureInfo` has no kickoff field, so a `hasproperty` fallback used `DateTime(slate_day)` | the entire entry-time axis was wrong (leads of −1125 min) while the gate passed 583/583 around it |
| `latents_invariant` filters on `eltype(col) <: Number`, but latent cells are `Vector{Float64}` in `Any` columns | **cannot fail** — returns "invariant" without comparing anything. A previous stream's clock-invariance claim rested on it |
| `L2Snapshot.fixtures` widened to `Dict{Int,Any}` | routed `stake_sheet` to its DataStore method, which returns an empty sheet for unplayed fixtures |
| `hasproperty(cap, :c)` — neither cap type has a field `c` | `recap_slates!` never ran; cross-instant entry rules kept over-exposed books |
| first tick ≠ tradeable book — feed ticks at T−334 then goes silent 193 min | rules aimed at T−180 would snap to T−120 and report under the wrong label |
| WP0 top-of-book size wrong by ~50× (`market_matched`, and ×10000 raw units) | would have mis-sized every capacity conclusion |
| `adaptive_grid` coarse/fine stub gap | re-read the same 3-minute tick |
| G4 summed de-vigged probabilities over the staked subset | 134 spurious failures |
| top-level `big` shadowing `Base.big` | broke an unrelated test file |
| `Diagnostics.extract_chains` throws on the inert dispersion config | log noise per fold |

Plus a training gate refinement: 79's global max R-hat is 1.616, but all three non-converged
folds sit **outside** the corpus window; restricted to the folds actually consumed, max R-hat is
1.0097 (79) and 1.0060 (718) with zero exceedances.

## 7. Graduation

**Recommended:**
- `PriceDrift` and `FillCost` → `src/backtesting/metrics/`. Both subtype
  `AbstractDistributionalMetric`, so it is additive, and both produced findings the existing
  metrics could not.
- The threaded `build_snapshots` pattern (23 min → 5m50s across slates, with the `_CARD_META`
  lock) if the replay driver graduates.

**Not recommended:**
- `MaxOdds` / `MaxClaim` — they lose in both leagues. Keeping them in the prototype documents a
  measured negative result; promoting them would ship a mistake.
- `ClosingLineValue` — useful for the entry-time study but **degenerate at the close**, where it
  equals minus the market's margin. It should not graduate without that limitation encoded, or
  it will be misread as an edge metric.

**No change needed:** `MinEdge`, the intervention that won, already ships in
`src/Portfolio/implementations/filters.jl`.

## 8. What to do next

1. ~~**`market_on = false` control on this corpus.**~~ **Done — see §9.** The anchoring carries
   real accuracy; the Layer-2 conclusions above do not depend on it.
2. **A `MinEdge` threshold sweep with a proper out-of-sample design.** The result replicated
   across leagues at one arbitrary threshold; that is promising and unvalidated.
3. **Let the order-book corpus grow.** The collector keeps running; the harness now judges any
   engine against it in minutes rather than days.
4. **Do not chase the size problem yet.** £100 is unfillable on half the legs, but that only
   binds once the edge is established at a size worth caring about.

## 9. WP10 — the `market_on = false` control (2026-08-13)

**Question:** two of the engine's four likelihood pillars are the market (supremacy and smile,
weight 0.4 each). Does removing them tank accuracy — the anchor was carrying it — or leave it
roughly unchanged — the anchor was suppressing a signal the xG/goals pillars would otherwise
express?

**Answer: the anchor carries real accuracy.** Same pinned DataStores, same sampler, the same
293/221-match Route 2 sample, the same de-vigged close — only `market_on` flips.

| | 79 | 718 | pooled |
|---|---|---|---|
| model LL, anchored | 0.5755 | 0.5749 | 0.5753 |
| model LL, unanchored | 0.5824 | 0.5773 | 0.5806 |
| gap (unanchored − anchored) | **+0.0069** | **+0.0024** | **+0.0053** |
| `w*` | 0.32 → 0.20 | 0.20 → 0.20 | 0.28 → 0.20 |
| skill vs. market | −0.0038 → −0.0107 | −0.0078 → −0.0102 | −0.0052 → −0.0105 |

Every comparison moves the same direction, independently in both leagues — which share no
fixtures and were fitted on different market feeds — and pooled. This is the clean outcome the
pre-registration priced in, not the ambiguous one. The unanchored engine is not uninformative
either: `w*` holds at 0.20 rather than collapsing to 0, so the xG/goals pillars alone still add
something over the raw market, just less than with the anchor attached.

One consequence worth naming rather than discovering by surprise: `log_φ` (the O/U smile shape)
appears only inside the now-zeroed smile pillar, so its posterior is its prior once
`market_on = false`. Totals default toward a flat/Poisson shape, and O/U family skill degrades
monotonically with strike level for the unanchored engine (0.5-line skill −0.006 → 4.5-line
−0.054) — consistent with losing the market-informed shape exactly where it matters most.

**But every Layer-2 finding in this document replicates on the unanchored engine anyway.**
Reference policy, no curation:

| | 79 anchored | 79 unanchored | 718 anchored | 718 unanchored |
|---|---|---|---|---|
| final | ×1.52 | ×1.50 | ×1.24 | **×1.29** |
| ROI | +7.3% | +6.9% | +6.5% | +7.4% |
| growth/slate | 0.00418 | 0.00403 | 0.00262 | 0.00313 |

Despite the calibration loss, Kelly performance is statistically indistinguishable from the
anchored engine — 718 is nominally *better* unanchored, and both leagues' ROI intervals overlap
almost entirely. The specific interventions replicate a third time on a genuinely different
engine: `MinEdge(0.02)` wins again (718: ×1.29 → ×1.49, ROI 7.4% → 12.8%); `MaxClaim`/`MaxOdds`
lose money again in both leagues; curation derived on 2025 still fails to transfer to 2026; the
`FlatTrust` homogeneity result still holds under a binding vs. slack risk model (§4.2's table
reproduces to the same four decimals).

**Reading.** The market anchor is load-bearing for calibration — a real, measured L1 fact, not a
modelling nicety free to relax. But the Layer-2 staking conclusions in §2-§4 do not depend on
it: they are a property of how this model class's directional signal interacts with Kelly
staking against this market, not an artefact of which pillars happen to be switched on. That is
a stronger claim for this document's verdict than the single-engine result supported on its own.

**Code:** `r08_train_ireland_noanchor.jl` (training, reuses WP2's pinned DataStores exactly),
`r09_route2_noanchor.jl` (head-to-head accuracy plus the full WP9 suite, reading r07's
`route2.jls` for the anchored side rather than recomputing it).
