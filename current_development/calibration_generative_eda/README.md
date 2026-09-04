# Generative rate calibration — EDA and Layer-2 overhaul

Prototype stream for the work package in
[`CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md`](CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md).

**Status — 2026-09-04.** Four runners executed on `mcmc-beast`. §5 is the
calibration result at the close, §6 the portfolio result at the close, §7 both
again at **T−25 — a price a bettor could actually have taken.**

> **CORRECTION, and it matters.** r01 and r02 both state that every return in them
> is "an upper bound" because they are priced at the close. **That claim has the
> sign backwards.** §7.2 measures it: staking the identical model, trust and
> fixtures at T−25 instead of the close returns **more**, not less — +113.60%
> against +79.88% on `m12` under flat trust, +159.32% against +121.13% under the
> canonical tiers. The closing price is the *converged* price; a model with real
> edge is better off taking the earlier, softer one. Closing-price staking was a
> **penalty** on these backtests, not a flattery. The caveat was right that the
> number was not tradeable and wrong about which way it erred.

**Headline (close-priced, §5–§6).** Generative rate calibration improves out-of-sample LogLoss on both
Scottish Lower candidates; the gain is concentrated almost entirely in the
**large-edge** regime; and the winning direction is **shrinkage**, not the Ireland
stream's conviction. On the portfolio it raises flat ROI, Sharpe and drawdown on
every arm while *lowering* total return — and arm F shows that shortfall is
**exposure, not skill**: at a drawdown matched to the raw model's, calibrated `m12`
returns **+151.09%** against raw's +126.09%. Out of sample it produces the best arm
measured anywhere in this stream, beating the production champion on return,
Sharpe and drawdown at once. It does **not** replace `CanonicalScottishLowerTrust`,
and it does not rescue Over 2.5 far enough to un-gate it.

**Headline (T−25, §7).** All three functional forms clear Gate 1 at tradeable
prices on both models — where at the close `m12` refused all three — because the
T−25 book is measurably less calibrated (ECE 0.0183) than the close (0.0119) and
there is more error left to remove. But **the optimum moves**: on T−25 rates the
*inverse* form wins on LogLoss, reversing §5, and r01's close-fitted pick gives up
0.0015–0.0020 LogLoss when transferred. On the portfolio, calibration at T−25 buys
risk-adjusted quality rather than return, and every arm — calibrated or not — has
**positive closing-line value**.

---

## 1. The question

Selection-level calibration (`src/Calibration/shift_models/basic_logit.jl`) moves
`P(Home)` and `P(Over 2.5)` independently, so the shifted board is no longer a
valid bivariate scoreline matrix and derivative markets can contradict each other.
This stream replaces it with a shift at the **generative intensity** level:

1. invert the de-vigged Betfair close back to `(λ_mkt_h, λ_mkt_a)` by Nelder-Mead
   on `Features.DoublePoissonMarketFeature`;
2. measure the log-rate discrepancy `Δ = log median(λ_model) − log λ_mkt`;
3. pool **every posterior draw** log-linearly, `log λ* = w(Δ)·log λ_model + (1−w)·log λ_mkt`;
4. price the shifted `CountLatents` through the same score-grid kernels, evaluator
   and portfolio the raw container goes through.

Coherence is then structural rather than checked: 1X2, every totals line and BTTS
are three partitions of one 12×12 tensor, so they cannot disagree.

The open question is **not** whether the construction is coherent — it is — but
whether it *helps*. The Ireland Premier calibration `(w_base, σ) = (0.25, 0.25)`
gained +145 bps LPD on extreme edges there and then **lost 16–22% of final wealth**
on Scottish Lower under matched policies
([`notes_rqs_01.md` §4](../orderbook_layer2/research_questions_explore/notes_rqs_01.md)).
So the grid sweeps three competing functional forms rather than tuning the one that
already failed.

| form | `w(0)` | `w(±∞)` | reads as |
|---|---|---|---|
| `:inverse_gaussian` | `w_base` | 1.0 | trust the market on noise, the model on structural edges |
| `:standard_gaussian` | `w_max` | `w_base` | optimiser's-curse shrinkage of extreme claims |
| `:static_geometric` | `w_base` | `w_base` | constant pool — the control for "does Δ-dependence buy anything" |

---

## 2. Files

| File | Is |
|---|---|
| [`l01_generative_calibrator.jl`](l01_generative_calibrator.jl) | Loader. `GenerativeCalibrationSpec` and the three weight laws, market-rate inversion with acceptance gates, the posterior shift, the coherence audit, proper scoring and the sweep grid. Definitions only. |
| [`r01_sweep_rate_calibration.jl`](r01_sweep_rate_calibration.jl) | Runner 1. The diagnostic and proper-score sweep, gates G-A…G-D, CSV artefacts. **Calibration only — no bankroll claim.** |
| [`r02_portfolio_direction_audit.jl`](r02_portfolio_direction_audit.jl) | Runner 2. The direction and trust-vector portfolio audit, arms A–F; Gate 2. |
| [`l02_point_in_time_book.jl`](l02_point_in_time_book.jl) | Loader. The T−25 point-in-time book, its staleness and completeness gates, book drift and closing-line value. |
| [`r03_t25_book_and_calibration.jl`](r03_t25_book_and_calibration.jl) | Runner 3. The T−25 book, rate re-inversion, and the calibration sweep at tradeable prices; Gate 1 restated. |
| [`r04_t25_portfolio.jl`](r04_t25_portfolio.jl) | Runner 4. The portfolio at T−25, the close-vs-T−25 attribution, and CLV. |
| [`l03_variance_schemes.jl`](l03_variance_schemes.jl) | Loader. The pool decomposed into a pooled location and a 2×2 residual map, seven dispersion schemes over it, the market-free Jensen tail diagnostics, and the gate that keeps `A_pool` identical to `l01`'s pool. |
| [`r05_variance_experiments.jl`](r05_variance_experiments.jl) | Runner 5. Dispersion held against location under two weight laws: Gate 1 with the tail audit, Gate 2 over the whole (λ, Kelly) risk surface, and the frontier read at a common drawdown. |
| `results/` | Replaceable CSV artefacts. Re-running overwrites them. |

Run:

```bash
julia --project -t 16                     # mcmc-beast; -t 8 on archpc
julia> include("current_development/calibration_generative_eda/r01_sweep_rate_calibration.jl")

R01_SMOKE=1 julia --project -t 16 ...     # 3-spec dry run, one model, not a result
```

Each runner has its own smoke switch on the same pattern — `R03_SMOKE`, `R04_SMOKE`,
`R05_SMOKE`. r04 needs `results/r03_best_per_form_t25.csv`; r05 needs it too, and
reads the location laws from it rather than restating them.

---

## 3. Design decisions that change what the numbers mean

These are recorded here because each one would silently alter a headline figure if
it were changed, and none of them is visible in a results table.

**The calibrating price is the close — in §5 and §6 only.** Rates there are
inverted from the Betfair close (TWA over `[−20 min, kick-off]`), the snapshot
experiment 06 scored against. A closing price is not available when a bet is
struck, which is why §7 rebuilds everything at T−25. **The direction of that error
was not what I assumed:** §7.2 measures the same strategies returning 22–38 points
MORE at T−25, so closing-price staking was a penalty. §5 and §6 understate the raw
model rather than flattering it.

**The fixture set is pinned to the published gate.** Gate 1 quotes LogLoss
`0.64337` and ECE `0.0100` from the 40-fold 24/25 + 25/26 study (2,899 scored
observations). The canonical runs were extended to **43 folds** with the 26/27
August programme on 2026-09-04
([EXTEND_2627_REPORT.md](../../experiments/scottish_lower/06_joint_player_lineup_fusion/EXTEND_2627_REPORT.md)),
so `restrict_latents` cuts the container back to `R01_GATE_SEASONS` before scoring
and the runner reports how many fixtures that dropped. If the recomputed baseline
still fails to reproduce the published number, **the recomputed number wins** and
the runner says so — the published constants describe a different fixture set.

**Two scoring scopes, kept apart.** `head_*` columns are
`Evaluation.DEFAULT_SCORED_MARKETS` (1X2 + O/U 2.5 + BTTS) — the only scope in
which the Gate-1 thresholds mean anything. `wide_*` columns are the 13-direction
book (1X2, O/U 0.5/1.5/2.5/3.5, BTTS) that r02 audits. Both come from one pricing
pass; the headline is that pass filtered to the narrow selections.

**Edge strata are anchored on the raw model.** Calibration shrinks edges toward the
book by construction, so self-anchored `|Δp| < 0.02` / `> 0.05` buckets would move
between grid points and two rows would score different observations under the same
column heading. Every stratified LPD uses the **uncalibrated** edges to bucket.

**The fallback is `w = 1`, bit for bit.** A fixture whose book cannot be inverted
(too few quotes, Nelder-Mead not converged, residual over `max_sse`, rates outside
`[0.05, 6.0]`) passes through unchanged — not dropped, not priced at a league mean.
Dropping would change which fixtures each spec scores; inventing a rate would price
a fixture from inputs the pipeline declined to use. A fixture at `w = 1.0` copies
its raw draws rather than computing `exp(1·log λ + 0)`, because `exp(log(x)) != x`
in Float64 and the `w_base = 1.00` grid point has to reproduce the baseline exactly
or it is not a control. **Gate D asserts this rather than assuming it.**

**Log-linear pooling destroys posterior variance, by `w²` exactly.** At the Ireland
median `w ≈ 0.41`, 83% of the posterior log-variance is gone. Kelly stake size reads
that variance. `weight_summary` reports `w` and `var_retention` quantiles for every
grid point, so no headline score can hide the contraction.

**The inversion objective weights each line once.** `Features.LINES` lists both
sides of every totals line while `_calculate_error(Val(:over_25), …)` already scores
the over *and* under keys, so the default tuple would count each totals line twice
against 1X2's once. `L01_INVERSION_LINES` passes one symbol per line.

---

## 4. Gates

| Gate | Where | Refuses |
|---|---|---|
| **G-A** | r01 §4 | Fixture inventory and filtration: OOS fixtures per model, how many the season restriction dropped, and every quoted `(market, line, selection)`. Warns on a scored selection the book never quotes. |
| **G-B** | r01 §5 | Inversion coverage and residual quality, with every refusal counted **by reason**. Warns below 90% coverage, because refused fixtures dilute the measured effect size without changing any score. |
| **G-C** | r01 §6 | Derivative-market coherence. Errors if two market families disagree on one fixture by more than 1e-9 — that cannot happen when every price is a partition of one tensor, so it would indict the pricing path, not the calibration. |
| **G-D** | r01 §7–8 | Identity control. The `w_base = 1.00` grid point must reproduce the uncalibrated baseline LogLoss to 1e-12. |
| **Gate 1** | r01 §9 | Proper scoring, judged against the **recomputed** baseline: LogLoss no worse, ECE no worse than the uncalibrated model, and ECE at or below the Betfair close's. |
| **Gate 1 at T−25** | r03 §7.3 | The same proper-score gate against the T−25 book. Passes on every form and both models. |
| **Gate 2 at T−25** | r04 §7.4 | The portfolio at tradeable prices, risk-matched against the raw T−25 arm. |
| **Gate 2** | r02 §6.7 | Bankroll > +130%, Sharpe ≥ 1.416, max drawdown no worse than −20.5%. Measured; and §6.7 argues the return threshold is mis-specified for a variance-contracting transform. |
| **G-0** | r05 §8.1 | Dispersion identity control. `A_pool` must reproduce `l01.calibrate_latents` on every draw of every fixture to 1e-9 relative, or the baseline of §8 is not the pool r01–r04 measured and no difference in that section means what its heading says. |

A Gate-1 PASS is a **calibration** result and entitles nobody to a bankroll claim.
The Ireland transfer improved its own league's calibration diagnostics and still
lost money here.

---

## 5. Results

Run: `mcmc-beast`, 16 threads, 2026-09-04 18:53:57, commit `f0e2b7d8`, branch
`feat/extend-scottish-lower-2627`, experiment `scottish_lower_joint_player_2426`
(read-only). Fixture set: **710** OOS fixtures in 24/25 + 25/26 after
`restrict_latents` dropped the 49 fixtures of the 26/27 extension; **2,899**
scored observations in the headline scope, 4,390 in the wide scope.

**`m05` does not pass the strict convergence gate**, on **tail ESS** alone
(251.6 against a 400 floor); its R̂ (1.0084), bulk ESS (647.3), divergence rate
(2.2e-05) and BFMI (0.538) all clear. `m12` passes every gate. Both are scored
anyway and flagged, because the refusal is a precision statement about the tails
of `m05`'s posterior, not a claim that its central tendency is wrong — but every
`m05` number below carries it.

**The G-D control reproduced the published 40-fold numbers exactly** — `m12`
LogLoss `0.64337`, ECE `0.0100`, Betfair close ECE `0.0139`, N `2,899`. The
restriction and the scoring scope are therefore right, and every delta below is
measured against a baseline that is the published one.

### 5.1 Inversion quality — `results/r01_market_inversion.csv`

| Fixtures | With any quote | Accepted | Coverage (all) | Coverage (quoted) | Median SSE | p90 SSE | Max SSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 710 | 633 | 623 | 87.7% | **98.4%** | 4.119e-04 | 1.329e-03 | 3.556e-03 |

| Refusal reason | Fixtures |
|---|---:|
| too few quoted selections (0 < 3) | 77 |
| too few quoted selections (2 < 3) | 10 |

`λ_mkt` home median **1.473** `[0.708, 3.582]`; away median **1.218**
`[0.479, 2.496]`. Inversion of all 710 fixtures took **0.3 s** on 16 threads.

**The G-B warning was too pessimistic and the gate is being corrected.** It
compares accepted fits against *every* fixture, so the 77 fixtures with no closing
quote at all count as failures. Those fixtures contribute no scored observation
either — they are dropped by `require_market` before any metric sees them — so the
number that measures dilution is coverage over *quoted* fixtures, **98.4%**. Only
10 fixtures were genuinely refused after having a book to invert.

### 5.2 Optimum per functional form — `results/r01_best_per_form.csv`

Headline scope (1X2 + O/U 2.5 + BTTS), 2,899 observations. `r01_best_per_form`
selects on **LogLoss alone**, which is why two of its three `m12` picks degrade
ECE; §5.3 is the selection that should drive Phase 2.

| Model | Method | Spec | LogLoss | Δ vs baseline | ECE | BF ECE | median `w` | Gate 1 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `m12` | — | uncalibrated | 0.64337 | — | 0.0100 | 0.0139 | 1.000 | — |
| `m12` | inverse_gaussian | `inv_w0.40_s1.00` | 0.64008 | −0.00329 | 0.0114 | 0.0139 | 0.404 | REFUSE (ECE) |
| `m12` | standard_gaussian | `std_w0.25_s0.15` | **0.63915** | **−0.00422** | 0.0138 | 0.0139 | 0.798 | REFUSE (ECE) |
| `m12` | static_geometric | `sta_w0.40` | 0.63994 | −0.00343 | 0.0149 | 0.0139 | 0.400 | REFUSE (ECE) |
| `m05` | — | uncalibrated | 0.64299 | — | 0.0149 | 0.0139 | 1.000 | — |
| `m05` | inverse_gaussian | `inv_w0.40_s1.00` | 0.64013 | −0.00286 | **0.0074** | 0.0139 | 0.404 | **PASS** |
| `m05` | standard_gaussian | `std_w0.25_s0.15` | **0.63933** | **−0.00366** | 0.0107 | 0.0139 | 0.819 | **PASS** |
| `m05` | static_geometric | `sta_w0.40` | 0.64004 | −0.00294 | 0.0094 | 0.0139 | 0.400 | **PASS** |

`m05` passes Gate 1 on all three forms; `m12` fails all three **on ECE alone**.
That asymmetry is not a defect of the calibration — `m12`'s uncalibrated ECE
(0.0100) is already better than the closing line's (0.0139), so there is little
calibration error left to remove, whereas `m05`'s (0.0149) is worse than the close
and calibration repairs it.

### 5.3 The joint-improvement frontier — `results/r01_sweep_scores.csv`

Specs improving LogLoss **and** ECE against the same model's own baseline. This is
the set that clears Gate 1, and it is what Phase 2 is nominated from.

| Model | Form | Best joint improver | LogLoss | ECE | median `w` | var retained | LPD large edge |
|---|---|---|---:|---:|---:|---:|---:|
| `m12` | inverse | `inv_w0.25_s0.75` | 0.64032 | 0.0097 | 0.259 | **0.067** | −0.58066 |
| `m12` | standard | **`std_w0.25_s0.25`** | **0.63976** | **0.0096** | 0.920 | **0.846** | −0.57886 |
| `m12` | static | *none* (`sta_w1.00` = identity) | — | — | — | — | — |
| `m05` | inverse | `inv_w0.40_s1.00` | 0.64013 | 0.0074 | 0.404 | 0.163 | −0.58099 |
| `m05` | standard | **`std_w0.25_s0.15`** | **0.63933** | 0.0107 | 0.819 | 0.671 | −0.58062 |
| `m05` | static | `sta_w0.40` | 0.64004 | 0.0094 | 0.400 | 0.160 | −0.58086 |

14 of 68 `m12` specs and 40 of 68 `m05` specs improve both.

**No constant pool improves `m12` on both axes.** The only `:static_geometric`
point in `m12`'s joint-improving set is the identity. Whatever the calibration is
buying on `m12`, it is buying it from the **Δ-dependence** — blending toward the
market at a fixed rate does not reproduce it. On `m05` a static pool does work,
which is consistent with `m05` having a uniform calibration error to remove rather
than an edge-dependent one.

**The two forms buy the same LogLoss at very different variance cost.** On `m12`,
`inv_w0.25_s0.75` and `std_w0.25_s0.25` land within 0.0006 LogLoss of each other,
but the inverse form destroys **93.3%** of posterior log-variance (median `w`
0.259) against the standard form's **15.4%** (median `w` 0.920). Kelly stake size
reads that variance. Phase 2 carries both precisely so the difference shows up
where it matters.

### 5.4 Edge-stratified LPD — `results/r01_sweep_scores.csv`

Buckets anchored on the raw model's edges, wide scope.

| Model | Spec | LPD `\|Δp\|<0.02` | N | LPD `\|Δp\|>0.05` | N | LPD all |
|---|---|---:|---:|---:|---:|---:|
| `m12` | uncalibrated | −0.54932 | 1600 | −0.58633 | 1483 | −0.57648 |
| `m12` | `inv_w0.40_s1.00` | −0.54930 | 1600 | −0.58003 | 1483 | −0.57362 |
| `m12` | `std_w0.25_s0.15` | −0.54914 | 1600 | **−0.57837** | 1483 | **−0.57281** |
| `m12` | `sta_w0.40` | −0.54930 | 1600 | −0.57976 | 1483 | −0.57350 |
| `m05` | uncalibrated | −0.55522 | 1605 | −0.58754 | 1406 | −0.57556 |
| `m05` | `inv_w0.40_s1.00` | −0.55529 | 1605 | −0.58099 | 1406 | −0.57336 |
| `m05` | `std_w0.25_s0.15` | −0.55499 | 1605 | **−0.58062** | 1406 | **−0.57276** |
| `m05` | `sta_w0.40` | −0.55530 | 1605 | −0.58086 | 1406 | −0.57331 |

**Every gain is in the large-edge bucket.** Small edges move by at most 2e-04 in
either direction — five of the eight rows move the wrong way — while large edges
gain **+63 to +80 bps** on `m12` and +65 to +69 bps on `m05`.

**Caveat, and it is a real one.** 225 of `m12`'s 1,483 large-edge rows (15.2%) and
217 of `m05`'s 1,406 (15.4%) are O/U 0.5 selections, whose closing prices are
structurally broken on this book — see §5.6. Phase 2 §2 re-scores these strata on
the tradeable market set; treat the magnitudes above as provisional and the sign
as robust (the effect is present in every family, §5.5).

### 5.5 Per-family scores — `results/r01_family_scores.csv`

`m12`, headline and wide families. `std` is `std_w0.25_s0.25`, `inv` is
`inv_w0.25_s0.75` — the two joint improvers of §5.3.

| Family | N | LogLoss raw | LogLoss `std` | LogLoss `inv` | BF LogLoss | ECE raw | ECE `std` | ECE `inv` | BF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1X2 | 1785 | 0.61636 | **0.61089** | 0.61154 | 0.61312 | 0.0155 | 0.0141 | 0.0163 | 0.0186 |
| O/U 1.5 | 430 | 0.53190 | 0.53025 | **0.52835** | 0.52728 | 0.0195 | 0.0150 | 0.0180 | 0.0103 |
| O/U 2.5 | 758 | 0.68732 | 0.68740 | 0.68757 | 0.68988 | 0.0100 | **0.0015** | 0.0066 | 0.0183 |
| O/U 3.5 | 528 | 0.61446 | **0.61145** | 0.61092 | 0.61046 | 0.0116 | 0.0051 | 0.0083 | 0.0148 |
| BTTS | 356 | 0.68520 | **0.68310** | 0.68399 | 0.68337 | 0.0087 | 0.0198 | 0.0230 | 0.0300 |

**The 1X2 leg is where the money-relevant gain is, and it crosses the close.**
Uncalibrated `m12` prices 1X2 *worse* than the Betfair close (0.61636 vs 0.61312);
calibrated it prices them *better* (0.61089). That is the only family in which
calibration turns a deficit against the market into a surplus.

**On the Over 2.5 question the answer is "calibration, not repricing."** O/U 2.5
LogLoss is unmoved to four decimals (0.68732 → 0.68740), but its ECE collapses
from 0.0100 to **0.0015** — a 6.7× reduction, and 12× better than the closing
line's 0.0183. Rate calibration does not tell you something new about totals; it
makes the number the staking layer reads mean what it says. Whether that converts
into the ROI the direction has historically lacked is Gate 2's question, not this
one.

### 5.6 A data defect this run exposed — O/U 0.5 de-vigging

`results/r01_odds_inventory.csv`:

| Line | Over rows | Under rows | Paired? |
|---|---:|---:|---|
| O/U 0.5 | 982 | 408 | **no — 574 one-sided** |
| O/U 1.5 | 634 | 631 | yes |
| O/U 2.5 | 1141 | 1141 | yes |
| O/U 3.5 | 728 | 728 | yes |
| O/U 5.5 | 137 | 294 | no |

`l01_betfair_closing_odds` de-vigs by normalising within `(match, market, line)`,
which is correct on a two-sided quote and **degenerate on a one-sided one**: a lone
`over_05` row normalises to `prob_fair_close = 1.0`. 574 O/U 0.5 fixtures are
one-sided, and the symptom is unmissable — the "market" LogLoss on that family is
**1.31832** against the model's 0.21098.

Consequences, stated exactly:

* The **headline scope is unaffected** — it contains no O/U 0.5 — so every Gate-1
  number above is clean.
* `wide_market_logloss` and `wide_market_ece` are contaminated. The model-side wide
  columns are not.
* The edge strata are contaminated at ~15% (§5.4), because a market probability of
  1.0 against a model probability near 0.93 lands in the large-edge bucket by
  construction.
* **Phase 2 does not bet O/U 0.5.** `l01_tradeable_markets()` excludes it and says
  why; `l01_wide_markets()` is left as it was so this run stays reproducible.

### 5.7 Verdict on Phase 1

1. **The construction works and is coherent to machine precision.** Maximum
   disagreement between market families on the same fixture: **5.55e-16**. The
   residual against 1.0 (1.7e-06 raw, 2.5e-06 calibrated) is the 12-goal grid
   truncation, shared by every family.
2. **Calibration improves LogLoss on both models, by 0.0029–0.0042.**
3. **The gain is a large-edge phenomenon.** Small edges are untouched.
4. **The Ireland direction is refuted on this league.** `:standard_gaussian` —
   shrink extreme claims toward consensus — beats `:inverse_gaussian` — hold
   conviction on extreme claims — on LogLoss for every model, at every `w_base`
   optimum, and on large-edge LPD. The optimiser's-curse reading of the data wins
   here; the "attack the market's structural bias" reading does not. Phase 2 must
   not quietly re-adopt the inverse form because it was the stream's original
   hypothesis.
5. **Gate 1: `m05` PASS on all three forms, `m12` REFUSE on ECE at the
   LogLoss-optimal points but PASS at 14 of 68 grid points**, best of which is
   `std_w0.25_s0.25` (LogLoss 0.63976, ECE 0.0096, 85% of posterior variance
   retained).
6. **None of this is a bankroll claim.** Ireland improved its own league's
   diagnostics and still lost 16–22% of final wealth here.

---

## 6. Phase 2 — portfolio and direction audit

Run: `mcmc-beast`, 16 threads, 2026-09-04 (CEST), commit `eab3ba8d`. Same 710
fixtures as §5; **631 priced, 79 skipped** (no closing quote, or no complete
market). Book: 11 tradeable directions (1X2, O/U 1.5/2.5/3.5, BTTS), `DeArb`,
`KellyLogUtility`, `FractionalKelly(0.30)`, 2% commission. Policy:
`SlateDrawdown(23.0)`, `FixedCap(0.25)`, `DailySlate()`. Only the latent container
and the trust model vary between arms.

**The reference arm reproduces the published champion.** Raw `m12` +
`CanonicalScottishLowerTrust` returns **+157.50%** at Sharpe **1.637**, MDD
**−20.47%**, against experiment 06's published +155.93% / 1.636 / −19.79%. The
small gap is the two directions this book drops. The comparison base is therefore
the real champion, not an approximation of it.

### 6.1 Full period — the 2×2 and the variance cost

| Model | container | trust | bets | return % | flat ROI % | Sharpe | MDD % | exposure |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `m12` | raw | flat | 1800 | +126.09 | +9.71 | 1.323 | −22.33 | 9.58% |
| `m12` | raw | canonical | 1261 | **+157.50** | +14.10 | 1.637 | −20.47 | 7.43% |
| `m12` | **std** | flat | 1763 | +110.29 | +11.58 | 1.683 | −15.99 | 6.93% |
| `m12` | **std** | canonical | 1256 | +123.62 | **+15.76** | **1.878** | **−13.71** | 5.47% |
| `m12` | inv | flat | 1464 | +32.41 | +9.43 | 1.377 | −8.63 | 3.11% |
| `m12` | inv | canonical | 1079 | +40.45 | +14.64 | 1.823 | −10.36 | 2.39% |
| `m05` | raw | flat | 1804 | +130.12 | +10.28 | 1.442 | −21.77 | 9.09% |
| `m05` | raw | canonical | 1234 | **+134.54** | +13.67 | 1.580 | −18.17 | 6.88% |
| `m05` | **std** | flat | 1681 | +74.39 | +11.83 | 1.753 | −11.70 | 4.96% |
| `m05` | **std** | canonical | 1197 | +75.66 | **+15.33** | **1.815** | **−9.87** | 3.87% |
| `m05` | inv | flat | 1556 | +49.18 | +10.58 | 1.568 | −10.15 | 3.96% |
| `m05` | inv | canonical | 1106 | +53.67 | +14.77 | 1.825 | −10.36 | 3.02% |

Calibration moves every arm the same way: **flat ROI up, Sharpe up, drawdown
shallower, total return down, exposure down.** That is the `w²` log-variance
contraction reaching the allocator — Kelly stake size is monotone in posterior
variance, so a calibrated container stakes less and compounds less at the same
risk budget. Mean exposure falls 28% (`m12` std) to 68% (`m12` inv), tracking the
variance retained (84.6% and 6.7%).

### 6.2 Arm F — scale against shape

Loosening λ on the calibrated container until its drawdown matches the raw arm's.
**In-sample λ selection; a mechanism demonstration, not a performance claim.**

| Model | container | trust | λ | return % | Sharpe | MDD % | raw MDD % | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `m12` | **std** | flat | 18.0 | **+151.09** | 1.655 | −19.93 | −22.33 | matched |
| `m12` | inv | flat | 8.0 | +100.80 | 1.285 | −21.93 | −22.33 | matched |
| `m05` | **std** | flat | 12.0 | **+167.26** | 1.691 | −20.92 | −21.77 | matched |
| `m05` | inv | flat | 10.0 | +128.60 | 1.493 | −21.20 | −21.77 | matched |
| `m12` | std | canonical | 0.5 | +137.20 | 1.828 | −16.80 | −20.47 | cap-saturated |
| `m05` | std | canonical | 0.5 | +83.45 | 1.775 | −10.96 | −18.17 | cap-saturated |

**This is the discriminating result of the whole stream.** At matched risk under
flat trust, calibrated `m12` returns **+151.09%** against raw's +126.09% — 25
points more, with a *shallower* drawdown and Sharpe 1.655 against 1.323. `m05`
goes +167.26% against +130.12%. The §6.1 return shortfall was exposure, not skill.

And the two functional forms come apart exactly here. `:standard_gaussian`
recovers and overtakes at matched risk; `:inverse_gaussian` does **not** — +100.80%
against raw's +126.09% on `m12`, still behind after the risk is equalised. Its
shortfall is real, not a scale artefact. Phase 1 preferred `std` on LogLoss and
large-edge LPD; the P&L agrees, and this is the arm where a stream that had
adopted the Ireland form on faith would have lost money.

**Why the canonical arms cannot be matched.** Under `CanonicalScottishLowerTrust`
the λ sweep moves maximum drawdown by only **0.7–3.1 percentage points**, against
**19.8–24.4** under flat trust. The tiers already gate most of the slate to zero,
so `FixedCap(0.25)` becomes the binding constraint and `SlateDrawdown` — being
homogeneous of degree 0 in the stakes it receives — cannot buy any more exposure.
Extending λ further would not move those rows; `mdd_span` in
`results/r02_risk_matched.csv` is the evidence. This is the same scale-invariance
law experiment 06 §2.1 records, observed from the other side.

### 6.3 Out of sample — slates after 2025-05-03

A true holdout: no trust rule and no calibration parameter was fitted on it.
(The r01 sweep selected `std_w0.25_s0.25` on the full period, so the *spec* is not
out of sample here; the trust vector of arm E is.)

| Model | container | trust | bets | return % | Sharpe | MDD % |
|---|---|---|---:|---:|---:|---:|
| `m12` | raw | flat | 978 | +14.48 | 0.452 | −22.33 |
| `m12` | raw | canonical | 641 | +32.49 | 0.982 | −20.47 |
| `m12` | **std** | flat | 953 | +24.63 | 1.074 | −13.19 |
| `m12` | **std** | **canonical** | 635 | **+33.84** | **1.402** | **−13.71** |
| `m12` | std | refit | 440 | +21.71 | 0.983 | −15.94 |
| `m12` | raw | refit | 441 | +8.75 | 0.313 | −26.02 |
| `m05` | raw | flat | 993 | +22.17 | 0.684 | −21.77 |
| `m05` | raw | canonical | 632 | +31.25 | 0.983 | −18.17 |
| `m05` | std | flat | 906 | +17.81 | 1.144 | −9.68 |
| `m05` | std | canonical | 604 | +18.93 | 1.193 | −9.87 |
| `m05` | raw | refit | 324 | −3.75 | −0.151 | −23.85 |

**`m12` + `std` + `canonical_P1` is the best arm measured anywhere in this
stream**, and it beats the production champion on all three axes at once: +33.84%
against +32.49%, Sharpe 1.402 against 0.982, drawdown −13.71% against −20.47%.

### 6.4 The 13-direction audit — `results/r02_direction_ledger.csv`

`m12`, flat trust, full period, so every direction is stakeable and comparable.

| direction | container | bets | Kelly ROI % | capital % | efficiency | calibration |
|---|---|---:|---:|---:|---:|---:|
| home | raw | 293 | +10.19 | 24.81 | 1.51 | −0.0086 |
| home | **std** | 299 | **+16.66** | 26.25 | 1.79 | +0.0065 |
| away | raw | 375 | +5.84 | 29.24 | 0.87 | −0.0670 |
| away | **std** | 378 | **+10.06** | 30.27 | 1.08 | −0.0489 |
| draw | raw | 366 | +10.30 | 13.50 | 1.53 | −0.0051 |
| draw | std | 347 | +6.63 | 11.78 | 0.71 | −0.0013 |
| under_25 | raw | 227 | +19.17 | 9.88 | 2.85 | −0.0047 |
| under_25 | std | 232 | +14.25 | 10.49 | 1.53 | +0.0054 |
| **over_25** | raw | 59 | **−10.99** | 2.15 | −1.63 | −0.0407 |
| **over_25** | **std** | 49 | **+0.75** | 2.10 | 0.08 | −0.0106 |
| **over_25** | **inv** | 29 | **+8.04** | 1.55 | 0.96 | +0.0655 |
| over_35 | raw | 86 | −2.02 | 2.45 | −0.30 | −0.0454 |
| over_35 | std | 79 | −0.91 | 2.73 | −0.10 | −0.0294 |
| btts_yes | raw | 62 | −7.17 | 2.92 | −1.06 | +0.0199 |
| btts_yes | std | 60 | −1.74 | 2.62 | −0.19 | +0.0172 |

**On Over 2.5 the answer is: rescued from destructive to break-even, and that is
not enough to un-gate it.** Kelly ROI goes −10.99% → +0.75% (`std`) → +8.04%
(`inv`), and the mechanism is visible in the calibration column: the raw model
over-rates the selection by 4.1 percentage points and calibration removes three
quarters of that. But `std`'s efficiency is **0.08** — it clears zero without
carrying its weight against a book averaging far more — and `inv`'s +8.04% rests
on **29 bets**. `CanonicalScottishLowerTrust` gates Over 2.5 to zero, and nothing
here justifies reversing that.

The wider pattern is the optimiser's curse at the P&L level: mean edge shrinks on
every direction (home 0.0677 → 0.0505, away 0.0702 → 0.0517) and Kelly ROI *rises*
on the two largest. The raw edges were inflated, and the shrinkage was removing
error rather than signal.

### 6.5 Arm E — the refitted trust vector fails

Fitted on slates to 2025-05-03, scored on the ones after. The rule tiers a
direction by Kelly ROI and capital efficiency on the selection window, holding the
audited 0.35 / 0.25 / 0.00 ladder fixed.

It selects `{home, away}` at 0.35, `under_25` at 0.25 and gates everything else —
including `draw`, which the canonical vector keeps at 0.25 and which returned
−11.4% on the selection window and is the difference between the two vectors.

**Out of sample it loses to both comparators, on every arm.** `m12` raw: refit
+8.75% (Sharpe 0.313) against flat +14.48% and canonical +32.49%. `m12` std: refit
+21.71% (0.983) against flat +24.63% and canonical +33.84%. `m05` raw: refit
−3.75% (Sharpe −0.151).

This is the same failure `MARKET_LINE_EDA_REPORT.md` §5.1 records for this rule
class, reproduced on a calibrated book: a per-direction selection rule fitted on
half a season of Scottish Lower slates does not generalise, and the audited vector
is not improved by refitting it. **Calibration does not make the trust vector
re-derivable, and it does not make it unnecessary.**

### 6.6 The O/U 0.5 exclusion, evidenced — arm R1

Raw `m12`, flat trust, the full 13-direction book including O/U 0.5:

| | return % | Sharpe | MDD % | bets |
|---|---:|---:|---:|---:|
| 11 directions (§6.1) | +126.09 | 1.323 | −22.33 | 1800 |
| 13 directions | +123.61 | 1.329 | −20.97 | 1894 |

| direction | bets | Kelly ROI % | capital % | mean `p_market` |
|---|---:|---:|---:|---:|
| over_05 | 33 | −3.11 | 2.85 | 0.9207 |
| under_05 | 64 | **−29.99** | 0.57 | 0.0596 |

Both sides lose, `under_05` catastrophically, and it is staked against a de-vigged
"fair" price that is a normalisation artefact on 574 of the fixtures (§5.6).
Excluding the line costs nothing and removes a fabricated input.

### 6.7 Gate 2

Thresholds from the work package: return > +130%, annual Sharpe ≥ 1.416, max
drawdown no worse than −20.5%.

| Arm | return | Sharpe | MDD | Verdict |
|---|---|---|---|---|
| `m12` raw + canonical | +157.50 ✓ | 1.637 ✓ | −20.47 ✓ | **PASS** |
| `m12` std + canonical | +123.62 ✗ | 1.878 ✓ | −13.71 ✓ | REFUSE |
| `m12` std + flat | +110.29 ✗ | 1.683 ✓ | −15.99 ✓ | REFUSE |
| `m12` raw + flat | +126.09 ✗ | 1.323 ✗ | −22.33 ✗ | REFUSE |

**Read literally, Gate 2 refuses calibration. Read honestly, the gate is
mis-specified for this transform.** Its return threshold is a *scale* criterion,
and a variance-contracting calibration fails it mechanically while improving every
*shape* criterion beside it — at a fixed λ it simply takes less risk. Arm F is the
disposal of that confound: at matched drawdown, `m12` std + flat returns +151.09%,
which clears +130% with Sharpe 1.655 and a shallower drawdown than the raw arm.

The gate should be restated as **return at matched drawdown**, or the λ should be
tuned per container before the return is read. Either way the number to carry
forward is §6.3's: out of sample, calibrated `m12` under the canonical trust beats
the production champion on all three axes simultaneously.

### 6.8 Verdict on Phase 2

1. **Q1 — calibration changes the bankroll, favourably, once risk is matched.**
   +151.09% against +126.09% on `m12`, +167.26% against +130.12% on `m05`, with
   equal or shallower drawdown. At a *fixed* λ it trades return for risk.
2. **Q2 — Over 2.5 is repaired but not rehabilitated.** −10.99% → +0.75% Kelly
   ROI, driven by a genuine calibration fix, but at efficiency 0.08 on 49 bets.
   Keep it gated.
3. **Q3 — the trust vector survives.** `CanonicalScottishLowerTrust` beats flat on
   every container, calibrated or not, in and out of sample; and refitting it
   loses to both. Calibration and staking trust remain **separate controls**, which
   is what `notes_rqs_01.md` §5 concluded and this run confirms rather than
   overturns.
4. **`:standard_gaussian` is the form to graduate, and `:inverse_gaussian` is
   not.** The Ireland form loses on LogLoss, on large-edge LPD, on full-period
   return, and — decisively — on return at matched risk, where the standard form
   overtakes the raw model and the inverse form still does not.
5. **Everything here is priced at the CLOSE and is an upper bound.** Before any of
   it reaches the MatchDay consoles the inversion must be re-run on the T−25 book.
   That is the next work package, and no line of this stream should be deployed
   before it.

---

## 7. Phase 3 — the T−25 book, at tradeable prices

Runners: [`r03_t25_book_and_calibration.jl`](r03_t25_book_and_calibration.jl),
[`r04_t25_portfolio.jl`](r04_t25_portfolio.jl); builder
[`l02_point_in_time_book.jl`](l02_point_in_time_book.jl). Run on `mcmc-beast`,
16 threads, 2026-09-04, commit `6fbc1e1a`.

**T−25 is the start of MatchDay's execution band** (T−25 to T−12, AGENTS.md §7.2)
and therefore the earliest instant a slate is committed — the most conservative
honest cutoff.

### 7.1 The book at T−25

**Not a windowed TWA.** Measured over 599,529 archived ticks on 1,641 fixtures, a
`(−30, −25)` window carries 8,644 selection groups against the close book's
26,341, at a **median of one tick** — so its "time-weighted average" is one number
with a weight, and widening it to recover coverage averages prices up to four
hours old. `l02` instead takes the **last tick at or before the cutoff**, the way
the replay console's `PreloadedBook` does, and carries its staleness as a column.

| | rows | fixtures | markets | median staleness | p90 staleness | median overround |
|---|---:|---:|---:|---:|---:|---:|
| close book | 14,617 | 1,627 | — | — | — | — |
| **T−25, bound 90 min** | **10,373** | **1,572** | 4,491 | **8.0 min** | 51.0 min | 1.0015 |
| T−25, bound 30 min | 6,879 | 1,341 | — | — | 20.4 min | — |
| T−25, bound 60 min | 9,050 | 1,510 | — | — | 34.5 min | — |
| T−25, bound 180 min | 12,841 | 1,621 | — | — | 94.0 min | — |

Market refusals at the 90-minute bound: 3,230 stale beyond it, 2,876 incomplete,
1,580 with no completeness contract (correct score, Asian handicap), **1** on
overround. The overround filter, which I expected to bind hardest pre-match,
turned out to be irrelevant — exchange traded prices are near-fair at T−25 as
well as at the close.

`l02` also **fixes the §5.6 de-vig defect at source**: completeness is checked
*before* normalisation, so a one-sided quote is refused by name instead of
normalising to a fair probability of exactly 1.0. `l01`'s builder is untouched so
r01 and r02 stay reproducible.

Every head-to-head below runs on the **7,270-row / 1,478-fixture intersection** of
the two books, so price is never confounded with coverage.

### 7.2 What the close knows that T−25 does not

| | median | p90 | sd |
|---|---:|---:|---:|
| \|log price drift\|, T−25 → close | 0.0221 | 0.0715 | — |
| fair-probability drift | +0.00033 | — | 0.01615 |
| inverted `log λ_mkt` drift, home | — | 0.1432 | 0.1010 |
| inverted `log λ_mkt` drift, away | — | 0.1826 | 0.1138 |

Per family: 1X2 median |log drift| 0.0260, O/U 0.0183, BTTS 0.0194.

**The λ drift is the same order as the smallest σ in the grid** (0.15), which is
why the parameters had to be re-swept rather than transferred — the weight law's
input is drawn from a different distribution at the two instants.

**And the price effect runs the other way from the caveat.** Same model, same
trust, same fixtures, staked at each instant:

| Model | trust | @close | @T−25 | Δ return | Δ Sharpe |
|---|---|---:|---:|---:|---:|
| `m12` | flat | +79.88 | **+113.60** | **+33.72** | +0.234 |
| `m12` | canonical | +121.13 | **+159.32** | **+38.20** | +0.204 |
| `m05` | flat | +76.04 | **+98.77** | +22.73 | +0.153 |
| `m05` | canonical | +108.97 | **+135.58** | +26.61 | +0.155 |

The close is the converged price. A model with genuine edge does better taking the
earlier, softer one — which is the same statement as the positive CLV in §7.5.

### 7.3 Calibration at T−25 — Gate 1 passes everywhere, and the optimum moves

Headline scope, matched rows, calibrated with **T−25 rates** and scored against
the **T−25 fair price**.

| Model | form | spec | LogLoss | Δ base | ECE | T−25 book ECE | median `w` | var retained |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `m12` | — | uncalibrated | 0.63488 | — | 0.0151 | 0.0183 | 1.000 | — |
| `m12` | **inverse** | `inv_w0.25_s0.35` | **0.63064** | −0.00424 | 0.0133 | 0.0183 | 0.291 | **0.085** |
| `m12` | standard | `std_w0.40_s0.15` | 0.63188 | −0.00300 | **0.0093** | 0.0183 | 0.840 | 0.706 |
| `m12` | static | `sta_w0.40` | 0.63089 | −0.00399 | 0.0093 | 0.0183 | 0.400 | 0.160 |
| `m05` | — | uncalibrated | 0.63381 | — | 0.0179 | 0.0183 | 1.000 | — |
| `m05` | **inverse** | `inv_w0.25_s0.35` | **0.63027** | −0.00354 | 0.0076 | 0.0183 | 0.287 | 0.082 |
| `m05` | standard | `std_w0.40_s0.15` | 0.63165 | −0.00216 | 0.0094 | 0.0183 | 0.856 | 0.732 |
| `m05` | static | `sta_w0.40` | 0.63076 | −0.00305 | **0.0030** | 0.0183 | 0.400 | 0.160 |

**All six (model × form) PASS Gate 1**, where at the close `m12` refused all
three. The reason is in the last-but-two column: the T−25 book's ECE is **0.0183**
against the close's **0.0119**. The market's calibration advantage is largely made
in the final 25 minutes, so at T−25 there is more error to remove and a model can
beat the price it can actually take.

**The optimum moved, and it moved back toward Ireland.** On T−25 rates the
**inverse** form wins on LogLoss for both models — reversing §5.3, where the
standard form won on closing rates.

| Model | r01's close-fitted pick | its LogLoss on T−25 rates | best T−25 spec | forgone |
|---|---|---:|---:|---:|
| `m12` | `std_w0.25_s0.25` | 0.63267 | 0.63064 | **+0.00203** |
| `m05` | `std_w0.25_s0.15` | 0.63178 | 0.63027 | **+0.00151** |

This is the same lesson the Ireland transfer taught, one level down: **which
functional form is right depends on how sharp the book you are pooling with is.**
Against a converged closing line, shrinking extreme claims wins. Against a softer
T−25 book, holding conviction on them wins. A stream that had fixed the form on
either result alone would have been wrong on the other, and that is now measured
rather than argued.

### 7.4 The portfolio at T−25 — full book, what a bettor actually has

`SlateDrawdown(23.0)`, `FixedCap(0.25)`, `DailySlate()`, 11 tradeable directions.

| Model | container | trust | bets | return % | flat ROI % | Sharpe | MDD % |
|---|---|---|---:|---:|---:|---:|---:|
| `m12` | raw | flat | 1592 | +111.70 | +9.79 | 1.220 | −23.45 |
| `m12` | **raw** | **canonical** | 1127 | **+151.52** | +15.04 | 1.592 | −16.15 |
| `m12` | inv | canonical | 975 | +65.64 | **+17.44** | **1.772** | **−7.76** |
| `m12` | std | flat | 1503 | +63.35 | +9.63 | 1.396 | −12.10 |
| `m12` | std | canonical | 1103 | +72.85 | +13.62 | 1.606 | −11.36 |
| `m12` | sta | canonical | 980 | +49.94 | +14.90 | 1.697 | −6.61 |
| `m05` | **raw** | **canonical** | 1113 | **+130.15** | +14.67 | 1.544 | −16.07 |
| `m05` | inv | canonical | 944 | +63.91 | **+18.28** | **1.943** | **−6.86** |
| `m05` | std | canonical | 1078 | +63.01 | +12.92 | 1.484 | −10.84 |

**At a fixed risk budget, calibration costs more return at T−25 than it did at the
close** — and the mechanism is the same `w²` contraction, now biting harder because
the T−25-optimal specs retain far less variance (0.085–0.160 for the inverse and
static forms, against 0.846 for r01's close-fitted standard pick).

Risk-matched to the raw T−25 arm (in-sample λ; a mechanism demonstration, not a
performance claim — and none of these reached the target inside the 0.5pp
tolerance, so each is the nearest λ below it):

| Model | container | λ | return % | Sharpe | MDD % | raw MDD % |
|---|---|---:|---:|---:|---:|---:|
| `m12` | **std** | 12.0 | **+134.17** | 1.314 | −22.22 | −23.45 |
| `m12` | sta | 10.0 | +88.93 | 1.156 | −19.49 | −23.45 |
| `m12` | inv | 15.0 | +64.09 | 1.043 | −21.05 | −23.45 |
| `m05` | sta | 10.0 | +85.32 | 1.182 | −19.78 | −22.73 |
| `m05` | std | 15.0 | +83.96 | 1.158 | −20.01 | −22.73 |
| `m05` | inv | 12.0 | +82.91 | 1.134 | −22.05 | −22.73 |

**`m12`'s standard-form advantage survives the move to tradeable prices**
(+134.17% against raw's +111.70%, at a shallower drawdown — the same ~+22pp
matched-risk gain §6.2 measured at the close). **`m05`'s does not** (+83.96%
against raw's +93.23%), though at 2.7pp less drawdown. The Phase 2 claim was
`m12`-specific and should not have been read as a property of the transform.

Note also that the **LogLoss-optimal form is the portfolio-worst one on both
models**: the inverse form wins §7.3 and finishes last in every risk-matched row.
Choosing a calibration on a proper score alone would have picked exactly wrong for
the allocator, and that is the third time this stream has recorded that gap.

### 7.5 Closing-line value

The diagnostic a T−25 strategy earns and a close-priced one cannot pose: did the
market move *toward* the bet after it was struck?

| Model | container | trust | bets | mean CLV % | stake-weighted % | % positive |
|---|---|---|---:|---:|---:|---:|
| `m12` | raw | canonical | 1014 | +0.266 | +0.988 | 50.5 |
| `m12` | **inv** | canonical | 885 | **+0.510** | +1.009 | **54.4** |
| `m12` | std | canonical | 993 | +0.354 | +1.025 | 52.0 |
| `m12` | sta | canonical | 889 | +0.500 | **+1.133** | 54.1 |
| `m05` | raw | canonical | 1010 | +0.327 | +0.940 | 51.8 |
| `m05` | **inv** | canonical | 854 | **+0.552** | +0.964 | **54.7** |
| `m05` | std | canonical | 982 | +0.471 | +0.996 | 53.4 |

**Every arm has positive CLV**, and the calibrated arms have roughly twice the raw
model's. Calibration is selecting bets the market subsequently agrees with — which
is the same fact as §7.2's price effect, seen per bet instead of per book.

Read this carefully. CLV says the calibrated arms pick *better*; §7.4 says they
stake *less*. Both are true, they are not in tension, and only the second one
compounds.

### 7.6 Out of sample — slates after 2025-05-03, T−25 prices

| Model | container | trust | return % | Sharpe | MDD % |
|---|---|---|---:|---:|---:|
| `m12` | **raw** | canonical | **+31.77** | 0.941 | −16.15 |
| `m12` | **std** | canonical | +24.63 | **1.269** | **−8.61** |
| `m12` | inv | canonical | +15.22 | 1.010 | −7.76 |
| `m05` | **raw** | canonical | **+30.09** | 0.945 | −16.07 |
| `m05` | **inv** | canonical | +18.84 | **1.351** | −6.86 |
| `m05` | std | canonical | +19.99 | 1.081 | −9.08 |

Raw wins on return, calibrated wins on Sharpe and halves the drawdown. That is the
same trade as everywhere else in this stream, and out of sample it is not resolved
in calibration's favour on return at any arm.

### 7.7 Verdict on Phase 3

1. **The T−25 book exists and is usable.** 1,572 of 1,627 fixtures, median
   staleness 8 minutes, near-fair overround. The cutoff is not the obstacle.
2. **Closing-price staking was a penalty, not a flattery.** Correcting r01/r02's
   caveat: the same strategies return **22–38 points more** at T−25. Everything
   this stream has published understated the raw model.
3. **Gate 1 passes at tradeable prices on every form and both models**, because
   the T−25 book is measurably less calibrated than the close.
4. **Calibration parameters do not transfer between price instants**, and the
   winning functional form flips with the sharpness of the book being pooled
   with. This reconciles the Ireland result rather than dismissing it.
5. **On the portfolio, calibration at T−25 buys risk-adjusted quality, not
   return.** `m12`'s matched-risk advantage survives; `m05`'s does not.
6. **Positive CLV everywhere, strongest on the calibrated arms.** The bet
   selection is genuinely better; the sizing is what gives the return back.
7. **What would actually change the recommendation:** a variance-preserving
   version of the pool. Every cost measured in §7.4 traces to `w²` contraction
   reaching Kelly, and nothing in the construction requires the posterior spread
   to shrink with its location. That is the next thing to build, and it is a
   modelling change rather than another sweep.

> **Item 7 is wrong, and §8 is the retraction.** The variance-preserving pool was
> built (`l03_variance_schemes.jl`) and it recovers nothing, because there was
> nothing to recover: restoring up to **11.8×** the posterior log-variance moves
> the staked exposure by **0.4%** (§8.5). The stake shrinkage §7.4 measured is
> caused by the LOCATION shift contracting the model's edges, not by the `w²`
> contraction of the width. Item 5 does not survive either — at a common drawdown
> with both risk knobs available, the `inv` container returns **+192.23%** against
> the raw model's +151.52%, so calibration at T−25 buys return as well as
> risk-adjusted quality once the risk budget is allowed to move (§8.7).

### 7.8 What still stands between this and a Saturday

* **Fill model.** Bets are struck at the archived traded price in whatever size
  the allocator asked for. The live system rests at the touch and the archive
  carries at most three levels (AGENTS.md §7.4). These returns are still an upper
  bound — a smaller one than r01/r02's, and now for this reason rather than the
  price instant.
* **Traded price, not the resting ladder.** `betfair.odds_history` archives what
  someone paid; the console prices off `betfair_live.order_book_1m`. A T−25 traded
  price is not necessarily what was showing on the side we would have taken.
* **Staleness.** Bounded at 90 minutes, median 8. A 40-minute-old price is the
  last trade, not a live quote.
* **In-sample spec selection.** r03 chose the calibration parameters over the full
  period; §7.6 holds out the slates, not the spec.

---

## 8. Phase 4 — variance preservation and dispersion transforms

Loader [`l03_variance_schemes.jl`](l03_variance_schemes.jl), runner
[`r05_variance_experiments.jl`](r05_variance_experiments.jl). Run on `mcmc-beast`,
16 threads, 2026-09-04, commit `72a39cca`, 449.9 s wall.

§7.7 closed Phase 3 with an open item: the log-linear pool contracts posterior
log-variance by `w²` as a side effect of the algebra, and the reading offered there
was that Fractional Kelly therefore stakes smaller and the calibrated arm compounds
less. Four hypotheses were on the table.

| | claim |
|---|---|
| **H1** | the contraction is an ARTEFACT. Move the location, keep the width, and the compounding returns at no cost in proper score |
| **H2** | the contraction is LOAD-BEARING. Restoring width re-inflates the Jensen term `E[e^(−Λ)] ≥ e^(−E[Λ])` (`eda/README.md` Discovery 2) and manufactures longshot mass the allocator then bets |
| **H3** | the question is malformed. `SlateDrawdown` absorbs uniform stake changes (Discovery 4), so the unused drawdown headroom is spendable with a risk knob and no posterior needs touching |
| **H4** | some asymmetric scheme beats all three |

**The result, in one line: H1's premise is false.** Restoring the posterior width
moves the Kelly stake by less than half a percent, so there was never a
compounding loss to recover. H2's mechanism is real, behaves exactly as predicted,
and is one order of magnitude too small to matter. H3 is supported and turns out to
be worth **+41 percentage points** of compound return. H4 survives as a small,
consistent, second-order effect — and it is not the effect anyone proposed.

### 8.1 Separating location from dispersion

`l01`'s pool applies `log λ̃⁽ˢ⁾ = w·log λ⁽ˢ⁾ + (1 − w)·log λ_mkt` to every draw,
which is exactly

```
log λ̃⁽ˢ⁾ = c + w·u⁽ˢ⁾ ,   c = w·m + (1 − w)·log λ_mkt ,   u⁽ˢ⁾ = log λ⁽ˢ⁾ − m
```

with `m` the raw posterior log-location. `l03` freezes `c` — the location law stays
`l01`'s, unchanged — and replaces `w·u` with `κ + M·u` for a 2×2 residual map `M`
and a scalar anchor `κ`. Every row below therefore differs from the baseline **in
dispersion and nothing else**, which is the one thing r01–r04 could not say.

`M` is 2×2 rather than two scalars because the two things a football posterior is
uncertain about are not `λ_home` and `λ_away` but **supremacy** `u_h − u_a` (what
1X2 prices) and **totals** `u_h + u_a` (what O/U and BTTS price). A retention pair
`(ρ_s, ρ_t)` maps back to a symmetric `M`, and reduces to `ρ·I` when `ρ_s = ρ_t`.

| scheme | `ρ_s` | `ρ_t` | anchor | is |
|---|---|---|---|---|
| `A_pool` | `w` | `w` | — | the production pool, `Var = w²σ²` |
| `B_full` | 1 | 1 | — | full mean-shift variance preservation (work package Scheme B) |
| `B_anch` | 1 | 1 | pool mean | the same, with the predictive rate anchored to `A_pool`'s |
| `C_sqrt` | `√w` | `√w` | — | `Var = wσ²` (work package Scheme C) |
| `D_sup` | 1 | `w̄` | — | supremacy preserved, totals contracted |
| `D_sup_anch` | 1 | `w̄` | pool mean | the same, anchored |
| `D_tot` | `w̄` | 1 | — | totals preserved, supremacy contracted — **falsification control** |

Two design points carry most of the interpretive weight.

**`C_sqrt` is not a midpoint, it is the coherent Bayesian answer.** If the log-rate
posterior is `N(m, σ²)` and the market is an independent noisy observation of the
same log-rate with precision `τ`, the conjugate posterior has mean
`(σ⁻²m + τ·log λ_mkt)/(σ⁻² + τ)` and variance `w·σ²`, where `w = σ⁻²/(σ⁻² + τ)` is
exactly the pool weight. So `C_sqrt` **is** "model the market as a likelihood with
explicit `τ_mkt`", and the pool's `w²σ²` is what you get when the same information
is counted twice — once in the location, once in the width.

**`κ` separates "wider" from "hotter".** `E[Λ] = E[e^(log Λ)]` grows with
`Var(log Λ)`, so a variance-preserving scheme does not only widen the posterior, it
also predicts more goals than the calibrated location said. `:pool_mean` anchoring
chooses `κ` on the draws so the scheme's mean rate equals `A_pool`'s exactly. An
anchored scheme and its unanchored twin then differ **only** in first predictive
moment, which turns out to be where almost all of the action is.

**G-0.** `A_pool` must reproduce `l01.calibrate_latents`, or the baseline is not the
pool r01–r04 measured and no difference in this section means what its heading says.
Maximum relative departure over every draw of every fixture:

| model | form | departure | bound |
|---|---|---:|---:|
| `m12` | std | 4.34e-16 | 1e-9 |
| `m12` | inv | 2.22e-16 | 1e-9 |
| `m05` | std | 4.39e-16 | 1e-9 |
| `m05` | inv | 4.11e-16 | 1e-9 |

Two location laws are run, not one, because the effect size should scale with how
much variance the pool destroys. r03's `std_w0.40_s0.15` (median `w` 0.84) retains
about 70%; its `inv_w0.25_s0.35` (median `w` 0.29) retains about 9%. A dispersion
effect that is real must be larger under `inv`.

### 8.2 The dispersion each scheme delivers — `results/r05_variance_dispersion.csv`

Retained log-variance against the RAW posterior, in three bases, median over the 580
fixtures with an accepted market inversion (`m12`; `m05` is within 0.04 everywhere):

| form | scheme | median `w` | ret side | ret sup | ret tot | rate ratio |
|---|---|---:|---:|---:|---:|---:|
| std | `A_pool` | 0.840 | 0.706 | 0.687 | 0.674 | 1.0000 |
| std | `B_full` | 0.840 | 1.000 | 1.000 | 1.000 | 1.0013 |
| std | `B_anch` | 0.840 | 1.000 | 1.000 | 1.000 | 1.0000 |
| std | `C_sqrt` | 0.840 | 0.840 | 0.816 | 0.815 | 1.0006 |
| std | `D_sup` | 0.840 | 0.819 | **1.000** | 0.665 | 1.0006 |
| std | `D_tot` | 0.840 | 0.849 | 0.665 | **1.000** | 1.0007 |
| inv | `A_pool` | 0.291 | **0.085** | 0.094 | 0.094 | 1.0000 |
| inv | `B_full` | 0.291 | 1.000 | 1.000 | 1.000 | 1.0038 |
| inv | `B_anch` | 0.291 | 1.000 | 1.000 | 1.000 | 1.0000 |
| inv | `C_sqrt` | 0.291 | 0.291 | 0.303 | 0.303 | 1.0009 |
| inv | `D_sup` | 0.291 | 0.511 | **1.000** | 0.092 | 1.0017 |
| inv | `D_tot` | 0.291 | 0.596 | 0.092 | **1.000** | 1.0021 |

The transforms do what they are supposed to. Under `inv` the pool destroys **91.5%**
of the posterior log-variance and `B_full` restores all of it — an **11.8×** change
in posterior width, which is the largest lever this study has. `D_sup` and `D_tot`
are exact mirrors: each preserves precisely what the other discards.

The `rate ratio` column is the predictive rate against `A_pool`'s. Unanchored
preservation makes the model **0.13% to 0.50% hotter**; the anchored twins sit at
1.0000 by construction. Keep that number in view — it is small, and it turns out to
be the only thing in this table that any downstream number responds to.

### 8.3 Gate 1 — proper scores are almost perfectly indifferent to dispersion

Headline scope (1X2 + O/U 2.5 + BTTS), T−25 book, 2,148 scored rows, market LogLoss
0.63306. `A_pool` reproduces r03's row for the same spec, as it must.

| model | container | LogLoss | ECE | Brier |
|---|---|---:|---:|---:|
| `m12` | `raw` | 0.63488 | 0.0151 | 0.22196 |
| `m12` | `std_A_pool` | 0.63188 | 0.0093 | 0.22066 |
| `m12` | `std_B_full` | 0.63188 | 0.0092 | 0.22066 |
| `m12` | `std_B_anch` | 0.63187 | 0.0098 | 0.22065 |
| `m12` | `std_C_sqrt` | 0.63188 | 0.0092 | 0.22066 |
| `m12` | `std_D_sup` | 0.63188 | 0.0092 | 0.22066 |
| `m12` | `std_D_tot` | 0.63188 | 0.0093 | 0.22066 |
| `m12` | `inv_A_pool` | **0.63064** | 0.0133 | 0.22001 |
| `m12` | `inv_B_full` | 0.63068 | 0.0144 | 0.22002 |
| `m12` | `inv_B_anch` | 0.63070 | 0.0131 | 0.22004 |
| `m12` | `inv_C_sqrt` | 0.63065 | 0.0133 | 0.22001 |
| `m12` | `inv_D_sup` | 0.63068 | 0.0136 | 0.22002 |
| `m12` | `inv_D_sup_anch` | 0.63069 | **0.0126** | 0.22003 |
| `m12` | `inv_D_tot` | **0.63064** | 0.0134 | 0.22001 |

Under `std` every scheme agrees to the fifth decimal place. Under `inv` — where the
posterior width changes by a factor of twelve — the whole spread of LogLoss is
**6e-5**, against a raw-to-calibrated gap of 4.2e-3, i.e. **seventy times larger**.
H1's "no cost in proper score" is satisfied, and so is its mirror image: no benefit
either.

One structure does survive the noise floor, and it is not the one that was proposed.
The LogLoss ordering under `inv` is **identical on both models**:

* `A_pool` and `D_tot` tie exactly (`m12` 0.63064, `m05` 0.63027);
* `B_full` and `D_sup` tie exactly (`m12` 0.63068, `m05` 0.63033/0.63034).

`D_tot` preserves **totals** dispersion and costs nothing. `D_sup` preserves
**supremacy** dispersion and costs the whole of the (tiny) penalty, the same penalty
`B_full` pays for preserving both. The per-family scores say the same thing from the
other end: on 1X2 (n = 1,560), `inv_A_pool` and `inv_D_tot` both score 0.61140 while
`inv_D_sup` scores 0.61145.

**So whatever small proper-score cost dispersion carries is a supremacy effect, not a
totals effect — the opposite of what the Jensen argument predicts.** The falsification
control earned its place: reporting `D_sup` alone would have produced a confident
story pointing the wrong way.

### 8.4 The Jensen term, measured — `results/r05_variance_jensen.csv`

Computed straight off the draws, market-free, over all 710 priced fixtures, because
Discovery 2 is a claim about the predictive distribution and not about any price.
`mixture − plugin` is `E[P(N ≤ n | Λ)] − P(N ≤ n | E[Λ])`, which **is** the Jensen
term written out. `m12`:

| container | ret tot | sd(log Λ_tot) | Jensen U0.5 | Jensen U1.5 | Jensen O3.5 |
|---|---:|---:|---:|---:|---:|
| `raw` | 1.000 | 0.1876 | +0.00118 | +0.00199 | +0.00040 |
| `inv_A_pool` | 0.094 | 0.0858 | **+0.00034** | +0.00056 | +0.00012 |
| `inv_C_sqrt` | 0.303 | 0.1222 | +0.00054 | +0.00091 | +0.00019 |
| `inv_D_sup` | 0.092 | 0.0895 | +0.00035 | +0.00059 | +0.00012 |
| `inv_D_tot` | 1.000 | 0.1871 | +0.00117 | +0.00198 | +0.00039 |
| `inv_B_full` | 1.000 | 0.1886 | **+0.00119** | +0.00200 | +0.00039 |

**H2's mechanism is confirmed, exactly and cleanly.** The Jensen term is a monotone
function of retained TOTALS dispersion and of nothing else: `D_sup`, which preserves
supremacy, sits on top of `A_pool`; `D_tot`, which preserves totals, sits on top of
`B_full`. The pool suppresses it by a factor of 3.5 and full preservation restores it
to the raw model's value to three significant figures. The scheme pair was designed
to isolate this and it isolated it.

**And it does not matter.** The whole term is **0.12 percentage points** of
probability at its largest. Set against it:

| quantity | `m12` |
|---|---:|
| realised goalless rate, 710 fixtures | 0.0620 |
| predicted `P(under 0.5)`, across all 15 containers | 0.0682 – 0.0693 |
| **bias (predicted − realised)** | **+0.0063 to +0.0074** |
| the entire Jensen term | ≤ 0.0012 |

The model over-predicts goalless draws by roughly 0.65pp under every scheme, and at
most a fifth of that is dispersion. **The `P(under 0.5)` bias is a location error,
not a dispersion error**, and no dispersion transform can fix it or meaningfully
worsen it.

Two scope notes are owed. `eda/README.md` Discovery 2 quotes a realised goalless rate
of **3.36%** against a predicted 6.97%; this study measures **6.20%** over its 710
gate-restricted fixtures and **6.82%** over the 44 of them the T−25 O/U 0.5 ladder
actually quotes. Those are different fixture sets answering different questions and
must not be read as a contradiction. On the quoted subset `m12`'s calibrated containers sit
*under* the realised rate (predicted 0.0662–0.0678, bias −0.0020 to −0.0004) and only
the raw model sits above it (0.0710, +0.0028), so the "manufactured longshot mass" of
Discovery 2 is not visible at T−25 on this book —
which is a statement about coverage and de-vigging (§7.1), not a refutation of it.

The Jensen term also cuts both ways, and the O/U 3.5 column shows it: restoring
totals dispersion moves `bias(over 3.5)` from −0.0069 toward −0.0047, i.e. it
*improves* the over-tail while worsening the under-tail. On the family scores the two
cancel — `ou_35` LogLoss is 0.66028 for `inv_A_pool` and 0.65991 for `inv_B_full`,
marginally in preservation's favour.

### 8.5 The premise fails — dispersion does not move the Kelly stake

This is the finding that settles H1, and it needs no drawdown argument. Mean
portfolio exposure, `m12`, canonical trust, production risk budget:

| container | ret side | mean exposure | vs `A_pool` |
|---|---:|---:|---:|
| `raw` | 1.000 | 0.06878 | — |
| `std_A_pool` | 0.706 | 0.04320 | — |
| `std_B_full` | 1.000 | 0.04307 | **−0.3%** |
| `std_C_sqrt` | 0.840 | 0.04315 | −0.1% |
| `std_D_sup` | 0.819 | 0.04312 | −0.2% |
| `inv_A_pool` | **0.085** | 0.03067 | — |
| `inv_B_full` | **1.000** | 0.03078 | **+0.4%** |
| `inv_C_sqrt` | 0.291 | 0.03068 | +0.0% |
| `inv_D_sup` | 0.511 | 0.03101 | +1.1% |

**Restoring 11.8× the posterior log-variance changes the staked exposure by 0.4%.**
Meanwhile the raw → calibrated step cuts exposure by **37%** (0.0688 → 0.0432) and
then by a further **29%** (→ 0.0307) — under a dispersion transform that is, by
construction, doing nothing to the width in the second step and everything to it in
the first.

The stake shrinkage §7.4 recorded is therefore caused by the **location shift**, not
by the variance contraction. That is not a subtle attribution: the model's edges
`p_model − p_market` shrink toward zero because the pool moves the location toward
the market, and Kelly stakes on edges. Posterior width enters the predictive
probability only through a mixture over `Λ` whose spread (`cv(Λ_tot)` ≈ 0.03–0.07)
is small beside the Poisson sampling variance the score grid already carries, so
changing it changes the probabilities in the fourth decimal place and the stakes with
them.

**H1 is refuted at its premise.** There was no Kelly sizing to recover, because none
was lost to dispersion.

### 8.6 Gate 2 — the portfolio at the production risk budget

`SlateDrawdown(23.0)`, `FixedCap(0.25)`, `DailySlate()`, `FractionalKelly(0.30)`, 11
tradeable directions, full T−25 book. `raw` and `*_A_pool` reproduce §7.4 exactly.

| model | container | trust | bets | return % | MDD % | Sharpe | Calmar |
|---|---|---|---:|---:|---:|---:|---:|
| `m12` | `raw` | canonical | 1127 | **+151.52** | −16.15 | 1.592 | 9.383 |
| `m12` | `std_A_pool` | canonical | 1103 | +72.85 | −11.36 | 1.606 | 6.414 |
| `m12` | `std_B_full` | canonical | 1100 | +73.38 | −11.22 | 1.609 | 6.537 |
| `m12` | `std_B_anch` | canonical | 1105 | **+73.65** | −11.25 | 1.613 | **6.548** |
| `m12` | `std_C_sqrt` | canonical | 1102 | +73.14 | −11.31 | 1.609 | 6.469 |
| `m12` | `std_D_sup` | canonical | 1099 | +73.20 | −11.26 | 1.606 | 6.502 |
| `m12` | `std_D_tot` | canonical | 1102 | +73.26 | −11.26 | 1.613 | 6.504 |
| `m12` | `inv_A_pool` | canonical | 975 | +65.64 | **−7.76** | 1.772 | 8.458 |
| `m12` | `inv_B_full` | canonical | 967 | +65.32 | −7.85 | 1.750 | 8.319 |
| `m12` | `inv_B_anch` | canonical | 984 | **+66.82** | −7.95 | **1.775** | 8.401 |
| `m12` | `inv_C_sqrt` | canonical | 973 | +65.52 | −7.75 | 1.765 | 8.455 |
| `m12` | `inv_D_sup` | canonical | 970 | +65.23 | −7.90 | 1.747 | 8.256 |
| `m12` | `inv_D_sup_anch` | canonical | 977 | +66.08 | −7.85 | 1.762 | 8.412 |
| `m12` | `inv_D_tot` | canonical | 974 | +65.52 | −7.78 | 1.771 | 8.421 |
| `m05` | `raw` | canonical | 1113 | **+130.15** | −16.07 | 1.544 | 8.101 |
| `m05` | `std_A_pool` | canonical | 1078 | +63.01 | −10.84 | 1.484 | 5.815 |
| `m05` | `std_B_anch` | canonical | 1080 | **+64.07** | −10.73 | **1.494** | **5.972** |
| `m05` | `inv_A_pool` | canonical | 944 | +63.91 | **−6.86** | 1.943 | 9.315 |
| `m05` | `inv_B_full` | canonical | 941 | +63.76 | −7.22 | 1.905 | 8.828 |
| `m05` | `inv_B_anch` | canonical | 956 | **+65.40** | −6.92 | **1.944** | **9.450** |

The whole spread across seven dispersion schemes is **1.6 points of return** on a
+65% base. The raw-to-calibrated gap on the same axis is **86 points**. Dispersion is
a rounding error on the thing it was proposed to fix.

There is nonetheless a consistent ordering inside the noise, and it holds on **both**
models and **both** location laws:

```
B_anch  >  D_sup_anch  ≳  A_pool ≈ C_sqrt ≈ D_tot  >  B_full  ≳  D_sup
```

The **anchored** preserving schemes are at the top and the **unanchored** ones at the
bottom, with the pool in the middle. `B_full` and `B_anch` have identical dispersion
in every basis and differ only in `κ` — 0.13–0.50% of predictive rate — and that
single difference is worth **+1.5 points** of return on `m12 inv` (+65.32 → +66.82)
and **+1.6** on `m05 inv` (+63.76 → +65.40), plus 0.04 of Sharpe.

**So the operative variable is not the width at all. It is the first predictive
moment that changing the width drags along with it.** That is a Jensen effect, as H2
said — but on `E[Λ]`, not on the zero-goal mass, and it is a *bias* rather than a
*tail* story.

### 8.7 The arbiter — return inside a common drawdown

A return quoted at a fixed λ compares two different amounts of risk taken, and H1 and
H3 cannot be separated by any such row. Panel F therefore sweeps the **whole (λ,
Kelly) surface** — λ ∈ {23, 18, 15, 12, 10, 8} × Kelly ∈ {0.30, 0.40, 0.50, 0.60},
24 settings per container — and reads off the **best return whose realised drawdown
is no deeper than the raw arm's at the production budget**. Same fixtures, same
book, same budget, every arm handed both knobs.

`m12`, canonical trust, budget −16.15%:

| container | λ | Kelly | return % | MDD % | Sharpe |
|---|---:|---:|---:|---:|---:|
| `inv_B_anch` | 8.0 | 0.60 | **+195.90** | −15.69 | 1.729 |
| `inv_D_sup_anch` | 8.0 | 0.60 | +192.73 | −15.38 | 1.713 |
| `inv_D_tot` | 8.0 | 0.60 | +192.42 | −15.35 | 1.729 |
| `inv_A_pool` | 8.0 | 0.60 | +192.23 | −15.32 | 1.727 |
| `inv_C_sqrt` | 8.0 | 0.60 | +191.68 | −15.30 | 1.720 |
| `inv_B_full` | 8.0 | 0.60 | +190.55 | −15.15 | 1.703 |
| `inv_D_sup` | 8.0 | 0.60 | +189.63 | −15.13 | 1.698 |
| **`raw`** | 23.0 | 0.30 | **+151.52** | −16.15 | 1.592 |
| `std_B_anch` | 18.0 | 0.40 | +98.29 | −14.25 | 1.577 |
| `std_A_pool` | 18.0 | 0.40 | +97.20 | −14.37 | 1.572 |

`m05`, canonical trust, budget −16.07%: `inv_B_anch` **+191.51%** at −14.87, against
`inv_A_pool` +186.70% and `raw` +130.15%. Same shape.

Three things follow, in descending order of importance.

**1. §7.4's headline reverses at matched risk, and by a wide margin.** The
inverse-form calibration returns **+192%** at a drawdown *shallower* than the raw
model's +151.52% at −16.15%. §7.4 reported the same container at +65.64% and
concluded "at a fixed risk budget, calibration costs more return at T−25 than it did
at the close". That conclusion was an artefact of the fixed budget: the calibrated
arm was leaving **8.4 percentage points of drawdown headroom unspent**, and r04's
λ-only ladder could not reach it because λ alone saturates (§8.8). Spending it with
both knobs is worth **+41 points** of compound return over raw and **+127 points**
over the same container at the production settings. **H3 is supported, and it is by
some distance the largest effect in this study.**

**2. r04's ranking of the location laws also reverses.** §7.4's risk-matched panel
put `std` ahead of `inv` on `m12` (+134.17 vs +64.09). Inside a common drawdown with
both knobs available, `inv` reaches +192.23 and `std` only +97.20 — because `inv`
strikes fewer, higher-conviction bets and so has far more headroom per unit of
drawdown to lever into. The earlier ranking was a statement about how far a
one-dimensional ladder happened to reach, not about the containers.

**3. Dispersion is worth ~2% of what the risk knob is worth.** The best scheme beats
the pool by **+3.67 points** (195.90 vs 192.23) on `m12` and **+4.81** on `m05`.
The risk knob beats the production setting by **+126.59 points** on the same
container. Both are real; they are not the same order of magnitude, and a project
with finite attention should spend it on the second.

Every `inv` row above is marked `reached = false` in
`results/r05_variance_frontier.csv`: those arms cannot spend the budget even at the
loosest setting on the grid (deepest reachable −15.15% to −15.69% against −16.15%).
Their returns are therefore a **lower bound** on what a matched-risk comparison would
give them, which strengthens the conclusion rather than weakening it.

### 8.8 The risk knob's ceiling, and why r04 could not find this

`SlateDrawdown` solves for a scalar `k` by bisection on `[0, 1]`
(`src/Portfolio/implementations/risk.jl`, `_bisect_k`), so **`k` can only shrink a
stake vector, never lever it up**. That produces two regimes, and only one knob is
live in each. Max drawdown against λ, `m12`, canonical trust:

| container | Kelly | λ=23 | λ=18 | λ=15 | λ=12 | λ=10 | λ=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw` | 0.30 | −16.15 | −19.06 | −19.29 | −19.29 | −19.29 | −19.29 |
| `raw` | 0.60 | −16.36 | −20.42 | −23.97 | −30.33 | −36.31 | −39.87 |
| `std_A_pool` | 0.30 | −11.36 | −14.03 | −14.03 | −14.03 | −14.03 | −14.03 |
| `std_A_pool` | 0.60 | −11.39 | −14.37 | −17.04 | −20.93 | −24.87 | −27.34 |
| `inv_A_pool` | 0.30 | −7.76 | −7.85 | −7.85 | −7.85 | −7.85 | −7.85 |
| `inv_A_pool` | 0.60 | −7.83 | −9.85 | −11.64 | −14.27 | −15.26 | **−15.32** |

* **While `k < 1`** the constraint binds, and it absorbs a uniform stake change
  exactly — `eda/README.md` Discovery 4 seen from the other side. At λ = 23, Kelly
  0.30 and 0.60 give −16.15% and −16.36%: the Kelly fraction is inert.
* **Once `k` pins at 1** the risk model is no longer doing anything, λ moves nothing
  (every `k = 0.30` row is flat from λ = 15 down), and the Kelly fraction becomes the
  only live risk knob.

This is why r04 §7's risk-matched panel reported "NOT MATCHED" on every row and had
to caveat itself: it swept λ only, so it hit the ceiling at λ ≈ 12–15 and could not
go further. The headroom it was looking for was real; the knob it was turning had
stopped working.

The ceiling is a container property. At the loosest setting on the grid the deepest
attainable drawdown is −39.87% for `raw`, −27.34% for `std_A_pool` and **−15.32%**
for `inv_A_pool` — which is why the `inv` arms cannot reach the raw arm's budget at
all. It is also the reason there is a real question here that this section does not
answer: whether λ = 8 with Kelly 0.60 is a configuration anyone should deploy, as
against one that happens to sit inside a backtested drawdown. See §8.11.

### 8.9 Directional audit — where the anchor's 1.5 points come from

Over 2.5, flat trust (the canonical trust gates this direction to zero, so it is
absent there by design, not for want of edge). `m12`, `inv` law:

| container | rate ratio | bets | win rate | Kelly ROI % | flat ROI % |
|---|---:|---:|---:|---:|---:|
| `inv_A_pool` | 1.0000 | 42 | 0.476 | +12.83 | +1.39 |
| `inv_B_full` | 1.0038 | **48** | 0.458 | +10.48 | **−1.69** |
| `inv_B_anch` | 1.0000 | **38** | 0.500 | **+14.32** | **+7.10** |

`m05`, `inv` law: `A_pool` 47 bets / +5.51 Kelly ROI / −2.09 flat; `B_full` **57
bets** / +2.96 / **−8.21**; `B_anch` 42 bets / +6.90 / −4.50.

**This is the mechanism behind §8.6's ordering, and it is legible.** Unanchored
preservation raises `E[Λ]` by 0.4%, which shifts probability mass toward the Over
side, which makes the allocator strike **six more Over 2.5 bets on `m12` and ten more
on `m05`** — and those marginal bets lose. Anchoring removes the rate inflation, the
marginal bets go away, and the direction's flat ROI improves by **8.8 points** on
`m12`. Over 2.5 is the direction `eda/README.md` Discovery 3 identifies as
retail-shaded and README §7 records as rescued by calibration; it is exactly where a
0.4% upward bias in the goal rate is most expensive.

Under 2.5, canonical trust, is the largest capital consumer (19–22% of stake) and is
indifferent: `m12 inv` Kelly ROI ranges +30.37 (`B_anch`) to +32.46 (`B_full`) across
all seven schemes, with no ordering that repeats on `m05`.

### 8.10 Closing-line value, and out of sample

CLV, `m12`, canonical trust, Panel P. Every arm remains positive; the calibrated arms
keep roughly twice the raw model's mean CLV, as §7.5 found.

| container | bets | mean CLV % | stake-weighted % | % positive |
|---|---:|---:|---:|---:|
| `raw` | 1009 | +0.314 | +1.029 | 50.7 |
| `std_A_pool` | 983 | +0.397 | +1.063 | 52.1 |
| `inv_A_pool` | 871 | **+0.573** | +1.056 | **55.0** |
| `inv_C_sqrt` | 869 | +0.570 | +1.052 | 54.9 |
| `inv_B_anch` | 879 | +0.553 | +1.048 | 54.9 |
| `inv_B_full` | 865 | +0.547 | +1.035 | 54.6 |
| `inv_D_sup` | 867 | +0.521 | +1.030 | 54.4 |

**No dispersion scheme improves CLV on the pool, and every one of them is at best
level with it.** The pool and its conjugate cousin `C_sqrt` are the top two on both
models. That is the expected result once §8.5 is believed: CLV is a property of
*which* bets get struck, which is a location question, and dispersion barely moves
the bet set (871 bets against 865–879).

Out of sample — slates after 2025-05-03, T−25 prices, canonical trust:

| model | container | return % | MDD % | Sharpe |
|---|---|---:|---:|---:|
| `m12` | `raw` | **+31.77** | −16.15 | 0.941 |
| `m12` | `std_A_pool` | +24.63 | −8.61 | 1.269 |
| `m12` | `std_B_anch` | +24.87 | −8.69 | **1.274** |
| `m12` | `inv_A_pool` | +15.22 | **−7.76** | 1.010 |
| `m12` | `inv_B_full` | +14.54 | −7.85 | 0.963 |
| `m12` | `inv_B_anch` | +15.38 | −7.95 | 1.008 |
| `m05` | `raw` | **+30.09** | −16.07 | 0.945 |
| `m05` | `inv_A_pool` | +18.84 | −6.86 | **1.351** |
| `m05` | `inv_B_anch` | +19.26 | −6.92 | 1.353 |
| `m05` | `inv_B_full` | +17.80 | −7.22 | 1.264 |

The out-of-sample window reproduces the ordering rather than contradicting it —
`B_anch` above `A_pool` above `B_full`, on both models and both laws — but the whole
spread is **0.8 points of return** over **50 slates**, which is far inside what
that many slates can resolve. It should be read as "not contradicted", not as
"confirmed".

### 8.11 Verdict on Phase 4

1. **H1 is refuted at its premise, not merely unsupported.** Restoring the posterior
   width — up to **11.8×** under the `inv` law — moves the Kelly stake by **0.4%**
   (§8.5). §7.7's reading, that the `w²` contraction shrinks Fractional Kelly stakes
   and costs compounding, is wrong. The stake shrinkage is caused by the location
   shift contracting the model's edges, and dispersion has essentially no channel to
   the allocator: posterior spread in `Λ` (`cv` ≈ 0.03–0.07) is small beside the
   Poisson variance already in the score grid. **This corrects a claim published in
   §7.7 of this document.**

2. **H2's mechanism is confirmed and quantified, and it is negligible.** The Jensen
   term tracks retained TOTALS dispersion monotonically and exactly — `D_sup` sits on
   `A_pool`, `D_tot` sits on `B_full` — and full preservation multiplies it by 3.5
   (§8.4). It is at most **0.0012** of probability, against a `P(under 0.5)` bias of
   **+0.0065** that no scheme changes. It also cuts both ways: it worsens the under
   tail and improves the over tail, and on family LogLoss the two cancel.

3. **H3 is supported and is the only first-order effect in this study.** Spending the
   calibrated arm's unused drawdown headroom with **both** risk knobs takes `m12 inv`
   from +65.64% to **+192.23%** at a drawdown shallower than the raw model's, beating
   raw's +151.52% by 41 points (§8.7). r04 could not find this because λ alone
   saturates once `SlateDrawdown`'s `k` pins at 1 (§8.8). This also **reverses two
   Phase 3 conclusions**: that calibration costs return at tradeable prices, and that
   `std` outranks `inv`.

4. **H4 survives, in a form nobody proposed.** The winning scheme is `B_anch` — full
   variance preservation with the predictive rate anchored back to the pool's — on
   both models, both location laws, at the production budget, at matched risk, and
   out of sample. But the anchor is doing all the work and the variance preservation
   none: `B_full` and `B_anch` have identical dispersion in every basis and differ
   only by 0.13–0.50% of predictive rate, and that difference is worth +1.5 points of
   return (§8.6) and +8.8 points of Over 2.5 flat ROI (§8.9). The supremacy/totals
   asymmetry the scheme family was built to test is real in the diagnostics and
   inert in the portfolio.

5. **The falsification control did its job.** `D_tot` was included so that a `D_sup`
   result could not be read as a supremacy story by default. It reversed the expected
   sign twice: the proper-score cost of dispersion turned out to track **supremacy**,
   while the Jensen tail term turned out to track **totals**. Half the experiment
   would have supported a confident and wrong conclusion.

6. **`C_sqrt` — the coherent Bayesian update — is indistinguishable from the pool.**
   If the market is an independent noisy observation with precision `τ`, the correct
   posterior variance is `wσ²`, not the pool's `w²σ²`. Every score and every
   portfolio number for `C_sqrt` sits on `A_pool`'s to three significant figures.
   The double-counting the log-linear pool commits is real and it is unmeasurable
   here, which is worth knowing before anyone rebuilds the calibrator to fix it.

**Recommendation.** Leave `calibrate_latents` exactly as it is. The pool's `w²`
contraction costs nothing worth recovering, and the one change that does pay —
anchoring the predictive rate so the calibrated container is not silently 0.4% hotter
than its own location says — is a two-line correction to the existing transform
rather than a new scheme. The engineering effort this stream has left belongs in the
risk budget, where §8.7 found 41 points, and not in the posterior, where §8.6 found
1.6.

### 8.12 What this section does not settle

* **The λ = 8 / Kelly 0.60 configuration is not validated, only measured.** §8.7
  reports the best point on a 24-setting surface, chosen against the same slates the
  return is read off. That is the identical in-sample selection bias r02 §6.2 and r04
  §7 flagged, and it is why those panels are labelled mechanism demonstrations. A
  deployable claim needs the risk setting chosen on one window and scored on another,
  and this runner does not do that.
* **A backtested drawdown is not a risk limit.** Deep Kelly fractions with a
  non-binding risk model are exactly the regime where a fill model matters most
  (§7.8), and the +192% row stakes roughly twice the production exposure into
  Scottish League Two liquidity.
* **The differences that survive are small relative to the sample.** 1.6 points of
  return over 99 slates, and 0.8 over the 50 out-of-sample ones, are consistent orderings rather
  than resolved ones. They repeat across two models and two location laws, which is
  why they are reported; they are not separately significant.
* **The O/U 0.5 evidence is thin.** The T−25 matched book quotes that ladder on 44 of
  the 710 priced fixtures (the full T−25 book on 312). The market-free Jensen audit
  in §8.4 has full coverage and is the load-bearing tail evidence; the `ou_05` family
  scores at n = 69 are context.
* **`observation_params` is `nothing` throughout.** These containers price a
  double-Poisson grid. A negative-binomial or copula observation carries its own
  dispersion, and nothing here says how these transforms interact with it.

---

## 9. Boundaries

* Reads `mcmc_experiments` (posteriors, via `PostgresStorage`) and `betdb`
  (odds, results). **Writes neither.** No run, portfolio or config registration.
* `betdb.paper_runbook` is never opened. The live console on **8085** and the
  replay console on **8086** are not this stream's business and were verified up
  and untouched while this code was written.
* Credentials are resolved by `PostgresStorage` / `Data` from the environment and
  never printed.
