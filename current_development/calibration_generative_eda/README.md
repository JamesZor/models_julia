# Generative rate calibration — EDA and Layer-2 overhaul

Prototype stream for the work package in
[`CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md`](CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md).

**Status — 2026-09-04.** Both phases executed on `mcmc-beast`. §5 is the
calibration result, §6 the portfolio result.

**Headline.** Generative rate calibration improves out-of-sample LogLoss on both
Scottish Lower candidates; the gain is concentrated almost entirely in the
**large-edge** regime; and the winning direction is **shrinkage**, not the Ireland
stream's conviction. On the portfolio it raises flat ROI, Sharpe and drawdown on
every arm while *lowering* total return — and arm F shows that shortfall is
**exposure, not skill**: at a drawdown matched to the raw model's, calibrated `m12`
returns **+151.09%** against raw's +126.09%. Out of sample it produces the best arm
measured anywhere in this stream, beating the production champion on return,
Sharpe and drawdown at once. It does **not** replace `CanonicalScottishLowerTrust`,
and it does not rescue Over 2.5 far enough to un-gate it.

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
| `results/` | Replaceable CSV artefacts. Re-running overwrites them. |

Run:

```bash
julia --project -t 16                     # mcmc-beast; -t 8 on archpc
julia> include("current_development/calibration_generative_eda/r01_sweep_rate_calibration.jl")

R01_SMOKE=1 julia --project -t 16 ...     # 3-spec dry run, one model, not a result
```

---

## 3. Design decisions that change what the numbers mean

These are recorded here because each one would silently alter a headline figure if
it were changed, and none of them is visible in a results table.

**The calibrating price is the close.** Rates are inverted from the Betfair
close (TWA over `[−20 min, kick-off]`) — the snapshot experiment 06 scored against.
A closing price is not available when a bet is struck, so **any P&L this stream
eventually reports is an upper bound** until the snapshot is moved back to T−25.

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
| **Gate 2** | r02 §6.7 | Bankroll > +130%, Sharpe ≥ 1.416, max drawdown no worse than −20.5%. Measured; and §6.7 argues the return threshold is mis-specified for a variance-contracting transform. |

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

## 7. Boundaries

* Reads `mcmc_experiments` (posteriors, via `PostgresStorage`) and `betdb`
  (odds, results). **Writes neither.** No run, portfolio or config registration.
* `betdb.paper_runbook` is never opened. The live console on **8085** and the
  replay console on **8086** are not this stream's business and were verified up
  and untouched while this code was written.
* Credentials are resolved by `PostgresStorage` / `Data` from the environment and
  never printed.
