# Generative rate calibration — EDA and Layer-2 overhaul

Prototype stream for the work package in
[`CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md`](CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md).

**Status — 2026-09-04.** Phase 1 executed on `mcmc-beast`. §5 carries measured
numbers. Phase 2 (`r02_portfolio_direction_audit.jl`) is written and its results
are in §6; until those are filled in, **nothing here supports a bankroll claim.**

**Headline.** Generative rate calibration improves out-of-sample LogLoss on both
Scottish Lower candidates, and the improvement is concentrated almost entirely in
the **large-edge** regime — but the winning direction is **shrinkage**, not the
Ireland stream's conviction. `:standard_gaussian`, which pulls extreme claims back
toward the close, beats `:inverse_gaussian`, which pushes them away from it, on
every model and every scope measured. See §5.5.

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
| `r02_portfolio_direction_audit.jl` | Runner 2. Not yet written. The 13-direction portfolio and trust-vector audit; Gate 2. |
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
| **Gate 2** | r02 | Bankroll > +130%, Sharpe ≥ 1.416, max drawdown no worse than −20.5%. Not yet implemented. |

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

## 6. Boundaries

* Reads `mcmc_experiments` (posteriors, via `PostgresStorage`) and `betdb`
  (odds, results). **Writes neither.** No run, portfolio or config registration.
* `betdb.paper_runbook` is never opened. The live console on **8085** and the
  replay console on **8086** are not this stream's business and were verified up
  and untouched while this code was written.
* Credentials are resolved by `PostgresStorage` / `Data` from the environment and
  never printed.
