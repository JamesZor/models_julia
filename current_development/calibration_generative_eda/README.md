# Generative rate calibration — EDA and Layer-2 overhaul

Prototype stream for the work package in
[`CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md`](CALIBRATION_GENERATIVE_RATE_EDA_PROMPT.md).

**Status — 2026-09-04.** Phase 1 code is written and parse-checked. **Nothing has
been executed and this README therefore carries no measured numbers.** Every
results table below is a placeholder with its columns fixed so a run can fill it
without changing the question being asked. Do not cite this file as evidence of
anything until §5 is filled in with run context.

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

**Not yet measured.** Fill each table from `results/` together with the run context
(date, host, threads, git commit, the `m12`/`m05` run UUIDs read from
`mcmc_experiments`, and the fixture and observation counts G-A reported).

### 5.1 Inversion quality — `results/r01_market_inversion.csv`

| Fixtures | Accepted | Coverage | Median SSE | p90 SSE | Refusals by reason |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### 5.2 Optimum per functional form — `results/r01_best_per_form.csv`

Headline scope (1X2 + O/U 2.5 + BTTS).

| Model | Method | `w_base` | `σ` | LogLoss | Δ vs baseline | ECE | BF ECE | median `w` | var retained |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — |

### 5.3 Edge-stratified LPD — `results/r01_sweep_scores.csv`

Buckets anchored on the raw model's edges.

| Model | Spec | LPD `|Δp|<0.02` | N | LPD `|Δp|>0.05` | N | LPD all |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### 5.4 Per-family scores — `results/r01_family_scores.csv`

The Over 2.5 row is the one the work package asks about: does rate calibration
rescue a direction that has been negative-ROI on this league?

| Model | Spec | Family | N | LogLoss | Brier | ECE |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### 5.5 Verdict

Not reached.

---

## 6. Boundaries

* Reads `mcmc_experiments` (posteriors, via `PostgresStorage`) and `betdb`
  (odds, results). **Writes neither.** No run, portfolio or config registration.
* `betdb.paper_runbook` is never opened. The live console on **8085** and the
  replay console on **8086** are not this stream's business and were verified up
  and untouched while this code was written.
* Credentials are resolved by `PostgresStorage` / `Data` from the environment and
  never printed.
