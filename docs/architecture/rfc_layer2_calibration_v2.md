# RFC — Layer-2 Calibration v2: generative rate calibration as a production tier

* **Status.** Accepted for implementation on `feat/modernize-calibration-layer2`.
* **Supersedes.** `src/Calibration/shift_models/basic_logit.jl` (`BasicLogitShift`), deprecated but retained.
* **Sources.** `current_development/calibration_generative_eda/` — `l01_generative_calibrator.jl`,
  `l02_point_in_time_book.jl`, `l03_variance_schemes.jl`, and the measured results in that
  stream's [`README.md`](../../current_development/calibration_generative_eda/README.md)
  (commits `f0e2b7d8` … `fd17cbe6`).
* **Work package.** `current_development/calibration_generative_eda/MODERNIZE_CALIBRATION_PROMPT.md`.

---

## 1. What is wrong with the module being replaced

`BasicLogitShift` fits a one-parameter GLM offset per market and applies it to scalar
outcome probabilities **independently**:

```
p* = logistic(logit(p) + c_market)
```

Applied to `over_25` and `under_25` with the two `c` values those two GLMs happen to
learn, `p*(over) + p*(under) != 1`. Applied to `home`/`draw`/`away`, the 1X2 sum drifts
off 1. There is no scoreline distribution behind the shifted numbers at all, so
"Over 2.5", "BTTS yes" and "Home" become three unrelated claims that a Kelly allocator
is then invited to hold simultaneously. Coherence is not merely unchecked; it is
**unrepresentable** in that construction.

The replacement moves the shift one level down, to the generative intensity. Every
derivative price is then read off **one** 12×12 score tensor, so 1X2, every totals line
and BTTS are three partitions of one object and cannot disagree. Coherence becomes
structural rather than audited — though §7 audits it anyway, because "structural" is a
claim about code that should be measured once.

---

## 2. The construction, in four steps

For each fixture `i`, each side `s ∈ {h, a}`, over the `D` posterior draws:

1. **Invert the tradeable book.** Nelder–Mead on `Features.DoublePoissonMarketFeature`
   recovers `(λ_mkt_h, λ_mkt_a)` from the de-vigged quotes, under four acceptance gates
   (quote count, residual SSE, convergence, rate bounds). A refused inversion is named,
   never imputed.
2. **Measure the discrepancy.** `Δ_s = log median_d(λ_s^(d)) − log λ_mkt_s`. The
   **median**, not the mean: the weight is a statement about where the bulk of a
   right-skewed rate posterior sits relative to the book.
3. **Pool every draw.** `log λ̃_s^(d) = c_s + [M · u^(d)]_s + κ_s`, where
   `m_s = mean_d log λ_s^(d)`, `u_s^(d) = log λ_s^(d) − m_s`,
   `w_s = law(Δ_s)` and `c_s = w_s·m_s + (1 − w_s)·log λ_mkt_s`.
   With `M = diag(w_h, w_a)` and `κ = 0` this is exactly the log-linear opinion pool
   `log λ̃ = w·log λ + (1−w)·log λ_mkt`.
4. **Price it through the unchanged pipeline.** The result is the *same concrete latent
   container type* the raw run held, so `Predictions.alloc_score_grid`,
   `Predictions.compute_score_grid!`, `Evaluation.build_evaluation_context`,
   `Portfolio.BookWorkspace` and `Portfolio.simulate_portfolio` need no new methods.

Step 4 is the whole design. Everything else is arithmetic on two matrices.

### 2.1 The three location laws

| law | `w(0)` | `w(±∞)` | reads as |
|---|---|---|---|
| `InverseGaussianLaw(w_base, σ)` | `w_base` | 1.0 | trust the market on noise, the model on structural edges |
| `StandardGaussianLaw(w_base, σ, w_max)` | `w_max` | `w_base` | optimiser's-curse shrinkage of extreme claims |
| `StaticGeometricLaw(w)` | `w` | `w` | a constant pool — the control for "does Δ-dependence buy anything" |

```
inverse  : w = w_base + (1 − w_base)·(1 − exp(−Δ²/2σ²))
standard : w = w_base + (w_max − w_base)·exp(−Δ²/2σ²)
static   : w = w
```

**Which one is right depends on the sharpness of the book being pooled with**, and that
is measured, not asserted. Against the Betfair *close* the standard (shrinkage) form
wins on LogLoss; against the softer **T−25** book the *inverse* (conviction) form wins,
reversing the ordering (stream README §7.3). The parameters do not transfer between
price instants: r01's close-fitted pick gives up 0.0015–0.0020 LogLoss when moved to
T−25 rates. A calibrator is therefore **named for the instant it was fitted at**, and
`GenerativeRateCalibrator` records `book_as_of_minutes` so a T−25 calibrator applied to a
closing book is a visible mismatch rather than a silent one.

### 2.2 The dispersion map `M`, and why it defaults to the pool

The pool contracts posterior log-variance by exactly `w²`. `l03_variance_schemes.jl`
made `M` and `κ` independently controllable so the question "is that contraction
load-bearing?" could be answered by experiment. It was, and the answer is **no**:

* restoring up to **11.8×** the posterior log-variance moves staked exposure by
  **0.4%** (README §8.5) — posterior spread in `Λ` (`cv ≈ 0.03–0.07`) is small beside the
  Poisson variance already in the score grid;
* the Jensen tail term the variance drives is at most **0.0012** of probability, against
  a `P(under 0.5)` bias of `+0.0065` that no scheme changes (§8.4);
* `C_sqrt` — the *coherent* Bayesian update, `Var = wσ²`, which the log-linear pool
  double-counts down to `w²σ²` — is indistinguishable from the pool to three significant
  figures on every score and every portfolio number (§8.6).

So the shipped default is `PoolDispersion()`, `M = diag(w_h, w_a)`, which reproduces
`l01.calibrate_latents` **bit for bit**. The other maps ship because they were built,
validated and measured, and because `B_anch` is the one that won §8.6 by 1.5 points of
return — an ordering that repeats across two models and two location laws but is *not*
separately significant over 99 slates. It is available; it is not the default.

### 2.3 The anchor, and an honest note about it

`κ` is the **Jensen anchor**: restoring dispersion at a fixed log-location raises the
predictive rate (`E[Λ] = E[e^{log Λ}]` grows with `Var(log Λ)`), so a wider container is
also a *hotter* one. `anchor = :pool_mean` chooses `κ` so the draw-mean rate equals the
rate the plain pool would have produced:

```
κ_s = log mean_d exp(w_s·u_s^(d)) − log mean_d exp([M·u^(d)]_s)
```

computed on the draws, not from a log-normal formula, so it is exact whatever the
posterior shape.

**On `PoolDispersion` this is identically zero**, because `M = diag(w_h, w_a)` makes the
two sums the same sum. That is not a defect: it is the statement that the production pool
is *already* rate-anchored, which is why §8.11's recommendation ("leave `calibrate_latents`
exactly as it is") and the anchor's measured value (`+8.8` points of Over 2.5 flat ROI,
`+1.5` of return — both **relative to `B_full`**, the unanchored variance-preserving
scheme) are consistent rather than in tension. The default is `:pool_mean` anyway, so
that the moment anyone selects a non-pool `M` the anchor is already on; and
`test_calibration_v2.jl` asserts the no-op on the pool rather than leaving a reader to
work it out.

### 2.4 The fallback is `w = 1`, bit for bit

A fixture whose book cannot be inverted passes through **unchanged** — the raw draws,
not a league-mean rate and not a dropped row. Dropping would change which fixtures two
calibrators score and make their rows incomparable; inventing a rate would price a
fixture from inputs the pipeline declined to use.

A side at `w == 1.0` **copies** its raw draws rather than computing `exp(1·log λ + 0)`,
because `exp(log(x)) != x` in Float64 and the identity calibrator has to reproduce the
raw container exactly or it is not a control. Asserted, not assumed (Gate D / test T3).

---

## 3. Type hierarchy — `src/Calibration/types.jl`

```julia
abstract type AbstractCalibrator end
abstract type AbstractGenerativeRateCalibrator <: AbstractCalibrator end

abstract type AbstractCalibrationWeightLaw end
struct StandardGaussianLaw <: AbstractCalibrationWeightLaw   # w_base=0.40, σ=0.15, w_max=1.0
struct InverseGaussianLaw  <: AbstractCalibrationWeightLaw   # w_base=0.25, σ=0.35
struct StaticGeometricLaw  <: AbstractCalibrationWeightLaw   # w=0.40

abstract type AbstractDispersionMap end
struct PoolDispersion     <: AbstractDispersionMap   # M = diag(w_h, w_a)   — l01, the default
struct PreservedDispersion<: AbstractDispersionMap   # M = I
struct ConjugateDispersion<: AbstractDispersionMap   # M = diag(√w_h, √w_a)
struct SupremacyDispersion<: AbstractDispersionMap   # (ρ_s, ρ_t) in the (u_h−u_a, u_h+u_a) basis

Base.@kwdef struct GenerativeRateCalibrator{L,D} <: AbstractGenerativeRateCalibrator
    name::String
    law::L
    dispersion::D            = PoolDispersion()
    anchor::Symbol           = :pool_mean
    fallback::Symbol         = :identity
    inversion::MarketInversionConfig = MarketInversionConfig()
    book_as_of_minutes::Float64      = -25.0
end
```

`calibration_weight(law, Δ)` is the single dispatch point for the location;
`residual_map(map, w_h, w_a) -> (m11, m12, m21, m22)` the single dispatch point for the
dispersion. Adding a law or a map is "add a struct + one method"; no existing file changes.

`fallback` is `:identity` (pass the raw draws through) or `:refuse` (throw, naming the
fixtures). There is deliberately no `:market` or `:league_mean` option — see §2.4.

### 3.1 `CalibratedLatents` — an alias, and why it is not a wrapper

The work package asks for `CalibratedLatents <: AbstractPosteriorLatents` wrapping
`CountLatents`. **This RFC ships it as `const CalibratedLatents = Models.AbstractPosteriorLatents`
instead**, and `calibrate_latents` returns the *same concrete container type* it was given.

The reason is `src/Portfolio/pricing.jl:113`:

```julia
if l isa Models.SmileLatents
    ...  # build a SmileScoreGrid so Over/Under prices off λ_tot·φ(K)
```

A wrapper type fails that `isa`. The book would still build, still price, still stake —
and it would silently **de-smile** the O/U ladder, pricing every totals line off the
score grid instead of the model's own per-strike intensity. That is precisely the failure
`src/predictions/score_computation/smile_poisson.jl` warns about, it produces a plausible
number, and no test that checks "the portfolio ran" would catch it. A wrapper would have
to shadow every `isa`-dispatch and every kernel method in `Predictions` and `Portfolio`,
for no gain: the provenance the wrapper was for lives on `CalibrationResult` (§4), which
is the object a caller actually holds.

So: **no wrapper, no new kernel methods, no `src/Portfolio/` change, and the smile route
keeps working by construction.**

---

## 4. The pipeline contract

```julia
book, refusals = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))
rates          = invert_market_rates(cal, book)                      # Dict{Int, MarketRateFit}
cal_fit        = calibrate_fit(cal, fit, book)                       # CalibratedFit
result, books, rep = run_portfolio_simulation(spec, policy, cal_fit, book, ds)
```

```
Fit (L1/L2)  ──calibrate_fit(cal, fit, book)──▶  CalibratedFit
                                                    │
                       ┌────────────────────────────┼────────────────────────────┐
                       ▼                            ▼                            ▼
              evaluate_predictions          run_portfolio_simulation      save_calibration_db
                (L3, unchanged)              (L4, unchanged)             mcmc_experiments
```

### 4.1 `CalibratedFit`

```julia
struct CalibratedFit{F1<:Training.Fit, F2<:Training.Fit, C<:AbstractCalibrator}
    fit::F1                              # a REAL Training.Fit carrying the calibrated latents
    base_fit::F2                         # the raw run, untouched
    calibrator::C
    market_rates::Dict{Int, MarketRateFit}
    rate_diagnostics::DataFrame          # one row per fixture: Δ, w, κ, var retention, λ before/after
    book_as_of_minutes::Float64
    coverage::NamedTuple
    created_at::DateTime
end
```

`cal_fit.fit` is an ordinary `Fit` — same config (with a `calibrator:` tag), same folds,
same `ConvergenceSummary`, same metadata — whose `latents` field holds the calibrated
container. So **`run_portfolio_simulation(spec, policy, cal_fit.fit, odds, ds)` already
works with zero new code**, and the two forwarding methods below exist only so the
`CalibratedFit` itself can be passed:

```julia
Portfolio.build_books_reported(spec, cf::CalibratedFit, odds, fixtures; kw...) =
    Portfolio.build_books_reported(spec, cf.fit, odds, fixtures; kw...)
Evaluation.fit_latents(cf::CalibratedFit)        = Evaluation.fit_latents(cf.fit)
Evaluation.convergence_verdict(cf::CalibratedFit) = Evaluation.convergence_verdict(cf.fit)
Evaluation.evaluate_predictions(cf::CalibratedFit, ds; kw...) = ...
```

These are methods **added from `Calibration` onto our own types**. `src/Portfolio/` is not
edited. `run_portfolio_simulation`'s `else` branch reaches
`build_books_reported(spec, source, odds, fixtures; require_result, quiet)`, which our
method answers.

### 4.2 Module load order

`Calibration` must be able to name `Portfolio`, `Evaluation` and `Training`. Its include
moves in `src/BayesianFootball.jl` from *before* `Predictions` to *after* `Portfolio`.
Nothing between the two positions references `Calibration` (checked: only
`types-interfaces.jl`'s `AbstractLayerTwoModelConfig`, which is defined earlier and
unaffected).

---

## 5. The tradeable T−25 book — `src/Calibration/book.jl`

Ported from `l02_point_in_time_book.jl` unchanged in substance.

* **Last traded price at or before the cutoff**, per (match, market, line, selection),
  with its `staleness_minutes` carried as a column. `<=`, not `<`: a tick stamped exactly
  at the cutoff is visible at it, and the frame is filtered *before* the group-wise
  `argmax`, so a later tick is unreachable rather than merely unqueried.
* **Not** `summarize_odds(window = (-30, -25))`. Measured on 599,529 ticks: a five-minute
  window at T−25 carries a third of the close book's coverage at a median of **one** tick,
  so its "time-weighted average" is one number with a weight; widening it to recover
  coverage averages prices up to four hours old.
* **Completeness is checked before normalisation.** De-vigging a one-sided quote yields a
  fair probability of exactly 1.0 and no error — which is how the O/U 0.5 ladder came to
  be scored against a fabricated price (README §5.6: the *closing line's own* LogLoss on
  that family was 1.31832 against the model's 0.21098, on 574 one-sided fixtures).
  `expected_selection_count` states the contract per family and `devig_book!` refuses
  ahead of the arithmetic.
* **Staleness and overround gates**, both refusing *by name* into a `refusals` frame.
* **`assert_book_as_of(book, minutes)`.** `Evaluation.build_odds_view` and
  `Portfolio.build_odds_index` read `:odds_close` / `:prob_fair_close` by name, so a T−25
  book must carry those names to pass through unmodified — and a frame named "close"
  holding a T−25 price is exactly what gets mixed up three weeks later. Every frame the
  builder returns also carries `:as_of_minutes`, and this assertion turns the mix-up into
  an error at the call site. `calibrate_fit` calls it.

Measured at T−25 on Scottish Lower: **1,572 of 1,627 fixtures**, median staleness
8 minutes, near-fair overround (README §7.1).

---

## 6. Experiment-database integration — `src/Calibration/db_storage.jl`

### 6.1 Registry

`config_registry` gains `config_type = 'calibrator'`. The column is `config_type`, not
`component_type` as the work package writes it; the work package's name does not exist in
this schema. `Training.save_calibrator(db, name, cal)` / `load_calibrator(db, key)` join
`save_book_spec` / `save_policy_spec` on the existing `_save_component` path, so a
calibrator is registered, hashed, tagged and searchable exactly as a `BookSpec` is.

`_truth_config_type` learns one branch — any type with an `AbstractCalibrator` in its
supertype chain classifies as `"calibrator"` — resolved by name so `Training` (which
loads before `Calibration`) need not reference the type.

### 6.2 New tables (`src/training/inference/db/schema.sql`, idempotent)

```sql
CREATE TABLE IF NOT EXISTS calibration_runs (
    id                   BIGSERIAL,
    calibration_run_id   UUID PRIMARY KEY,
    model_run_id         UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    experiment_name      VARCHAR NOT NULL,
    calibrator_name      TEXT NOT NULL,
    calibrator_hash      VARCHAR(64) NOT NULL,
    config_json          JSONB NOT NULL,
    book_as_of_minutes   DOUBLE PRECISION,
    n_fixtures           INT, n_inverted INT,
    log_loss             DOUBLE PRECISION,
    ece                  DOUBLE PRECISION,
    brier                DOUBLE PRECISION,
    clv_mean_pct         DOUBLE PRECISION,
    clv_weighted_pct     DOUBLE PRECISION,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS calibration_artifacts (
    calibration_run_id      UUID PRIMARY KEY REFERENCES calibration_runs(calibration_run_id) ON DELETE CASCADE,
    calibrator_blob         BYTEA NOT NULL,
    calibrated_latents_blob BYTEA,
    diagnostics_blob        BYTEA
);
```

The work-package schema is followed with four additions: `experiment_name` (every other
table in this database is experiment-scoped and a calibration run that is not would be
the one row you cannot filter), `book_as_of_minutes` and the two coverage counts (the
price instant is the single fact that makes two calibration runs incomparable — §2.1 —
and burying it in `config_json` makes the most important filter a JSON path), and
`diagnostics_blob` (the per-fixture `Δ`/`w`/`κ` frame is what a post-mortem reads and it
does not reconstruct from the calibrator).

`log_loss`/`ece`/`brier` are **headline scope** (`Evaluation.DEFAULT_SCORED_MARKETS`:
1X2 + O/U 2.5 + BTTS), because that is the only scope in which the published Gate-1
thresholds mean anything. Wide-book scores go in `metadata`.

### 6.3 Linking a portfolio run

`Portfolio.save_portfolio_db` already merges a caller's `metadata` into its JSONB column,
so:

```julia
save_portfolio_db(result, model_run_id, db;
                  book_spec = spec, policy_spec = policy,
                  metadata = (; calibration_run_id = string(cal_run_id),
                                calibrator = cal.name,
                                book_as_of_minutes = -25.0))
```

links the two with **zero** change to `src/Portfolio/`. `Calibration.link_portfolio_run`
wraps exactly that call so the key spelling is in one place, and
`portfolio_runs_for_calibration(db, cal_run_id)` reads it back with a JSONB predicate.
No foreign key: `portfolio_runs` must remain insertable without a calibration tier.

### 6.4 Boundaries

`betdb` stays read-only and `betdb.paper_runbook` is never opened. Everything this module
writes goes to `mcmc_experiments` on `mcmc-beast:5432` through
`Training.PostgresStorage`, which resolves its own credentials.

---

## 7. Verification — `test/test_calibration_v2.jl`

| # | Gate | Asserts |
|---|---|---|
| T1 | Weight laws | closed forms at `Δ = 0`, `Δ → ∞`; monotonicity in `|Δ|`; `w ∈ [0,1]`; constructor refusals |
| T2 | **Equivalence with `l01`** | `PoolDispersion` + `:none` reproduces `l01.calibrate_latents` to `< 1e-12` on a synthetic posterior — the prototype's own code, `include`d and run beside it |
| T3 | Identity | `w_base = 1.0` returns draws **bit-identical** (`===` on values) to the raw container |
| T4 | Anchor | `κ ≡ 0` on `PoolDispersion`; on `PreservedDispersion` the draw-mean rate matches the pool's to `< 1e-12`, and the unanchored twin does not |
| T5 | Dispersion algebra | `PreservedDispersion` retains variance 1.0; `PoolDispersion` retains `w²`; `SupremacyDispersion(ρ_s, ρ_t)` retains exactly those in the (s, t) basis |
| T6 | Fallback | a fixture with no accepted inversion passes through bit-identically and is counted, not dropped |
| T7 | **Coherence** | 1X2, four totals lines and BTTS each sum to the same grid mass per fixture, spread `< 1e-12`; the same audit on a `BasicLogitShift` board fails, which is the comparison being made |
| T8 | Book | last-tick-at-or-before; one-sided market refused before de-vigging; staleness and overround gates; `assert_book_as_of` refuses a close book |
| T9 | **Portfolio** | `CountModelBuilder` → `fit_model` → `calibrate_fit` → `run_portfolio_simulation` runs end to end, and `run_portfolio_simulation(…, cal_fit, …)` is field-identical to `run_portfolio_simulation(…, cal_fit.fit, …)` |
| T10 | **DB round-trip** | `ensure_schema!` → `save_calibrator` → `save_calibration_db` → `load_calibration_db` recovers an identical calibrator and container; `save_portfolio_db` links; `portfolio_runs_for_calibration` finds it |
| T11 | Deprecation | `BasicLogitShift` still fits and applies, and `@warn`s once |

T9/T10 skip **with a message** when `mcmc_experiments` or the DataStore cache is out of
reach — a "passed" line from a tier that skipped is not evidence.

---

## 8. What this RFC does not claim

* **The calibration parameters are not validated out of sample.** r03 chose them over the
  full period; README §7.6 holds out the *slates*, not the *spec*. The shipped defaults
  are the measured T−25 optima and should be refitted per segment and per price instant.
* **The T−25 returns are still an upper bound**, now for fill-model reasons rather than
  price-instant ones: bets are struck at the archived *traded* price in whatever size the
  allocator asked for, while the live system rests at the touch and the archive carries at
  most three levels (AGENTS.md §7.4).
* **The +192.2% figure is a mechanism demonstration, not a performance claim.** The
  `λ = 8` / Kelly 0.60 risk setting was chosen against the same slates the return is read
  off (README §8.12). What this module ships is the calibrated *container*; the risk
  budget stays a `PolicySpec` decision made elsewhere.
* **`observation_params` is `nothing` throughout the validated work.** These transforms
  are defined for `CountLatents` and `SmileLatents` grid intensities. A negative-binomial
  or copula observation carries its own dispersion and nothing here says how the two
  interact; `calibrate_latents` therefore carries `observation_params` through unchanged
  and says so.
* **`BasicLogitShift` is deprecated, not deleted.** It warns and keeps working, because a
  legacy script that stops running is a worse outcome than one that prints a line.
