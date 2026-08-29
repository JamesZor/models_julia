# 08 — Unified Evaluation Framework

Six scoring rules computed directly from `06`'s typed posterior containers and `07`'s
`Fit`, with convergence as a gate rather than a footnote, and every legacy
`src/evaluation/` call site kept working.

```
julia --project current_development/08_unified_evaluation_framework/r01_demo.jl
```

Runs in seconds, needs no database, no cache and no MCMC, and exits non-zero if any
gate fails. **78 gates, all passing.**

---

## The problem

`src/evaluation/metrics_methods/*.jl` all have the same four-step shape, and each of
the four is paid **once per metric**:

```julia
ppd            = Predictions.model_inference(latents_raw)          # (1)
model_features = transform(ppd.df, :distribution => ByRow(mean))   # (2)
analysis_df    = innerjoin(ds.odds, model_features, on = [4 cols]) # (3)
dropmissing!(analysis_df, [...])                                   # (4)
```

| | |
|---|---|
| **(1) prices everything** | `DEFAULT_MARKET_CONFIG` is 40-odd markets — five scalar ones, eleven Over/Under lines and the whole Asian-handicap ladder. A 1X2 log-loss pays for all of it, and stores every posterior draw as a `Vector{Float64}` inside a `Vector{Any}` DataFrame column — the fragmentation `06` exists to remove, reintroduced immediately downstream of it. |
| **(2) collapses, then discards** | Every one of those vectors is reduced to a mean, including the ~90% the metric throws away three lines later. |
| **(3) joins twice the frame** | A four-column hash join between a ~50,000-row odds frame and an ~80,000-row PPD frame, materialising a third with every column of both. |
| **(4) copies it again** | `dropmissing!`. |

And `evaluate_experiments` calls `Experiments.extract_oos_predictions(ds, exp)` on
every run — re-deriving the split boundaries from the `DataStore` and rebuilding every
fold's feature set, to recover a posterior the run already had in hand.

**The audit is absent, not merely optional.** `ExperimentResults` has no convergence
field, and `Experiments.Diagnostics.check_convergence` needs the `DataStore` and the
splitter to rebuild what the run threw away. So every leaderboard this repository has
produced ranks runs that mixed alongside runs that did not, with nothing in the output
to tell them apart.

That is not cosmetic. An unconverged chain gives a posterior that is too **narrow** and
biased toward wherever the sampler stuck, and both effects *flatter* the model on
exactly these metrics — LPD and log-loss improve, MIQ's winner/loser gap widens,
GLMEdge's `spread_fair` inflates. An unconverged run does not produce noise. It
produces a run that **wins**.

## The replacement

Three dense typed indexes, built **once for a whole batch**:

```julia
sc = evaluate_fits([LogLoss(), LPD(), CRPS(), RQR(), GLMEdge(), MIQ()], fits, ds)
```

```
--- Convergence ---
  MODEL                VERDICT  MAX RHAT     MIN ESS   N DIV  FAILED GATES
  -----------------------------------------------------------------------
  poisson_baseline     PASS      1.0077      1069.2       0  —
  negbin_dispersed     PASS      1.0045      1102.4       0  —
  poisson_unconverged  FAIL      2.8586         4.7       0  R-hat, bulk ESS, tail ESS
```

`poisson_unconverged` is **excluded from the scored rows and still present in the
convergence frame** — because "eleven models, three of which did not converge" is a
different message from "eight models".

| index | is |
|---|---|
| `OddsView` | `ds.odds` as six concretely-typed parallel vectors plus presence bitmasks |
| `MatchOutcomes` | `match_id → (home_score, away_score)` |
| `MarketProbabilities` | one `n_draws × n_fixtures × n_selections` tensor, priced by `06`'s zero-allocation kernels over the markets the metrics **actually named** |

Every kernel is then one pass over the odds rows with two integer lookups. No join, no
`dropmissing`, no intermediate frame, no `Vector{Any}`.

## Files

| file | contents |
|---|---|
| `l01_types.jl` | `AbstractScoringRule`, the six triggers, the result containers, `MetricScorecard`, and the selection ↔ market inverse |
| `l02_scoring_rules.jl` | `OddsView`, `MatchOutcomes`, `MarketProbabilities`, `EvaluationContext`, the six kernels, `marginals` by dispatch, and the public `compute_metric` entry points |
| `l03_batch_runner.jl` | `evaluate_fits` with convergence filtering, the result flattener, `display_summary_metric`, `display_convergence`, `leaderboard` |
| `l04_compat_bridge.jl` | `module UnifiedEvaluation` + every legacy name |
| `l05_parity.jl` | the parity harness against the live `src` kernels, the cost measurement, three live defect probes, and the runner's deterministic fixtures |
| `r01_demo.jl` | the runner: 12 sections, 78 gates |

`l04` includes `l03` includes … `l01`, and `l01` pulls in
[`07_unified_inference_framework`](../07_unified_inference_framework/) and, through it,
[`06_typed_posterior_latents`](../06_typed_posterior_latents/) and
[`05_composable_count_builder`](../scottish_lower/05_composable_count_builder/). One
`include` loads everything.

`l05` is included from inside `l04`'s module body rather than wrapping it: it holds
both implementations at once and neither is reachable from the other's namespace. The
briefing's file *numbering* is kept; the include *order* follows the dependency.

## Using it

```julia
include("current_development/08_unified_evaluation_framework/l04_compat_bridge.jl")
using .UnifiedEvaluation

fits = load_fits(list_experiments("experiments"))          # 07
sc   = evaluate_fits([LogLoss(), LPD(), CRPS(), MIQ()], fits, ds)

leaderboard(sc, :lpd_overall_diff_lpd; higher_is_better = true)
sc.excluded          # the runs that did not converge
sc.errors            # the (fit, metric) pairs that raised, with the message
```

One fit, one metric:

```julia
compute_metric(LogLoss(), fit, ds)                          # off fit.latents — a read
compute_metric(LogLoss(), fit, ds; require_converged = true) # or a ConvergenceRefusal
```

### The triggers

```julia
LogLoss(; markets = DEFAULT_SCORED_MARKETS, selections = Symbol[])
LPD(; markets, selections, target = :market)   # or target = :score
CRPS(; max_goals = 30)
RQR(; n_sims = 1000, seed = 42)
GLMEdge(; target_selection = :all, min_edge = 0.0)
MIQ(; markets = MIQ_DEFAULT_MARKETS)
```

Every legacy construction still works — `LogLoss(:over_25)`, `LPD([:btts_yes,
:btts_no])`, `GLMEdge(:home)` — and now additionally tells the pricer which markets it
needs, because a selection symbol determines its market uniquely.

**`markets` is the change that matters.** `src` names a metric's scope with a
post-hoc selection filter, so the pipeline prices forty markets to answer a question
about three. Here `markets` is *what to price* and `selections` is *what to score*.

## Convergence gating

```julia
evaluate_fits(metrics, fits, ds)                            # excludes, and says so
evaluate_fits(metrics, fits, ds; require_converged = false) # scores, and flags
compute_metric(m, fit, ds; require_converged = true)        # raises ConvergenceRefusal
```

The verdict comes off `fit.diagnostics` — a **field**, audited by the run that produced
it (`07/l02_convergence.jl`) — so gating a batch of two hundred fits loaded from disk
needs no chains, no `DataStore` and no re-audit.

**An unaudited container counts as not converged.** A `Fit` carrying something other
than a `ConvergenceSummary` returns `(false, ["no audit"])`, for the same reason
`07` abstains on an unmeasured gate: letting a container earn a clean bill of health by
recording nothing is precisely backwards.

The scored frame carries `converged` and `max_rhat` alongside the metrics, so a row
pulled out into a plot or a CSV carries the reason it should or should not be believed.

## What `r01_demo.jl` establishes

**Parity, leaf by leaf, against the live `src` kernels.** Not a transcription: the
legacy side builds a real `Experiments.ExperimentResults`, wraps a real
`Experiments.LatentStates`, and runs the real `model_inference` → four-column
`innerjoin` → `dropmissing!` path, fed a `DataFrame` built from the same typed
container the new side reads.

```
  PARITY — NegBin container, src vs typed kernels
  metric                 leaves     exact      max |Δ|   max ULP  verdict  worst leaf
  logloss_all                 4     2/4      1.110e-16        64  pass     .overall.diff_ll
  lpd_all                     8     2/8      2.842e-14       128  pass     .overall.diff_lpd
  crps                        3     3/3      0.000e+00         0  pass     —
  rqr                        18    18/18     0.000e+00         0  pass     —
  glmedge_all                13     1/13     2.931e-14      1092  pass     .intercept.p_value
  miq                        84    83/84     2.776e-17         1  pass     .all.std
  logloss_over_25             4     4/4      0.000e+00         0  pass     —
  lpd_btts_yes_btts_no        8     8/8      0.000e+00         0  pass     —
```

**The gate is `1e-12`, not 0 ULP, and that is a weaker claim than `06` makes.** Said
plainly because the reason is specific: `src` accumulates its per-row scores in the
order `DataFrames.innerjoin` emitted them, and that order is documented as unspecified.
Floating-point addition is not associative, so the last bit of a `mean` is not
reproducible without reimplementing a hash join. Most leaves come out bit-identical
anyway — the `exact` column says how many, so "passed on tolerance" is never mistaken
for "identical".

The two large ULP figures next to a 1e-16 difference are amplification, not movement:
`diff_ll` is a cancellation of two O(0.6) numbers, and `GLMEdge` runs iteratively
reweighted least squares. §6.4 is a pair of negative controls — one sized to fire
(a 1e-9 relative change in one fixture's λ, caught at 2.8e-12) and one sized to show
where the gate stops (a single ULP, 1.1e-16, below it and said so).

**The one floating-point assumption is measured, not assumed.** The legacy frame stores
a posterior as a contiguous `Vector`; the typed container stores it as a matrix row.
§4d checks that `mean(view(M, i, :)) === mean(M[i, :])` for every row, because if it
ever failed, every CRPS and RQR row would fail and the reason would be three files
away. §4e does the same for the draw-major tensor, and §4f re-checks at 0 ULP that the
tensor holds exactly `06`'s prices.

**Marginals by dispatch, and it matters.** A `CountLatents{Float64, Nothing}` reaches
`Poisson(λ̄)`; a `CountLatents{Float64, <:NamedTuple}` reaches
`NegativeBinomial(r̄, r̄/(r̄+λ̄))`. `src` selects the same two with
`hasproperty(df, :r)` / `hasproperty(df, :r_h)` and an `Inf` sentinel meaning "Poisson"
(`rqr.jl:58-68`) — a container with no dispersion reaches the Poisson branch by falling
off the end of an `if`. Here it cannot reach the negative-binomial method at all.

**Reproducibility.** The same inputs give the same numbers twice, including RQR — which
in `src` they do not. Evaluating never perturbs the caller's global RNG.

**The gate gates.** A three-fit batch with one offset-chain run: excluded by default,
flagged on request, refused outright by a single-fit call that asked for it, and
present with its failed gates named in the convergence frame either way.

**A legacy call site runs verbatim.** §9 is a `LegacyCallSite` module whose body is the
pattern every evaluation runner writes. §9.1 is the strongest form of the claim: both
translators are run over both sides' results and the produced column names are required
to be identical.

**Cost, measured.** 24 fixtures × 1,200 draws, five markets:

```
  path                    legacy s       new s  speedup   legacy KiB      new KiB   shrink
  1 metric  (LogLoss)       0.8776      0.0679   12.92×      15137.8          2.1  7339.5×
  6 metrics (all)           0.8263      0.0702   11.77×      15137.8       2477.1     6.1×
```

The bytes are the posterior-probability materialisation: `src`'s PPD frame over every
market in `DEFAULT_MARKET_CONFIG`, against this framework's tensor over the five the
metrics named. The one-metric row is 2 KiB because `LogLoss` needs only means
(`needs_draws` is `false`) and the tensor is not built. On a real fold the ratio grows
with the markets the store carries and shrinks with the markets a batch actually wants.

### Not established

That any model fits anything, or that any of these numbers is good. Posteriors are
prior draws with a fixed seed (`06/l04_parity.jl` §9) and the synthetic market is the
model's own prices, perturbed and vigged — so the model beats it by construction and
every `diff_ll` in the transcript is an artefact of the fixture.

No live `DataStore` is touched. The store is assembled from `matches` and `odds` alone,
which is every domain an evaluation kernel reads; the legacy
`extract_oos_predictions` re-derivation path (the third branch of `_ue_as_fit`) is
therefore reached structurally but not exercised against a database.

## Backward compatibility

A legacy call site's **body** is unchanged. Its import line changes:

```julia
# before
using BayesianFootball
df = Evaluation.evaluate_experiments([Evaluation.LogLoss(), Evaluation.CRPS()],
                                     experiments, ds)
Evaluation.display_summary_metric(df, :logloss)

# after — the same lines, one different import
import BayesianFootball
using .UnifiedEvaluation.Legacy      # binds `Evaluation`
…identical body…
```

The second line is needed because `BayesianFootball` *exports* the name `Evaluation`,
and Julia refuses to rebind an imported name. Nothing can make two modules answer to
one name; the honest claim is that everything *after* the import is untouched.
`r01_demo.jl` §9 proves it with a `LegacyCallSite` module.

| legacy expression | still works |
|---|---|
| `evaluate_experiments(metrics, exps, ds)` | returns the wide `DataFrame`, sorted by `:model` |
| `compute_metric(metric, exp, ds)` | extracts, then scores |
| `compute_metric(metric, exp, ds, latents)` | scores what you have |
| `to_dataframe_row(exp, metric, result)` | same column names, character for character |
| `to_dataframe_row(exp, result)` | 2-arg aggregate form |
| `display_summary_metric(df, :logloss)` | same curated columns, same regex sweeps |
| `get_metric_method_name(metric_or_result)` | same strings |
| `LogLoss(:over_25)` / `LogLoss([:a, :b])` | same filter semantics |
| `CRPSResults` / `CRPSResult` | both spellings |

`latents` in the four-argument form may be a typed container, either `LatentStates`, or
a raw legacy `DataFrame`; `as_typed_latents` reconciles them.

**Two deliberate behaviour changes.**

1. `evaluate_experiments` **warns** about unconverged fits. It does not exclude them —
   that would change the frame a legacy caller gets — but it will not stay silent about
   a row that should not be believed. `evaluate_fits` is where the gate filters.

2. A metric that raises no longer costs the model its whole row. `src`'s runner catches,
   warns, and then `push!`es nothing (`batch_runner.jl:44-51`), so one missing odds
   column silently removes a model from the comparison. Here the failing metric's
   columns are `missing` for that fit, everything else survives, and the failure is
   reported in `scorecard.errors`.

## What this framework does differently, and the defects behind each

Reproduced live in `r01_demo.jl` §10 — not described, so the claims cannot go stale
silently. **None is fixed in `src` by this prototype.**

1. **`CRPS` and `RQR` cannot be computed for a Poisson model.**
   `Predictions.get_latent_column_symbols` has methods for `AbstractNegBinModel`
   (`negativebinomial.jl:29`) and the Frank-copula NegBin model (`frank_copula.jl:77`)
   and nothing else; `crps.jl:69` and `rqr.jl:89` call it unconditionally. Every
   `AbstractPoissonModel` engine raises `MethodError` inside `evaluate_experiments`'
   `try`, which drops the model's entire row with a `@warn` and no other trace.
   *Here:* `crps_parameters` and `marginals` dispatch on the container and have a
   Poisson method by construction.

2. **An `MIQResult` with an empty selection group cannot be flattened.** `MIQStats`'
   fields are `Union{Missing, Float64}` (`miq.jl:12-18`) and `Evaluation.unroll` has
   methods for `Real` and `AbstractMetricComponent` only (`translator.jl:6,11`).
   `MIQResult` reports twelve selections including Over/Under 1.5 and 3.5, so any store
   that does not quote those lines makes `to_dataframe_row` raise — again inside the
   `try`, again dropping the whole model.
   *Here:* `l03` §1 adds the one-line `Missing` method.

3. **`src`'s RQR is not reproducible.** `rqr.jl:50` draws from the unseeded global RNG,
   so two consecutive calls on identical inputs disagree — the diagnostic cannot be
   re-checked, and two models' RQR rows are incomparable unless computed in the same
   session in the same order.
   *Here:* `RQR(seed = …)` uses a private `Xoshiro` stream, so evaluating also never
   perturbs the caller's RNG.

Three more, carried but **not** changed, because changing them would make new numbers
incomparable with every number this repository has already recorded:

4. **CRPS is plug-in, not posterior-predictive.** `crps.jl:88` builds one marginal from
   the posterior MEAN λ, rather than averaging the CDF over draws. The Bayesian form is
   strictly better calibrated. Reproduced verbatim; see `CRPS`'s docstring.

5. **`GLMEdge` checks `n_obs < 10` before dropping missing odds** (`glm_edge.jl:75-82`),
   so a metric with 12 rows of which 9 have no `odds_close` reaches `glm` with 3.
   Reproduced verbatim.

6. **`CRPSResults.all` averages home and away; `RQRResult.all` pools them.** Two
   different conventions in two adjacent files. Both preserved.

## Where this deviates from the briefing, and why

1. **LogLoss is binary cross-entropy, not multi-class.** The briefing writes
   `−Σᵢ yᵢ log pᵢ`; `logloss.jl:49` computes the binary form, and they are different
   numbers. Three reasons the binary form stays: every leaderboard in `data/` was
   written with it; it generalises to Over/Under and BTTS, which are two-outcome
   markets with no partition in the odds table; and it keeps `market_ll` a like-for-like
   baseline.

2. **LPD has two targets, and `:market` is the default.** The briefing specifies the
   joint density of the realised scoreline; `src` computes the market-selection LPD.
   Both are implemented — `LPD(target = :score)` is the briefing's, read straight off
   the score grid — and the default is `src`'s so every existing caller gets the number
   it already has. `:score` reports `NaN` for the market baseline rather than inventing
   a correct-score line nobody quoted.

3. **`RQR.n_sims` averages SUMMARIES, not residuals.** Averaging the residuals across
   randomisations would shrink them toward the mid-quantile normal score and manufacture
   normality. Each replicate is summarised in full and the summaries are averaged;
   `n_sims = 1` reproduces `src` draw for draw, which is what the parity row uses.

4. **The triggers carry `markets` *and* `selections`.** The briefing's `LogLoss` has
   only `markets`; `src`'s has only `selections`. Both are needed — one decides what to
   price, the other what to score — and dropping either would break a call site.

5. **`CRPSResults`, with `CRPSResult` as an alias.** The briefing names it
   `CRPSResult`; `src` names it `CRPSResults`. Both bind to one type, so neither
   spelling is a `UndefVarError`.

6. **`GLMEdge.target_selection` maps onto `selections`.** The briefing's field name is
   kept as a keyword; the storage is the legacy vector, so `GLMEdge(:home)` and
   `GLMEdge(target_selection = :home)` are the same object.

7. **`l05_parity.jl` is included from `l04`, and also holds the runner's fixtures.**
   The parity harness must see both implementations, so it lives inside the module; the
   deterministic `DataStore` and chain builders live there too, because the prototype's
   rule is definitions in the loaders and execution in the runner, and the briefing's
   loader list has no other place for them.

## Graduating this to `src/`

In dependency order, and each step is independently useful:

1. **`unroll(::String, ::Missing)`** → `src/evaluation/translator.jl`. One line, fixes
   defect 2, and unblocks MIQ in every existing leaderboard.
2. **`get_latent_column_symbols(::AbstractPoissonModel, df)`** →
   `src/predictions/score_computation/poisson.jl`. Three lines, fixes defect 1, and
   makes CRPS and RQR computable for the entire Poisson engine ladder.
3. **`l01_types.jl` + `l02_scoring_rules.jl`** → `src/evaluation/`. The kernels can
   replace `metrics_methods/` one file at a time: each new `compute_metric` is
   parity-checked by `l05` against the one it replaces, and the legacy signature stays.
4. **`l03_batch_runner.jl`** → `src/evaluation/batch_runner.jl`, once `07`'s `Fit` has
   graduated and `fit.diagnostics` exists in `src`. Until then the gate has nothing to
   read.
5. **`RQR`'s seed** can go in on its own, ahead of everything else. It changes no number
   that anyone should have been relying on.

## Notes carried forward

The three `src` defects `06` documented are unchanged: `DynamicPxGRecombModel`'s OOS
extractor cannot be called; three extractors mis-size their output under multiple
chains; nine call sites use a `haskey(::Chains, ::Symbol)` that MCMCChains 7.7 removed.
See [`06/README.md`](../06_typed_posterior_latents/README.md) and
[`07/README.md`](../07_unified_inference_framework/README.md).
