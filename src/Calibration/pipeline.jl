# ==============================================================================
# src/Calibration/pipeline.jl — the calibrator tier, and its contract with L3/L4
# ==============================================================================
#
#   Raw Fit / CountLatents (L1/L2)
#             |
#             v
#     calibrate_fit(cal, fit, book)      <- inverts the book, pools the rates
#             |
#             v
#       CalibratedFit
#             |
#     +-------+-------------------------------+
#     v                                       v
#   Evaluation (L3)                     Portfolio (L4)
#   evaluate_predictions(cf, ds)        run_portfolio_simulation(spec, policy, cf, book, ds)
#             |                                       |
#             v                                       v
#   calibration_runs                          portfolio_runs
#                                             (linked through metadata)
#
# ------------------------------------------------------------------------------
# HOW "ZERO MODIFICATIONS TO src/Portfolio/" IS ACHIEVED
# ------------------------------------------------------------------------------
#
# `CalibratedFit.fit` is a REAL `Training.Fit` — same config (plus provenance tags), same
# folds, same `ConvergenceSummary`, same metadata — whose `latents` field holds the
# calibrated container. So
#
#     run_portfolio_simulation(spec, policy, cf.fit, book, ds)
#
# already works with no new code at all: it takes `Portfolio`'s own `Training.Fit` branch,
# including the convergence gate.
#
# The two forwarding methods at the bottom of this file exist only so the `CalibratedFit`
# ITSELF can be passed, which is what a caller will naturally write. They are methods
# added FROM this module ONTO our own type — the ordinary Julia extension mechanism, not
# piracy — and `src/Portfolio/` is not edited.
#
# ------------------------------------------------------------------------------
# WHY THE BOOK INSTANT IS ASSERTED
# ------------------------------------------------------------------------------
#
# Calibration parameters do NOT transfer between price instants and the winning
# functional form flips with the sharpness of the book being pooled with (README §7.3:
# the standard form wins at the close, the inverse form at T-25, and each gives up
# 0.0015-0.0020 LogLoss when transferred). Every book frame in this pipeline carries
# `:as_of_minutes`, and `calibrate_fit` checks it against the calibrator's own
# `book_as_of_minutes` before it inverts anything.
# ==============================================================================


# ==============================================================================
# 1. THE CONTAINER
# ==============================================================================

"""
    CalibratedFit

What [`calibrate_fit`](@ref) returns: a calibrated run, the raw run it came from, and
every fact needed to explain the difference.

| field | is |
|---|---|
| `fit` | a real `Training.Fit` carrying the CALIBRATED latents — the object L3 and L4 consume |
| `base_fit` | the raw run, untouched |
| `calibrator` | the recipe |
| `market_rates` | `match_id -> MarketRateFit`, refusals included and named |
| `rate_diagnostics` | one row per fixture: delta, w, kappa, variance retention, rate ratio, lambda before and after |
| `book_as_of_minutes` | the instant of the book this was calibrated against |
| `coverage` | what [`inversion_coverage`](@ref) measured over this container's fixtures |
| `created_at` | wall clock at construction |

`cf.latents` is `cf.fit.latents`, the calibrated container. `cf.raw_latents` is the raw
one. Neither is a wrapper type — see [`CalibratedLatents`](@ref).

`rate_diagnostics` is the object a post-mortem reads and the one thing that does NOT
reconstruct from the calibrator plus the book, which is why `calibration_artifacts`
stores it.
"""
struct CalibratedFit{F1, F2, C <: AbstractCalibrator}
    fit::F1
    base_fit::F2
    calibrator::C
    market_rates::Dict{Int, MarketRateFit}
    rate_diagnostics::DataFrame
    book_as_of_minutes::Float64
    coverage::NamedTuple
    created_at::DateTime
end

@inline function Base.getproperty(cf::CalibratedFit, s::Symbol)
    s === :latents && return getfield(getfield(cf, :fit), :latents)
    s === :raw_latents && return getfield(getfield(cf, :base_fit), :latents)
    s === :name && return Training.fit_name(getfield(cf, :fit))
    return getfield(cf, s)
end

Base.propertynames(::CalibratedFit) =
    (fieldnames(CalibratedFit)..., :latents, :raw_latents, :name)

"The calibrated `Training.Fit`. Pass this anywhere a `Fit` is expected."
calibrated_fit(cf::CalibratedFit) = getfield(cf, :fit)

"The raw `Training.Fit` this was calibrated from."
base_fit(cf::CalibratedFit) = getfield(cf, :base_fit)

function Base.show(io::IO, ::MIME"text/plain", cf::CalibratedFit)
    cov = cf.coverage
    ws = weight_summary(cf.rate_diagnostics)
    print(io, "CalibratedFit(\"", Training.fit_name(cf.fit), "\")")
    print(io, "\n  calibrator  : ", cf.calibrator.name, "  [", calibrator_label(cf.calibrator), "]")
    print(io, "\n  book instant: T", @sprintf("%+.0f", cf.book_as_of_minutes), " min")
    print(io, "\n  fixtures    : ", cov.n_fixtures, " held, ", cov.n_accepted, " inverted (",
          @sprintf("%.1f%%", 100 * cov.coverage), " of all, ",
          @sprintf("%.1f%%", 100 * cov.coverage_quoted), " of quoted)")
    if ws.n_shifted > 0
        print(io, "\n  weight w    : median ", @sprintf("%.3f", ws.w_median),
              "  [p10 ", @sprintf("%.3f", ws.w_p10), ", p90 ", @sprintf("%.3f", ws.w_p90), "]")
        print(io, "\n  var retained: median ", @sprintf("%.3f", ws.var_retention_median))
    else
        print(io, "\n  weight w    : nothing shifted — this container equals its raw source")
    end
    print(io, "\n  convergence : ", cf.fit.diagnostics.passed ? "passed" : "FAILED")
    return nothing
end

Base.show(io::IO, cf::CalibratedFit) =
    print(io, "CalibratedFit(\"", Training.fit_name(cf.fit), "\", ",
          calibrator_label(cf.calibrator), ")")


# ==============================================================================
# 2. THE ENTRY POINT
# ==============================================================================

"""
    calibrate_fit(cal, fit, market_odds; check_book_instant = true,
                  rates = nothing, quiet = false) -> CalibratedFit

Calibrate a completed inference run against a tradeable book.

`market_odds` is a de-vigged book frame carrying `:match_id`, `:selection` and
`:prob_fair_close` — normally the output of [`point_in_time_book`](@ref). The inversion
is restricted to the fixtures `fit.latents` actually holds, so a book covering three
seasons costs nothing extra on a one-fold container.

# Refusals, and what they are for

* **The instant.** If the frame carries `:as_of_minutes` it must match the calibrator's
  `book_as_of_minutes`. A T-25 calibrator meeting a closing book is an error here rather
  than a plausible wrong number in a table three weeks later. `check_book_instant = false`
  suppresses the check and should be accompanied by a comment saying why.
* **The posterior.** A container is required: `fit.latents === nothing` means the run
  never extracted one and there is nothing to calibrate.
* **Convergence is NOT gated here.** Calibration does not size a bet; `Portfolio` gates
  the bankroll and does it on `cf.fit.diagnostics`, which this function carries across
  unchanged. Gating twice, in two places, with two defaults, is how the two come to
  disagree.

# Reusing an inversion

`rates` accepts a `Dict{Int, MarketRateFit}` computed earlier. The inversion depends on
the BOOK ONLY — not the model, not the law, not the dispersion map — so a sweep over
calibrators should invert once and pass it in.

```julia
book, _ = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))
rates   = invert_market_rates(cal, book; match_ids = latent_match_ids(fit.latents))
cfs     = [calibrate_fit(c, fit, book; rates = rates) for c in candidates]
```
"""
function calibrate_fit(cal::AbstractGenerativeRateCalibrator,
                       fit::Training.Fit,
                       market_odds::AbstractDataFrame;
                       check_book_instant::Bool = true,
                       rates::Union{Nothing, AbstractDict{Int, MarketRateFit}} = nothing,
                       quiet::Bool = false)
    lat = getfield(fit, :latents)
    lat === nothing && error(
        "calibrate_fit: fit \"$(Training.fit_name(fit))\" carries no latent container, so " *
        "there is no posterior to calibrate. `fit_model` records why in `fit.config.tags` " *
        "(look for a `latents:` tag).")

    if check_book_instant && hasproperty(market_odds, :as_of_minutes)
        assert_book_as_of(market_odds, cal.book_as_of_minutes)
    elseif check_book_instant
        quiet || @warn(
            "calibrate_fit: this book carries no `:as_of_minutes`, so the price instant " *
            "cannot be checked against the calibrator's. Calibration parameters do not " *
            "transfer between instants (README §7.3). Build the book with " *
            "`point_in_time_book` or pass `check_book_instant = false` deliberately.",
            calibrator = cal.name, expected = cal.book_as_of_minutes)
    end

    ids = Models.latent_match_ids(lat)
    r = rates === nothing ? invert_market_rates(cal, market_odds; match_ids = ids) : rates
    cov = inversion_coverage(r, ids)

    calibrated, diag = calibrate_latents(cal, lat, r)

    quiet || cov.n_accepted > 0 || @warn(
        "calibrate_fit: not one fixture of this container had an accepted market " *
        "inversion, so the calibrated posterior equals the raw one. Check the book's " *
        "coverage and the inversion gates before reading any difference as a result.",
        calibrator = cal.name, n_fixtures = cov.n_fixtures, n_quoted = cov.n_quoted,
        refusals = inversion_refusals(r))

    return CalibratedFit(_retag_fit(fit, cal, calibrated), fit, cal, r, diag,
                         cal.book_as_of_minutes, cov, now())
end

"""
    _retag_fit(fit, cal, latents) -> Training.Fit

The same run with the calibrated container in the `latents` slot and the calibration
recorded in `config.tags`.

The NAME is deliberately unchanged: `Training.fit_name` still identifies the model run,
which is the key `calibration_runs.model_run_id` points at and the key every existing
report joins on. The tags carry the calibration, because a tag is queryable and a mangled
name is not.

`folds`, `diagnostics`, `metadata` and `save_path` are carried across verbatim. The
chains are the chains; calibration does not touch them, and a `CalibratedFit` whose
convergence verdict differed from its source's would be lying about which posterior was
audited.
"""
function _retag_fit(fit::Training.Fit, cal::AbstractCalibrator, latents)
    cfg = getfield(fit, :config)
    tags = vcat(cfg.tags, ["calibrated",
                           "calibrator:" * cal.name,
                           "calibrator_label:" * calibrator_label(cal),
                           "calibrator_hash:" * calibrator_hash(cal)[1:16],
                           @sprintf("book_as_of:%+.0f", cal.book_as_of_minutes)])
    cfg2 = Training.FitConfig(name = cfg.name, model = cfg.model, splitter = cfg.splitter,
                              sampler = cfg.sampler, execution = cfg.execution,
                              tags = tags, description = cfg.description,
                              save_dir = cfg.save_dir)
    return Training.Fit(cfg2, getfield(fit, :folds), latents,
                        getfield(fit, :diagnostics), getfield(fit, :metadata),
                        getfield(fit, :save_path))
end

"""
    calibrate_fit(cal, fit, ds::Data.DataStore; book_config, kw...) -> CalibratedFit

Convenience: build the point-in-time book from the store at the calibrator's own instant,
then calibrate against it. Returns the `CalibratedFit` only; call
[`point_in_time_book`](@ref) yourself when you also want the refusal frame.
"""
function calibrate_fit(cal::AbstractGenerativeRateCalibrator, fit::Training.Fit,
                       ds::Data.DataStore;
                       book_config::PointInTimeBookConfig =
                           PointInTimeBookConfig(as_of_minutes = cal.book_as_of_minutes),
                       kw...)
    book, _ = point_in_time_book(ds; config = book_config)
    return calibrate_fit(cal, fit, book; kw...)
end


# ==============================================================================
# 3. THE PORTFOLIO BRIDGE  (no `src/Portfolio/` file is edited)
# ==============================================================================
#
# `Portfolio.run_portfolio_simulation` routes a source that is neither a `Training.Fit`
# nor a `Tuple` to
#
#     build_books_reported(spec, source, odds, fixtures; require_result, quiet)
#
# so one method is all that is needed. It forwards to `Portfolio`'s own `Fit` method,
# which means the convergence gate, the `BuildReport`, the fallback-market warning and the
# book ordering are `Portfolio`'s and not a parallel implementation.

"""
    Portfolio.build_books_reported(spec, cf::CalibratedFit, odds, fixtures; ...)

Build a staking book from a calibrated run. Delegates to `Portfolio`'s `Training.Fit`
method on `cf.fit`, so everything downstream — gate, report, ordering — is unchanged.

`require_converged` defaults to `false` here, matching `Portfolio`'s own `Fit` method and
the keyword `run_portfolio_simulation` forwards.
"""
Portfolio.build_books_reported(spec::Portfolio.BookSpec, cf::CalibratedFit, odds, fixtures;
                               kw...) =
    Portfolio.build_books_reported(spec, calibrated_fit(cf), odds, fixtures; kw...)

Portfolio.build_books(spec::Portfolio.BookSpec, cf::CalibratedFit, odds, fixtures; kw...) =
    first(Portfolio.build_books_reported(spec, cf, odds, fixtures; kw...))

"""
    Portfolio.stake_sheet(sys::PortfolioSystem, cf::CalibratedFit, odds, fixtures; kw...)

The match-day sheet from a calibrated run, for symmetry with the backtest path. Same
delegation, so the sheet an operator bets from is produced by the code path that was
audited against history rather than by a parallel one.
"""
Portfolio.stake_sheet(sys::Portfolio.PortfolioSystem, cf::CalibratedFit, odds, fixtures;
                      kw...) =
    Portfolio.stake_sheet(sys, calibrated_fit(cf), odds, fixtures; kw...)


# ==============================================================================
# 4. THE EVALUATION BRIDGE
# ==============================================================================

Evaluation.fit_latents(cf::CalibratedFit) = Evaluation.fit_latents(calibrated_fit(cf))
Evaluation.convergence_verdict(cf::CalibratedFit) =
    Evaluation.convergence_verdict(calibrated_fit(cf))

"""
    Evaluation.evaluate_predictions(cf::CalibratedFit, ds; kw...) -> PredictionScores

Proper scores for a calibrated run against `ds.odds`.

**Score against the book you calibrated with, not `ds.odds`,** when the two are different
instants — `evaluate_predictions(cf, book, ds.matches)` below does that. Scoring T-25
rates against closing prices measures the drift between the two books as well as the
model, and the drift is the larger term.
"""
Evaluation.evaluate_predictions(cf::CalibratedFit, ds::Data.DataStore; kw...) =
    Evaluation.evaluate_predictions(calibrated_fit(cf), ds; kw...)

"""
    calibration_context(cf, odds_df, matches_df; markets, metrics, threaded) -> EvaluationContext

An `EvaluationContext` over the calibrated container and a chosen book. One pricing pass;
filter it with `evaluate_predictions(ctx; selections = ...)` for a narrower scope rather
than pricing twice.
"""
function calibration_context(cf::CalibratedFit, odds_df::AbstractDataFrame,
                             matches_df::AbstractDataFrame;
                             markets = nothing,
                             metrics = Evaluation.AbstractScoringRule[Evaluation.PredictionScore()],
                             threaded::Bool = true)
    return Evaluation.build_evaluation_context(Evaluation.fit_latents(cf), odds_df,
                                               matches_df, metrics;
                                               markets = markets, threaded = threaded)
end

"""
    calibration_scores(cf, odds_df, matches_df; selections, n_bins, markets) -> NamedTuple

The headline proper scores a calibration run is recorded by: LogLoss, ECE, Brier and RPS,
each beside the same quantity computed from the book, over
`Evaluation.DEFAULT_SCORED_MARKETS` (1X2 + O/U 2.5 + BTTS).

**That scope is not a default chosen for convenience.** It is the only scope in which the
published Gate-1 thresholds mean anything, and it is what `calibration_runs.log_loss` /
`.ece` / `.brier` store. A wide-book score belongs in `metadata`, under its own name.
"""
function calibration_scores(cf::CalibratedFit, odds_df::AbstractDataFrame,
                            matches_df::AbstractDataFrame;
                            selections = nothing, n_bins::Integer = 10,
                            markets = nothing, threaded::Bool = true)
    ctx = calibration_context(cf, odds_df, matches_df; markets = markets,
                              threaded = threaded)
    s = Evaluation.evaluate_predictions(ctx; selections = selections, n_bins = n_bins)
    return (; n_obs = s.model.n_obs,
            logloss = s.model.logloss, market_logloss = s.market.logloss,
            ece = s.model.ece, market_ece = s.market.ece,
            mce = s.model.mce,
            brier = s.model.brier, market_brier = s.market.brier,
            rps = s.model.rps, market_rps = s.market.rps)
end
