# src/evaluation/compat.jl
#
# The bridge from the legacy evaluation path to the typed evaluator.
#
# NOTHING IN `metrics_methods/`, `interfaces.jl`, `translator.jl` OR `batch_runner.jl` IS
# SHADOWED OR REPLACED. `compute_metric(metric, exp::ExperimentResults, ds, latents)`,
# `evaluate_experiments`, `to_dataframe_row` and `display_summary_metric` behave exactly
# as they did; this file adds the entry points that reach the typed kernels and the
# conversions that let one path's inputs feed the other's.
#
# The two worlds differ in ONE input: the legacy kernels take an
# `Experiments.LatentStates` (or a raw `DataFrame`) and rebuild a PPD from it; the typed
# kernels take an `AbstractPosteriorLatents` and read it. `as_typed_latents` is the
# conversion, and it needs the MODEL when it is handed a frame, because the family — and
# therefore the schema — is a property of the engine and not of the columns: a frame with
# `λ_h`, `λ_a` and no `r` is a Poisson container for one model and a mis-extracted NegBin
# one for another.

export ConvergenceRefusal, convergence_verdict, fit_latents, as_typed_latents, as_fit


# ==============================================================================
# 1. READING A CONTAINER WITHOUT ASSUMING WHAT IT IS
# ==============================================================================

"Duck-typed model reader, so the bridges work on a `Fit`, an `ExperimentResults`, or
anything else that carries a `.config` — which is every container this repository has
ever put a model in."
_fit_model(fit) = hasproperty(fit, :config) && hasproperty(fit.config, :model) ?
                  fit.config.model : nothing

"Duck-typed name reader. Falls back to the type name so a report row is never blank."
_fit_display_name(fit) = hasproperty(fit, :config) && hasproperty(fit.config, :name) ?
                         String(fit.config.name) : string(typeof(fit))

"""
    ConvergenceRefusal(fit, failed_gates, detail)

Raised when `require_converged = true` and the fit did not pass its gates.

An exception rather than a `NaN` result: a metric computed on a chain that did not mix is
not a worse number, it is not a number, and a leaderboard that ranks it alongside
converged runs is worse than one that is missing a row.
"""
struct ConvergenceRefusal <: Exception
    fit::String
    failed_gates::Vector{String}
    detail::Vector{String}
end

function Base.showerror(io::IO, e::ConvergenceRefusal)
    print(io, "ConvergenceRefusal: fit `", e.fit, "` did not converge — failed gate(s): ",
          isempty(e.failed_gates) ? "unknown" : join(e.failed_gates, ", "), ".")
    for d in e.detail
        print(io, "\n    ", d)
    end
    print(io, "\n  Pass `require_converged = false` to score it anyway (the result will ",
              "be flagged, not trusted).")
end

"""
    convergence_verdict(fit) -> (passed::Bool, failed_gates, detail)

Read `fit.diagnostics` without assuming it is a `ConvergenceSummary`.

A `Fit` built by `fit_model` always carries one. A `Fit` reconstructed from an old
serialisation, or one someone built by hand, may carry anything — and this function's
contract is that it NEVER THROWS, because a missing audit must degrade to "unknown", not
to a crash inside the gate that exists to prevent crashes.

An unknown verdict counts as NOT PASSED under `require_converged = true`, for the same
reason the audit abstains on an unmeasured gate.
"""
function convergence_verdict(fit)
    diag = try
        getfield(fit, :diagnostics)
    catch
        nothing
    end
    diag isa ConvergenceSummary || return (false, ["no audit"],
        ["this container carries no ConvergenceSummary — re-run `fit_model`, or " *
         "`audit_convergence(fit)` if you have the folds."])
    return (diag.passed, copy(diag.failed_gates), copy(diag.failures))
end

"""
    fit_latents(fit) -> AbstractPosteriorLatents

The typed OOS container a `Fit` carries, with an actionable error when it does not.
"""
function fit_latents(fit)
    lat = try
        getfield(fit, :latents)
    catch
        nothing
    end
    lat isa AbstractPosteriorLatents && return lat
    lat === nothing && error(
        "this Fit carries no typed latents, so there is nothing to score. `fit_model` " *
        "records the reason in `fit.config.tags` when a model's family is not " *
        "registered with `Models.latent_family`.")
    error("expected a typed posterior container, got a $(typeof(lat)).")
end

"""
    as_typed_latents(latents, model = nothing) -> AbstractPosteriorLatents

Normalise whatever a legacy caller passed into a typed container.

| given                                | how                              |
|--------------------------------------|----------------------------------|
| an `AbstractPosteriorLatents`        | returned as-is                   |
| anything with a `.latents` container | unwrapped (`LatentStates`)       |
| anything with a `.df`                | `latents_from_legacy_dataframe`  |
| a `DataFrame`                        | `latents_from_legacy_dataframe`  |

The frame route needs the MODEL — see the file header for why the columns alone are not
enough to name the family.
"""
function as_typed_latents(latents, model = nothing)
    latents isa AbstractPosteriorLatents && return latents
    if hasproperty(latents, :latents)
        inner = getproperty(latents, :latents)
        inner isa AbstractPosteriorLatents && return inner
    end
    df = latents isa AbstractDataFrame ? latents :
         hasproperty(latents, :df) ? getproperty(latents, :df) : nothing
    df === nothing && error(
        "as_typed_latents: cannot read a $(typeof(latents)). Expected a typed " *
        "container, a LatentStates, or a legacy DataFrame.")
    m = model
    if m === nothing && hasproperty(latents, :model)
        m = getproperty(latents, :model)
    end
    m === nothing && error(
        "as_typed_latents: rebuilding a typed container from a legacy DataFrame needs " *
        "the model — the family determines the schema. Pass the model, or hand over " *
        "`fit.latents` instead.")
    return latents_from_legacy_dataframe(m, df)
end


# ==============================================================================
# 2. THE TYPED ENTRY POINTS
# ==============================================================================

"""
    compute_metric(metric, latents::AbstractPosteriorLatents, odds_df, matches_df;
                   max_goals = 12, threaded = true, kwargs...)

Score one metric off a typed container and two frames. Builds a context for this one
metric.

For MORE than one metric use [`evaluate_fits`](@ref), or build a context yourself with
`build_evaluation_context` — the pricing sweep is the expensive part and it is shared
there. Any extra keywords are forwarded to the kernel, so `compute_metric(LPD(), …;
target = :score)` and `compute_metric(CRPS(), …; max_goals = 40)` work.
"""
function compute_metric(metric::AbstractScoringRule,
                        latents::AbstractPosteriorLatents,
                        odds_df::AbstractDataFrame,
                        matches_df::AbstractDataFrame;
                        max_goals::Integer = Predictions.TPL_MAX_GOALS,
                        threaded::Bool = true,
                        kwargs...)
    ctx = build_evaluation_context(latents, odds_df, matches_df, [metric];
                                   max_goals = max_goals, threaded = threaded)
    return compute_metric(metric, ctx; kwargs...)
end

"""
    compute_metric(metric, fit::Fit, ds::DataStore; require_converged = false, kwargs...)

Score one run straight off its typed latents.

No `extract_oos_predictions`, no re-derived boundaries, no rebuilt feature sets — the
container was extracted by the run that produced it and is READ, not recomputed.
"""
function compute_metric(metric::AbstractScoringRule, fit::Fit, ds::DataStore;
                        require_converged::Bool = false,
                        max_goals::Integer = Predictions.TPL_MAX_GOALS,
                        threaded::Bool = true,
                        kwargs...)
    if require_converged
        passed, gates, detail = convergence_verdict(fit)
        passed || throw(ConvergenceRefusal(fit_name(fit), gates, detail))
    end
    return compute_metric(metric, fit_latents(fit), ds.odds, ds.matches;
                          max_goals = max_goals, threaded = threaded, kwargs...)
end

"""
    compute_metric(metric, fit::Fit, ds::DataStore, latents; kwargs...)

The four-argument shape, for a caller that already holds its latents. `latents` may be a
typed container, an `Experiments.LatentStates`, or a raw legacy `DataFrame`;
[`as_typed_latents`](@ref) reconciles them.
"""
function compute_metric(metric::AbstractScoringRule, fit::Fit, ds::DataStore, latents;
                        require_converged::Bool = false,
                        max_goals::Integer = Predictions.TPL_MAX_GOALS,
                        threaded::Bool = true,
                        kwargs...)
    if require_converged
        passed, gates, detail = convergence_verdict(fit)
        passed || throw(ConvergenceRefusal(fit_name(fit), gates, detail))
    end
    typed = as_typed_latents(latents, _fit_model(fit))
    return compute_metric(metric, typed, ds.odds, ds.matches;
                          max_goals = max_goals, threaded = threaded, kwargs...)
end


# ==============================================================================
# 3. A LEGACY CONTAINER, AS A FIT
# ==============================================================================

"""
    as_fit(exp, ds::DataStore; latents = nothing) -> Fit

Whatever was submitted, as a `Fit` the typed path can score.

A `Fit` passes through. Anything else goes through `Training.upgrade_to_fit`, which
recovers the folds, flattens `training_config.sampler`, and — the point — AUDITS
CONVERGENCE, giving the legacy container a verdict it never had a field for.

THE LATENTS ARE THEN LOOKED FOR IN THREE PLACES, CHEAPEST FIRST, because a legacy
container carries neither them nor the feature sets they would be rebuilt from:

  1. the `latents` keyword, if the caller has them in hand;
  2. `oos_latents.jls` in the run's own directory — what
     `save_experiment(...; compute_oos = true)` wrote, and the usual case for a run that
     has been evaluated before;
  3. `Experiments.extract_oos_predictions(ds, exp)` — the full re-derivation, which needs
     a live `DataStore` carrying every column the feature builders read.

Only (3) can fail on a store assembled for evaluation rather than fetched from the
database, and it fails naming the column it wanted, which is the right error.
"""
as_fit(f::Fit, ::DataStore; latents = nothing) = f

function as_fit(exp, ds::DataStore; latents = nothing)
    fit = upgrade_to_fit(exp; save_path = _exp_save_path(exp))
    getfield(fit, :latents) === nothing || return fit

    model = getfield(fit, :config).model
    resolved = if latents !== nothing
        as_typed_latents(latents, model)
    else
        cached = load_latents(getfield(fit, :save_path))
        cached isa AbstractPosteriorLatents ? cached :
            as_typed_latents(Experiments.extract_oos_predictions(ds, exp), model)
    end

    return Fit(getfield(fit, :config), getfield(fit, :folds), resolved,
               getfield(fit, :diagnostics), getfield(fit, :metadata),
               getfield(fit, :save_path))
end

_exp_save_path(exp) = hasproperty(exp, :save_path) ? String(exp.save_path) : ""
