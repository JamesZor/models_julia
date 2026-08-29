# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : THE BACKWARD-COMPATIBILITY BRIDGE
# ==============================================================================
#
# Loader, and the module wrapper. `include`ing THIS file loads the whole prototype:
#
#     include("current_development/08_unified_evaluation_framework/l04_compat_bridge.jl")
#     using .UnifiedEvaluation
#
# l04 → l03 → l02 → l01 → 07_unified_inference_framework
#                          → 06_typed_posterior_latents
#                          → 05_composable_count_builder
#
# `l05_parity.jl` is included from INSIDE this module's body, at the end, rather than
# wrapping it. It is a harness over the kernels above and the legacy ones in `src`, so
# it has to see both; and it must not be the file that opens the module, because a
# reader looking for "where does `UnifiedEvaluation` begin" should find it in the
# compatibility bridge, next to the names it exists to bind. The briefing's file
# NUMBERING is preserved; the include ORDER follows the dependency.
#
# ------------------------------------------------------------------------------
# WHY THIS IS A MODULE
# ------------------------------------------------------------------------------
#
# `BayesianFootball` EXPORTS the name `Evaluation` (src/BayesianFootball.jl:64), so in
# any scope that has done `using BayesianFootball` — every runner in this repository,
# and every file of `05`, `06` and `07` this prototype includes — that name is already
# bound, and Julia refuses to rebind an imported name:
#
#     const Evaluation = @__MODULE__
#     ERROR: cannot assign a value to imported variable Main.Evaluation
#
# Same problem `07` has with `Experiments` and `Training`, same shape of answer: the
# prototype lives in a module, and the one colliding name lives in a nested `Legacy`
# submodule that does not `using BayesianFootball` and can therefore bind it.
#
# Everything ELSE a legacy evaluation call site names — `LogLoss`, `LPD`, `CRPS`,
# `RQR`, `GLMEdge`, `MIQ`, `compute_metric`, `evaluate_experiments`,
# `to_dataframe_row`, `display_summary_metric`, every `*Result` and `*Component` — is
# exported by `Evaluation` but NOT re-exported by `BayesianFootball` (which never does
# `using .Evaluation`). Those bind here directly, with nothing to collide with.
#
# ------------------------------------------------------------------------------
# WHAT "100% BACKWARD COMPATIBLE" MEANS HERE, EXACTLY
# ------------------------------------------------------------------------------
#
# A legacy call site's BODY is unchanged. Its import line changes.
#
#     # before
#     using BayesianFootball
#     df = Evaluation.evaluate_experiments(
#              [Evaluation.LogLoss(), Evaluation.LPD(:over_25), Evaluation.CRPS()],
#              experiments, ds)
#     Evaluation.display_summary_metric(df, :logloss)
#
#     # after — the same three lines, one different import
#     import BayesianFootball
#     using .UnifiedEvaluation.Legacy       # binds `Evaluation`
#     …identical body…
#
# Preserved, name for name and column for column:
#
#   | legacy expression                                  | still works           |
#   |----------------------------------------------------|-----------------------|
#   | `evaluate_experiments(metrics, exps, ds)`          | returns a `DataFrame` |
#   | `evaluate_experiments(metric, exps, ds)`           | single-metric form    |
#   | `compute_metric(metric, exp, ds)`                  | extracts, then scores |
#   | `compute_metric(metric, exp, ds, latents)`         | scores what you have  |
#   | `to_dataframe_row(exp, metric, result)`            | same column names     |
#   | `to_dataframe_row(exp, result)`                    | 2-arg aggregate form  |
#   | `display_summary_metric(df, :logloss)`             | same curated columns  |
#   | `get_metric_method_name(metric_or_result)`         | same strings          |
#   | `LogLoss(:over_25)` / `LogLoss([:a, :b])`          | same filter semantics |
#   | `CRPSResults` / `CRPSResult`                        | both spellings        |
#
# and `latents` in the four-argument form may be a typed container (06), either
# `LatentStates`, or a raw legacy `DataFrame` — `as_typed_latents` reconciles them.
#
# ------------------------------------------------------------------------------
# THE TWO DELIBERATE BEHAVIOUR CHANGES
# ------------------------------------------------------------------------------
#
# 1. `evaluate_experiments` WARNS about unconverged fits. It does not exclude them —
#    that would change the frame a legacy caller gets — but it will not stay silent
#    about a leaderboard row that should not be believed. `evaluate_fits` is where the
#    gate actually filters, and it defaults to filtering.
#
# 2. A metric that raises no longer costs the model its whole row. `src`'s runner
#    catches, warns, and then `push!`es nothing (batch_runner.jl:44-51), so one missing
#    odds column silently removes a model from the comparison. Here the failing
#    metric's columns are `missing` for that fit and everything else survives. Strictly
#    more output than before; nothing that previously appeared has gone.
#
# ==============================================================================

module UnifiedEvaluation

# The whole prototype. l03 chains down through l02 → l01, and l01 pulls in
# `07_unified_inference_framework` and, through it, `06` and `05`.
include(joinpath(@__DIR__, "l03_batch_runner.jl"))

using DataFrames
using Printf
using Statistics


# ==============================================================================
# 1. THE NEW API
# ==============================================================================

export AbstractScoringRule, AbstractEvaluationResult, AbstractMetricComponent
export LogLoss, LPD, CRPS, RQR, GLMEdge, MIQ
export LogLossComponent, LogLossResult
export LPDComponent, LPDResult
export CRPSComponent, CRPSResults, CRPSResult
export DistributionStats, RQRResult
export GLMCoefComponent, GLMEdgeResult
export MIQStats, MIQResult
export MetricScorecard, EvaluationError, EvaluationContext, ConvergenceRefusal
export OddsView, MatchOutcomes, MarketProbabilities
export compute_metric, evaluate_fits, evaluation_context, market_probabilities
export display_summary_metric, display_convergence, leaderboard
export convergence_verdict, fit_latents, as_typed_latents
export scored_markets, scored_selections, needs_outcomes, needs_draws
export get_metric_method_name, metric_column_suffix, flatten_result, unroll
export market_for_selection, selections_to_markets, market_selections
export DEFAULT_SCORED_MARKETS, MIQ_DEFAULT_MARKETS
export marginals, crps_parameters, posterior_mean, prob_mean, prob_draws
export calc_logloss, calc_lpd_scalar, calc_lpd_samples!, compute_crps, compute_rqr
export summarize_stats, get_miq, evaluate_group_edge, probability_bytes

# Re-exported from `07` so a runner needs ONE `using`. These are the names an
# evaluation call site reaches for immediately after scoring something.
export Fit, FitConfig, FoldFit, ConvergenceSummary, ConvergenceGates,
       audit_convergence, fit_name, chains, format_elapsed
export ExperimentResults, ExperimentConfig, ExperimentTask
export AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents,
       n_matches, n_draws, latent_match_ids, observation_family, extract_latents,
       to_legacy_dataframe, latents_from_legacy_dataframe


# ==============================================================================
# 2. LEGACY SHIM — `evaluate_experiments`
# ==============================================================================

"""
    evaluate_experiments(metrics, experiments, ds::DataStore; kwargs...) -> DataFrame

`src/evaluation/batch_runner.jl:11`, under its old name and with its old return type.

Returns the WIDE `DataFrame` sorted by `:model`, exactly as before. The convergence
columns `evaluate_fits` attaches are stripped unless `with_convergence = true`, because
a caller doing `select(df, Not(:model))` or writing the frame straight to CSV has a
column list it did not ask to change.

`experiments` may hold `Fit`s, genuine `BayesianFootball.Experiments.ExperimentResults`,
or a mixture. The legacy containers are upgraded per fit (§4).

THE ONE BEHAVIOUR CHANGE: an unconverged fit produces a `@warn` naming the gates it
failed. It is still scored and still appears in the frame — see the header.
"""
function evaluate_experiments(metrics::AbstractVector,
                              experiments::AbstractVector,
                              ds::UE_D.DataStore;
                              with_convergence::Bool = false,
                              require_converged::Bool = false,
                              quiet::Bool = false,
                              show_tables::Bool = false,
                              kwargs...)
    fits = [_ue_as_fit(e, ds) for e in experiments]

    for f in fits
        passed, gates, _ = convergence_verdict(f)
        passed || @warn string(
            "evaluate_experiments: `", _ue_fit_name(f), "` did not pass its convergence ",
            "gates (", isempty(gates) ? "unknown" : join(gates, ", "), ") and is being ",
            "scored anyway, because that is what this legacy entry point has always ",
            "done. Use `evaluate_fits(...)` — which excludes it by default — to gate.")
    end

    sc = evaluate_fits(collect(AbstractScoringRule, metrics), fits, ds;
                       require_converged = require_converged,
                       quiet = quiet, show_tables = show_tables, kwargs...)

    df = sc.rows
    if !with_convergence
        drop = intersect([:converged, :max_rhat, :min_ess_bulk], Symbol.(names(df)))
        isempty(drop) || (df = select(df, Not(drop)))
    end
    return df
end

evaluate_experiments(metric::AbstractScoringRule, experiments::AbstractVector,
                     ds::UE_D.DataStore; kwargs...) =
    evaluate_experiments([metric], experiments, ds; kwargs...)

evaluate_experiments(metrics::AbstractVector, experiment, ds::UE_D.DataStore; kwargs...) =
    evaluate_experiments(metrics, [experiment], ds; kwargs...)

export evaluate_experiments


# ==============================================================================
# 3. LEGACY SHIM — `to_dataframe_row`
# ==============================================================================

"""
    to_dataframe_row(exp, metric, result) -> NamedTuple

`src/evaluation/translator.jl:41`. The three-argument form, with the per-selection
suffix.

`exp` is read only for its name — `Experiments.get_model_name(exp)` is
`exp.config.name` (helpers.jl:4) — so anything carrying a `.config.name` works,
including a `Fit`, a legacy `ExperimentResults`, or a bare `NamedTuple`. A plain
`String` is accepted too, because the flattener never needed a container in the first
place and requiring one made `to_dataframe_row` untestable on its own.
"""
to_dataframe_row(exp, metric::AbstractScoringRule, result::AbstractEvaluationResult) =
    flatten_result(_ue_row_name(exp), metric, result)

"""
    to_dataframe_row(exp, result) -> NamedTuple

`src/evaluation/translator.jl:59`. The two-argument form: aggregate column names, no
selection suffix, regardless of what the metric was filtered to.
"""
function to_dataframe_row(exp, result::AbstractEvaluationResult)
    ks = propertynames(result)
    name = get_metric_method_name(result)
    flat = merge((unroll("$(name)_$(k)", getproperty(result, k)) for k in ks)...)
    return merge((model = _ue_row_name(exp),), flat)
end

_ue_row_name(exp::AbstractString) = String(exp)
_ue_row_name(exp) = _ue_fit_name(exp)

export to_dataframe_row


# ==============================================================================
# 4. LEGACY SHIM — the three-argument `compute_metric`
# ==============================================================================
#
# `l02_scoring_rules.jl` §8 already defines the `Fit` and four-argument forms. This adds
# the one that has to reach for the latents itself.

"""
    compute_metric(metric, exp, ds::DataStore; require_converged = false)

`src/evaluation/interfaces.jl:17`, under its old name: score a run whose latents are
not in hand.

For a `Fit` (the common case) this is `l02` §8's method — the latents are a FIELD and
the call is a read. For a genuine legacy `BayesianFootball.Experiments.ExperimentResults`
this falls back to `Experiments.extract_oos_predictions(ds, exp)`, which is the full
re-derivation this framework exists to avoid; it warns, once, saying so and naming the
fix (`fit_model`, or `load_fit` on the run's directory).
"""
function compute_metric(metric::AbstractScoringRule, exp, ds::UE_D.DataStore;
                        require_converged::Bool = false, kwargs...)
    if require_converged
        passed, gates, detail = convergence_verdict(exp)
        passed || throw(ConvergenceRefusal(_ue_fit_name(exp), gates, detail))
    end
    @warn string(
        "compute_metric: `", _ue_fit_name(exp), "` is a $(typeof(exp)), not a `Fit`, so ",
        "its out-of-sample latents have to be rebuilt from the DataStore — the ",
        "boundaries are re-derived and every fold's feature set is reconstructed. ",
        "`fit_model` (07) extracts them during the run; `load_fit` on the run ",
        "directory upgrades an existing one.") maxlog = 1
    legacy = UE_BF.Experiments.extract_oos_predictions(ds, exp)
    return compute_metric(metric, exp, ds, legacy; kwargs...)
end

"""
    _ue_as_fit(exp, ds; latents = nothing) -> Fit

Whatever was submitted, as a `Fit`.

A `Fit` passes through. Anything else goes through `07`'s `upgrade_to_fit`, which
recovers the folds, flattens `training_config.sampler`, and — the point — AUDITS
CONVERGENCE, giving the legacy container a verdict it never had a field for.

THE LATENTS ARE THEN LOOKED FOR IN THREE PLACES, cheapest first, because a legacy
container carries neither them nor the feature sets they would be rebuilt from:

  1. the `latents` keyword, if the caller has them in hand;
  2. `oos_latents.jls` in the run's own directory — what
     `save_experiment(...; compute_oos = true)` wrote, and the usual case for a run
     that has been evaluated before;
  3. `Experiments.extract_oos_predictions(ds, exp)` — the full re-derivation, which
     needs a live `DataStore` with every column the feature builders read.

Only (3) can fail on a store that was assembled for evaluation rather than fetched from
the database, and it fails with the column it wanted, which is the right error.
"""
_ue_as_fit(f::Fit, ::UE_D.DataStore; latents = nothing) = f

function _ue_as_fit(exp, ds::UE_D.DataStore; latents = nothing)
    fit = upgrade_to_fit(exp; save_path = _ue_save_path(exp))
    getfield(fit, :latents) === nothing || return fit

    model = getfield(fit, :config).model
    resolved = if latents !== nothing
        as_typed_latents(latents, model)
    else
        cached = load_latents(getfield(fit, :save_path))
        cached !== nothing ? cached :
            as_typed_latents(UE_BF.Experiments.extract_oos_predictions(ds, exp), model)
    end

    return Fit(getfield(fit, :config), getfield(fit, :folds), resolved,
               getfield(fit, :diagnostics), getfield(fit, :metadata),
               getfield(fit, :save_path))
end

_ue_save_path(exp) = hasproperty(exp, :save_path) ? String(exp.save_path) : ""

export _ue_as_fit


# ==============================================================================
# 5. THE PARITY HARNESS
# ==============================================================================
#
# Included here, inside the module, because it holds BOTH implementations at once —
# the kernels above and `BayesianFootball.Evaluation`'s — and neither is reachable
# from the other's namespace.

include(joinpath(@__DIR__, "l05_parity.jl"))

export MetricParityRow, parity_leaves, parity_results, metric_parity_table,
       parity_report, parity_scope_ok, legacy_experiment, legacy_latent_states,
       legacy_metric, legacy_compute, clear_ppd_cache!
export CostRow, cost_table, measure_cost, speedup, shrink
export probe_poisson_latent_columns, probe_miq_translator, probe_rqr_nondeterminism
export demo_nuts_chain, simulate_scores, synthetic_matches, synthetic_odds,
       synthetic_datastore


# ==============================================================================
# 6. THE COLLIDING NAME
# ==============================================================================

"""
    UnifiedEvaluation.Legacy

The one name that cannot be bound in a scope that has done `using BayesianFootball`.

```julia
import BayesianFootball
using .UnifiedEvaluation.Legacy      # Evaluation
```

`Evaluation` is `UnifiedEvaluation` itself, so `Evaluation.evaluate_experiments`,
`Evaluation.LogLoss` and `Evaluation.display_summary_metric` all resolve to the shims
above. Anything else the legacy module offered resolves to whatever `UnifiedEvaluation`
binds under that name, or raises `UndefVarError` naming it.

`parentmodule(@__MODULE__)` rather than `import ..UnifiedEvaluation`: this submodule is
elaborated while its parent's body is still executing, and `parentmodule` needs no
binding to already exist.
"""
module Legacy

const Evaluation = parentmodule(@__MODULE__)

export Evaluation

end # module Legacy

end # module UnifiedEvaluation
