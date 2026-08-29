# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : BATCH EVALUATION AND CONVERGENCE GATING
# ==============================================================================
#
# `evaluate_fits(metrics, fits, ds)` — the leaderboard, and the gate in front of it.
#
# ------------------------------------------------------------------------------
# THE GATE IS THE POINT
# ------------------------------------------------------------------------------
#
# `Evaluation.evaluate_experiments` (src/evaluation/batch_runner.jl:11) does not look
# at convergence. It cannot: `ExperimentResults` has no field for it, and
# `Experiments.Diagnostics.check_convergence` needs the `DataStore` and the splitter to
# rebuild what the run already had. So every leaderboard this repository has produced
# ranks runs that mixed alongside runs that did not, and there is nothing in the output
# that distinguishes them.
#
# That is not a cosmetic gap. An unconverged chain produces λ posteriors that are too
# NARROW (the sampler never reached the tails) and biased toward wherever it got stuck.
# Both effects flatter the model on exactly the metrics here:
#
#   * LPD and log-loss improve, because a narrow posterior concentrated near the mode
#     puts more mass on the modal outcome;
#   * MIQ's winner/loser gap widens, because the quantiles are computed against a
#     distribution that is too tight;
#   * GLMEdge's `spread_fair` coefficient inflates, because the "edge" is measured
#     against a market price using a model probability whose uncertainty was thrown away.
#
# So an unconverged run does not merely produce noise. It produces a run that WINS the
# leaderboard. `07`'s `Fit` made the verdict a field; this makes it a filter.
#
#     evaluate_fits(metrics, fits, ds)                          # excludes, and says so
#     evaluate_fits(metrics, fits, ds; require_converged = false) # flags, and says so
#
# Either way the `convergence` frame has a row for EVERY submitted fit, so "eleven
# models, three of which did not converge" never silently becomes "eight models".
#
# ==============================================================================

include(joinpath(@__DIR__, "l02_scoring_rules.jl"))


# ==============================================================================
# 1. FLATTENING A RESULT
# ==============================================================================
#
# `src/evaluation/translator.jl`, with one behavioural difference (§1.2) and the same
# column names for every legacy construction.

"""
    unroll(prefix, value) -> NamedTuple

Recursively flatten a result into `prefix_field_subfield => scalar` pairs.

`Real` and `Missing` are the leaves; `AbstractMetricComponent` is the recursion step.
`Missing` is a leaf here and is NOT one in `src` — `MIQStats` fields are
`Union{Missing,Float64}` and `unroll(::String, ::Real)` does not accept a `missing`, so
`to_dataframe_row` on an `MIQResult` with an empty selection group raises `MethodError`
inside `evaluate_experiments`' `try`, which drops the model's whole row. Recorded in
`README.md`; accepting `Missing` here is the fix, and it is additive — nothing that
worked before behaves differently.
"""
unroll(prefix::AbstractString, val::Real) = NamedTuple{(Symbol(prefix),)}((val,))
unroll(prefix::AbstractString, ::Missing) = NamedTuple{(Symbol(prefix),)}((missing,))

function unroll(prefix::AbstractString, comp::AbstractMetricComponent)
    ks = propertynames(comp)
    return merge((unroll("$(prefix)_$(k)", getproperty(comp, k)) for k in ks)...)
end

"""
    flatten_result(model_name, metric, result) -> NamedTuple

One leaderboard row for one (fit, metric): `(; model = …, <family>_<path> = …)`.

The column prefix is `get_metric_method_name(result) * metric_column_suffix(metric)`,
which is `src/evaluation/translator.jl:46` with a suffix rule that additionally
distinguishes non-default OPTIONS (`l01_types.jl` §4.2) and is empty for every legacy
construction.
"""
function flatten_result(model_name::AbstractString,
                        metric::AbstractScoringRule,
                        result::AbstractEvaluationResult)
    ks = propertynames(result)
    name = get_metric_method_name(result) * metric_column_suffix(metric)
    flat = merge((unroll("$(name)_$(k)", getproperty(result, k)) for k in ks)...)
    return merge((model = String(model_name),), flat)
end


# ==============================================================================
# 2. THE CONVERGENCE FRAME
# ==============================================================================

"""
    convergence_row(fit) -> NamedTuple

One fit's verdict, flattened for the scorecard's `convergence` frame.

`max_rhat`, `min_ess_bulk` and the rest come straight off the `ConvergenceSummary`
`07` computed at run time — no chains are re-read and no `DataStore` is touched, which
is what makes it possible to gate a batch of two hundred fits loaded from disk.
"""
function convergence_row(fit)
    name = _ue_fit_name(fit)
    diag = try
        getfield(fit, :diagnostics)
    catch
        nothing
    end
    if !(diag isa ConvergenceSummary)
        return (model = name, converged = false, audited = false,
                max_rhat = NaN, min_ess_bulk = NaN, min_ess_tail = NaN,
                n_divergent = 0, divergence_rate = NaN, min_bfmi = NaN,
                n_folds = 0, failed_gates = "no audit")
    end
    return (model = name,
            converged = diag.passed,
            audited = true,
            max_rhat = diag.max_rhat,
            min_ess_bulk = diag.min_ess_bulk,
            min_ess_tail = diag.min_ess_tail,
            n_divergent = diag.n_divergent,
            divergence_rate = diag.divergence_rate,
            min_bfmi = diag.min_bfmi,
            n_folds = diag.n_folds,
            failed_gates = isempty(diag.failed_gates) ? "—" :
                           join(diag.failed_gates, ", "))
end


# ==============================================================================
# 3. THE BATCH RUNNER
# ==============================================================================

"""
    evaluate_fits(metrics, fits, ds; require_converged = true, …) -> MetricScorecard

Score every fit on every metric, gated on convergence.

| keyword             | default | meaning                                            |
|---------------------|---------|----------------------------------------------------|
| `require_converged` | `true`  | exclude unconverged fits; `false` scores and flags |
| `threaded`          | `true`  | one task per fit                                    |
| `max_goals`         | `12`    | score-grid truncation                               |
| `quiet`             | `false` | suppress the progress lines                         |
| `show_tables`       | `true`  | print the convergence and leaderboard tables        |

WHAT IS SHARED AND WHAT IS NOT. Each fit gets ONE `EvaluationContext` — one pricing
sweep over the union of every metric's markets, one `OddsView`, one outcome index —
and every metric reads it. Six metrics on one fit therefore price the posterior once,
where `evaluate_experiments` prices it six times (or once, if
`Predictions._PPD_CACHE` happens to hit) and joins it six times regardless.

THREADED OVER FITS, NOT OVER METRICS. Fits are independent and roughly equal in cost;
metrics are neither. The inner pricing sweep drops to single-threaded when more than
one fit is in flight, so the two levels do not oversubscribe.

ERRORS ARE COLLECTED, NOT SWALLOWED. A metric that raises loses its OWN columns for
that fit and nothing else — the other metrics' columns survive, the row survives, and
the failure is reported in `scorecard.errors` and printed. `evaluate_experiments`
drops the entire model on any single metric's failure (`batch_runner.jl:44-51`), which
is how a leaderboard silently loses a model to one missing odds column.
"""
function evaluate_fits(metrics::AbstractVector,
                       fits::AbstractVector,
                       ds::UE_D.DataStore;
                       require_converged::Bool = true,
                       threaded::Bool = true,
                       max_goals::Integer = TPL_MAX_GOALS,
                       quiet::Bool = false,
                       show_tables::Bool = true)
    t0 = time()
    ms = AbstractScoringRule[m for m in metrics]
    isempty(ms) && error("evaluate_fits: no metrics given.")
    isempty(fits) && error("evaluate_fits: no fits given.")

    quiet || begin
        println("=" ^ 78)
        println(" Unified evaluation — ", length(fits), " fit(s) × ", length(ms), " metric(s)")
        println("   metrics : ", join(get_metric_method_name.(ms), ", "))
        println("   gate    : require_converged = ", require_converged,
                require_converged ? "  (unconverged fits are EXCLUDED)" :
                                    "  (unconverged fits are FLAGGED)")
        println("=" ^ 78)
    end

    conv_rows = [convergence_row(f) for f in fits]
    conv_df = DataFrame(conv_rows)

    keep_idx = Int[]
    excluded = String[]
    for (i, r) in enumerate(conv_rows)
        if require_converged && !r.converged
            push!(excluded, r.model)
        else
            push!(keep_idx, i)
        end
    end

    if !quiet && !isempty(excluded)
        println()
        for r in conv_rows
            r.converged && continue
            @printf("  ⛔ EXCLUDED  %-28s  %s\n", r.model, r.failed_gates)
        end
    end

    n_keep = length(keep_idx)
    rows   = Vector{Union{Nothing, Dict{Symbol, Any}}}(nothing, n_keep)
    errs   = [EvaluationError[] for _ in 1:n_keep]
    inner_threaded = threaded && n_keep <= 1

    function score_one(slot::Int)
        fit  = fits[keep_idx[slot]]
        name = _ue_fit_name(fit)
        row  = Dict{Symbol, Any}(:model => name)
        local ctx
        try
            ctx = evaluation_context(fit_latents(fit), ds.odds, ds.matches, ms;
                                     max_goals = max_goals, threaded = inner_threaded)
        catch e
            push!(errs[slot], EvaluationError(name, "<context>", sprint(showerror, e)))
            rows[slot] = row
            return nothing
        end
        for metric in ms
            try
                result = compute_metric(metric, ctx)
                flat = flatten_result(name, metric, result)
                for k in propertynames(flat)
                    k === :model && continue
                    row[k] = getproperty(flat, k)
                end
            catch e
                push!(errs[slot],
                      EvaluationError(name, get_metric_method_name(metric),
                                      sprint(showerror, e)))
            end
        end
        rows[slot] = row
        return nothing
    end

    if threaded && n_keep > 1
        @sync for slot in 1:n_keep
            Threads.@spawn score_one(slot)
        end
    else
        for slot in 1:n_keep
            score_one(slot)
        end
    end

    if !quiet
        println()
        for slot in 1:n_keep
            r = conv_rows[keep_idx[slot]]
            mark = isempty(errs[slot]) ? "✅" : "⚠️ "
            flag = r.converged ? "" : "   [UNCONVERGED: $(r.failed_gates)]"
            @printf("  %s %-28s %d metric(s)%s\n", mark, r.model,
                    length(ms) - length(errs[slot]), flag)
        end
    end

    all_errors = reduce(vcat, errs; init = EvaluationError[])
    master = _ue_assemble(rows)
    if nrow(master) > 0
        sort!(master, :model)
        # The verdict travels WITH the numbers. A row pulled out of this frame into a
        # plot or a CSV carries the reason it should or should not be trusted.
        vmap = Dict(r.model => r for r in conv_rows)
        master.converged = Bool[vmap[m].converged for m in master.model]
        master.max_rhat  = Float64[vmap[m].max_rhat for m in master.model]
        master.min_ess_bulk = Float64[vmap[m].min_ess_bulk for m in master.model]
    end

    sc = MetricScorecard(master, conv_df, ms, excluded, all_errors,
                         require_converged, time() - t0)

    if show_tables && !quiet
        println()
        display_convergence(sc)
        for fam in unique(_ue_family_symbol.(ms))
            display_summary_metric(sc, fam)
        end
        println()
        @printf("  evaluated in %s\n", format_elapsed(sc.elapsed))
        isempty(all_errors) || println("  ", length(all_errors), " metric error(s) — see `scorecard.errors`")
    end

    return sc
end

evaluate_fits(metric::AbstractScoringRule, fits::AbstractVector, ds::UE_D.DataStore; kwargs...) =
    evaluate_fits([metric], fits, ds; kwargs...)

evaluate_fits(metrics::AbstractVector, fit::Fit, ds::UE_D.DataStore; kwargs...) =
    evaluate_fits(metrics, [fit], ds; kwargs...)

"""
    _ue_assemble(rows) -> DataFrame

Union the row dictionaries into one wide frame, filling absent columns with `missing`.

A metric that failed for ONE fit leaves that fit's cells empty rather than removing the
column from every other fit — which is what a `Vector{NamedTuple}` → `DataFrame`
conversion would force, since it requires identical keys.
"""
function _ue_assemble(rows::Vector{Union{Nothing, Dict{Symbol, Any}}})
    present = Dict{Symbol, Any}[r for r in rows if r !== nothing]
    isempty(present) && return DataFrame(model = String[])
    cols = Symbol[:model]
    for r in present, k in keys(r)
        k === :model && continue
        k in cols || push!(cols, k)
    end
    df = DataFrame()
    for c in cols
        df[!, c] = Any[get(r, c, missing) for r in present]
    end
    # Narrow every column that came out fully populated and homogeneous, so the frame
    # sorts, plots and writes to CSV like a normal one rather than as `Vector{Any}`.
    for c in names(df)
        v = df[!, c]
        any(ismissing, v) && continue
        T = mapreduce(typeof, promote_type, v)
        T === Any && continue
        df[!, c] = convert(Vector{T}, v)
    end
    return df
end

_ue_family_symbol(m::AbstractScoringRule) = Symbol(_ue_family_name(m))


# ==============================================================================
# 4. DISPLAY
# ==============================================================================

"""
    display_convergence(scorecard)

The gate's own table: every submitted fit, its verdict and the number that produced it.
"""
function display_convergence(sc::MetricScorecard)
    df = sc.convergence
    nrow(df) == 0 && return nothing
    println("\n--- Convergence ---")
    w = max(maximum(length, df.model), 5)
    @printf("  %-*s  %-6s  %8s  %10s  %6s  %s\n",
            w, "MODEL", "VERDICT", "MAX RHAT", "MIN ESS", "N DIV", "FAILED GATES")
    println("  ", "-" ^ (w + 52))
    for r in eachrow(df)
        @printf("  %-*s  %-6s  %8.4f  %10.1f  %6d  %s\n",
                w, r.model, r.converged ? "PASS" : "FAIL",
                r.max_rhat, r.min_ess_bulk, r.n_divergent, r.failed_gates)
    end
    return nothing
end

"""
    display_summary_metric(df_or_scorecard, family::Symbol)

A curated column subset per metric family.

The five legacy families (`:rqr`, `:logloss`, `:glmedge`, `:crps`, `:lpd`) select
EXACTLY the columns `src/evaluation/batch_runner.jl:71-112` selects, in the same order,
including the regex sweeps for per-selection variants. `:miq` and `:convergence` are
new — `src` has no MIQ branch at all, so `display_summary_metric(df, :miq)` there
prints "Unknown metric family" and returns.
"""
function display_summary_metric(df::AbstractDataFrame, metric_family::Symbol)
    cols, title = _ue_summary_columns(df, metric_family)
    if cols === nothing
        println("Unknown metric family: $metric_family")
        return nothing
    end
    println("\n", title)
    existing = filter(c -> string(c) in names(df), cols)
    if length(existing) <= 1
        @warn "No data found for metric family '$metric_family' in the provided DataFrame."
        return nothing
    end
    display(select(df, existing...))
    return nothing
end

display_summary_metric(sc::MetricScorecard, metric_family::Symbol) =
    metric_family === :convergence ? display_convergence(sc) :
                                     display_summary_metric(sc.rows, metric_family)

function _ue_summary_columns(df::AbstractDataFrame, family::Symbol)
    nm = names(df)
    if family === :rqr
        return ([:model, :rqr_all_mean, :rqr_all_std, :rqr_all_skewness,
                 :rqr_all_kurtosis, :rqr_all_shapiro_w, :rqr_all_shapiro_p],
                "--- RQR Summary (mean ≈ 0, std ≈ 1, shapiro_p > 0.05 is calibrated) ---")
    elseif family === :logloss
        cols = [:model, :logloss_overall_model_ll, :logloss_overall_market_ll,
                :logloss_overall_diff_ll]
        append!(cols, Symbol.(filter(n -> occursin(r"^logloss_.+_overall_diff_ll$", n) &&
                                          n != "logloss_overall_diff_ll", nm)))
        return (cols, "--- LogLoss Summary (Lower Diff is Better) ---")
    elseif family === :glmedge
        cols = [:model]
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_intercept_coef", n), nm)))
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_spread_fair_coef", n), nm)))
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_spread_fair_p_value", n), nm)))
        return (cols, "--- GLM Edge Summary (spread_fair > 0 with small p = an edge) ---")
    elseif family === :crps
        return ([:model, :crps_home_mean, :crps_away_mean, :crps_all_mean],
                "--- CRPS Summary (Lower is Better) ---")
    elseif family === :lpd
        cols = [:model, :lpd_overall_model_lpd, :lpd_overall_model_std,
                :lpd_overall_model_skewness, :lpd_overall_model_kurtosis,
                :lpd_overall_market_lpd, :lpd_overall_diff_lpd,
                :lpd_overall_elpd, :lpd_overall_n_obs]
        append!(cols, Symbol.(filter(n -> occursin(r"^lpd_.+_overall_diff_lpd$", n) &&
                                          n != "lpd_overall_diff_lpd", nm)))
        append!(cols, Symbol.(filter(n -> occursin(r"^lpd(_.+)?_score_overall_model_lpd$", n), nm)))
        return (cols, "--- LPD Summary (Higher Diff is Better; Higher ELPD is Better) ---")
    elseif family === :miq
        cols = [:model, :miq_all_mean_gap, :miq_all_ks_d_stat, :miq_all_p_value,
                :miq_all_n_winners, :miq_all_n_losers]
        append!(cols, Symbol.(filter(n -> occursin(r"^miq_(home|draw|away)_mean_gap$", n), nm)))
        return (cols, "--- MIQ Summary (Positive mean_gap = market underprices winners) ---")
    end
    return (nothing, "")
end

"""
    leaderboard(scorecard, column; higher_is_better = false) -> DataFrame

Rank the scored fits on one column, carrying the convergence verdict alongside.

The verdict column is not optional and not removable, because the failure mode this
whole file exists to prevent is a ranked table that does not say which of its rows can
be believed.
"""
function leaderboard(sc::MetricScorecard, column::Symbol; higher_is_better::Bool = false)
    string(column) in names(sc.rows) ||
        error("leaderboard: no column `$column`. Available: $(join(names(sc.rows), ", "))")
    keep = [:model, column]
    for extra in (:converged, :max_rhat)
        string(extra) in names(sc.rows) && push!(keep, extra)
    end
    out = select(sc.rows, keep...)
    sort!(out, column; rev = higher_is_better)
    return out
end
