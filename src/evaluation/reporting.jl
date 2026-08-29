# src/evaluation/reporting.jl
#
# `EvaluationReport` — a batch's numbers, the convergence verdict that decided which of
# them exist, and the formatters that render both.
#
# ------------------------------------------------------------------------------
# THE GATE IS THE POINT
# ------------------------------------------------------------------------------
#
# `evaluate_experiments` does not look at convergence. It cannot: `ExperimentResults` has
# no field for it, and `Experiments.Diagnostics.check_convergence` needs the `DataStore`
# and the splitter to rebuild what the run already had. So every leaderboard this
# repository has produced ranks runs that mixed alongside runs that did not, with nothing
# in the output to tell them apart.
#
# That is not a cosmetic gap. An unconverged chain gives λ posteriors that are too NARROW
# (the sampler never reached the tails) and biased toward wherever it got stuck, and both
# effects FLATTER the model on exactly these metrics:
#
#   * LPD and log-loss improve, because a narrow posterior concentrated near the mode
#     puts more mass on the modal outcome;
#   * MIQ's winner/loser gap widens, because the quantiles are computed against a
#     distribution that is too tight;
#   * GLMEdge's `spread_fair` inflates, because the "edge" is measured with a model
#     probability whose uncertainty was thrown away.
#
# So an unconverged run does not produce noise. It produces a run that WINS. `Fit` made
# the verdict a field (src/training/inference/); this makes it a filter.
#
# Either way the `convergence` frame has a row for EVERY submitted fit, so "eleven
# models, three of which did not converge" never silently becomes "eight models".

export EvaluationReport, evaluate_fits, convergence_row, flatten_result,
       metric_column_suffix, display_convergence, leaderboard,
       report_table, markdown_report


# ==============================================================================
# 1. FLATTENING A RESULT
# ==============================================================================

"""
    metric_column_suffix(metric) -> String

Appended to a result's column prefix so two rules of the same family in one batch do not
overwrite each other's columns.

Identical to `translator.jl`'s `_metric_selection_suffix`, and deliberately so: a legacy
construction must produce a legacy column name, character for character, or every saved
evaluation CSV in `data/` stops lining up with the new ones.
"""
metric_column_suffix(m::AbstractScoringRule) = _metric_selection_suffix(m)

"""
    flatten_result(model_name, metric, result) -> NamedTuple

One leaderboard row for one (fit, metric): `(; model = …, <family>_<path> = …)`.

The column prefix is `get_metric_method_name(result) * metric_column_suffix(metric)`,
which is exactly `to_dataframe_row`'s rule — so a row produced here and a row produced
by the legacy translator carry the same column names.
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

One fit's verdict, flattened for the report's `convergence` frame.

`max_rhat`, `min_ess_bulk` and the rest come straight off the `ConvergenceSummary` the
run computed — no chains are re-read and no `DataStore` is touched, which is what makes
it possible to gate a batch of two hundred fits loaded from disk.

An UNAUDITED container reports `converged = false, audited = false`, for the same reason
the convergence audit abstains on an unmeasured gate: letting a container earn a clean
bill of health by recording nothing is precisely backwards.
"""
function convergence_row(fit)
    name = _fit_display_name(fit)
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
# 3. THE REPORT
# ==============================================================================

"""
    EvaluationReport

The result of [`evaluate_fits`](@ref): one wide row per fit, plus the convergence verdict
that decided whether that row exists at all.

| field               | is                                                             |
|---------------------|----------------------------------------------------------------|
| `rows`              | `:model` + every flattened metric column, sorted by `:model`    |
| `convergence`       | one row per fit: verdict, R-hat, ESS, divergences, failed gates |
| `metrics`           | the rules that were run                                         |
| `excluded`          | fits filtered out by the convergence gate, by name              |
| `errors`            | every `(fit, metric)` that raised, with the message             |
| `require_converged` | whether the gate was filtering or only flagging                 |
| `elapsed`           | wall-clock seconds                                              |

`convergence` HAS A ROW FOR EVERY FIT, INCLUDING THE EXCLUDED ONES. A report that simply
omitted the unconverged runs would look identical to one where they were never
submitted.
"""
struct EvaluationReport
    rows::DataFrame
    convergence::DataFrame
    metrics::Vector{AbstractScoringRule}
    excluded::Vector{String}
    errors::Vector{EvaluationError}
    require_converged::Bool
    elapsed::Float64
end

DataFrames.DataFrame(r::EvaluationReport) = r.rows
DataFrames.nrow(r::EvaluationReport) = nrow(r.rows)
Base.length(r::EvaluationReport) = nrow(r.rows)

"The fits that produced a scored row, in report order."
scored_models(r::EvaluationReport) =
    nrow(r.rows) == 0 ? String[] : String.(r.rows.model)

function Base.show(io::IO, r::EvaluationReport)
    print(io, "EvaluationReport(", nrow(r.rows), " scored, ",
          length(r.excluded), " excluded, ", length(r.errors), " errors)")
end

function Base.show(io::IO, ::MIME"text/plain", r::EvaluationReport)
    println(io, "EvaluationReport")
    println(io, "  metrics    : ", join(get_metric_method_name.(r.metrics), ", "))
    println(io, "  scored     : ", nrow(r.rows), " fit(s)")
    println(io, "  gate       : ", r.require_converged ?
                "require_converged = true  (unconverged fits EXCLUDED)" :
                "require_converged = false (unconverged fits FLAGGED)")
    isempty(r.excluded) ||
        println(io, "  excluded   : ", join(r.excluded, ", "))
    if !isempty(r.errors)
        println(io, "  errors     : ", length(r.errors))
        for e in r.errors
            println(io, "      ", e.model, " / ", e.metric, " — ", e.message)
        end
    end
    print(io, "  elapsed    : ", round(r.elapsed, digits = 2), "s")
end


# ==============================================================================
# 4. THE BATCH RUNNER
# ==============================================================================

"""
    evaluate_fits(metrics, fits, ds; require_converged = true, …) -> EvaluationReport

Score every fit on every metric, gated on convergence.

| keyword             | default | meaning                                            |
|---------------------|---------|----------------------------------------------------|
| `require_converged` | `true`  | exclude unconverged fits; `false` scores and flags |
| `threaded`          | `true`  | one task per fit                                   |
| `max_goals`         | `12`    | score-grid truncation                              |
| `quiet`             | `false` | suppress the progress lines                        |
| `show_tables`       | `true`  | print the convergence and leaderboard tables       |

WHAT IS SHARED AND WHAT IS NOT. Each fit gets ONE `EvaluationContext` — one pricing sweep
over the union of every metric's markets, one `OddsView`, one outcome index — and every
metric reads it. Six metrics on one fit therefore price the posterior once, where
`evaluate_experiments` prices and joins it six times.

THREADED OVER FITS, NOT OVER METRICS. Fits are independent and roughly equal in cost;
metrics are neither. The inner pricing sweep drops to single-threaded when more than one
fit is in flight, so the two levels do not oversubscribe.

ERRORS ARE COLLECTED, NOT SWALLOWED. A metric that raises loses its OWN columns for that
fit and nothing else — the other metrics' columns survive, the row survives, and the
failure is reported in `report.errors` and printed. `evaluate_experiments` drops the
entire model on any single metric's failure, which is how a leaderboard silently loses a
model to one missing odds column.
"""
function evaluate_fits(metrics::AbstractVector,
                       fits::AbstractVector,
                       ds::DataStore;
                       require_converged::Bool = true,
                       threaded::Bool = true,
                       max_goals::Integer = Predictions.TPL_MAX_GOALS,
                       quiet::Bool = false,
                       show_tables::Bool = true)
    t0 = time()
    ms = AbstractScoringRule[m for m in metrics]
    isempty(ms) && error("evaluate_fits: no metrics given.")
    isempty(fits) && error("evaluate_fits: no fits given.")

    quiet || begin
        println("="^78)
        println(" Unified evaluation — ", length(fits), " fit(s) × ", length(ms), " metric(s)")
        println("   metrics : ", join(get_metric_method_name.(ms), ", "))
        println("   gate    : require_converged = ", require_converged,
                require_converged ? "  (unconverged fits are EXCLUDED)" :
                                    "  (unconverged fits are FLAGGED)")
        println("="^78)
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
            @printf("  EXCLUDED  %-28s  %s\n", r.model, r.failed_gates)
        end
    end

    n_keep = length(keep_idx)
    rows = Vector{Union{Nothing, Dict{Symbol, Any}}}(nothing, n_keep)
    errs = [EvaluationError[] for _ in 1:n_keep]
    inner_threaded = threaded && n_keep <= 1

    function score_one(slot::Int)
        fit = fits[keep_idx[slot]]
        name = _fit_display_name(fit)
        row = Dict{Symbol, Any}(:model => name)
        local ctx
        try
            ctx = build_evaluation_context(fit_latents(fit), ds.odds, ds.matches, ms;
                                           max_goals = max_goals,
                                           threaded = inner_threaded)
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
            mark = isempty(errs[slot]) ? "  ok " : "  !! "
            flag = r.converged ? "" : "   [UNCONVERGED: $(r.failed_gates)]"
            @printf("%s%-28s %d metric(s)%s\n", mark, r.model,
                    length(ms) - length(errs[slot]), flag)
        end
    end

    all_errors = reduce(vcat, errs; init = EvaluationError[])
    master = _assemble_rows(rows)
    if nrow(master) > 0
        sort!(master, :model)
        # The verdict travels WITH the numbers. A row pulled out of this frame into a
        # plot or a CSV carries the reason it should or should not be trusted.
        vmap = Dict(r.model => r for r in conv_rows)
        master.converged = Bool[vmap[m].converged for m in master.model]
        master.max_rhat = Float64[vmap[m].max_rhat for m in master.model]
        master.min_ess_bulk = Float64[vmap[m].min_ess_bulk for m in master.model]
    end

    report = EvaluationReport(master, conv_df, ms, excluded, all_errors,
                              require_converged, time() - t0)

    if show_tables && !quiet
        println()
        display_convergence(report)
        for fam in unique(_family_symbol.(ms))
            display_summary_metric(report, fam)
        end
        println()
        @printf("  evaluated in %.2fs\n", report.elapsed)
        isempty(all_errors) ||
            println("  ", length(all_errors), " metric error(s) — see `report.errors`")
    end

    return report
end

evaluate_fits(metric::AbstractScoringRule, fits::AbstractVector, ds::DataStore; kwargs...) =
    evaluate_fits([metric], fits, ds; kwargs...)

evaluate_fits(metrics::AbstractVector, fit::Fit, ds::DataStore; kwargs...) =
    evaluate_fits(metrics, [fit], ds; kwargs...)

"""
    _assemble_rows(rows) -> DataFrame

Union the row dictionaries into one wide frame, filling absent columns with `missing`.

A metric that failed for ONE fit leaves that fit's cells empty rather than removing the
column from every other fit — which is what a `Vector{NamedTuple}` → `DataFrame`
conversion would force, since that requires identical keys.
"""
function _assemble_rows(rows::Vector{Union{Nothing, Dict{Symbol, Any}}})
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

_family_symbol(m::AbstractScoringRule) = Symbol(_family_name(m))

_family_name(::LogLoss) = "logloss"
_family_name(::LPD) = "lpd"
_family_name(::CRPS) = "crps"
_family_name(::RQR) = "rqr"
_family_name(::GLMEdge) = "glmedge"
_family_name(::MIQ) = "miq"
_family_name(::PredictionScore) = "predictions"
_family_name(m::AbstractScoringRule) = lowercase(String(nameof(typeof(m))))


# ==============================================================================
# 5. TABULAR FORMATTERS
# ==============================================================================

"""
    display_convergence(report; io = stdout)

The gate's own table: every submitted fit, its verdict, and the number that produced it.
"""
function display_convergence(r::EvaluationReport; io::IO = stdout)
    df = r.convergence
    nrow(df) == 0 && return nothing
    println(io, "\n--- Convergence ---")
    w = max(maximum(length, df.model), 5)
    @printf(io, "  %-*s  %-7s  %8s  %10s  %6s  %s\n",
            w, "MODEL", "VERDICT", "MAX RHAT", "MIN ESS", "N DIV", "FAILED GATES")
    println(io, "  ", "-"^(w + 52))
    for row in eachrow(df)
        @printf(io, "  %-*s  %-7s  %8.4f  %10.1f  %6d  %s\n",
                w, row.model, row.converged ? "PASS" : "FAIL",
                row.max_rhat, row.min_ess_bulk, row.n_divergent, row.failed_gates)
    end
    return nothing
end

# The legacy `display_summary_metric(df, family)` in batch_runner.jl is untouched; these
# two methods add the report overload and the two families it has no branch for.
display_summary_metric(r::EvaluationReport, family::Symbol) =
    family === :convergence ? display_convergence(r) :
                              display_summary_metric(r.rows, family)

"""
    report_table(report; columns = nothing) -> DataFrame

The scored frame with the convergence verdict pulled to the front, for printing.

`columns` selects a subset by name; absent ones are dropped rather than raising, because
a batch that lost one metric to an error should still tabulate the rest.
"""
function report_table(r::EvaluationReport; columns = nothing)
    nrow(r.rows) == 0 && return r.rows
    nm = names(r.rows)
    want = columns === nothing ? Symbol.(nm) : Symbol[Symbol(c) for c in columns]
    lead = Symbol[:model]
    for c in (:converged, :max_rhat)
        string(c) in nm && push!(lead, c)
    end
    rest = [c for c in want if !(c in lead) && string(c) in nm]
    return select(r.rows, vcat(lead, rest)...)
end

"""
    leaderboard(report, column; higher_is_better = false) -> DataFrame

Rank the scored fits on one column, carrying the convergence verdict alongside.

The verdict column is not optional and not removable, because the failure this whole file
exists to prevent is a ranked table that does not say which of its rows can be believed.
"""
function leaderboard(r::EvaluationReport, column::Symbol; higher_is_better::Bool = false)
    string(column) in names(r.rows) ||
        error("leaderboard: no column `$column`. Available: $(join(names(r.rows), ", "))")
    keep = [:model, column]
    for extra in (:converged, :max_rhat)
        string(extra) in names(r.rows) && push!(keep, extra)
    end
    out = select(r.rows, keep...)
    sort!(out, column; rev = higher_is_better)
    return out
end


# ==============================================================================
# 6. MARKDOWN
# ==============================================================================

"""
    markdown_report(report; columns = nothing, title = "Evaluation") -> String

The whole report as GitHub-flavoured Markdown: the convergence table, the scored table,
and the collected errors.

Written for pasting into a ticket or a findings document, which is where an evaluation
run actually ends up. The convergence table comes FIRST and is never omitted — a
leaderboard pasted without it is the exact artefact this framework exists to stop
producing.
"""
function markdown_report(r::EvaluationReport; columns = nothing,
                         title::AbstractString = "Evaluation")
    io = IOBuffer()
    println(io, "# ", title)
    println(io)
    println(io, "- metrics: ", join(get_metric_method_name.(r.metrics), ", "))
    println(io, "- scored: ", nrow(r.rows), " fit(s)")
    println(io, "- gate: `require_converged = ", r.require_converged, "`")
    isempty(r.excluded) ||
        println(io, "- excluded: ", join(r.excluded, ", "))
    println(io, "- elapsed: ", round(r.elapsed, digits = 2), "s")
    println(io)

    println(io, "## Convergence")
    println(io)
    _markdown_table(io, r.convergence)
    println(io)

    println(io, "## Scores")
    println(io)
    _markdown_table(io, report_table(r; columns = columns))

    if !isempty(r.errors)
        println(io)
        println(io, "## Errors")
        println(io)
        for e in r.errors
            println(io, "- **", e.model, "** / `", e.metric, "` — ",
                    replace(e.message, "\n" => " "))
        end
    end
    return String(take!(io))
end

"One `DataFrame` as a Markdown table. Numbers rounded to four places; `missing` as `—`."
function _markdown_table(io::IO, df::AbstractDataFrame)
    if nrow(df) == 0 || ncol(df) == 0
        println(io, "_(no rows)_")
        return nothing
    end
    cols = names(df)
    println(io, "| ", join(cols, " | "), " |")
    println(io, "|", join(fill("---", length(cols)), "|"), "|")
    for row in eachrow(df)
        println(io, "| ", join((_md_cell(row[c]) for c in cols), " | "), " |")
    end
    return nothing
end

_md_cell(::Missing) = "—"
_md_cell(x::AbstractFloat) = isfinite(x) ? string(round(x, digits = 4)) : string(x)
_md_cell(x) = string(x)
