# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : THE MATHEMATICAL PARITY HARNESS
# ==============================================================================
#
# Does the typed kernel produce the same number as the one it replaces?
#
# ------------------------------------------------------------------------------
# THE COMPARABILITY CONTRACT
# ------------------------------------------------------------------------------
#
# Both sides are fed ONE set of numbers. The typed container and the legacy
# `latents.df` are the same posterior — `to_legacy_dataframe` builds the frame FROM the
# container, and `06/r01_demo.jl` holds that conversion at 0 ULP against the real
# `extract_oos_predictions` output. So a difference reported here is the METRIC's, and
# nothing else's.
#
# The legacy side always goes through the LIVE `BayesianFootball.Evaluation` kernels —
# a real `Experiments.ExperimentResults`, a real `Experiments.LatentStates`, the real
# `Predictions.model_inference` and the real four-column `innerjoin`. Nothing is
# transcribed. The tables therefore track `src` as `src` changes, which a hand-copied
# reference implementation would not.
#
# ------------------------------------------------------------------------------
# WHERE BIT-IDENTITY IS AND IS NOT ACHIEVABLE, AND WHY
# ------------------------------------------------------------------------------
#
# `06` gates at 0 ULP because it compares PRICES: the same arithmetic on the same
# operands in the same order. This file compares AGGREGATES over a set of rows, and
# three things can legitimately move the last bit:
#
#   1. ROW ORDER. `mean` over `n` numbers is a pairwise sum whose value depends on the
#      order. The kernels here walk `ds.odds` top to bottom; `src` walks whatever order
#      `innerjoin(ds.odds, model_features, …)` emitted, which `DataFrames` documents as
#      unspecified. Most rows come out bit-identical anyway — the join does, in this
#      implementation, follow the left frame — but that is an observation, not a
#      contract, and a gate must not be written as though it were one.
#
#   2. THE RNG. `src`'s RQR draws from the unseeded GLOBAL rng (`rqr.jl:50`), so its
#      output differs between two runs of the same code. Parity is only meaningful
#      against a snapshot of a seeded global stream, which is what `parity_report`
#      hands to both sides.
#
#   3. NOTHING ELSE. The scalar formulae are copied verbatim (`l02` §5), the posterior
#      means are `mean(view(M, i, :))` — bit-identical to `mean` of the `Vector` the
#      legacy frame stored, checked in `r01_demo.jl` §4d — and the market probabilities
#      come from `06`'s kernels, already at 0 ULP against `compute_market_probs`
#      (checked again, through this framework's tensor, in §4f).
#
# So the gate is `max |Δ| ≤ 1e-12` — the briefing's — and the table additionally prints
# how many leaves came out BIT-IDENTICAL, so that "passed on tolerance" is never
# mistaken for "identical". `MetricParityRow` argues the case in full.
#
# ==============================================================================


# ==============================================================================
# 1. LEAF EXTRACTION
# ==============================================================================
#
# Two result structs from two different modules, with the same field names. Compared
# by walking both to their scalars and matching on the PATH, so a field reordered on
# one side is a name mismatch rather than a silently wrong comparison.

"""
    parity_leaves(prefix, x) -> Vector{Pair{String, Any}}

Every scalar inside a result, keyed by its dotted path. `Real` and `Missing` are
leaves; anything else is recursed through `propertynames`.

Deliberately NOT `unroll`: that dispatches on `AbstractMetricComponent`, which the two
sides define separately, so a legacy `LogLossComponent` would fall through to the
scalar method and raise. This walks by structure, not by type.
"""
function parity_leaves(prefix::AbstractString, x)
    x isa Real    && return [String(prefix) => Float64(x)]
    x isa Missing && return [String(prefix) => missing]
    out = Pair{String, Any}[]
    for k in propertynames(x)
        append!(out, parity_leaves("$(prefix).$(k)", getproperty(x, k)))
    end
    return out
end

parity_leaves(x) = parity_leaves("", x)

"""
    MetricParityRow

One metric's comparison, reported with the two numbers that matter and one that is
easy to misread.

| field       | is                                                                  |
|-------------|---------------------------------------------------------------------|
| `n_leaves`  | scalars compared                                                     |
| `n_exact`   | of those, how many are BIT-IDENTICAL                                 |
| `max_abs`   | largest absolute difference — **the gate**                           |
| `max_ulp`   | largest ULP distance — reported, NOT gated (see below)               |
| `worst`     | the leaf path that produced `max_ulp`                                |

WHY THE GATE IS ABSOLUTE AND THE ULP COLUMN IS NOT.

`06/l04_parity.jl` gates at 0 ULP and argues, correctly, that a tolerance-only gate can
pass a change that has really happened. That argument holds for its subject — PRICES,
which are the same arithmetic on the same operands in the same order, so bit-identity
is achievable and anything less is a bug.

It does not transfer here, for one specific reason: `src` accumulates its per-row scores
in the order `DataFrames.innerjoin` emitted them, and that order is documented as
unspecified. Floating-point addition is not associative, so the two means can differ in
the last bit for a reason that is not a defect and cannot be removed without
reimplementing a hash join.

That one-ULP difference is then AMPLIFIED in two places, which is why the ULP column
reads large next to an absolute difference of 1e-16:

  * `diff_ll` and `diff_lpd` are differences of two O(0.6) numbers. A 1.1e-16 error on
    each is 1 ULP of the operands and ~256 ULP of a 0.0024 difference. The number has
    not moved; the yardstick has shrunk.
  * `GLMEdge` runs iteratively reweighted least squares. A last-bit change in an input
    moves the converged coefficients by tens of ULP, and would whatever the kernels did.

So the gate is `max_abs <= tol` with the briefing's `tol = 1e-12`, and `n_exact` is
printed so a reader can see how much of the result is bit-identical anyway rather than
taking "within tolerance" on trust. §6.4 of `r01_demo.jl` is a negative control sized to
THAT gate — a perturbation small enough to be invisible in a chart and large enough to
break 1e-12 — because a control the gate cannot detect tests nothing.
"""
struct MetricParityRow
    check::String
    n_leaves::Int
    n_exact::Int
    max_abs::Float64
    max_ulp::Int64
    worst::String
    tol::Float64
    pass::Bool
end

"""
    parity_results(check, legacy, new; tol = 1e-12, skip = String[]) -> MetricParityRow

Compare two evaluation results leaf by leaf.

`NaN == NaN` and `missing == missing` COUNT AS EQUAL and as EXACT. Both are legitimate
metric outputs — an empty selection group reports `missing`, a `GLMEdge` with too few
rows reports `NaN` — and a harness that failed on them could not check the thing most
worth checking, which is that the two implementations agree about *when a number does
not exist*.

`skip` drops leaves by path substring, for a comparison deliberately not being made.
"""
function parity_results(check::AbstractString, legacy, new;
                        tol::Float64 = 1e-12, skip::Vector{String} = String[])
    a = parity_leaves(legacy)
    b = parity_leaves(new)

    keep(p) = !any(s -> occursin(s, p), skip)
    a = filter(kv -> keep(first(kv)), a)
    b = filter(kv -> keep(first(kv)), b)

    fail(tag) = MetricParityRow(check * tag, 0, 0, Inf, typemax(Int64), "—", tol, false)

    (length(a) == length(b) && first.(a) == first.(b)) || return fail(" [SHAPE MISMATCH]")
    isempty(a) && return fail(" [nothing compared]")

    max_abs = 0.0
    max_ulp = Int64(0)
    worst = "—"
    n_exact = 0
    for ((path, x), (_, y)) in zip(a, b)
        if x === missing || y === missing
            (x === missing && y === missing) || return fail(" [missing mismatch: $path]")
            n_exact += 1
            continue
        end
        fx = Float64(x); fy = Float64(y)
        if isnan(fx) || isnan(fy)
            (isnan(fx) && isnan(fy)) || return fail(" [NaN mismatch: $path]")
            n_exact += 1
            continue
        end
        fx === fy && (n_exact += 1)
        d = abs(fx - fy)
        d > max_abs && (max_abs = d)
        u = ulp_distance(fx, fy)
        if u > max_ulp
            max_ulp = u
            worst = path
        end
    end
    return MetricParityRow(check, length(a), n_exact, max_abs, max_ulp, worst,
                           tol, max_abs <= tol)
end

"""
    metric_parity_table(rows; title, tol) -> Bool

Print the report and return whether every row passed.

`exact` is `n_exact / n_leaves`. A row can pass with `exact` below 100% — see
`MetricParityRow`'s docstring for the two reasons — and the column is there so that
"passed on tolerance" is never mistaken for "identical".
"""
function metric_parity_table(rows::Vector{MetricParityRow};
                             title::AbstractString = "PARITY")
    width = maximum(length(r.check) for r in rows; init = 20)
    rule = "-" ^ (width + 62)
    println()
    println("  ", title)
    println("  ", rule)
    @printf("  %-*s %8s %9s %12s %9s  %-8s %s\n", width,
            "metric", "leaves", "exact", "max |Δ|", "max ULP", "verdict", "worst leaf")
    println("  ", rule)
    ok = true
    for r in rows
        r.pass && r.n_leaves > 0 || (ok = false)
        verdict = r.n_leaves == 0 ? "FAIL" : (r.pass ? "pass" : "FAIL (>tol)")
        absstr = isfinite(r.max_abs) ? @sprintf("%.3e", r.max_abs) : "Inf"
        ulpstr = r.max_ulp == typemax(Int64) ? "n/a" : string(r.max_ulp)
        @printf("  %-*s %8d %5d/%-3d %12s %9s  %-8s %s\n", width,
                r.check, r.n_leaves, r.n_exact, r.n_leaves, absstr, ulpstr,
                verdict, r.max_ulp == 0 ? "—" : r.worst)
    end
    println("  ", rule)
    @printf("  gate: every leaf within %.0e absolute.  %d of %d rows bit-identical throughout.\n",
            first(rows).tol, count(r -> r.n_exact == r.n_leaves && r.n_leaves > 0, rows),
            length(rows))
    return ok
end


# ==============================================================================
# 2. THE LEGACY SIDE
# ==============================================================================
#
# A genuine `BayesianFootball.Experiments.ExperimentResults`, built the way
# `Experiments.run_experiment` builds one, so `Evaluation.compute_metric`'s typed
# signature accepts it and every line it executes is the production line.

"""
    legacy_experiment(name, model, chains, metas, splitter; sampler, save_dir)
        -> BayesianFootball.Experiments.ExperimentResults

The real `src` container. Not a mock: `Evaluation.compute_metric`'s signature is
`(::AbstractScoringRule, ::ExperimentResults, ::DataStore, ::Any)`, so a stand-in would
not dispatch, and a harness that had to loosen the signature to run would no longer be
testing the production path.
"""
function legacy_experiment(name::AbstractString, model, chs::AbstractVector,
                           metas::AbstractVector, splitter;
                           sampler = UE_BF.Samplers.NUTSConfig(),
                           save_dir::AbstractString = tempdir())
    cfg = UE_BF.Experiments.ExperimentConfig(
        name = String(name),
        model = model,
        splitter = splitter,
        training_config = UE_BF.Training.TrainingConfig(
            sampler = sampler, strategy = UE_BF.Training.Independent()),
        save_dir = String(save_dir))
    tr = UE_BF.Training.TrainingResults([(chs[k], metas[k]) for k in eachindex(chs)])
    return UE_BF.Experiments.ExperimentResults(cfg, tr, nothing, String(save_dir))
end

"""
    legacy_latent_states(latents, model) -> BayesianFootball.Experiments.LatentStates

The typed container as the legacy wrapper `Predictions.model_inference` dispatches on.

The frame comes from `to_legacy_dataframe` (06), which `06/r01_demo.jl` holds at 0 ULP
against a frame built by `src`'s own `_latent_state_dict_to_df` — so the two sides of
every row below start from the same bits.
"""
legacy_latent_states(l::AbstractPosteriorLatents, model) =
    UE_BF.Experiments.LatentStates(to_legacy_dataframe(l), model)

"""
    legacy_metric(metric) -> the corresponding `BayesianFootball.Evaluation` trigger

Translate this framework's rule into `src`'s, so the comparison is over the same
SCOPE. The selection filter carries across directly; `markets` has no legacy
counterpart, which is precisely why `parity_report` also restricts the odds frame — see
its docstring.
"""
legacy_metric(m::LogLoss) = UE_Eval.LogLoss(copy(m.selections))
legacy_metric(m::LPD)     = UE_Eval.LPD(copy(m.selections))
legacy_metric(m::GLMEdge) = UE_Eval.GLMEdge(copy(m.selections))
legacy_metric(::CRPS)     = UE_Eval.CRPS()
legacy_metric(::RQR)      = UE_Eval.RQR()
legacy_metric(::MIQ)      = UE_Eval.MIQ()

"""
    legacy_compute(metric, exp, ds, latent_states)

Call `src`'s kernel. A thin wrapper so every call site in `r01_demo.jl` reads the same
and the `BayesianFootball.Evaluation` qualification lives in one place.
"""
legacy_compute(metric, exp, ds::UE_D.DataStore, ls) =
    UE_Eval.compute_metric(metric, exp, ds, ls)

"Clear `Predictions._PPD_CACHE`, so a legacy timing measures work rather than a hit."
function clear_ppd_cache!()
    try
        empty!(UE_Pred._PPD_CACHE)
    catch
        # The cache is an implementation detail of `src`; if it is renamed, a timing
        # row becomes optimistic for the LEGACY side, which is the safe direction.
    end
    return nothing
end


# ==============================================================================
# 3. THE PARITY REPORT
# ==============================================================================

"""
    parity_report(metrics, latents, exp, ds; tol, ulp_budget, rqr_seed) -> Vector{ParityRow}

One row per metric: `src`'s answer against this framework's.

SCOPE ALIGNMENT IS THE SUBTLE PART. `src` prices every market in
`DEFAULT_MARKET_CONFIG` and then keeps whatever survives the join with `ds.odds`; this
framework prices `scored_markets(metric)`. The two therefore score the same rows only
when the odds frame contains no market outside `scored_markets(metric)` that also
survives the legacy selection filter.

`r01_demo.jl` satisfies that by building its odds frame over exactly the markets the
metrics name — which is not a special case but the normal one: a real `ds.odds` for
these leagues carries 1X2, Over/Under and BTTS and the model prices all of them. The
requirement is checked rather than assumed by `parity_scope_ok`, below.

RQR TAKES A SNAPSHOT OF THE SEEDED GLOBAL STREAM. `src` calls the unseeded global rng,
so the only way to compare is to seed it, hand the same state to both sides, and let
each draw its own copy — which is what happens here.
"""
function parity_report(metrics::AbstractVector,
                       latents::AbstractPosteriorLatents,
                       exp,
                       ds::UE_D.DataStore;
                       tol::Float64 = 1e-12,
                       rqr_seed::Int = 20240808,
                       model = nothing)
    mdl = model === nothing ? _ue_fit_model(exp) : model
    ls  = legacy_latent_states(latents, mdl)
    rows = MetricParityRow[]

    for metric in metrics
        # `get_metric_method_name` already carries the selection filter; appending
        # `metric_column_suffix` on top would print it twice.
        name = get_metric_method_name(metric)
        metric isa LPD && metric.target === :score && (name *= "_score")
        ctx = evaluation_context(latents, ds.odds, ds.matches, [metric]; threaded = false)

        legacy_result, new_result = if metric isa RQR
            Random.seed!(rqr_seed)
            lg = legacy_compute(legacy_metric(metric), exp, ds, ls)
            Random.seed!(rqr_seed)
            nw = compute_metric(metric, ctx; rng = copy(Random.default_rng()))
            (lg, nw)
        else
            (legacy_compute(legacy_metric(metric), exp, ds, ls),
             compute_metric(metric, ctx))
        end

        push!(rows, parity_results(name, legacy_result, new_result; tol = tol))
    end
    return rows
end

"""
    parity_control(metric, reference_latents, perturbed_latents, exp, ds;
                   model, tol) -> MetricParityRow

A NEGATIVE CONTROL: `src`'s answer on the REFERENCE container against this framework's
answer on a PERTURBED one.

Both sides must not be perturbed. Running the perturbed container through
`parity_report` would hand it to `src` as well, the two implementations would agree
about the perturbed posterior exactly as they agree about the unperturbed one, and the
control would report "pass" while testing nothing — which is the trap this function
exists to stay out of, and the one an earlier version of `r01_demo.jl` fell into.
"""
function parity_control(metric::AbstractScoringRule,
                        reference::AbstractPosteriorLatents,
                        perturbed::AbstractPosteriorLatents,
                        exp, ds::UE_D.DataStore;
                        model = nothing, tol::Float64 = 1e-12)
    mdl = model === nothing ? _ue_fit_model(exp) : model
    legacy = legacy_compute(legacy_metric(metric), exp, ds,
                            legacy_latent_states(reference, mdl))
    new = compute_metric(metric, perturbed, ds.odds, ds.matches; threaded = false)
    return parity_results(get_metric_method_name(metric), legacy, new; tol = tol)
end

"""
    parity_scope_ok(metric, odds_df) -> (ok::Bool, offenders::Vector{Symbol})

Whether `src` and this framework would score the SAME ROWS for `metric` on this odds
frame.

They do when every selection the legacy filter would let through is one this
framework's `scored_markets(metric)` prices. An offender is a selection present in the
odds frame, passing the legacy filter, and unpriced here — each one is a row `src`
would include and this framework would not, and the metric would differ for a reason
that has nothing to do with the kernels.

Called by `r01_demo.jl` BEFORE the parity table, so a scope mismatch is reported as a
scope mismatch rather than showing up as a mysterious 1e-3 in `max |Δ|`.
"""
function parity_scope_ok(metric::AbstractScoringRule, odds_df::AbstractDataFrame)
    priced = Set{Symbol}(market_selections(scored_markets(metric)))
    filt = _ue_selection_filter(metric)
    offenders = Symbol[]
    for s in unique(odds_df.selection)
        sym = s isa Symbol ? s : Symbol(s)
        _ue_passes(filt, sym) || continue
        sym in priced || (sym in offenders || push!(offenders, sym))
    end
    return (isempty(offenders), offenders)
end


# ==============================================================================
# 4. COST
# ==============================================================================

"""
    CostRow

One measured comparison of the two paths. `speedup` and `shrink` are legacy ÷ new, so
greater than one is an improvement in both columns.
"""
struct CostRow
    what::String
    legacy_seconds::Float64
    new_seconds::Float64
    legacy_bytes::Int
    new_bytes::Int
end

speedup(r::CostRow) = r.new_seconds > 0 ? r.legacy_seconds / r.new_seconds : NaN
shrink(r::CostRow)  = r.new_bytes > 0 ? r.legacy_bytes / r.new_bytes : NaN

"""
    cost_table(rows; title) -> nothing

Print the bill. No verdict is returned and no gate is attached: a timing on a
24-fixture synthetic fold is an indication, not a measurement of production cost, and
gating a CI run on it would make the run flaky for a reason unrelated to correctness.
"""
function cost_table(rows::Vector{CostRow}; title::AbstractString = "COST")
    width = maximum(length(r.what) for r in rows; init = 20)
    rule = "-" ^ (width + 60)
    println()
    println("  ", title)
    println("  ", rule)
    @printf("  %-*s %11s %11s %8s %12s %12s %8s\n", width,
            "path", "legacy s", "new s", "speedup", "legacy KiB", "new KiB", "shrink")
    println("  ", rule)
    for r in rows
        @printf("  %-*s %11.4f %11.4f %7.2f× %12.1f %12.1f %7.1f×\n", width,
                r.what, r.legacy_seconds, r.new_seconds, speedup(r),
                r.legacy_bytes / 1024, r.new_bytes / 1024, shrink(r))
    end
    println("  ", rule)
    return nothing
end

"""
    measure_cost(what, metrics, latents, exp, ds; model) -> CostRow

Time and size both paths over the SAME metric list.

The legacy side pays `model_inference` (every market in `DEFAULT_MARKET_CONFIG`) plus
one four-column `innerjoin` PER METRIC; the new side pays one pricing sweep over the
union of `scored_markets` plus one pass over the odds vectors per metric. The PPD cache
is cleared first, so the legacy number is work and not a memo hit — clearing it is the
honest choice in both directions, since a real batch of eleven experiments misses that
cache on all eleven.

Bytes are the POSTERIOR PROBABILITY MATERIALISATION on each side: `tpl_dataframe_bytes`
of the legacy PPD frame against `probability_bytes` of the tensor. That is the term the
container change actually moves; the score grids are transient on both sides.
"""
function measure_cost(what::AbstractString, metrics::AbstractVector,
                      latents::AbstractPosteriorLatents, exp, ds::UE_D.DataStore;
                      model = nothing)
    mdl = model === nothing ? _ue_fit_model(exp) : model
    ls = legacy_latent_states(latents, mdl)

    # --- legacy ---------------------------------------------------------------
    clear_ppd_cache!()
    t_legacy = @elapsed for metric in metrics
        legacy_compute(legacy_metric(metric), exp, ds, ls)
    end
    ppd = UE_Pred.model_inference(ls)
    legacy_bytes, _ = tpl_dataframe_bytes(ppd.df)
    clear_ppd_cache!()

    # --- new ------------------------------------------------------------------
    ms = collect(AbstractScoringRule, metrics)
    t_new = @elapsed begin
        ctx = evaluation_context(latents, ds.odds, ds.matches, ms; threaded = false)
        for metric in ms
            compute_metric(metric, ctx)
        end
    end
    ctx2 = evaluation_context(latents, ds.odds, ds.matches, ms; threaded = false)

    return CostRow(String(what), t_legacy, t_new, legacy_bytes, probability_bytes(ctx2.probs))
end


# ==============================================================================
# 5. LIVE DEFECT PROBES
# ==============================================================================
#
# Three failures in `src/evaluation/` that this framework does not inherit. Each is
# REPRODUCED rather than described, so the claim decays with the code: if one is fixed
# upstream, the corresponding probe stops returning a failure and `r01_demo.jl` says so
# instead of continuing to assert a defect that no longer exists.

"""
    probe_poisson_latent_columns(model, latents) -> (raised::Bool, message::String)

`Predictions.get_latent_column_symbols` has exactly two methods — one for
`AbstractNegBinModel` (negativebinomial.jl:29) and one for the Frank-copula NegBin
model (frank_copula.jl:77). `Evaluation`'s CRPS and RQR both call it unconditionally
(`crps.jl:69`, `rqr.jl:89`).

So **`CRPS` and `RQR` cannot be computed for any Poisson model.** Every
`AbstractPoissonModel` engine — which is most of the team-level ladder and every
`ComposableCountModel` with a `PoissonObservation` — raises `MethodError` inside
`evaluate_experiments`' `try`, which drops the model's entire row from the leaderboard
with a `@warn` and no other trace.

This framework reaches the same numbers through `crps_parameters` / `marginals`, which
dispatch on the CONTAINER (`l02` §6) and have a Poisson method by construction.
"""
function probe_poisson_latent_columns(model, latents::AbstractPosteriorLatents)
    df = to_legacy_dataframe(latents)
    try
        UE_Pred.get_latent_column_symbols(model, df)
        return (false, "get_latent_column_symbols($(nameof(typeof(model))), df) succeeded")
    catch e
        return (true, sprint(showerror, e) |> s -> first(split(s, '\n')))
    end
end

"""
    probe_miq_translator(exp, miq_result) -> (raised::Bool, message::String)

`MIQStats`' fields are `Union{Missing, Float64}` (`miq.jl:12-18`) and
`Evaluation.unroll` has methods for `Real` and `AbstractMetricComponent` only
(`translator.jl:6,11`).

So **an `MIQResult` with any empty selection group cannot be flattened.** In a batch
that is guaranteed: `MIQResult` reports twelve selections including Over/Under 1.5 and
3.5, and a store without those lines makes four of the twelve all-`missing`, at which
point `to_dataframe_row` raises `MethodError(unroll, (String, Missing))` — again inside
the `try`, again dropping the whole model.

`l03_batch_runner.jl` §1 adds the one-line `Missing` method.
"""
function probe_miq_translator(exp, miq_result)
    try
        UE_Eval.to_dataframe_row(exp, UE_Eval.MIQ(), miq_result)
        return (false, "to_dataframe_row on an MIQResult succeeded")
    catch e
        return (true, sprint(showerror, e) |> s -> first(split(s, '\n')))
    end
end

"""
    probe_rqr_nondeterminism(exp, ds, ls) -> (differs::Bool, delta::Float64)

`rqr.jl:50` calls `rand(Uniform(...))` on the unseeded global rng, so **`src`'s RQR
table is different on every run.** Two consecutive calls on identical inputs return
different Shapiro-Wilk statistics, which makes the diagnostic impossible to
re-check and makes two models' RQR rows incomparable unless they were computed in the
same session in the same order.

`RQR(seed = …)` here is reproducible by construction; `r01_demo.jl` §7 also asserts
that two calls agree exactly.
"""
function probe_rqr_nondeterminism(exp, ds::UE_D.DataStore, ls)
    a = UE_Eval.compute_metric(UE_Eval.RQR(), exp, ds, ls)
    b = UE_Eval.compute_metric(UE_Eval.RQR(), exp, ds, ls)
    δ = abs(a.all.shapiro_w - b.all.shapiro_w) + abs(a.all.mean - b.all.mean)
    return (δ > 0, δ)
end


# ==============================================================================
# 6. DETERMINISTIC FIXTURES FOR THE RUNNER
# ==============================================================================
#
# `r01_demo.jl` needs a `DataStore` and a set of `Fit`s and must touch no database.
# The builders live here rather than in the runner because the prototype's rule is
# definitions in the loaders and execution in the runner, and because `parity_report`
# above has a scope contract (`parity_scope_ok`) that only holds for a store built the
# way `synthetic_datastore` builds one.
#
# WHAT IS SYNTHETIC AND WHAT IS NOT. The posteriors are prior draws with a fixed seed —
# `06/l04_parity.jl` §9's `tpl_synthetic_chain`, reused unchanged — and the odds are
# generated by perturbing the model's own prices and adding a fixed vig. Everything
# else is the real code path: real market types, real `_enrich_market_data!`, real
# `extract_parameters`, real `fit_model`, real `Evaluation` kernels.
#
# WHAT THAT DOES AND DOES NOT TEST. It tests that two implementations of six metrics
# agree, that the gate fires, and that the legacy surface still compiles and returns.
# It does NOT test that any model fits anything or that any of these numbers is good.
# A synthetic market built from the model's own prices is one the model beats by
# construction; the log-loss `diff_ll` below is an artefact of that and is not evidence
# of anything.

"The eight HMC internals a Turing NUTS run records, and the audit in `07` reads."
const UE_INTERNALS = ["lp", "n_steps", "is_accept", "acceptance_rate", "step_size",
                      "tree_depth", "numerical_error", "hamiltonian_energy"]

"""
    demo_nuts_chain(colnames; n_draws, n_chains, seed, chain_offset, …) -> Chains

A deterministic `Chains` carrying the requested parameter sites AND the internals the
convergence audit reads, so a `Fit` built from it has a real verdict rather than an
abstention.

`chain_offset` shifts every parameter by a per-chain constant. That is how the
unconverged fit in `r01_demo.jl` §8 is made: between-chain variance with no
within-chain justification is exactly what R-hat measures, and a run built this way
fails the R-hat gate while remaining a perfectly well-formed `Chains`.
"""
function demo_nuts_chain(colnames::Vector{String};
                         n_draws::Int = 200, n_chains::Int = 2, seed::Int = 20240808,
                         chain_offset::Float64 = 0.0, energy_ar::Float64 = 0.50,
                         max_depth::Int = 10)
    rng = MersenneTwister(seed)
    p = length(colnames)
    all_names = vcat(colnames, UE_INTERNALS)
    vals = Array{Float64, 3}(undef, n_draws, length(all_names), n_chains)

    for c in 1:n_chains
        for (j, nm) in enumerate(colnames)
            vals[:, j, c] = tpl_synthetic_site(nm, n_draws, rng) .+ chain_offset * (c - 1)
        end
        E = Vector{Float64}(undef, n_draws)
        E[1] = 40.0 + randn(rng)
        for i in 2:n_draws
            E[i] = 40.0 + energy_ar * (E[i - 1] - 40.0) + sqrt(1 - energy_ar^2) * randn(rng)
        end
        depth = fill(Float64(max_depth - 3), n_draws)
        base = p
        vals[:, base + 1, c] = -520.0 .+ randn(rng, n_draws)      # lp
        vals[:, base + 2, c] = 2 .^ depth                          # n_steps
        vals[:, base + 3, c] .= 1.0                                # is_accept
        vals[:, base + 4, c] = 0.80 .+ 0.05 .* rand(rng, n_draws)  # acceptance_rate
        vals[:, base + 5, c] .= 0.05                               # step_size
        vals[:, base + 6, c] = depth                               # tree_depth
        vals[:, base + 7, c] .= 0.0                                # numerical_error
        vals[:, base + 8, c] = E                                   # hamiltonian_energy
    end

    return Chains(vals, Symbol.(all_names),
                  Dict(:parameters => Symbol.(colnames),
                       :internals  => Symbol.(UE_INTERNALS)))
end

"""
    simulate_scores(latents; seed) -> Vector{Tuple{Int,Int}}

A realised scoreline per fixture, drawn from the container's own posterior-mean
marginals with a fixed seed.

Drawn from the MODEL, so the metrics below measure the kernels rather than a
misspecification the fixtures happened to contain. It also means the model is correctly
specified for this data, which is why the RQR statistics in §5 of the runner sit near
`mean ≈ 0, std ≈ 1` — that is a property of the fixture, not a result.
"""
function simulate_scores(l::AbstractPosteriorLatents; seed::Int = 424243)
    rng = MersenneTwister(seed)
    out = Tuple{Int, Int}[]
    for i in 1:n_matches(l)
        dh, da = marginals(l, i)
        push!(out, (rand(rng, dh), rand(rng, da)))
    end
    return out
end

"""
    synthetic_matches(latents, scores; first_date) -> DataFrame

The `ds.matches` columns every evaluation path reads: `match_id`, `home_score`,
`away_score`, plus the five `crps.jl:78` / `rqr.jl:98` additionally `select`.
"""
function synthetic_matches(l::AbstractPosteriorLatents,
                           scores::Vector{Tuple{Int, Int}};
                           first_date::Date = Date(2025, 1, 5),
                           teams::Union{Nothing, AbstractDataFrame} = nothing)
    ids = latent_match_ids(l)
    n = length(ids)
    dates = [first_date + Day(11 * (i - 1)) for i in 1:n]
    home = teams === nothing ? ["TEAM_$(lpad(1 + (i - 1) % 8, 2, '0'))" for i in 1:n] :
                               String.(teams.home_team)
    away = teams === nothing ? ["TEAM_$(lpad(1 + i % 8, 2, '0'))" for i in 1:n] :
                               String.(teams.away_team)
    return DataFrame(
        match_id      = copy(ids),
        match_date    = dates,
        match_month   = [Dates.month(d) for d in dates],
        home_score    = [s[1] for s in scores],
        away_score    = [s[2] for s in scores],
        tournament_id = fill(1, n),
        season        = fill("24/25", n),
        home_team     = home,
        away_team     = away,
        match_week    = [(i - 1) ÷ 8 + 1 for i in 1:n],
    )
end

"""
    _ue_is_winner(market, selection, gh, ga) -> Bool

Settle one selection against a realised scoreline. Deliberately explicit rather than
derived from the pricing kernels: a settlement bug that agreed with the pricer would
make every metric look calibrated.
"""
function _ue_is_winner(m::Market1X2, sel::Symbol, gh::Int, ga::Int)
    sel === :home && return gh > ga
    sel === :draw && return gh == ga
    return ga > gh
end
_ue_is_winner(::MarketBTTS, sel::Symbol, gh::Int, ga::Int) =
    sel === :btts_yes ? (gh > 0 && ga > 0) : !(gh > 0 && ga > 0)
_ue_is_winner(m::MarketOverUnder, sel::Symbol, gh::Int, ga::Int) =
    startswith(String(sel), "over_") ? (gh + ga > m.line) : (gh + ga < m.line)

"""
    synthetic_odds(latents, markets, scores; seed, vig, noise) -> DataFrame

A `ds.odds`-shaped long frame over exactly `markets`.

Built by perturbing the model's OWN mean prices in log-odds space, renormalising, and
applying a flat `vig` — then handed to the real
`Data.Markets._enrich_market_data!`, which computes `prob_implied_*`, `overround_*`,
`prob_fair_*`, `fair_odds_*`, `vig_*` and the two CLM columns exactly as the production
fetcher does. Reproducing that arithmetic here instead would be a second
implementation of the thing the metrics are scored against.

`noise = 0` would make the market and the model identical and every `diff` exactly
zero, which passes a parity test while testing nothing; `0.12` puts the two a realistic
distance apart.
"""
function synthetic_odds(l::AbstractPosteriorLatents,
                        markets::AbstractVector,
                        scores::Vector{Tuple{Int, Int}};
                        seed::Int = 424244, vig::Float64 = 0.05, noise::Float64 = 0.12)
    rng = MersenneTwister(seed)
    probs = market_probabilities(l, markets; keep_draws = false, threaded = false)
    ids = latent_match_ids(l)

    mid = Int[]; mname = String[]; mline = Float64[]; sel = Symbol[]
    oopen = Float64[]; oclose = Float64[]; won = Bool[]

    for (i, id) in enumerate(ids)
        gh, ga = scores[i]
        for m in markets
            keys_m = market_keys(m)
            p = Float64[probs.means[i, probs.col_of[k]] for k in keys_m]
            # The model's price, perturbed in log space and re-normalised, then vigged
            # up. `open` is perturbed further, so `clm_prob` is non-degenerate.
            pc = p .* exp.(noise .* randn(rng, length(p)))
            pc ./= sum(pc)
            po = p .* exp.((noise * 1.6) .* randn(rng, length(p)))
            po ./= sum(po)
            for (j, k) in enumerate(keys_m)
                push!(mid, id)
                push!(mname, market_group(m))
                push!(mline, market_line(m))
                push!(sel, k)
                push!(oopen,  1.0 / (po[j] * (1.0 + vig)))
                push!(oclose, 1.0 / (pc[j] * (1.0 + vig)))
                push!(won, _ue_is_winner(m, k, gh, ga))
            end
        end
    end

    df = DataFrame(match_id = mid, market_name = mname, market_line = mline,
                   selection = sel, odds_open = oopen, odds_close = oclose,
                   is_winner = won)
    UE_D.Markets._enrich_market_data!(df)
    return df
end

"""
    synthetic_datastore(latents, markets; seed, …) -> (ds, scores)

A `Data.DataStore` carrying only `matches` and `odds` — the two domains every
evaluation kernel reads — and empty frames for the six it does not.

The segment is a real one (`Data.ScottishLower()`) because `DataStore`'s field is
typed to `DataTournemantSegment` and because `create_experiment_task` reads
`tournament_ids(ds.segment)`; nothing here queries it.
"""
function synthetic_datastore(l::AbstractPosteriorLatents, markets::AbstractVector;
                             seed::Int = 424243, vig::Float64 = 0.05,
                             noise::Float64 = 0.12,
                             fixtures::Union{Nothing, AbstractDataFrame} = nothing)
    scores = simulate_scores(l; seed = seed)
    matches = synthetic_matches(l, scores; teams = fixtures)
    odds = synthetic_odds(l, markets, scores; seed = seed + 1, vig = vig, noise = noise)
    ds = UE_D.DataStore(UE_D.ScottishLower(), matches, DataFrame(), odds,
                        DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame())
    return (ds, scores)
end
