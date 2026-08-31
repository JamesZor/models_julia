# ==============================================================================
# r62 — The gauntlet: out-of-sample log loss under a full team-strength model
# ==============================================================================
#
# WHAT THIS IS
#   The decision. r60 and r61 measured linear association and incremental R-squared
#   against a two-covariate baseline. Both are weak proxies for the real question,
#   because the count model already carries team strength in dyn.alpha/dyn.beta — a
#   covariate's job is to explain what those CANNOT.
#
#   So the baseline here is not "nothing". It is a Poisson ridge with the engine's own
#   linear predictor:
#
#       log lambda_h = mu + gamma + alpha_h + beta_a + SUM_k q_k^h
#       log lambda_a = mu         + alpha_a + beta_h + SUM_k q_k^a
#
#   fitted by IRLS on the history block with a ridge penalty on the team effects, then
#   scored on held-out fixtures as multiclass 1X2 log loss. A candidate that improves on
#   THAT has earned something.
#
#   Every comparison is a paired bootstrap over held-out matches. Differences of a few
#   thousandths of a nat are routine and meaningless without one.
#
# WHAT THIS IS NOT
#   Not the engine. This is a maximum-a-posteriori Poisson stand-in with independent
#   sides, no dispersion, no time decay and no correlation structure. It ranks candidates
#   honestly and cheaply; r40 remains the arbiter for anything that graduates.
#
# USAGE
#   source .env            # only needed for the English confirmation pass
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r62_feature_gauntlet.jl")
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))
include(joinpath(@__DIR__, "l50_player_norm.jl"))
include(joinpath(@__DIR__, "l60_novel_features.jl"))
include(joinpath(@__DIR__, "l61_holdout_gauntlet.jl"))

Random.seed!(62)

const R62_HISTORY_FRAC = 0.80
const R62_RIDGE = 4.0
const R62_RUN_ENGLISH = get(ENV, "R62_ENGLISH", "1") == "1"

eda_banner("r62 · the gauntlet")

# %%
# ==============================================================================
# 2. Candidate construction, as one reusable pass
# ==============================================================================

"""
    r62_columns(ds) -> (frame, ids, mask, train, columns)

Every candidate column for one store, in one place. `columns` maps a name to
`(vector, role)` where role is `:supremacy` or `:level`.
"""
function r62_columns(ds)
    schedule = l60_schedule(ds)
    ids = [r.id for r in schedule]
    frame = eda_match_frame(ds; ordered_ids = ids)
    cut = clamp(round(Int, R62_HISTORY_FRAC * length(ids)), 1, length(ids) - 1)
    target_ids = Set(ids[(cut + 1):end])
    mask = Bool[(id in target_ids) for id in ids] .& frame.played
    train = Bool[!(id in target_ids) for id in ids] .& frame.played
    history = Set(ids[1:cut])

    records = l60_records(ds)
    align(fs, key) = begin
        order = Dict(id => i for (i, id) in enumerate(Int.(fs.data[:ordered_match_ids])))
        Float64[fs.data[key][order[id]] for id in ids]
    end
    boundary = eda_boundary(ds)
    pxg_fs = eda_features(ds, Features.AbstractFeatureConfig[PxGFeature()]; boundary = boundary)
    wealth_fs = eda_features(ds, Features.AbstractFeatureConfig[ProductionWealthFeature()];
                             boundary = boundary)

    pick(lookup, field) = l60_standardise(
        Float64[getproperty(get(lookup, id, NamedTuple{(field,)}((0.0,))), field) for id in ids])

    adjusted = l60_adjusted_pxg(ds, records)
    slow = l60_exponential(ds, records, l60_pxg; half_life = 20.0)
    squad = l60_squad(ds)
    bench = l60_bench(ds)
    late = l60_late_share(ds)

    rapm_xg, _ = l60_rapm_column(ds, history, ids; target = :y_xg)
    rapm_shots, _ = l60_rapm_column(ds, history, ids; target = :y_shots)
    rapm_xg = l60_standardise(rapm_xg); rapm_shots = l60_standardise(rapm_shots)
    wealth = l60_standardise(align(wealth_fs, :flat_delta_production_wealth))

    columns = Dict{String, Tuple{Vector{Float64}, Symbol}}(
        "wealth"        => (wealth, :supremacy),
        "pxg_shipped"   => (l60_standardise(align(pxg_fs, :flat_pxg_supremacy)), :supremacy),
        "adj_pxg_sup"   => (pick(adjusted, :sup), :supremacy),
        "adj_pxg_level" => (pick(adjusted, :level), :level),
        "slow_sup"      => (pick(slow, :sup), :supremacy),
        "continuity"    => (pick(squad, :continuity), :supremacy),
        "bench_value"   => (pick(bench, :depth_value), :supremacy),
        "late_share"    => (pick(late, :sup), :supremacy),
        "rapm_xg"       => (rapm_xg, :supremacy),
        "rapm_shots"    => (rapm_shots, :supremacy),
        # H11: non-linear synergy. Wealth and RAPM are near-orthogonal (r93: r = +0.336,
        # both significant beyond the other). If an expensive squad converts money into
        # results only when the players are individually good, the product carries signal
        # neither main effect does.
        "wealth_x_rapm" => (l60_standardise(wealth .* rapm_xg), :supremacy),
    )
    return frame, ids, mask, train, columns
end

# %%
# ==============================================================================
# 3. The gauntlet itself
# ==============================================================================

"""
    r62_gauntlet(label, ds) -> Vector{NamedTuple}

Fit three reference models and then one model per candidate, all with full team effects,
and report held-out 1X2 metrics with a paired bootstrap against the incumbent.
"""
function r62_gauntlet(label::AbstractString, ds)
    eda_section(label, "candidate construction")
    t0 = time()
    frame, ids, mask, train, columns = r62_columns(ds)
    @printf("  %d matches | history %d | held out %d | built in %.1fs\n",
            length(ids), count(train), count(mask), time() - t0)

    run(cols, roles) = l61_run(frame, mask, cols, roles; lambda = R62_RIDGE)

    teams_only, probs_teams, _ = run(Vector{Float64}[], Symbol[])
    incumbent_cols = [columns["wealth"][1], columns["pxg_shipped"][1]]
    incumbent_roles = [:supremacy, :supremacy]
    incumbent, probs_incumbent, _ = run(incumbent_cols, incumbent_roles)

    @printf("\n  %-24s | %9s | %8s | %8s | %8s | %8s\n",
            "model", "logloss", "brier", "rps", "acc", "AUC")
    println("  " * repeat('-', 78))
    @printf("  %-24s | %9.5f | %8.5f | %8.5f | %8.4f | %8.4f\n",
            "team strength only", teams_only.logloss, teams_only.brier,
            teams_only.rps, teams_only.accuracy, teams_only.auc)
    @printf("  %-24s | %9.5f | %8.5f | %8.5f | %8.4f | %8.4f\n",
            "+ wealth + pxg (INCUMBENT)", incumbent.logloss, incumbent.brier,
            incumbent.rps, incumbent.accuracy, incumbent.auc)

    candidates = ["adj_pxg_sup", "adj_pxg_level", "slow_sup", "continuity",
                  "bench_value", "late_share", "rapm_xg", "rapm_shots", "wealth_x_rapm"]
    results = NamedTuple[]
    for name in candidates
        col, role = columns[name]
        metrics, probs, _ = run(vcat(incumbent_cols, [col]), vcat(incumbent_roles, [role]))
        d, lo, hi = l61_bootstrap_logloss(probs, probs_incumbent, frame[mask, :])
        push!(results, (name = name, role = role, metrics = metrics,
                        d_logloss = d, lo = lo, hi = hi))
        @printf("  %-24s | %9.5f | %8.5f | %8.5f | %8.4f | %8.4f\n",
                "+ " * name, metrics.logloss, metrics.brier, metrics.rps,
                metrics.accuracy, metrics.auc)
    end

    println("\n  Change in held-out log loss against the incumbent (NEGATIVE is better),")
    println("  paired bootstrap over held-out matches, 90% interval:")
    @printf("  %-16s | %-9s | %11s | %24s | %s\n",
            "candidate", "role", "d logloss", "90% CI", "verdict")
    println("  " * repeat('-', 92))
    for r in sort(results, by = x -> x.d_logloss)
        verdict = !isfinite(r.lo) ? "—" :
                  r.hi < 0 ? "IMPROVES" : r.lo > 0 ? "HURTS" : "no effect"
        @printf("  %-16s | %-9s | %+11.5f | [%+.5f, %+.5f] | %s\n",
                r.name, String(r.role), r.d_logloss, r.lo, r.hi, verdict)
    end

    # The kitchen sink: everything at once, to check for joint effects the one-at-a-time
    # pass cannot see.
    sink_names = candidates
    sink_cols = vcat(incumbent_cols, [columns[n][1] for n in sink_names])
    sink_roles = vcat(incumbent_roles, [columns[n][2] for n in sink_names])
    sink, probs_sink, _ = run(sink_cols, sink_roles)
    d, lo, hi = l61_bootstrap_logloss(probs_sink, probs_incumbent, frame[mask, :])
    @printf("\n  all %d candidates together: logloss %.5f, d = %+.5f [%+.5f, %+.5f] — %s\n",
            length(sink_names), sink.logloss, d, lo, hi,
            hi < 0 ? "IMPROVES" : lo > 0 ? "HURTS" : "no effect")
    @printf("  incumbent gain over team strength alone: %+.5f nats\n",
            incumbent.logloss - teams_only.logloss)
    return results
end

# %%
# ==============================================================================
# 4. Scottish Lower — the deployment target
# ==============================================================================

ds_scot = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
results_scot = r62_gauntlet("1/2 · SCOTTISH LOWER (deployment target)", ds_scot)

# %%
# ==============================================================================
# 5. English tiers — confirmation at four times the sample
# ==============================================================================

results_eng = NamedTuple[]
if R62_RUN_ENGLISH
    ds_eng = l50_store()
    results_eng = r62_gauntlet("2/2 · ENGLISH TIERS (confirmation)", ds_eng)
else
    println("\n  [SKIPPED] English confirmation pass (set R62_ENGLISH=1 to enable).")
end

# %%
# ==============================================================================
# 6. Verdict
# ==============================================================================

eda_section("VERDICT", "What survives both stores")

let
    eng = Dict(r.name => r for r in results_eng)
    @printf("  %-16s | %-26s | %-26s | %s\n",
            "candidate", "SCOTTISH d logloss", "ENGLISH d logloss", "verdict")
    println("  " * repeat('-', 104))
    winners = String[]
    for r in sort(results_scot, by = x -> x.d_logloss)
        e = get(eng, r.name, nothing)
        scot_good = isfinite(r.hi) && r.hi < 0
        eng_good = e !== nothing && isfinite(e.hi) && e.hi < 0
        eng_bad = e !== nothing && isfinite(e.lo) && e.lo > 0
        verdict = if e === nothing
            scot_good ? "improves (Scottish only tested)" : "no effect"
        elseif scot_good && eng_good
            "IMPROVES ON BOTH"
        elseif scot_good && !eng_bad
            "improves on target, neutral elsewhere"
        elseif scot_good && eng_bad
            "conflicting"
        else
            "no effect"
        end
        (scot_good && !eng_bad) && push!(winners, r.name)
        @printf("  %-16s | %+9.5f [%+.5f,%+.5f] | %s | %s\n",
                r.name, r.d_logloss, r.lo, r.hi,
                e === nothing ? repeat(" ", 26) :
                @sprintf("%+9.5f [%+.5f,%+.5f]", e.d_logloss, e.lo, e.hi),
                verdict)
    end
    println()
    if isempty(winners)
        println("  [VERDICT] Nothing improves held-out log loss once full team strength is in")
        println("            the model. Every candidate's interval spans zero on the deployment")
        println("            store. That is the honest result, and it is worth more than a")
        println("            marginal winner selected by looking at ten intervals at once.")
    else
        println("  [VERDICT] Carried forward to r40: ", join(winners, ", "))
        println("            Ten candidates were tested at a 90% level, so roughly one false")
        println("            positive is expected by chance. Re-confirm any winner on a")
        println("            different split before graduating it.")
    end
end
eda_rule(100, '=')
