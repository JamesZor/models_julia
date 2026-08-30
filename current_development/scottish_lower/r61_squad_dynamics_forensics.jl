# ==============================================================================
# r61 — Congestion, continuity, depth and the late game
# ==============================================================================
#
# HYPOTHESES — all about information sitting in `ds.lineups` that no covariate reads.
#
#   H5  SQUAD CONTINUITY. A settled eleven understands itself. The share of today's XI
#       that also started the previous match, home minus away, should carry positive
#       signal beyond wealth.
#
#   H6  REST ASYMMETRY. Days since the previous fixture, home minus away. Midweek
#       fixtures are unevenly distributed by cup progress, so this is not constant.
#
#   H7  MINUTES LOAD. Accumulated starting-XI minutes over the previous fortnight. A
#       side that has played three games in eight days should tire. Distinct from H6:
#       rest measures the gap, load measures the accumulation.
#
#   H8  BENCH DEPTH. Every wealth covariate in the builder reads the STARTING XI only,
#       so bench quality is invisible. Two sides with identical elevens are not equally
#       strong if one can bring on a better bench.
#
#   H9  LATE-GAME DROP-OFF. If depth and fatigue matter, they should show up
#       concentrated after the 70th minute — a side's historical share of pxG created
#       late should predict its late performance and therefore its results.
#
#   H10 REFEREE. Checked for data availability first; reported as a dead end if the
#       store carries no referee identity.
#
# WHAT THIS IS NOT
#   Not the decision. r62 puts the survivors through a Poisson ridge with full team
#   effects and out-of-sample log loss.
#
# USAGE
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r61_squad_dynamics_forensics.jl")
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

Random.seed!(61)

const R61_HISTORY_FRAC = 0.80

eda_banner("r61 · congestion, continuity, depth and the late game")

# %%
# ==============================================================================
# 2. Store and split
# ==============================================================================

eda_section("1/5", "Store, split and the incumbent baseline")

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
schedule = l60_schedule(ds)
ids = [r.id for r in schedule]
frame = eda_match_frame(ds; ordered_ids = ids)
cut = clamp(round(Int, R61_HISTORY_FRAC * length(ids)), 1, length(ids) - 1)
target_ids = Set(ids[(cut + 1):end])
mask = Bool[(id in target_ids) for id in ids] .& frame.played
train = Bool[!(id in target_ids) for id in ids] .& frame.played
@printf("  %d matches | history %d | held out %d\n", length(ids), count(train), count(mask))

shipped = eda_features(ds, Features.AbstractFeatureConfig[PxGFeature()]; boundary = eda_boundary(ds))
order = Dict(id => i for (i, id) in enumerate(Int.(shipped.data[:ordered_match_ids])))
pxg_shipped = Float64[shipped.data[:flat_pxg_supremacy][order[id]] for id in ids]
wealth_fs = eda_features(ds, Features.AbstractFeatureConfig[ProductionWealthFeature()];
                         boundary = eda_boundary(ds))
worder = Dict(id => i for (i, id) in enumerate(Int.(wealth_fs.data[:ordered_match_ids])))
wealth = Float64[wealth_fs.data[:flat_delta_production_wealth][worder[id]] for id in ids]
const R61_BASELINE = [wealth, pxg_shipped]

# %%
# ==============================================================================
# 3. H10 first — is there any referee data at all?
# ==============================================================================

eda_section("2/5", "H10 · referee identity")

let columns = names(ds.matches)
    if "referee_id" in columns
        n = count(x -> !ismissing(x) && x > 0, ds.matches.referee_id)
        @printf("  referee_id present on %d of %d matches (%.1f%%)\n",
                n, nrow(ds.matches), 100 * n / nrow(ds.matches))
        n == 0 && println("  [DEAD END] column exists but is empty; H10 cannot be tested.")
    else
        println("  [DEAD END] `ds.matches` carries no referee_id column on this segment.")
        println("  `Features.RefereeOfficiatingFeature` reads that column and would emit a")
        println("  constant index here. H10 is untestable without a schema change upstream;")
        println("  recording it rather than silently dropping it.")
    end
end

# %%
# ==============================================================================
# 4. Building the squad columns
# ==============================================================================

eda_section("3/5", "H5–H9 · construction and coverage")

squad = l60_squad(ds)
bench = l60_bench(ds)
late = l60_late_share(ds)

grab(lookup, field) = Float64[getproperty(get(lookup, id,
        NamedTuple{(field,)}((0.0,))), field) for id in ids]

continuity = l60_standardise(grab(squad, :continuity))
rest       = l60_standardise(grab(squad, :rest))
load       = l60_standardise(grab(squad, :load))
depth_val  = l60_standardise(grab(bench, :depth_value))
depth_cnt  = l60_standardise(grab(bench, :depth_count))
late_sup   = l60_standardise(Float64[get(late, id, (sup = 0.0,)).sup for id in ids])

@printf("  squad continuity/rest/load available on %.1f%% of held-out matches\n",
        100 * mean(grab(squad, :available)[mask]))
@printf("  bench depth available on %.1f%%\n", 100 * mean(grab(bench, :available)[mask]))
@printf("  late-game share available on %.1f%%\n",
        100 * mean(Float64[get(late, id, (available = 0.0,)).available for id in ids][mask]))

eda_print_describe([
    eda_describe("continuity", continuity[mask]),
    eda_describe("rest", rest[mask]),
    eda_describe("load", load[mask]),
    eda_describe("bench value", depth_val[mask]),
    eda_describe("bench count", depth_cnt[mask]),
    eda_describe("late pxG share", late_sup[mask]),
]; title = "\n  Held-out distributions (standardised):")

names_used = ["continuity", "rest", "load", "bench_value", "bench_count", "late_share",
              "wealth", "pxg_shipped"]
columns_used = [continuity[mask], rest[mask], load[mask], depth_val[mask],
                depth_cnt[mask], late_sup[mask], wealth[mask], pxg_shipped[mask]]
println()
eda_print_corr(names_used, eda_corr_matrix(names_used, columns_used); title = "Correlation:", flag = 0.5)

# %%
# ==============================================================================
# 5. Evaluation
# ==============================================================================

eda_section("4/5", "Association and incremental value")

candidates = [
    ("continuity",  continuity, :supremacy, "H5 share of XI retained from the last match"),
    ("rest",        rest,       :supremacy, "H6 days since previous fixture, capped at 21"),
    ("load",        load,       :supremacy, "H7 starting-XI minutes in the previous 14 days"),
    ("bench_value", depth_val,  :supremacy, "H8 log bench market value differential"),
    ("bench_count", depth_cnt,  :supremacy, "H8 named substitutes differential"),
    ("late_share",  late_sup,   :supremacy, "H9 historical share of pxG created after 70'"),
    ("load_total",  load,       :total,     "H7 does congestion suppress total goals?"),
]

rows = [l60_evaluate(name, column, frame, train, mask, R61_BASELINE; response = response)
        for (name, column, response, _) in candidates]
@printf("  %-13s | %-9s | %7s | %9s | %9s | %8s | %10s | %7s\n",
        "candidate", "response", "n_live", "r", "rho", "AUC", "dR2 (oos)", "t")
println("  " * repeat('-', 92))
for (r, spec) in zip(rows, candidates)
    @printf("  %-13s | %-9s | %7d | %9s | %9s | %8s | %10s | %7s\n",
            r.name, String(spec[3]), r.n_live,
            eda_fmt(r.r, "%+.4f"), eda_fmt(r.rho, "%+.4f"), eda_fmt(r.auc, "%.4f"),
            eda_fmt(r.delta_r2, "%+.5f"), eda_fmt(r.t, "%+.2f"))
end

println("\n  Bootstrap of the out-of-sample incremental R-squared, 90% interval:")
@printf("  %-13s | %11s | %22s | %s\n", "candidate", "dR2", "90% CI", "verdict")
println("  " * repeat('-', 76))
survivors = String[]
for (name, column, response, _) in candidates
    m, lo, hi = l60_bootstrap_delta_r2(column, frame, train, mask, R61_BASELINE; response = response)
    verdict = !isfinite(lo) ? "—" : lo > 0 ? "ADDS SIGNAL" :
              hi < 0 ? "HURTS" : "indistinguishable from zero"
    lo > 0 && push!(survivors, name)
    @printf("  %-13s | %+11.5f | [%+.5f, %+.5f] | %s\n", name, m, lo, hi, verdict)
end

# %%
# ==============================================================================
# 6. Verdicts
# ==============================================================================

eda_section("5/5", "Verdicts")

byname = Dict(r.name => r for r in rows)
for (tag, key, expectation) in (
        ("H5 continuity ", "continuity",  "positive — a settled side is stronger"),
        ("H6 rest       ", "rest",        "positive — more rest helps"),
        ("H7 load       ", "load",        "negative — accumulated minutes tire a side"),
        ("H8 bench value", "bench_value", "positive — depth is unmeasured strength"),
        ("H9 late share ", "late_share",  "positive — sides that finish strong win more"))
    r = byname[key]
    @printf("  %s : r = %+.4f, AUC %.4f  (expected %s)\n", tag, r.r, r.auc, expectation)
end
println()
if isempty(survivors)
    println("  [VERDICT] None of H5–H9 adds anything beyond production wealth and the shipped")
    println("            pxG column. Every interval spans zero. These are dead ends, and the")
    println("            most useful thing about them is that they are now measured rather")
    println("            than assumed.")
else
    println("  [VERDICT] Survivors carried to r62's gauntlet: ", join(survivors, ", "))
end
eda_rule(100, '=')
