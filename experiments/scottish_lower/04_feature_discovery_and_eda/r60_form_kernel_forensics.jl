# ==============================================================================
# r60 — Three things the pxG covariate throws away
# ==============================================================================
#
# HYPOTHESES
#
#   H1  OPPONENT ADJUSTMENT. `att_h` is a raw rolling mean of pxG created, unadjusted for
#       who the side played. The engine adjusts GOALS for opponent via dyn.alpha/beta;
#       nothing adjusts pxG. Fitting attack and defence ratings on the pxG matrix itself
#       should dominate the raw mean, because pxG carries two to three times the
#       information of a scoreline per match and should identify the same latent faster.
#
#   H2  VOLUME vs QUALITY. pxG = shots x mean xG per shot. Shot volume is a persistent
#       team property; finishing quality is famously not. Fusing them makes a side taking
#       20 poor shots indistinguishable from one taking 8 good ones. Split, the volume
#       half should carry most of the signal and the quality half should be near-noise.
#
#   H3  OVER-PERFORMANCE. `goals - pxG` is the classic mean-reversion signal. The ladder
#       already computes both quantities and discards the residual. A side that has been
#       out-scoring its chances should regress, so this column should carry a NEGATIVE
#       association with future supremacy.
#
#   H4  DUAL HORIZON. A single half-life must serve both short-run form and long-run
#       baseline quality, and is optimal for neither. A fast kernel and a slow kernel
#       entered together should beat any single kernel — testable as whether the fast one
#       adds anything once the slow one is present.
#
# WHAT THIS IS NOT
#   Not the decision. This runner measures each candidate's standalone association and
#   its incremental R-squared over the shipped pair. r62 puts every survivor through a
#   Poisson ridge with full team effects and out-of-sample log loss, which is what
#   actually decides.
#
# USAGE
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r60_form_kernel_forensics.jl")
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

Random.seed!(60)

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R60_HISTORY_FRAC = 0.80
const R60_K = 8
const R60_KAPPA = 3.0
const R60_FAST = 2.0
const R60_SLOW = 20.0
const R60_ADJ_HALF_LIFE = 180.0
const R60_ADJ_LAMBDA = 4.0

eda_banner("r60 · what the pxG covariate throws away")

# %%
# ==============================================================================
# 3. Data and the shipped reference columns
# ==============================================================================

eda_section("1/6", "Store, schedule and the incumbent columns")

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
schedule = l60_schedule(ds)
ids = [r.id for r in schedule]
frame = eda_match_frame(ds; ordered_ids = ids)
cut = clamp(round(Int, R60_HISTORY_FRAC * length(ids)), 1, length(ids) - 1)
target_ids = Set(ids[(cut + 1):end])
mask = Bool[(id in target_ids) for id in ids] .& frame.played
train = Bool[!(id in target_ids) for id in ids] .& frame.played
@printf("  %d matches | history %d | held out %d (played %d)\n",
        length(ids), cut, length(ids) - cut, count(mask))

records = l60_records(ds)
@printf("  records: %d matches, %d with live-text shot coverage (%.1f%%)\n",
        length(records), count(r -> r.covered, values(records)),
        100 * count(r -> r.covered, values(records)) / length(records))

# The incumbents: the shipped pxG supremacy column and production wealth.
shipped = eda_features(ds, Features.AbstractFeatureConfig[PxGFeature()];
                       boundary = eda_boundary(ds))
shipped_ids = Int.(shipped.data[:ordered_match_ids])
reorder = Dict(id => i for (i, id) in enumerate(shipped_ids))
pxg_shipped = Float64[shipped.data[:flat_pxg_supremacy][reorder[id]] for id in ids]

wealth_fs = eda_features(ds, Features.AbstractFeatureConfig[ProductionWealthFeature()];
                         boundary = eda_boundary(ds))
wealth_order = Dict(id => i for (i, id) in enumerate(Int.(wealth_fs.data[:ordered_match_ids])))
wealth = Float64[wealth_fs.data[:flat_delta_production_wealth][wealth_order[id]] for id in ids]

const R60_BASELINE = [wealth, pxg_shipped]
println("  incumbent baseline for every incremental test: production wealth + shipped pxG supremacy")

# %%
# ==============================================================================
# 4. H1 — opponent-adjusted pxG ratings
# ==============================================================================

eda_section("2/6", "H1 · opponent-adjusted pxG")

t0 = time()
adjusted = l60_adjusted_pxg(ds, records; half_life_days = R60_ADJ_HALF_LIFE,
                            lambda = R60_ADJ_LAMBDA)
@printf("  refit once per distinct kickoff, %.1fs\n", time() - t0)
adj_sup = l60_standardise(Float64[get(adjusted, id, (sup = 0.0,)).sup for id in ids])
adj_level = l60_standardise(Float64[get(adjusted, id, (level = 0.0,)).level for id in ids])
@printf("  available on %.1f%% of held-out matches\n",
        100 * mean(Float64[get(adjusted, id, (available = 0.0,)).available for id in ids][mask]))

# %%
# ==============================================================================
# 5. H2, H3 — decomposition
# ==============================================================================

eda_section("3/6", "H2 · volume vs quality   H3 · over-performance")

raw_pxg   = l60_rolling(ds, records, l60_pxg;      k = R60_K, kappa = R60_KAPPA)
volume    = l60_rolling(ds, records, l60_volume;   k = R60_K, kappa = R60_KAPPA)
quality   = l60_rolling(ds, records, l60_quality;  k = R60_K, kappa = R60_KAPPA)
overperf  = l60_rolling(ds, records, l60_overperf; k = R60_K, kappa = R60_KAPPA)

pick(lookup, field) = l60_standardise(
    Float64[getproperty(get(lookup, id, (sup = 0.0, level = 0.0)), field) for id in ids])

pxg_sup     = pick(raw_pxg, :sup)
volume_sup  = pick(volume, :sup)
quality_sup = pick(quality, :sup)
overperf_sup = pick(overperf, :sup)
pxg_level   = pick(raw_pxg, :level)
volume_level = pick(volume, :level)

@printf("  volume / quality correlation on held-out: %+.4f\n",
        eda_pearson(volume_sup[mask], quality_sup[mask]))
@printf("  volume / shipped pxG correlation:         %+.4f\n",
        eda_pearson(volume_sup[mask], pxg_shipped[mask]))
println("  (if volume and the shipped column are nearly identical, the shipped column is")
println("   a shot counter and H2's split has nothing left to find)")

# %%
# ==============================================================================
# 6. H4 — dual-horizon kernels
# ==============================================================================

eda_section("4/6", "H4 · fast form against slow baseline")

fast_lookup, slow_lookup = l60_dual_horizon(ds, records, l60_pxg;
                                            fast = R60_FAST, slow = R60_SLOW,
                                            kappa = R60_KAPPA)
fast_sup = pick(fast_lookup, :sup)
slow_sup = pick(slow_lookup, :sup)
@printf("  fast (half-life %.0f) vs slow (half-life %.0f) correlation: %+.4f\n",
        R60_FAST, R60_SLOW, eda_pearson(fast_sup[mask], slow_sup[mask]))
println("  A correlation near 1 would mean the two horizons are the same signal and H4 is")
println("  dead on arrival; well below 1 means there is a genuine decomposition to exploit.")

# %%
# ==============================================================================
# 7. Standalone and incremental evaluation
# ==============================================================================

eda_section("5/6", "Association and incremental value")

# A LEVEL candidate moves the TOTAL, not the result, so it is scored against total
# goals. Scoring it against supremacy asks it to do a job it was not built for and
# guarantees it looks harmful.
candidates = [
    ("adj_pxg_sup",   adj_sup,      :supremacy, "H1 opponent-adjusted pxG supremacy"),
    ("adj_pxg_level", adj_level,    :total,     "H1 opponent-adjusted pxG level"),
    ("pxg_sup",       pxg_sup,      :supremacy, "raw rolling pxG supremacy (reference build)"),
    ("volume_sup",    volume_sup,   :supremacy, "H2 shot-volume supremacy"),
    ("quality_sup",   quality_sup,  :supremacy, "H2 xG-per-shot supremacy"),
    ("overperf_sup",  overperf_sup, :supremacy, "H3 goals minus pxG, expect NEGATIVE"),
    ("fast_sup",      fast_sup,     :supremacy, "H4 fast kernel"),
    ("slow_sup",      slow_sup,     :supremacy, "H4 slow kernel"),
    ("pxg_level",     pxg_level,    :total,     "raw rolling pxG level"),
    ("volume_level",  volume_level, :total,     "shot-volume level"),
]

rows = [l60_evaluate(name, column, frame, train, mask, R60_BASELINE; response = response)
        for (name, column, response, _) in candidates]
@printf("  %-15s | %-9s | %7s | %9s | %9s | %8s | %9s | %8s\n",
        "candidate", "response", "n_live", "r", "rho", "AUC", "dR2", "t")
println("  " * repeat('-', 94))
for (r, spec) in zip(rows, candidates)
    @printf("  %-15s | %-9s | %7d | %9s | %9s | %8s | %9s | %8s\n", r.name, String(spec[3]), r.n_live,
            eda_fmt(r.r, "%+.4f"), eda_fmt(r.rho, "%+.4f"), eda_fmt(r.auc, "%.4f"),
            eda_fmt(r.delta_r2, "%+.5f"), eda_fmt(r.t, "%+.2f"))
end
println("\n  dR2 is over production wealth + the shipped pxG column, so it answers")
println("  'what does this add that the builder does not already have'.")

println("\n  Bootstrap of the incremental R-squared, 90% interval:")
@printf("  %-15s | %11s | %22s | %s\n", "candidate", "dR2", "90% CI", "verdict")
println("  " * repeat('-', 78))
for (name, column, response, _) in candidates
    m, lo, hi = l60_bootstrap_delta_r2(column, frame, train, mask, R60_BASELINE; response = response)
    verdict = !isfinite(lo) ? "—" : lo > 0 ? "ADDS SIGNAL" :
              hi < 0 ? "HURTS" : "indistinguishable from zero"
    @printf("  %-15s | %+11.5f | [%+.5f, %+.5f] | %s\n", name, m, lo, hi, verdict)
end

# %%
# ==============================================================================
# 8. Hypothesis verdicts
# ==============================================================================

eda_section("6/6", "Verdicts")

byname = Dict(r.name => r for r in rows)
let
    @printf("  H1 opponent adjustment : adj r = %+.4f vs raw pxG r = %+.4f  -> %s\n",
            byname["adj_pxg_sup"].r, byname["pxg_sup"].r,
            abs(byname["adj_pxg_sup"].r) > abs(byname["pxg_sup"].r) ? "SUPPORTED" : "not supported")
    @printf("  H2 volume vs quality   : volume r = %+.4f, quality r = %+.4f  -> %s\n",
            byname["volume_sup"].r, byname["quality_sup"].r,
            abs(byname["volume_sup"].r) > abs(byname["quality_sup"].r) ?
            "volume dominates, as predicted" : "quality dominates, against prediction")
    @printf("  H3 over-performance    : r = %+.4f  -> %s\n", byname["overperf_sup"].r,
            byname["overperf_sup"].r < 0 ? "NEGATIVE as predicted (mean reversion)" :
            "positive — reversion NOT observed")
    @printf("  H4 dual horizon        : fast r = %+.4f, slow r = %+.4f, corr %+.4f\n",
            byname["fast_sup"].r, byname["slow_sup"].r,
            eda_pearson(fast_sup[mask], slow_sup[mask]))
    joint = eda_ols(hcat(slow_sup[mask], fast_sup[mask]), Float64.(frame[mask, :supremacy]))
    slow_only = eda_ols(reshape(slow_sup[mask], count(mask), 1), Float64.(frame[mask, :supremacy]))
    @printf("                           fast adds dR2 = %+.5f over slow alone (t = %+.2f) -> %s\n",
            joint.r2 - slow_only.r2, joint.t[3],
            abs(joint.t[3]) >= 2 ? "SUPPORTED" : "not supported")
end
eda_rule(100, '=')
