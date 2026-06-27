#=
r01_roleneutral_eda.jl — Option-1 EDA: does out-of-position deployment hurt ROLE-NEUTRAL output?

Re-runs the out-of-position test (l00 Gate 3 was null on the role-conditioned SofaScore rating) on every
objective per-player output column the league actually has (xG, xA, bigChanceCreated, touchesInOppBox,
shots, defensive actions…), each ROLE-STANDARDISED first (z within position) so we measure role-fit, not
the mechanical baseline. Per (league × target): within-player FE off_modal coef in role-sd units.

READ: a target with off_modal coef < 0 and |t| ≥ 2, consistent across leagues, = a genuine
out-of-position penalty the single rating cannot see → worth turning into a feature (Option-1 build) and
testing vs market (per-line LogLoss + GLMEdge). Null everywhere = the rating already captured everything;
move to Option 2 (prior-informed RAPM).

Server REPL (kaimon): git pull, restart REPL, then
    include("current_development/position_aware_ratings/r01_roleneutral_eda.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Printf

include("current_development/position_aware_ratings/l00_position_helpers.jl")
include("current_development/position_aware_ratings/l01_roleneutral_helpers.jl")

const Data = BayesianFootball.Data

const SEGMENTS = [
    Data.ScottishLower(), Data.Ireland(), Data.IrelandFirstDivision(),
    Data.SouthKorea(), Data.Norway(), Data.Veikkausliiga(),
]
const MIN_APPS = 5
seg_name(s) = string(nameof(typeof(s)))

# (league × target) results
results = DataFrame(league=String[], target=String[], coverage=Float64[],
                    n=Int[], coef=Float64[], t=Float64[], p=Float64[])

for seg in SEGMENTS
    name = seg_name(seg)
    println("\n", "#"^84, "\n# LEAGUE: $name\n", "#"^84)
    ds = try
        Data.load_datastore_cached(seg)
    catch e
        println("[SKIP] load failed: $(sprint(showerror, e))"); continue
    end

  try
    df = prepare_starter_lineups(ds; from_year=2023)
    if nrow(df) == 0; println("[SKIP] no 2023+ starters."); continue; end
    add_modal_position!(df; min_apps=MIN_APPS)
    df = add_opponent_strength!(df)

    targets = candidate_target_columns(ds; min_coverage=0.3)
    if isempty(targets); println("[SKIP] no role-neutral target columns ≥30% coverage."); continue; end
    println("  discovered targets (coverage%): ",
            join(["$(t)=$(target_coverage(ds,t))" for t in targets], ", "))

    df = attach_targets(df, ds, targets)
    add_role_zscores!(df, targets)

    println("\n  role-fit regression  z_<target> ~ off_modal + is_home + minutes + opp_str  (within-player FE)")
    println("  off_modal coef in role-sd units; <0 & |t|≥2 = underperforms the role when out of position")
    tbl = DataFrame(target=String[], n=Int[], coef=Float64[], t=Float64[], p=Float64[], flag=String[])
    for t in targets
        zcol = Symbol("z_", t)
        res = role_fit_regression(df, zcol)
        res === nothing && continue
        flag = (res.coef < 0 && abs(res.t) >= 2) ? "** penalty **" :
               (res.coef > 0 && abs(res.t) >= 2) ? "(+, unexpected)" : ""
        push!(tbl, (string(t), res.n, res.coef, res.t, res.p, flag))
        push!(results, (name, string(t), target_coverage(ds, t), res.n, res.coef, res.t, res.p))
    end
    isempty(tbl) ? println("  (no target had ≥50 usable rows)") :
                   (sort!(tbl, :t); show(tbl; allrows=true, allcols=true, truncate=0); println())
  catch e
    println("[ERROR] $name failed: $(sprint(showerror, e))")
  end
end

# ============================================================================
# CROSS-LEAGUE SUMMARY — per target, how consistently does off-position hurt?
# ============================================================================
println("\n", "="^84, "\nCROSS-LEAGUE SUMMARY (per target)\n", "="^84)
if nrow(results) == 0
    println("no results.")
else
    g = combine(groupby(results, :target),
                :coef => (x -> round(mean(x), digits=4)) => :mean_coef,
                :t    => (x -> round(mean(x), digits=2)) => :mean_t,
                nrow => :n_leagues,
                :t => (x -> count(<=(-2), x)) => :n_leagues_penalty,
                :t => (x -> count(>=(2),  x)) => :n_leagues_pos)
    sort!(g, :mean_t)
    show(g; allrows=true, allcols=true, truncate=0); println()
end

println("""

$("="^84)
READ:
 • A target with mean_coef<0, mean_t≤−2, and n_leagues_penalty on most leagues = a REAL out-of-position
   output penalty the single rating misses → build it into a position-aware feature (Option-1 model) and
   judge vs market (per-line LogLoss + GLMEdge), focusing on totals/derivative lines.
 • Mixed signs / |t|<2 everywhere = the role-conditioned rating already captured it; no feature edge here
   → pivot to Option 2 (prior-informed Bayesian RAPM on team xG-differential).
$("="^84)
""")
