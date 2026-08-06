#=
r02 — NIGHT 1: the FAMILY bake-off for Scottish Upper (54/55).

ONE axis: which engine family, at a fixed (hl, hs). Half-life is deliberately NOT swept — the 56/57
stream spent 30h on that axis and found a monotone gradient favouring long memory (hl60 → hl365), so
we inherit hl=365 and carry a single hl=180 control to check the gradient still points that way on a
higher-turnover division.

Cells (see l01_upper.jl::family_specs, run cheapest-first):
  none_pois_hl365          structural baseline, NO market pillar
  none_pois_hl180          half-life gradient control
  rating_pois_hl365        + SofaScore player-rating pillar (on top of team dynamics)
  funnel_pois_hl365        BBC shots → goals thinned Poisson, NO market pillar
  none_nb_hl365            NegBin dispersion reference
  iso_pois_mw40_hl365      isotropic market pillar — the 56/57 PRODUCTION WINNER
  smile_pois_sup100_sw50   Ireland's keeper pillar at team level  ⚠ EXPENSIVE, opt-out

Spec (identical across every cell — comparability is the point):
  GroupedCVConfig [54,55] pooled · targets = last 2 seasons · history_seasons=2 · match_biweek
  warmup_period=0 (season-start folds kept: week-1 prediction off decayed prior seasons IS the
  operational regime, and the season starts in days) · 800 samples / 300 warmup × 4 chains
  · max_depth=10 (NEVER capped) · queued execution.

⚠ EDIT BEFORE RUNNING — set from the r01 smoke output:
    INCLUDE_SMILE   false if r01's projected total wall exceeds the night
    HISTORY_SEASONS 1 if r00/r01 showed the 22/23 block is unusable (apply UNIFORMLY, never per-arm)

A convergence gate is printed per cell and written to r02_convergence.txt.
DO NOT read the r03 eval tables for any cell below 95%.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_upper/r02_grid_family.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using Dates

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_upper/l01_upper.jl"))

# ---- knobs (edit after r01) ----
const INCLUDE_SMILE   = true
const HISTORY_SEASONS = 2
const SAMPLES         = 800
const WARMUP          = 300
const CHAINS          = 4
const DYN_COL         = :match_biweek

println("[INFO] Loading ScottishUpper DataStore...")
ds = Data.load_datastore_cached(Data.ScottishUpper())
save_dir = joinpath(ROOT, "data/scottish_upper_family/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
const TARGETS = season_strings[max(1, end-1):end]
println("[INFO] targets = ", TARGETS, "  history_seasons = ", HISTORY_SEASONS,
        "  dynamics_col = ", DYN_COL)

specs = family_specs(include_smile = INCLUDE_SMILE)
println("[INFO] Grid: $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# CONVERGENCE GATE (per fold, on the raw chains)
# ==========================================
# Share of folds whose raw-chain max R-hat ≤ 1.01. The curated diagnostic table drops engine-level
# params, so this reads the raw chain per fold.
function _fold_convergence(res)
    n_ok = 0; worst = 0.0; n = length(res.training_results.items)
    for it in res.training_results.items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        mr = isempty(vals) ? NaN : maximum(vals)
        isnan(mr) && continue
        worst = max(worst, mr)
        mr <= 1.01 && (n_ok += 1)
    end
    return n, n_ok, worst
end

# ==========================================
# RUN (sequential cells; queued chains inside each)
# ==========================================
gate_lines = String[]
t_start = time()
for (name, model) in specs
    println("\n", "#"^72,
            "\n# CELL: $name  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = HISTORY_SEASONS,
            warmup_period   = 0,
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_ok, worst = _fold_convergence(res)
        pct = n == 0 ? 0.0 : round(100n_ok / n, digits=1)
        gate = "$name: folds=$n converged(R-hat≤1.01)=$n_ok ($(pct)%) worst=$(round(worst, digits=4))" *
               (n == 0 ? "  ⚠ SILENT DROP — no items!" : pct < 95 ? "  ⚠ BELOW GATE" : "  ✅")
        println("[GATE] ", gate)
        push!(gate_lines, gate)
    catch e
        msg = "$name: FAILED ($(typeof(e)))"
        println("[GATE] ", msg)
        push!(gate_lines, msg)
        @error "cell failed: $name" exception=(e, catch_backtrace())
    end
end

open(joinpath(ROOT, "current_development/scottish_upper/r02_convergence.txt"), "w") do io
    println(io, "r02 family grid convergence gate — ", string(now()))
    println(io, "targets=", TARGETS, " hs=", HISTORY_SEASONS, " dyn=", DYN_COL,
               " samples=", SAMPLES, "/", WARMUP, " x", CHAINS, " chains, depth=10")
    foreach(l -> println(io, l), gate_lines)
end

println("\n[INFO] Family grid complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate written to r02_convergence.txt. Next: r03_eval_family.jl")
