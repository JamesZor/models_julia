#=
r02 — GRID A: time decay × history depth on the fast DP reference (ScottishLower).

Cells (canonical NOTES.md naming):
  none_pois_hl{60,120,180,365}_hs{1,2,3}   — 12 cells on TeamDPGoalsModel
  none_nb_hl180_hs2                        — src DynamicGoalsTimeDecayModel  (nb reference)
  iso_nb_mw100_hl180_hs2                   — src DynamicMarketGoalsTimeDecayModel (nb+iso reference;
                                             uses the l01 required_features phantom fix)

Question this grid answers: how much history should the static-rating time-decay engine see
(hs ∈ 1..3 seasons) and how fast should it forget (half-life 60→365 days)? Decided on the CHEAP
structural model; the winning (hl*, hs*) is then fixed for the Stage-B smile grid (r04).

Settings: targets 23/24→25/26 (walk-forward, GroupedCVConfig [56,57] pooled), match_biweek
(48 folds, r00), 800 samples / 300 warmup × 4 chains, depth 10, queued execution.
Expected ≈ 20–40 min/cell on the 16c/32t box → ~5–8 h total. Run overnight.

A convergence gate is printed per cell (share of folds with max R-hat ≤ 1.01) and written to
r02_convergence.txt — DO NOT read the r03 eval tables for any cell below 95%.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r02_grid_decay_history.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

# ==========================================
# 1. DATA + GRID SPEC
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_decay_grid/")
mkpath(save_dir)

TARGETS     = ["23/24", "24/25", "25/26"]
DYN_COL     = :match_biweek
SAMPLES     = 800
WARMUP      = 300
CHAINS      = 4

HLS = [60.0, 120.0, 180.0, 365.0]
HSS = [1, 2, 3]

# (name, model, history_seasons)
specs = Tuple{String, Any, Int}[]
for hl in HLS, hs in HSS
    push!(specs, ("none_pois_hl$(Int(hl))_hs$(hs)",
                  TeamDPGoalsModel(dynamics_config = PreGame.TimeDecayDynamics(days_half_life = hl)),
                  hs))
end
# nb reference row (src engines, defaults hl=180 / hs=2)
push!(specs, ("none_nb_hl180_hs2",
    PreGame.DynamicGoalsTimeDecayModel(
        interception_config  = PreGame.HierarchicalMonthlyInterception(),
        dynamics_config      = PreGame.TimeDecayDynamics(days_half_life = 180.0),
        dispersion_config    = PreGame.HomeAwayDispersion(),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage()), 2))
push!(specs, ("iso_nb_mw100_hl180_hs2",
    PreGame.DynamicMarketGoalsTimeDecayModel(
        interception_config  = PreGame.HierarchicalMonthlyInterception(),
        dynamics_config      = PreGame.TimeDecayDynamics(days_half_life = 180.0),
        dispersion_config    = PreGame.HomeAwayDispersion(),
        homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
        market_weight        = 1.0), 2))

println("[INFO] Grid A: $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 2. RUN (sequential cells; queued chains inside each)
# ==========================================
# Per-fold convergence: share of folds whose raw-chain max R-hat ≤ 1.01.
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

gate_lines = String[]
t_start = time()
for (name, model, hs) in specs
    println("\n", "#"^72, "\n# CELL: $name  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = hs,
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

open(joinpath(ROOT, "current_development/scottish_lower_smile/r02_convergence.txt"), "w") do io
    println(io, "r02 Grid A convergence gate — ", string(now()))
    foreach(l -> println(io, l), gate_lines)
end

println("\n[INFO] Grid A complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate written to r02_convergence.txt. Next: r03_eval_decay.jl")
