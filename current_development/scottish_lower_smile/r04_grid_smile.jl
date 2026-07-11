#=
r04 — GRID B: supremacy_weight × smile_weight on the team smile engine (ScottishLower).

⚠ EDIT FIRST: set BEST_HL / BEST_HS below to the Grid-A winner from r03 (recorded in NOTES.md).

Cells (canonical NOTES.md naming; all at the Grid-A winning decay/history):
  smile_pois_sup{40,70,100}_sw{0,40,50}   — 9 cells on TeamSmileDPGoalsModel
      (the sw=0 column doubles as the supremacy-only rung, li_sup_only analogue)
  none_pois_hl*_hs*                       — structural control (RE-USED from Grid A if the
      winner cell is already saved there; re-run here only if hl*/hs* was NOT a Grid-A cell)
  iso_pois_mw100_hl*_hs*                  — isotropic-pillar control on the SAME Poisson base
      (this is the "old pillar vs smile pillar" A/B at team level)

Settings identical to r02: targets 23/24→25/26, GroupedCVConfig [56,57], match_biweek,
800/300 × 4 chains, depth 10. 11 cells ≈ 4–7 h → overnight.

Convergence gate per cell → r04_convergence.txt (≥95% folds R-hat ≤ 1.01 to be read in r05).

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r04_grid_smile.jl"))
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
# 0. GRID-A WINNER — EDIT AFTER r03 ⚠
# ==========================================
const BEST_HL = 180.0    # ⚠ days_half_life winner from r03
const BEST_HS = 2        # ⚠ history_seasons winner from r03
const RERUN_CONTROLS = false   # true if (BEST_HL, BEST_HS) was NOT a Grid-A cell (none_pois
                               # control then needs training here); iso_pois control ALWAYS runs
                               # (Grid A only had the nb iso reference).
# ⚠ RUNTIME BUDGET (r01 finding: smile trees run deep — median 127 leapfrogs/iter at depth 10,
# ≈20× the DP base; see r01b probe). Set MAX_DEPTH to the r01b winner (5 or 6). Grid-B cell
# wall ≈ 6 × r01b probe wall (192 tasks = 6 waves of 32 threads); if still too heavy, trim
# CHAINS to 3 and/or SAMPLES to 600, or drop TARGETS to the last 2 seasons (32 folds).
const MAX_DEPTH = 10     # ⚠ set to r01b probe winner before launching

# ==========================================
# 1. DATA + GRID SPEC
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_smile_grid/")
mkpath(save_dir)

TARGETS = ["23/24", "24/25", "25/26"]
DYN_COL = :match_biweek
SAMPLES = 800
WARMUP  = 300
CHAINS  = 4

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = BEST_HL)
_tag = "hl$(Int(BEST_HL))_hs$(BEST_HS)"

SUPS = [0.4, 0.7, 1.0]
SWS  = [0.0, 0.4, 0.5]

specs = Tuple{String, Any}[]
for sup in SUPS, sw in SWS
    push!(specs, ("smile_pois_sup$(Int(100sup))_sw$(Int(100sw))_$(_tag)",
                  TeamSmileDPGoalsModel(dynamics_config = dyn_cfg,
                                        supremacy_weight = sup, smile_weight = sw)))
end
push!(specs, ("iso_pois_mw100_$(_tag)",
              TeamIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 1.0)))
if RERUN_CONTROLS
    push!(specs, ("none_pois_$(_tag)_ctl", TeamDPGoalsModel(dynamics_config = dyn_cfg)))
end

println("[INFO] Grid B: $(length(specs)) cells (history_seasons=$BEST_HS) -> ",
        join(first.(specs), ", "))

# ==========================================
# 2. RUN (identical loop to r02)
# ==========================================
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
for (name, model) in specs
    println("\n", "#"^72, "\n# CELL: $name  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = BEST_HS,
            warmup_period   = 0,   # match r02: include season-start folds (the operational regime)
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = MAX_DEPTH,
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

open(joinpath(ROOT, "current_development/scottish_lower_smile/r04_convergence.txt"), "w") do io
    println(io, "r04 Grid B convergence gate — ", string(now()), "  (hl=$(BEST_HL), hs=$(BEST_HS))")
    foreach(l -> println(io, l), gate_lines)
end

println("\n[INFO] Grid B complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate written to r04_convergence.txt. Next: r05_eval_smile.jl")
