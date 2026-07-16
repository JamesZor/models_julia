#=
r07 — ISO market_weight SWEEP + mw100 hard-gate confirm (Grid-B endgame, option A 2026-07-17).

Why: Grid B verdict (RESULTS §3) — the isotropic level pillar owns totals (the only money
family); smile/supremacy add nothing. But Grid B only tested mw=1.0 at 3 chains (gate marginal:
60% ≤1.01, worst 1.027), and the Ireland lesson says mw 0.25–0.4 is optimal ("raising it
backfires" — totals-compression/denoising finding). This sweep tests that on 56/57 and fixes
the gate.

Cells (all TeamIsoDPGoalsModel, depth 10, hl365/hs2, HARD gate ≥95% folds R-hat ≤ 1.01):
  iso_pois_mw{25,40,70}  — 1200/300 × 3 chains (Grid-B spec, comparable to §3 tables)
  iso_pois_mw100_c4      — 1200/300 × 4 chains (re-run of the Grid-B cell; the gate fix)

Spec: targets 24/25→25/26 (~40 folds), GroupedCVConfig [56,57], match_biweek, warmup_period=0.
Budget @16t: 3 × ~5.6h + ~7.5h ≈ 24h ≈ one overnight.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r07_grid_iso_mw.jl"))
Then: r08_eval_iso.jl (stdout-redirect pattern).
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

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

const BEST_HL = 365.0
const BEST_HS = 2

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_iso_grid/")
mkpath(save_dir)

TARGETS = ["24/25", "25/26"]
DYN_COL = :match_biweek
SAMPLES = 1200
WARMUP  = 300

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = BEST_HL)
_tag = "hl$(Int(BEST_HL))_hs$(BEST_HS)"

# (name, model, chains)
specs = Tuple{String, Any, Int}[]
for mw in (0.25, 0.4, 0.7)
    push!(specs, ("iso_pois_mw$(Int(100mw))_$(_tag)",
                  TeamIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = mw), 3))
end
push!(specs, ("iso_pois_mw100_c4_$(_tag)",
              TeamIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 1.0), 4))

println("[INFO] iso mw sweep: $(length(specs)) cells -> ",
        join([s[1] * " [$(s[3])ch]" for s in specs], ", "))

function _fold_convergence(res)
    n = length(res.training_results.items); n_ok = 0; worst = 0.0
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
for (name, model, chains) in specs
    println("\n", "#"^72, "\n# CELL: $name  chains=$chains  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = BEST_HS,
            warmup_period   = 0,
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = chains,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_ok, worst = _fold_convergence(res)
        pct = n == 0 ? 0.0 : round(100n_ok / n, digits=1)
        gate = "$name [$(chains)ch]: folds=$n converged(R-hat≤1.01)=$n_ok ($(pct)%) worst=$(round(worst, digits=4))" *
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

open(joinpath(ROOT, "current_development/scottish_lower_smile/r07_convergence.txt"), "w") do io
    println(io, "r07 iso mw sweep convergence gate — ", string(now()),
            "  (hl=$(BEST_HL), hs=$(BEST_HS), targets=$(TARGETS), depth 10, HARD gate)")
    foreach(l -> println(io, l), gate_lines)
end

println("\n[INFO] iso mw sweep complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate written to r07_convergence.txt. Next: r08_eval_iso.jl")
