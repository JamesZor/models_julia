# current_development/scottish_wealth/r03_grid_wealth.jl
#
# RUNNER: Multi-Season Grouped Cross-Validation Grid for Scottish Lower Wealth Models
#
# Compares Wealth-Augmented Models against Baseline Controls across 40 folds:
# 1. funnel_apm_ctl:       Goals Funnel Baseline (No Wealth)
# 2. funnel_apm_wealth:    Goals Funnel Baseline + Starting-XI Wealth Delta
# 3. pxg_apm:              Arm A Proxy xG + RAPM (No Wealth)
# 4. pxg_apm_wealth:       Arm A Proxy xG + RAPM + Starting-XI Wealth Delta
# 5. funnel_pxg_apm:       Arm B Champion 3-Layer (No Wealth)
# 6. funnel_pxg_apm_wealth:Arm B Champion 3-Layer + Starting-XI Wealth Delta

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using Dates
using Printf

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))
include("l02_wealth_engines.jl")

const HL      = 365.0
const HS      = 2
const TARGETS = ["24/25", "25/26"]
const DYN_COL = :match_biweek
const SAMPLES = 1000
const WARMUP  = 300
const CHAINS  = 3

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_wealth_grid/"); mkpath(save_dir)

dyn  = PreGame.TimeDecayDynamics(days_half_life = HL)
_tag = "hl$(Int(HL))_hs$(HS)"

specs = Tuple{String, Any}[
    # Wealth Models to Train (Controls are already saved in data/scottish_pxg_grid/)
    ("pxg_apm_wealth_$(_tag)",
     TeamPxGGoalsAPMWealthModel(dynamics_config = dyn)),

    ("funnel_pxg_apm_wealth_$(_tag)",
     TeamFunnelPxGGoalsAPMWealthModel(dynamics_config = dyn)),
]

println("[INFO] $(length(specs)) cells configured:")
for (name, _) in specs
    println("  - $name")
end

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

function _pool(res, sym)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(sym) in keys(ch) && append!(out, vec(Array(ch[Symbol(sym)])))
    end
    return out
end

gate_lines = String[]
diag_lines = String[]
t_start = time()

for (name, model) in specs
    println("\n", "#"^76)
    println("# EXECUTING CELL: $name (Elapsed: $(round((time() - t_start) / 60, digits = 1)) min)")
    println("#"^76)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = HS,
            warmup_period   = 0,
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = 9,
            max_concurrent_tasks = 12,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_ok, worst = _fold_convergence(res)
        pct  = n == 0 ? 0.0 : round(100n_ok / n, digits = 1)
        gate = "$name: folds=$n converged(R-hat≤1.01)=$n_ok ($(pct)%) worst=$(round(worst, digits = 4))" *
               (n == 0 ? "  ⚠ SILENT DROP — no items!" : pct < 95 ? "  ⚠ BELOW GATE" : "  ✅")
        println("[GATE] ", gate); push!(gate_lines, gate)

        for (sym, label) in (("log_κ", "kappa (exp)"), ("w_wealth", "w_wealth"), ("w_att", "w_att"), ("w_def", "w_def"))
            v = _pool(res, sym); isempty(v) && continue
            v = sym in ("log_κ",) ? exp.(v) : v
            line = "  $name  $(rpad(label, 12)) mean=$(round(mean(v), digits = 4)) " *
                   "90%=[$(round(quantile(v, 0.05), digits = 4)), $(round(quantile(v, 0.95), digits = 4))]"
            println("[DIAG]", line); push!(diag_lines, line)
        end
    catch e
        msg = "$name: FAILED ($(typeof(e)))"
        println("[GATE] ", msg); push!(gate_lines, msg)
        @error "Cell failed: $name" exception = (e, catch_backtrace())
    end
end

open(joinpath(ROOT, "current_development/scottish_wealth/r03_convergence.txt"), "w") do io
    println(io, "r03 Scottish Lower Wealth Grid Convergence Gate — ", string(now()))
    println(io, "spec: hl=$HL hs=$HS targets=$TARGETS $(SAMPLES)/$(WARMUP) x $(CHAINS)ch depth=9 warmup_period=0")
    foreach(l -> println(io, l), gate_lines)
    println(io, "\n--- parameter diagnostics ---")
    foreach(l -> println(io, l), diag_lines)
end

println("\n[INFO] Wealth grid complete in $(round((time() - t_start) / 3600, digits = 2)) h.")
println("[INFO] Gate + diagnostics written to r03_convergence.txt.")
