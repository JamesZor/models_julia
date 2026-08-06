#=
r03 — WP5 THE GRID. ~25h. Canonical 56/57 spec, matched to scottish_lower_smile/r07 so every number
is comparable to the rest of the stream.

CELLS
  1 funnel_apm_ctl   src DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel — the INCUMBENT, re-run on
                     these exact folds. Never compare against its previously published numbers.
  2 pxg_apm          Arm A: proxy-xG Gamma + goals + RAPM.            <- the headline
  3 pxg_noapm        Arm A with the RAPM pillar off.                  <- isolates the xG pillar
  4 funnel_pxg_apm   Arm B: shots + conditional-xG + goals + RAPM.    <- replace vs add
  5 pxg_apm_linvar   Arm A with the linear-variance Gamma.            <- ONLY if r01-E4 says linear

⚠ EDIT BEFORE RUNNING: set WARMUP from r02 check 7, and RUN_LINVAR from r01-E4.

THE COVERAGE CONFOUND, AND THE FREE FIX. With history_seasons = 2, target 24/25 pulls history from
22/23 — which has NO commentary. Cells 2-5 therefore see goals only there, while cell 1 still gets
its ds.bbc shot counts (six seasons). That is the operationally honest fight, but it confounds DATA
with STRUCTURE. r04 splits LogLoss by target season at zero extra compute: the 25/26 folds have
fully-covered history for every cell, the 24/25 folds do not. Winning on 25/26 and losing on 24/25
is the coverage story, cleanly separated.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r03_grid.jl"))
Then: r04_eval.jl (stdout-redirect pattern).
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
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))

# --- canonical 56/57 spec (r07-matched) ---
const HL      = 365.0          # Grid-A winner
const HS      = 2              # Grid-A winner
const TARGETS = ["24/25", "25/26"]
const DYN_COL = :match_biweek  # ~40 folds
const SAMPLES = 1200
const WARMUP  = 300            # ⚠ SET FROM r02 CHECK 7
const CHAINS  = 3
const RUN_LINVAR = false       # ⚠ SET FROM r01 E4 (slope < 1.5 => true)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_pxg_grid/"); mkpath(save_dir)

dyn  = PreGame.TimeDecayDynamics(days_half_life = HL)
_tag = "hl$(Int(HL))_hs$(HS)"

specs = Tuple{String, Any}[
    ("funnel_apm_ctl_$(_tag)",
     PreGame.DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel(dynamics_config = dyn)),
    ("pxg_apm_$(_tag)",
     TeamPxGGoalsAPMModel(dynamics_config = dyn)),
    ("pxg_noapm_$(_tag)",
     TeamPxGGoalsAPMModel(dynamics_config = dyn, apm_on = false)),
    ("funnel_pxg_apm_$(_tag)",
     TeamFunnelPxGGoalsAPMModel(dynamics_config = dyn)),
]
RUN_LINVAR && push!(specs,
    ("pxg_apm_linvar_$(_tag)",
     TeamPxGGoalsAPMModel(dynamics_config = dyn, variance_law = :linear)))

println("[INFO] $(length(specs)) cells: ", join(first.(specs), ", "))
println("[INFO] spec: targets=$(TARGETS) hs=$HS $(SAMPLES)/$(WARMUP) x $(CHAINS)ch depth=10 " *
        "warmup_period=0")

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

"""Pool a scalar parameter over every fold — the headline diagnostics live here, not in the gate."""
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
    println("\n", "#"^76,
            "\n# CELL: $name   (elapsed $(round((time() - t_start) / 60, digits = 1)) min)\n", "#"^76)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = HS,
            warmup_period   = 0,            # season-START folds INCLUDED (the operational regime)
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = 10,           # never cap depth on this data
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_ok, worst = _fold_convergence(res)
        pct  = n == 0 ? 0.0 : round(100n_ok / n, digits = 1)
        gate = "$name: folds=$n converged(R-hat≤1.01)=$n_ok ($(pct)%) worst=$(round(worst, digits = 4))" *
               (n == 0 ? "  ⚠ SILENT DROP — no items!" : pct < 95 ? "  ⚠ BELOW GATE" : "  ✅")
        println("[GATE] ", gate); push!(gate_lines, gate)

        # Parameter diagnostics that are findings in their own right, win or lose.
        for (sym, label) in (("log_κ", "kappa (exp)"), ("ν_xg", "nu_xg"), ("θ_xg", "theta_xg"),
                             ("σ_q", "sigma_q"), ("q_raw", "q (logistic)"), ("p2_raw", "p2 (logistic)"),
                             ("w_att", "w_att"), ("w_def", "w_def"))
            v = _pool(res, sym); isempty(v) && continue
            v = sym in ("log_κ",) ? exp.(v) :
                sym in ("q_raw", "p2_raw") ? 1 ./ (1 .+ exp.(-v)) : v
            line = "  $name  $(rpad(label, 12)) mean=$(round(mean(v), digits = 4)) " *
                   "90%=[$(round(quantile(v, 0.05), digits = 4)), $(round(quantile(v, 0.95), digits = 4))]"
            println("[DIAG]", line); push!(diag_lines, line)
        end
    catch e
        msg = "$name: FAILED ($(typeof(e)))"
        println("[GATE] ", msg); push!(gate_lines, msg)
        @error "cell failed: $name" exception = (e, catch_backtrace())
    end
end

open(joinpath(ROOT, "current_development/scottish_proxy_xg/r03_convergence.txt"), "w") do io
    println(io, "r03 proxy-xG grid convergence gate — ", string(now()))
    println(io, "spec: hl=$HL hs=$HS targets=$TARGETS $(SAMPLES)/$(WARMUP) x $(CHAINS)ch " *
                "depth=10 warmup_period=0  HARD gate ≥95% folds R-hat ≤ 1.01")
    foreach(l -> println(io, l), gate_lines)
    println(io, "\n--- parameter diagnostics ---")
    foreach(l -> println(io, l), diag_lines)
end

println("\n[INFO] grid complete in $(round((time() - t_start) / 3600, digits = 2)) h.")
println("[INFO] Gate + diagnostics written to r03_convergence.txt. Next: r04_eval.jl")
println("""
[READ] Before any eval table:
  • Exclude cells below the HARD gate.
  • sigma_q vs its prior mean (~0.12) is a FINDING either way — it answers "is there team-level
    shot quality on 56/57?" independently of whether Arm B wins anything.
  • kappa should sit near 1 in every cell. A drift means the cell table stopped being calibrated.
""")
