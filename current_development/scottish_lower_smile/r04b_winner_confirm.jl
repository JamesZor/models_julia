#=
r04b — WINNER CONFIRMATION at depth 10, full spec (the hard-gate run before graduation).

⚠ EDIT FIRST: set SUP_W / SMILE_W to the per-family winner cell from r05 (NOTES.md).

Why this exists: Grid B (r04) ranks the smile cells at max_depth=6 — r01b proved depth-6
posteriors are unbiased (means match depth-10 to 4 decimals) but sluggish (R-hat up to ~1.08),
so the ranking gate there is ≤1.05. NOTHING graduates on a relaxed gate. This runner re-trains
ONLY the winner at the Grid-A reference spec:
  depth 10, 3 target seasons (23/24→25/26, ~60 folds), 800/300 × 4 chains — hard gate
  ≥95% folds all-param R-hat ≤ 1.01, same as r02.
Budget: ≈25h at -t 16 (one long night; r01b task ≈107m × 240 tasks / 16 threads).

Afterwards: re-run r05 with INCLUDE_CONFIRM=true — the _confirm row must reproduce the
winner's per-family LogLoss story (its pooled numbers also cover 23/24, so compare SIGNS and
per-line pattern, not exact values, vs the 2-season Grid-B row).

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r04b_winner_confirm.jl"))
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

# ==========================================
# 0. WINNER — EDIT AFTER r05 ⚠
# ==========================================
const SUP_W   = 1.0      # ⚠ per-family winner from r05
const SMILE_W = 0.5      # ⚠ per-family winner from r05
const BEST_HL = 365.0    # Grid-A winner (fixed)
const BEST_HS = 2        # Grid-A winner (fixed)

# ==========================================
# 1. RUN (Grid-A reference spec, hard gate)
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_smile_confirm/")
mkpath(save_dir)

_tag = "hl$(Int(BEST_HL))_hs$(BEST_HS)"
name = "smile_pois_sup$(Int(100SUP_W))_sw$(Int(100SMILE_W))_$(_tag)_confirm"
model = TeamSmileDPGoalsModel(
    dynamics_config  = PreGame.TimeDecayDynamics(days_half_life = BEST_HL),
    supremacy_weight = SUP_W,
    smile_weight     = SMILE_W,
)

println("[INFO] Confirmation cell: $name  (depth 10, 3 seasons, 800/300×4 — expect ~25h @16t)")
t0 = time()
task = Experiments.create_experiment_task(
    ds, model, name, save_dir;
    target_seasons  = ["23/24", "24/25", "25/26"],
    history_seasons = BEST_HS,
    warmup_period   = 0,
    dynamics_col    = :match_biweek,
    samples         = 800,
    warmup          = 300,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,
)
res = Experiments.run_experiment(task)
Experiments.save_experiment(res)

# ==========================================
# 2. HARD GATE + posterior read
# ==========================================
n = length(res.training_results.items); n_ok = 0; worst = 0.0
for it in res.training_results.items
    er = DataFrame(MCMCChains.ess_rhat(it[1]))
    rcol = :rhat in propertynames(er) ? :rhat :
           first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
    vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
    mr = isempty(vals) ? NaN : maximum(vals)
    isnan(mr) && continue
    global worst = max(worst, mr)
    mr <= 1.01 && (global n_ok += 1)
end
pct = n == 0 ? 0.0 : round(100n_ok / n, digits=1)

function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
σsm = mean(_pool(res, "σ_smile")); σsp = mean(_pool(res, "σ_sup"))
gap = mean(_pool(res, "δ_league_raw[1]") .- _pool(res, "δ_league_raw[2]"))

gate = "$name: folds=$n converged(R-hat≤1.01)=$n_ok ($(pct)%) worst=$(round(worst, digits=4))" *
       (n == 0 ? "  ⚠ SILENT DROP" : pct >= 95 ? "  ✅ HARD GATE PASSED" : "  ❌ BELOW HARD GATE — do NOT graduate")
println("[GATE] ", gate)
println("[POSTERIOR] σ_smile=$(round(σsm, digits=4))  σ_sup=$(round(σsp, digits=4))  δ_gap=$(round(gap, digits=4))",
        "  (r01 depth-10 smoke ref: 0.0516 / 0.2493 / 0.035)")

open(joinpath(ROOT, "current_development/scottish_lower_smile/r04b_confirm.txt"), "w") do io
    println(io, "r04b winner confirmation — ", string(now()))
    println(io, gate)
    println(io, "σ_smile=$(σsm)  σ_sup=$(σsp)  δ_gap=$(gap)")
    println(io, "wall = $(round((time()-t0)/3600, digits=2)) h")
end

println("\n[INFO] Done in $(round((time()-t0)/3600, digits=2)) h → r04b_confirm.txt. ",
        "Next: re-run r05 with INCLUDE_CONFIRM=true, then Stage-4 graduation if ✅.")
