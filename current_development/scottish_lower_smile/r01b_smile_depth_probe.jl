#=
r01b — SMILE RUNTIME PROBE: can max_depth cap fix the 20× smile slowdown without breaking
convergence?

r01 finding: same 5-split window — none_pois 11m, iso_pois 40m, smile_pois 3h35m (converged,
R-hat ≤ 1.015). Chain internals: median tree_depth 4/5/7 (max 8), median leapfrogs/iter
15/31/127 — the smile pillar's stiff geometry (σ_smile→0.05) makes NUTS take ~8.5× more
gradient evals/iter, plus ~2.3× per-eval (the [n×5] smile matrix) ⇒ the 20×. The @model code
follows docs/turing_ad_performance_guide.md (broadcast-only, masks, views) — this is leapfrog
COUNT, not an AD defect.

Since trees max at depth 8, capping at 8 is a no-op — the binding caps are 6 (≤63 steps, ~2×)
and 5 (≤31 steps, ~4×). Parent-stream lesson: depth caps broke convergence only when σ was
FIXED; here σ_sup/σ_smile are SAMPLED (release valve) — this probe verifies a cap still mixes.

Trains the SAME smile config as r01 (target 25/26, warmup_period 16, 600/600×4) at
max_depth ∈ {6, 5}; reports wall + convergence + posterior drift vs the depth-10 r01 reference
(σ_smile≈0.052, σ_sup≈0.249, δ₅₆−δ₅₇≈0.035; wall 215m).

DECISION RULE (Grid-B cell wall ≈ 6 × probe wall — 192 tasks = 6 waves of 32 threads):
  • depth 6 passes (R-hat ≤ 1.01 + no drift) → probe ≈ ~110m ⇒ ~11h/cell: still heavy — combine
    with trimmed settings (chains=3 and/or samples 600, or 2 target seasons) OR prefer depth 5.
  • depth 5 passes → probe ≈ ~55m ⇒ ~5.5h/cell ⇒ 9 cells ≈ 2 overnights at full settings.
  • both fail → keep depth 10 and shrink Grid B instead (2 target seasons / chains=3 / fewer
    cells) — discuss before running.

Run on the server (kaimon REPL) after git pull + REPL restart:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r01b_smile_depth_probe.jl"))
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

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_smoke/")
mkpath(save_dir)

season_strings = sort(unique(String.(ds.matches.season)))
TARGET = season_strings[end]

model = TeamSmileDPGoalsModel(
    dynamics_config  = PreGame.TimeDecayDynamics(days_half_life = 180.0),
    supremacy_weight = 1.0,
    smile_weight     = 0.5,
)

# depth-10 r01 reference posteriors (for drift check)
const REF = (σ_smile = 0.0516, σ_sup = 0.2493, δ_gap = 0.035)

function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end

rows = NamedTuple[]
for depth in (6, 5)
    name = "smile_pois_depth$(depth)_probe"
    println("\n", "#"^68, "\n# RUN: $name\n", "#"^68)
    t0 = time()
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = [TARGET],
            history_seasons = 2,
            warmup_period   = 16,
            dynamics_col    = :match_biweek,
            samples         = 600,
            warmup          = 600,
            chains          = 4,
            use_queue       = true,
            max_depth       = depth,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        mins = round((time() - t0) / 60, digits=1)

        # convergence on new params, worst over folds
        worst_new = 0.0
        for it in res.training_results.items
            er = DataFrame(MCMCChains.ess_rhat(it[1]))
            rcol = :rhat in propertynames(er) ? :rhat :
                   first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
            for p in vcat(["σ_sup", "σ_smile", "δ_league_raw[1]", "δ_league_raw[2]"],
                          ["log_φ[$k]" for k in 1:5])
                r = er[er.parameters .== Symbol(p), rcol]
                isempty(r) || ismissing(r[1]) || isnan(r[1]) || (worst_new = max(worst_new, r[1]))
            end
        end
        σsm = mean(_pool(res, "σ_smile")); σsp = mean(_pool(res, "σ_sup"))
        gap = mean(_pool(res, "δ_league_raw[1]") .- _pool(res, "δ_league_raw[2]"))
        drift_ok = abs(σsm - REF.σ_smile) < 0.01 && abs(σsp - REF.σ_sup) < 0.05 &&
                   abs(gap - REF.δ_gap) < 0.03
        push!(rows, (; depth, mins, worst_new = round(worst_new, digits=4),
                       σ_smile = round(σsm, digits=4), σ_sup = round(σsp, digits=4),
                       δ_gap = round(gap, digits=4), drift_ok,
                       pass = worst_new <= 1.01 && drift_ok))
        println("[PROBE] depth=$depth  wall=$(mins)m  worst new-param R-hat=$(round(worst_new, digits=4))" *
                "  σ_smile=$(round(σsm, digits=4)) (ref $(REF.σ_smile))  σ_sup=$(round(σsp, digits=4))" *
                " (ref $(REF.σ_sup))  δgap=$(round(gap, digits=4)) (ref $(REF.δ_gap))")
    catch e
        push!(rows, (; depth, mins = round((time() - t0) / 60, digits=1), worst_new = NaN,
                       σ_smile = NaN, σ_sup = NaN, δ_gap = NaN, drift_ok = false, pass = false))
        @error "probe failed at depth $depth" exception=(e, catch_backtrace())
    end
end

println("\n", "="^70, "\nR01B DEPTH PROBE SUMMARY (depth-10 reference: 215m wall, R-hat ≤ 1.007)\n", "="^70)
df = DataFrame(rows)
show(df; allrows=true, allcols=true, truncate=0); println()
best = filter(r -> r.pass, rows)
println(isempty(best) ?
    ">> NO depth passed — keep MAX_DEPTH=10 in r04 and shrink Grid B (discuss options)." :
    ">> USE max_depth=$(minimum(r.depth for r in best)) in r04 (edit MAX_DEPTH_SMILE) — " *
    "estimated Grid-B cell wall ≈ (probe mins) × 48/5 folds × 9 cells; sanity-check before launch.")
