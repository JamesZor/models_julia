#=
r07b — funnel+iso PILOT (reconditioned). The first r07 launch stalled: funnel+iso has a
SPURIOUS collapsed basin (p₂→~0.03, σ_market→~1.6) that `UniformInit(-2,2)` chains fall into.
It is NOT the posterior mode — MAP (pure optimization) cleanly finds the TRUE mode
(p₂=0.147, σ_market=0.22, lp=11107); every gradient at the collapsed point points away from it,
so it's a sampling trap, not a density mode. Chains trapped there also go glacial (ε→2e-4).

FIX (validated on a fold): initialise each chain at the MAP (`init_type=:map`). NUTS launched
from the true mode STAYS there (p₂=0.147, σ_market=0.23, ε=0.18, n_div=0, ~1 min/fold). The
true mode is well-conditioned; the tiny-ε "stiffness" was only ever a symptom of the bad basin.

`create_experiment_task` hardcodes `UniformInit`, so this builds the task MANUALLY with a
`QueuedNUTSConfig(init_type=:map)`. Reconditioned sampler (max_depth 6, accept 0.6) for a speed
margin — not strictly needed once MapInit keeps us in the good basin, but cheap insurance.

PILOT SCOPE (user pick): ONE cell, funnel_iso_mw40, on a SINGLE season (~20 folds) — confirms
the fix works end-to-end at scale and gives a first read. Reuses the stored iso_pois_mw40 as the
none+iso reference (NOTE: stored iso is 2-season/40-fold, pilot is 1-season/~20-fold, so the
funnel_iso−iso delta is approximate — a clean matched comparison waits for the full grid).

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r07b_funnel_iso_pilot.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using StatsFuns: logit

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l06_funnel_iso.jl"))

_r(x, d=4) = round(x, digits=d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
@assert ds.segment isa Data.ScottishLower "wrong segment — reload before running!"
save_dir = joinpath(ROOT, "data/funnel_iso_pilot/")
mkpath(save_dir)

# ---- manual task builder with MAP initialisation (the fix create_experiment_task can't express) ----
function make_mapinit_task(ds, model, name, save_dir;
                           targets, hs, samples, warmup, chains, max_depth, accept_rate, map_iters)
    sampler_conf = Samplers.QueuedNUTSConfig(
        n_samples = samples, n_chains = chains, n_warmup = warmup,
        accept_rate = accept_rate, max_depth = max_depth,
        init_type = :map, map_iters = map_iters, show_progress = false)
    train_cfg = Training.Independent(parallel = true, max_concurrent_tasks = Threads.nthreads())
    training_config = Training.TrainingConfig(sampler = sampler_conf, strategy = train_cfg)
    cv = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = targets, history_seasons = hs,
        dynamics_col = :match_biweek, warmup_period = 0)
    config = Experiments.ExperimentConfig(
        name = name, model = model, splitter = cv,
        training_config = training_config, save_dir = save_dir)
    return Experiments.ExperimentTask(ds, config)
end

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 365.0)
PILOT_TARGET = ["25/26"]     # single season ⇒ ~20 folds

# ==========================================
# TRAIN the one pilot cell
# ==========================================
println("\n", "#"^72, "\n# PILOT CELL: funnel_iso_mw40 (MapInit, depth 6, accept 0.6)\n", "#"^72)
t0 = time()
task = make_mapinit_task(ds, TeamFunnelIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 0.40),
                         "funnel_iso_mw40_pilot", save_dir;
                         targets = PILOT_TARGET, hs = 2,
                         samples = 600, warmup = 1000, chains = 4,
                         max_depth = 6, accept_rate = 0.6, map_iters = 200)
res = Experiments.run_experiment(task)
Experiments.save_experiment(res)
wall = _r((time() - t0)/60, 1)

# convergence + the crucial mode check (did any fold land in the collapsed basin?)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]; Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    out
end
n = length(res.training_results.items); n_ok = 0; worst = 0.0
for it in res.training_results.items
    er = DataFrame(MCMCChains.ess_rhat(it[1]))
    rcol = :rhat in propertynames(er) ? :rhat :
           first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
    vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
    mr = isempty(vals) ? NaN : maximum(vals)
    isnan(mr) && continue
    global worst = max(worst, mr); mr <= 1.01 && (global n_ok += 1)
end
p2v = _pool(res, "p2_raw"); smv = _pool(res, "σ_market")
p2 = isempty(p2v) ? NaN : _r(mean(1 ./ (1 .+ exp.(-p2v))))
σm = isempty(smv) ? NaN : _r(mean(smv))
gate = "funnel_iso_mw40_pilot  folds=$n  R-hat≤1.01: $n_ok/$n ($(_r(100n_ok/max(n,1),1))%)  worst=$(_r(worst))\n" *
       "  MODE CHECK: p₂=$p2 (want ~0.147, collapse~0.03)   σ_market=$σm (want ~0.22, collapse~1.6)\n" *
       "  wall=$wall min"
println("\n", gate)
open(joinpath(@__DIR__, "r07b_convergence.txt"), "w") do io; println(io, gate); end

# ==========================================
# LOAD stored none+iso reference + EVAL
# ==========================================
all_results = Experiments.ExperimentResults[res]
try
    folders = Experiments.list_experiments("scottish_iso_grid"; data_dir = joinpath(ROOT, "data"))
    want = filter(f -> occursin("iso_pois_mw40", f), folders)
    isempty(want) || append!(all_results, Experiments.load_experiments(want))
catch e
    @warn "could not load stored iso reference" exception=e
end
println("[INFO] models in table: ", join([r.config.name for r in all_results], ", "))

selections = [:home,:draw,:away,:btts_yes,:btts_no,
              :over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
              :over_35,:under_35,:over_45,:under_45]
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    E = Evaluation.evaluate_experiments(metric, all_results, ds)
    present = sort(unique(E.model))
    _c(m, c) = (c in names(E) ?
        (r = E[E.model .== m, c]; (isempty(r) || ismissing(r[1])) ? NaN :
         round(Float64(r[1]), digits=4)) : NaN)
    fam = [(:x12, [:home,:draw,:away]), (:btts, [:btts_yes,:btts_no]),
           (:totals, [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
                      :over_35,:under_35,:over_45,:under_45])]
    fm = DataFrame(model = present)
    for (f, sels) in fam
        fm[!, f] = [round(mean(filter(!isnan, [_c(m, "logloss_$(s)_overall_diff_ll") for s in sels])), digits=4) for m in present]
    end
    println("\n", "="^72, "\n📊 Family-pooled LogLoss diff vs Bet365 close (negative = beats it)\n",
            "NOTE: pilot funnel is 1-season (~20 folds); stored iso is 2-season — delta approximate\n", "="^72)
    show(fm; allrows=true, allcols=true, truncate=0); println()
    open(joinpath(@__DIR__, "r07b_results.txt"), "w") do io
        println(io, "r07b funnel+iso PILOT — family-pooled LogLoss diff vs Bet365 close")
        println(io, gate); println(io)
        show(io, fm; allrows=true, allcols=true, truncate=0); println(io)
        println(io, "\nPer line:")
        ll = DataFrame(model = present)
        for s in selections; ll[!, s] = [_c(m, "logloss_$(s)_overall_diff_ll") for m in present]; end
        show(io, ll; allrows=true, allcols=true, truncate=0); println(io)
    end
catch e
    @error "eval failed" exception=(e, catch_backtrace())
end

println("\n", "="^72, "\nR07b PILOT DONE\n", "="^72)
println("READ: (1) MODE CHECK line — p₂≈0.147 on ALL folds = MapInit fixed the collapse.\n",
        "      (2) family table — does funnel+iso look sane vs stored iso? If yes → full 4-cell grid.")
