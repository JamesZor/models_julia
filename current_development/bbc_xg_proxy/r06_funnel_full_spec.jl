#=
r06 — FULL-SPEC funnel run on ScottishLower, matching the scottish_lower_smile grid protocol.

WHY: everything in r03–r05 is a 5-fold / 66-match smoke. The 1X2 gain (funnel −0.0071 vs
none_pois) is the only result big enough to be worth trusting provisionally, and it needs a
real sample before it means anything. This runs the funnel at the stream's production spec.

SPLITTER SPEC — identical to scottish_lower_smile/r02 Grid A (so folds match the stored cells
exactly and the eval tables are directly comparable):

    target_seasons  = ["23/24", "24/25", "25/26"]     (walk-forward, [56,57] pooled)
    history_seasons = 2                                (Grid A winner: hl365_hs2)
    warmup_period   = 0                                → ~60 biweek folds/cell
    dynamics_col    = :match_biweek

SAMPLER — deliberately NOT Grid A's 800/300. The funnel needs ≥1000 warmup: its conversion
posterior is ~15× tighter than the prior (sd ≈ 0.03 on the logit), and at 200–300 warmup the
chain is still in the burn-in transient (measured: p₂ = 0.249 against a true 0.330 with a +339
gradient still pushing toward the mode). Folds come from the splitter, so this changes nothing
about comparability.

    samples = 600, warmup = 1000, chains = 4, max_depth = 8

COST: the 5-fold smoke was 27.9 min; ~60 folds ⇒ ≈ 5–6 h per cell, 3 cells ≈ 15–18 h overnight.

CELLS (see the specs block): funnel_pois, then cw=0 with SoT, then cw=0 without SoT. The last
two exist because the 5-fold smoke (r05, 66 OOS matches) CANNOT rank variants — the routing
question and the "is SoT worth anything" question both need the full sample to be answerable.
References are LOADED from disk, never retrained: none_pois_hl365_hs2 (Grid A winner) and the
iso winner (Grid B / r07).

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r06_funnel_full_spec.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using StatsFuns: logit          # explicit: the no-SoT cell needs logit(0.145)

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l05_funnel_flex.jl"))   # pulls in l03 + l01

_r(x, d=4) = round(x, digits=d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
@assert ds.segment isa Data.ScottishLower "wrong segment — reload before running!"
save_dir = joinpath(ROOT, "data/funnel_full/")
mkpath(save_dir)

# ---- Grid-A protocol ----
TARGETS = ["23/24", "24/25", "25/26"]
HS      = 2
HL      = 365.0
DYN_COL = :match_biweek
SAMPLES, WARMUP, CHAINS, DEPTH = 600, 1000, 4, 8

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = HL)

# THREE cells, ordered by priority — each is saved the moment it finishes, so if the night runs
# short the most important results are already on disk.
#
#   1. funnel_pois          — the headline: full-spec funnel vs the stored none_pois_hl365_hs2.
#   2. flex_cw0_sot         — routing: goals onto λ_s. (2 vs 1) = does joining goals to team
#                             strength fix the totals deficit?
#   3. flex_cw0_nosot       — two-layer shots→goals, p₁ ≡ 1 so p₂ is goals per SHOT (≈0.145).
#                             (3 vs 2) = is the SoT layer worth anything, at ~1000 OOS matches
#                             instead of the 66 where it cannot be answered.
#
# Contrast 2-vs-1 needs cell 2; contrast 3-vs-2 needs both — hence this order.
specs = Tuple{String, Any}[
    ("funnel_pois_hl365_hs2",
     TeamFunnelDPGoalsModel(dynamics_config = dyn_cfg)),
    ("funnel_flex_cw0_sot_hl365_hs2",
     TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                                cascade_weight = 0.0, sot_on = true)),
    ("funnel_flex_cw0_nosot_hl365_hs2",
     TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                                cascade_weight = 0.0, sot_on = false,
                                p2_prior = Normal(logit(0.145), 0.5))),
]
println("[INFO] cells: ", join(first.(specs), ", "))

# per-fold convergence gate, same definition as scottish_lower_smile/r02
function _fold_convergence(res)
    n_ok = 0; worst = 0.0; n = length(res.training_results.items)
    for it in res.training_results.items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        mr = isempty(vals) ? NaN : maximum(vals)
        isnan(mr) && continue
        worst = max(worst, mr); mr <= 1.01 && (n_ok += 1)
    end
    return n, n_ok, worst
end

# ==========================================
# TRAIN
# ==========================================
trained = Experiments.ExperimentResults[]
gate_lines = String[]
t_start = time()
for (name, model) in specs
    println("\n", "#"^72, "\n# CELL: $name  (elapsed $(_r((time()-t_start)/60, 1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons = TARGETS, history_seasons = HS, warmup_period = 0,
            dynamics_col = DYN_COL,
            samples = SAMPLES, warmup = WARMUP, chains = CHAINS,
            use_queue = true, max_depth = DEPTH)
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(trained, res)
        n, n_ok, worst = _fold_convergence(res)
        line = "$(rpad(name, 30)) folds=$n  R-hat≤1.01: $n_ok/$n ($(_r(100n_ok/max(n,1), 1))%)  worst=$(_r(worst))"
        println("  ", line); push!(gate_lines, line)
        println("  wall = $(_r((time()-t_start)/60, 1)) min cumulative")
    catch e
        @error "CELL FAILED: $name" exception=(e, catch_backtrace())
        push!(gate_lines, "$(rpad(name, 30)) FAILED")
    end
end
open(joinpath(@__DIR__, "r06_convergence.txt"), "w") do io
    println(io, "r06 full-spec funnel — per-fold convergence gate (read no eval below 95%)")
    for l in gate_lines; println(io, l); end
end

# ==========================================
# LOAD stored references (never retrain)
# ==========================================
println("\n", "="^72, "\nREFERENCES (stored Grid-A / iso winners)\n", "="^72)
all_results = copy(trained)
for (dir, pats) in (("scottish_decay_grid", ["none_pois_hl365_hs2"]),
                    ("scottish_iso_grid",   ["iso_pois_mw40", "iso_pois_mw25"]))
    try
        folders = Experiments.list_experiments(dir; data_dir = joinpath(ROOT, "data"))
        want = filter(f -> any(occursin(p, f) for p in pats), folders)
        isempty(want) || append!(all_results, Experiments.load_experiments(want))
    catch e
        @warn "could not load references from $dir" exception=e
    end
end
println("[INFO] models: ", join([r.config.name for r in all_results], ", "))

# ==========================================
# EVAL vs the Bet365 fair close
# ==========================================
selections = [:home,:draw,:away,:btts_yes,:btts_no,
              :over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
              :over_35,:under_35,:over_45,:under_45]
R06_EVAL = nothing
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    global R06_EVAL = Evaluation.evaluate_experiments(metric, all_results, ds)
    present = sort(unique(R06_EVAL.model))
    _c(m, c) = (c in names(R06_EVAL) ?
        (r = R06_EVAL[R06_EVAL.model .== m, c]; (isempty(r) || ismissing(r[1])) ? NaN :
         round(Float64(r[1]), digits=4)) : NaN)
    fam = [(:x12, [:home,:draw,:away]), (:btts, [:btts_yes,:btts_no]),
           (:totals, [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
                      :over_35,:under_35,:over_45,:under_45])]
    println("\n", "="^72,
            "\n📊 Family-pooled LogLoss diff vs Bet365 close (negative = beats it)\n",
            "Grid-A reference for none_pois_hl365_hs2: x12 0.0143 / btts 0.0014 / totals 0.0002\n",
            "="^72)
    fm = DataFrame(model = present)
    for (f, sels) in fam
        fm[!, f] = [round(mean(filter(!isnan, [_c(m, "logloss_$(s)_overall_diff_ll") for s in sels])),
                          digits=4) for m in present]
    end
    show(fm; allrows=true, allcols=true, truncate=0); println()
    println("\n", "="^72, "\n📉 Per line\n", "="^72)
    ll = DataFrame(model = present)
    for s in selections; ll[!, s] = [_c(m, "logloss_$(s)_overall_diff_ll") for m in present]; end
    show(ll; allrows=true, allcols=true, truncate=0); println()
catch e
    @error "eval failed" exception=(e, catch_backtrace())
end

println("\n", "="^72, "\nR06 DONE — gate in r06_convergence.txt\n", "="^72)
println("READ: does the smoke's 1X2 gain (−0.0071 vs none_pois) survive at ~60 folds × 3 " *
        "seasons, and does the totals deficit (+0.0078) hold or shrink?")
