#=
r07 — FUNNEL + ISO vs NONE + ISO. The r06 verdict left one question open: the two-layer funnel
(cw=0, sot off) is a real STRUCTURAL win over none_pois on 1X2, but the iso market pillar owns
totals. Those levers are orthogonal — so does a sharper structural core add anything once the
market pillar sits on top? This runs the 2×2.

PROTOCOL — deliberately MATCHES scottish_lower_smile/r07 (the stored iso grid) EXACTLY, so the
stored none+iso cells (iso_pois_mw25/mw40) are reused as-is on identical folds:

    target_seasons  = ["24/25", "25/26"]   (~40 biweek folds, [56,57] pooled)
    history_seasons = 2 ,  HL = 365 ,  warmup_period = 0 ,  dynamics_col = :match_biweek

SAMPLER — funnel cells need warmup=1000 (tight conversion posterior); none_pois is cheap at 300.
Fold set comes from the splitter, so sampler settings don't affect comparability.

THE 2×2 (identical ~40 folds):

                    no market                +iso (mw 0.25 / 0.40)
    none core       none_pois_2s (new)       iso_pois_mw25/mw40  (STORED, reused)
    funnel core     funnel_cw0_2s (new)      funnel_iso_mw25/mw40 (new)

Reads:
  • funnel_iso  −  iso_pois   at matched mw  → does the sharper core help WITH the market on top?
  • funnel_iso  −  funnel_cw0            → iso lift on the funnel core (does a good core make the
                                            market pillar redundant, or still additive?)
  • iso_pois    −  none_pois_2s          → iso lift on the none core (the known big totals win)

Cells ordered by priority — each saved the moment it finishes.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r07_funnel_iso_grid.jl"))
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

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l06_funnel_iso.jl"))   # l06 → l05 → l03 → l01

_r(x, d=4) = round(x, digits=d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
@assert ds.segment isa Data.ScottishLower "wrong segment — reload before running!"
save_dir = joinpath(ROOT, "data/funnel_iso_grid/")
mkpath(save_dir)

# ---- protocol: IDENTICAL to scottish_lower_smile/r07 (the stored iso grid) ----
TARGETS = ["24/25", "25/26"]
HS      = 2
HL      = 365.0
DYN_COL = :match_biweek
dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = HL)

# per-cell sampler: (samples, warmup, chains, depth). Funnel needs warmup≥1000; none_pois cheap.
FUNNEL_SAMP = (600, 1000, 4, 8)
NONE_SAMP   = (800, 300, 4, 10)

# (name, model, sampler) — priority order; each saved on completion.
specs = Tuple{String, Any, NTuple{4,Int}}[
    ("funnel_iso_mw40_hl365_hs2",
     TeamFunnelIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 0.40), FUNNEL_SAMP),
    ("funnel_iso_mw25_hl365_hs2",
     TeamFunnelIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 0.25), FUNNEL_SAMP),
    ("funnel_cw0_2s_hl365_hs2",
     TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                                cascade_weight = 0.0, sot_on = false,
                                p2_prior = Normal(logit(0.145), 0.5)), FUNNEL_SAMP),
    ("none_pois_2s_hl365_hs2",
     TeamDPGoalsModel(dynamics_config = dyn_cfg), NONE_SAMP),
]
println("[INFO] training cells: ", join(first.(specs), ", "))

function _fold_convergence(res)
    n = length(res.training_results.items); n_ok = 0; worst = 0.0
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
for (name, model, (S, W, C, D)) in specs
    println("\n", "#"^72, "\n# CELL: $name  (elapsed $(_r((time()-t_start)/60, 1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons = TARGETS, history_seasons = HS, warmup_period = 0,
            dynamics_col = DYN_COL,
            samples = S, warmup = W, chains = C,
            use_queue = true, max_depth = D)
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(trained, res)
        n, n_ok, worst = _fold_convergence(res)
        line = "$(rpad(name, 28)) folds=$n  R-hat≤1.01: $n_ok/$n ($(_r(100n_ok/max(n,1), 1))%)  worst=$(_r(worst))"
        println("  ", line); push!(gate_lines, line)
        println("  wall = $(_r((time()-t_start)/60, 1)) min cumulative")
    catch e
        @error "CELL FAILED: $name" exception=(e, catch_backtrace())
        push!(gate_lines, "$(rpad(name, 28)) FAILED")
    end
end
open(joinpath(@__DIR__, "r07_convergence.txt"), "w") do io
    println(io, "r07 funnel+iso grid — per-fold convergence gate (read no eval below 95%; worst≤1.05 ok)")
    for l in gate_lines; println(io, l); end
end

# ==========================================
# LOAD stored none+iso (reused on identical folds) — NEVER retrain
# ==========================================
println("\n", "="^72, "\nREFERENCES (stored iso grid — none+iso)\n", "="^72)
all_results = copy(trained)
try
    folders = Experiments.list_experiments("scottish_iso_grid"; data_dir = joinpath(ROOT, "data"))
    want = filter(f -> any(occursin(p, f) for p in ["iso_pois_mw25", "iso_pois_mw40"]), folders)
    isempty(want) || append!(all_results, Experiments.load_experiments(want))
catch e
    @warn "could not load stored iso references" exception=e
end
println("[INFO] models in table: ", join([r.config.name for r in all_results], ", "))

# ==========================================
# EVAL vs the Bet365 fair close
# ==========================================
selections = [:home,:draw,:away,:btts_yes,:btts_no,
              :over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
              :over_35,:under_35,:over_45,:under_45]
R07_EVAL = nothing
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    global R07_EVAL = Evaluation.evaluate_experiments(metric, all_results, ds)
    present = sort(unique(R07_EVAL.model))
    _c(m, c) = (c in names(R07_EVAL) ?
        (r = R07_EVAL[R07_EVAL.model .== m, c]; (isempty(r) || ismissing(r[1])) ? NaN :
         round(Float64(r[1]), digits=4)) : NaN)
    fam = [(:x12, [:home,:draw,:away]), (:btts, [:btts_yes,:btts_no]),
           (:totals, [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
                      :over_35,:under_35,:over_45,:under_45])]
    _famval(m, sels) = round(mean(filter(!isnan, [_c(m, "logloss_$(s)_overall_diff_ll") for s in sels])), digits=4)

    println("\n", "="^72,
            "\n📊 Family-pooled LogLoss diff vs Bet365 close (negative = beats it)\n", "="^72)
    fm = DataFrame(model = present)
    for (f, sels) in fam; fm[!, f] = [_famval(m, sels) for m in present]; end
    show(fm; allrows=true, allcols=true, truncate=0); println()

    # --- the three contrasts, computed directly ---
    println("\n", "="^72, "\n🔑 KEY CONTRASTS (Δ family-pooled LogLoss; negative = first model better)\n", "="^72)
    fams = [:x12, :btts, :totals]
    val(m) = Dict(f => _famval(m, s) for (f, s) in fam)
    function _delta(a, b, label)
        (a in present && b in present) || (println("  [skip] $label — missing $(a in present ? b : a)"); return)
        va, vb = val(a), val(b)
        println("  $label")
        for f in fams; println("      $(rpad(string(f),7)) $(rpad(string(_r(va[f])),9)) − $(rpad(string(_r(vb[f])),9)) = $(_r(va[f]-vb[f]))"); end
    end
    _delta("funnel_iso_mw40_hl365_hs2", "iso_pois_mw40_hl365_hs2",
           "funnel+iso vs none+iso  @ mw0.40  (does a sharper core help WITH market on top?)")
    _delta("funnel_iso_mw25_hl365_hs2", "iso_pois_mw25_hl365_hs2",
           "funnel+iso vs none+iso  @ mw0.25")
    _delta("funnel_iso_mw40_hl365_hs2", "funnel_cw0_2s_hl365_hs2",
           "iso LIFT on funnel core  (funnel+iso mw40 − funnel no-market)")
    _delta("iso_pois_mw40_hl365_hs2", "none_pois_2s_hl365_hs2",
           "iso LIFT on none core    (iso mw40 − none no-market)")

    println("\n", "="^72, "\n📉 Per line\n", "="^72)
    ll = DataFrame(model = present)
    for s in selections; ll[!, s] = [_c(m, "logloss_$(s)_overall_diff_ll") for m in present]; end
    show(ll; allrows=true, allcols=true, truncate=0); println()

    # --- persist the whole eval to disk so the overnight run needs NO stdout capture ---
    open(joinpath(@__DIR__, "r07_results.txt"), "w") do io
        println(io, "r07 funnel+iso — family-pooled LogLoss diff vs Bet365 close (negative = beats it)")
        show(io, fm; allrows=true, allcols=true, truncate=0); println(io)
        println(io, "\nKEY CONTRASTS (Δ family-pooled; negative = first model better):")
        for (a, b, label) in [
            ("funnel_iso_mw40_hl365_hs2", "iso_pois_mw40_hl365_hs2", "funnel+iso vs none+iso @ mw0.40"),
            ("funnel_iso_mw25_hl365_hs2", "iso_pois_mw25_hl365_hs2", "funnel+iso vs none+iso @ mw0.25"),
            ("funnel_iso_mw40_hl365_hs2", "funnel_cw0_2s_hl365_hs2", "iso LIFT on funnel core"),
            ("iso_pois_mw40_hl365_hs2",   "none_pois_2s_hl365_hs2",   "iso LIFT on none core")]
            (a in present && b in present) || continue
            va, vb = val(a), val(b)
            println(io, "  $label")
            for f in fams; println(io, "      $(rpad(string(f),7)) $(_r(va[f])) − $(_r(vb[f])) = $(_r(va[f]-vb[f]))"); end
        end
        println(io, "\nPer line:")
        show(io, ll; allrows=true, allcols=true, truncate=0); println(io)
    end
catch e
    @error "eval failed" exception=(e, catch_backtrace())
end

println("\n", "="^72, "\nR07 DONE — gate in r07_convergence.txt\n", "="^72)
println("READ: at matched market_weight, does funnel+iso beat none+iso? And does the iso pillar " *
        "still lift the (already-sharp) funnel core, or does a good structural core absorb it?")
