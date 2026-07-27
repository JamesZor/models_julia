#=
r05 — variant grid on the flexible funnel (l05). Answers the two open questions cheaply.

Q1 (totals deficit): Stage 1's cascade makes goals conditionally independent of λ_s, so team
    strength is fitted to shots alone and the goal LEVEL rides on one global constant. Does
    routing goals through the marginal Poisson — where they meet λ_s — recover totals?
Q2 (is SoT pulling its weight?): with global p₁, SoT is ANCILLARY for team strength once shots
    are observed. Its team-level content is σ_p1 ≈ 0.034 (CI [0.004, 0.074], r04). Is that
    worth anything in prediction?

Cells (all 95 params, Stage-1 cost ≈ 28 min each):

  cw100_sot  = cascade_weight 1.0, sot_on   — ALREADY RUN as funnel_pois_smoke (loaded, not retrained)
  cw050_sot  = cascade_weight 0.5, sot_on   — half-and-half routing
  cw000_sot  = cascade_weight 0.0, sot_on   — goals joint with shots on λ_s, SoT still fitted
  cw000_nosot= cascade_weight 0.0, sot off  — two-layer shots→goals (p₂ ≈ 0.145 per SHOT)

  cw000_nosot vs cw000_sot isolates Q2: same goals routing, SoT layer in or out.
  cw000_sot   vs cw100_sot isolates Q1: same data, goals routed differently.

Also loads none_pois_smoke and funnel_hier_smoke so the table is complete.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r05_funnel_variants.jl"))
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
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l05_funnel_flex.jl"))

verdicts = Tuple{String, Bool}[]
_mark(n, ok) = (push!(verdicts, (n, ok)); println(ok ? "✅ $n" : "❌ $n"))
_r(x, d=4) = round(x, digits=d)

# NOTE: load our own ds — never trust a global from an earlier session.
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
@assert ds.segment isa Data.ScottishLower "wrong segment!"
save_dir = joinpath(ROOT, "data/funnel_smoke/")
TARGET = sort(unique(String.(ds.matches.season)))[end]

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 365.0)
cells = [
    ("flex_cw050_sot",   TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                            cascade_weight = 0.5, sot_on = true)),
    ("flex_cw000_sot",   TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                            cascade_weight = 0.0, sot_on = true)),
    # two-layer: p₂ is now goals per SHOT (≈0.145), and p₁ ≡ 1
    ("flex_cw000_nosot", TeamFunnelFlexDPGoalsModel(dynamics_config = dyn_cfg,
                            cascade_weight = 0.0, sot_on = false,
                            p2_prior = Normal(logit(0.145), 0.5))),
]

# ==========================================
# TRAIN
# ==========================================
trained = Experiments.ExperimentResults[]
for (name, model) in cells
    println("\n", "#"^68, "\n# RUN: $name\n", "#"^68)
    t0 = time()
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons = [TARGET], history_seasons = 2, warmup_period = 16,
            dynamics_col = :match_biweek,
            samples = 600, warmup = 1000, chains = 4, use_queue = true, max_depth = 8)
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(trained, res)
        println("  wall = $(_r((time()-t0)/60, 1)) min, items = $(length(res.training_results.items))")
        _mark("train $name (items=$(length(res.training_results.items)))",
              length(res.training_results.items) > 0)
    catch e
        _mark("train $name", false)
        @error "FAILED: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# CONVERGENCE + conversion posteriors
# ==========================================
println("\n", "="^70, "\nCONVERGENCE + conversion\n", "="^70)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]; Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    out
end
for res in trained
    nm = res.config.name
    try
        conv = Experiments.Diagnostics.check_convergence(Experiments.Diagnostics.extract_chains(ds, res))
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        _mark("$nm max R-hat ≤ 1.05 (=$(_r(worst)))", worst <= 1.05)
        p1 = _pool(res, "p1_raw"); p2 = _pool(res, "p2_raw")
        s(x) = isempty(x) ? NaN : _r(mean(1 ./ (1 .+ exp.(-x))))
        println("    p₁=$(s(p1))  p₂=$(s(p2))  ⇒ goals/shot = " *
                "$(_r((res.config.model.sot_on ? s(p1) : 1.0) * s(p2)))  (data ≈ 0.1456)")
    catch e
        _mark("$nm convergence", false); @error "conv failed $nm" exception=(e, catch_backtrace())
    end
end

# ==========================================
# LOAD references + three-way-plus eval
# ==========================================
all_results = copy(trained)
try
    folders = Experiments.list_experiments("funnel_smoke"; data_dir = joinpath(ROOT, "data"))
    want = filter(f -> occursin("funnel_pois_smoke", f) || occursin("none_pois_smoke", f) ||
                       occursin("funnel_hier_smoke", f), folders)
    append!(all_results, Experiments.load_experiments(want))
catch e
    @error "reference load failed" exception=(e, catch_backtrace())
end
println("[INFO] models: ", join([r.config.name for r in all_results], ", "))

# team-strength spread
team_of = Dict(Int(r.match_id) => String(r.home_team) for r in eachrow(ds.matches))
println("\n", "="^70, "\nTEAM-STRENGTH SPREAD\n", "="^70)
for res in all_results
    try
        df = Experiments.extract_oos_predictions(ds, res).df
        λh = [mean(r.λ_h) for r in eachrow(df)]
        bt = Dict{String, Vector{Float64}}()
        for (i, r) in enumerate(eachrow(df))
            push!(get!(bt, get(team_of, Int(r.match_id), "?"), Float64[]), log(λh[i]))
        end
        pt = [mean(v) for (_, v) in bt if length(v) >= 2]
        println("  $(rpad(res.config.name, 20)) λ_goals(h)=$(_r(mean(λh),3)) " *
                "sd log λ=$(_r(length(pt) >= 3 ? std(pt) : NaN)) ($(length(pt)) teams)")
    catch e
        @error "latents failed $(res.config.name)" exception=(e, catch_backtrace())
    end
end

selections = [:home,:draw,:away,:btts_yes,:btts_no,
              :over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
              :over_35,:under_35,:over_45,:under_45]
R05_EVAL = nothing
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    global R05_EVAL = Evaluation.evaluate_experiments(metric, all_results, ds)
    present = sort(unique(R05_EVAL.model))
    _c(m, c) = (c in names(R05_EVAL) ?
        (r = R05_EVAL[R05_EVAL.model .== m, c]; (isempty(r) || ismissing(r[1])) ? NaN :
         round(Float64(r[1]), digits=4)) : NaN)
    fam = [(:x12, [:home,:draw,:away]), (:btts, [:btts_yes,:btts_no]),
           (:totals, [:over_05,:under_05,:over_15,:under_15,:over_25,:under_25,
                      :over_35,:under_35,:over_45,:under_45])]
    println("\n", "="^70, "\n📊 Family-pooled LogLoss diff vs Bet365 close (negative = beats it)\n", "="^70)
    fm = DataFrame(model = present)
    for (f, sels) in fam
        fm[!, f] = [round(mean(filter(!isnan, [_c(m, "logloss_$(s)_overall_diff_ll") for s in sels])),
                          digits=4) for m in present]
    end
    show(fm; allrows=true, allcols=true, truncate=0); println()
    println("\n", "="^70, "\n📉 Per line\n", "="^70)
    ll = DataFrame(model = present)
    for s in selections; ll[!, s] = [_c(m, "logloss_$(s)_overall_diff_ll") for m in present]; end
    show(ll; allrows=true, allcols=true, truncate=0); println()
    _mark("eval table produced", true)
catch e
    _mark("eval table produced", false)
    @error "eval failed" exception=(e, catch_backtrace())
end

println("\n", "="^70, "\nR05 SUMMARY\n", "="^70)
for (n, ok) in verdicts; println(ok ? "✅ $n" : "❌ $n"); end
println("\nREAD:\n  Q1 cw000_sot vs funnel_pois (cw=1): does routing goals onto λ_s fix totals?" *
        "\n  Q2 cw000_nosot vs cw000_sot: is the SoT layer worth anything at all?")
