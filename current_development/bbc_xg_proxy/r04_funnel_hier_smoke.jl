#=
r04 — STAGE 2 smoke: hierarchical per-team conversion on the funnel cascade.

    logit p₁_ij = p1_μ + a₁_i + d₁_j    (shooting accuracy vs opponent shot-blocking)
    logit p₂_ij = p2_μ + a₂_i + d₂_j    (finishing vs goalkeeping)

WHY (Stage-1 result, RESULTS_bbc_xg_proxy.md): the cascade log-likelihood is additively
separable in (λ_s, p₁, p₂) and α/β live only inside λ_s, so goals contribute ZERO gradient to
team strength — Stage 1's team strength is 100% shots-driven. That won 1X2 (Δ −0.0071 vs
none_pois) but lost totals (Δ +0.0078), because the goal LEVEL runs through a single global
conversion constant. Per-team conversion is the channel that lets goals back into the
goal-rate prediction. THE READ: does totals recover while 1X2 holds?

Prior expectation from [[hierarchical-smile-sigma-null]]: hierarchical σ on Ireland bought
nothing (τ pulled below its prior). So σ_p1/σ_p2 vs prior is the first thing to look at — a
collapsed σ means conversion really is a league constant and Stage 1 was right.

Trains ONLY the hierarchical engine; the Stage-1 funnel and none_pois runs are LOADED from
data/funnel_smoke (identical splitter config ⇒ identical folds ⇒ comparable tables), with a
retrain fallback if deserialisation fails.

Run on the server (kaimon REPL) after git pull + REPL restart:
    include(joinpath(pkgdir(BayesianFootball), "current_development/bbc_xg_proxy/r04_funnel_hier_smoke.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using DynamicPPL, LogDensityProblems, ReverseDiff

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/bbc_xg_proxy/l04_funnel_hier.jl"))

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))
_r(x, d=3) = round(x, digits=d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/funnel_smoke/")
mkpath(save_dir)
TARGET = sort(unique(String.(ds.matches.season)))[end]

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = 365.0)
m_hier  = TeamFunnelHierDPGoalsModel(dynamics_config = dyn_cfg)

# ==========================================
# CHECK 0 — build + gradient + short probe
# ==========================================
println("\n", "="^70, "\nCHECK 0 — hierarchical engine builds and is identified\n", "="^70)
probe = nothing
try
    cv = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons    = [TARGET], history_seasons = 2,
        dynamics_col      = :match_biweek, warmup_period = 16)
    fs = Features.create_features(last(Data.create_id_boundaries(ds, cv))[1], ds, m_hier, :match_biweek)
    tm = PreGame.build_turing_model(m_hier, fs)

    vi = DynamicPPL.VarInfo(tm); tm(vi); θ = vi[:]
    lf = DynamicPPL.LogDensityFunction(tm)
    tape = ReverseDiff.compile(ReverseDiff.GradientTape(x -> LogDensityProblems.logdensity(lf, x), θ))
    g = similar(θ); ReverseDiff.gradient!(g, tape, θ)
    t_grad = minimum(@elapsed(ReverseDiff.gradient!(g, tape, θ)) for _ in 1:20)
    println("  gradient = $(_r(t_grad*1000, 3)) ms over $(length(θ)) params " *
            "(Stage 1: 0.83 ms / 95 params)")
    _mark("0a. gradient < 2 ms and finite", t_grad < 2e-3 && all(isfinite, g))

    global probe = sample(tm, NUTS(1000, 0.65; max_depth = 8,
                                   adtype = AutoReverseDiff(compile = true)), 500)
    _mark("0b. probe chain finite", all(isfinite, vec(Array(probe[:lp]))))

    # global conversion must still land on the data-implied MLE
    d = _unpack_funnel(fs.data, m_hier)
    Ss = d.suff_h.S_sot  + d.suff_a.S_sot;  Sm = d.suff_h.S_miss + d.suff_a.S_miss
    Sg = d.suff_h.S_goal + d.suff_a.S_goal; Sv = d.suff_h.S_save + d.suff_a.S_save
    p1m, p2m = Ss/(Ss+Sm), Sg/(Sg+Sv)
    p1μ = 1 ./ (1 .+ exp.(-vec(Array(probe[:p1_μ]))))
    p2μ = 1 ./ (1 .+ exp.(-vec(Array(probe[:p2_μ]))))
    s1 = vec(Array(probe[:σ_p1])); s2 = vec(Array(probe[:σ_p2]))
    println("  p1_μ = $(_r(mean(p1μ), 4)) (MLE $(_r(p1m, 4)))   p2_μ = $(_r(mean(p2μ), 4)) (MLE $(_r(p2m, 4)))")
    println("  σ_p1 = $(_r(mean(s1), 4)) 90%=[$(_r(quantile(s1,.05),4)), $(_r(quantile(s1,.95),4))]  " *
            "σ_p2 = $(_r(mean(s2), 4)) 90%=[$(_r(quantile(s2,.05),4)), $(_r(quantile(s2,.95),4))]")
    println("  (prior half-Normal(0, 0.3): mean ≈ 0.239. σ ≪ that ⇒ conversion is a league " *
            "constant and Stage 1 was right; σ ≳ that ⇒ real per-team conversion.)")
    _mark("0c. p1_μ/p2_μ still hit the pooled MLE (±0.03)",
          abs(mean(p1μ) - p1m) < 0.03 && abs(mean(p2μ) - p2m) < 0.03)
    _mark("0d. σ_p1, σ_p2 not collapsed to 0 (mean > 0.01)", mean(s1) > 0.01 && mean(s2) > 0.01)
catch e
    _mark("0. hierarchical probe ran", false)
    @error "probe FAILED" exception=(e, catch_backtrace())
end

# ==========================================
# 1. TRAIN the hierarchical engine
# ==========================================
println("\n", "="^70, "\nTRAIN — funnel_hier, target=$TARGET\n", "="^70)
res_hier = nothing
t0 = time()
try
    task = Experiments.create_experiment_task(
        ds, m_hier, "funnel_hier_smoke", save_dir;
        target_seasons = [TARGET], history_seasons = 2, warmup_period = 16,
        dynamics_col = :match_biweek,
        samples = 600, warmup = 1000, chains = 4, use_queue = true, max_depth = 8)
    global res_hier = Experiments.run_experiment(task)
    Experiments.save_experiment(res_hier)
    println("  wall = $(_r((time()-t0)/60, 1)) min, items = $(length(res_hier.training_results.items))")
    _mark("1. funnel_hier trained (items > 0)", length(res_hier.training_results.items) > 0)
catch e
    _mark("1. funnel_hier trained", false)
    @error "training FAILED" exception=(e, catch_backtrace())
end

# ==========================================
# 2. LOAD the Stage-1 comparators (identical folds)
# ==========================================
println("\n", "="^70, "\nLOAD — Stage-1 funnel + none_pois\n", "="^70)
all_results = Experiments.ExperimentResults[]   # NOT Any[] — evaluate_experiments is typed
res_hier === nothing || push!(all_results, res_hier)
try
    folders = Experiments.list_experiments("funnel_smoke"; data_dir = joinpath(ROOT, "data"))
    want = filter(f -> (occursin("funnel_pois_smoke", f) || occursin("none_pois_smoke", f)) &&
                       !occursin("hier", f), folders)
    loaded = Experiments.load_experiments(want)
    append!(all_results, loaded)
    _mark("2. Stage-1 comparators loaded (n=$(length(loaded))/2)", length(loaded) == 2)
catch e
    _mark("2. Stage-1 comparators loaded", false)
    @error "load failed — rerun r03 first, or retrain here" exception=(e, catch_backtrace())
end
println("[INFO] models in play: ", join([r.config.name for r in all_results], ", "))

# ==========================================
# 3. CONVERGENCE
# ==========================================
println("\n", "="^70, "\nCHECK 3 — convergence\n", "="^70)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
if res_hier !== nothing
    try
        conv = Experiments.Diagnostics.check_convergence(Experiments.Diagnostics.extract_chains(ds, res_hier))
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        _mark("3a. funnel_hier global max R-hat ≤ 1.05 (=$(_r(worst,4)))", worst <= 1.05)

        raw = res_hier.training_results.items[1][1]
        er  = DataFrame(MCMCChains.ess_rhat(raw))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        _rhat(p) = (rows = er[er.parameters .== Symbol(p), rcol]; isempty(rows) ? NaN : rows[1])
        ps = ["p1_μ", "p2_μ", "σ_p1", "σ_p2", "δ_league_raw[1]"]
        for p in ps; println("    $(rpad(p, 18)) rhat=$(_r(_rhat(p), 4))"); end
        _mark("3b. conversion params R-hat ≤ 1.01",
              all(!isnan(_rhat(p)) && _rhat(p) <= 1.01 for p in ps))
    catch e
        _mark("3. convergence check ran", false)
        @error "convergence failed" exception=(e, catch_backtrace())
    end
end

# ==========================================
# 4. DOES CONVERSION ACTUALLY VARY BY TEAM?
# ==========================================
println("\n", "="^70, "\nCHECK 4 — per-team conversion spread (the Stage-2 question)\n", "="^70)
if res_hier !== nothing
    for (s, lbl) in (("σ_p1", "σ_p1 (SoT|shot)"), ("σ_p2", "σ_p2 (goal|SoT)"))
        v = _pool(res_hier, s)
        isempty(v) && continue
        println("  $lbl: mean=$(_r(mean(v),4)) 90%=[$(_r(quantile(v,.05),4)), $(_r(quantile(v,.95),4))]" *
                "   prior half-N(0,0.3) mean ≈ 0.2394")
    end
    # translate σ_p2 into a p₂ range: ±1sd on the logit around the pooled mean
    p2μv = _pool(res_hier, "p2_μ"); s2v = _pool(res_hier, "σ_p2")
    if !isempty(p2μv) && !isempty(s2v)
        m, s = mean(p2μv), mean(s2v)
        lo, hi = 1/(1+exp(-(m-s))), 1/(1+exp(-(m+s)))
        println("  ⇒ ±1sd team finishing spans p₂ ∈ [$(_r(lo,3)), $(_r(hi,3))] " *
                "(pooled $(_r(1/(1+exp(-m)),3)))")
    end
    _mark("4. conversion spread reported", true)
end

# ==========================================
# 5. TEAM-STRENGTH SPREAD + OOS EVAL (three-way)
# ==========================================
println("\n", "="^70, "\nCHECK 5 — spread + eval vs Bet365 close\n", "="^70)
team_of = Dict(Int(r.match_id) => String(r.home_team) for r in eachrow(ds.matches))
for res in all_results
    try
        df = Experiments.extract_oos_predictions(ds, res).df
        λh = [mean(r.λ_h) for r in eachrow(df)]
        by_team = Dict{String, Vector{Float64}}()
        for (i, r) in enumerate(eachrow(df))
            push!(get!(by_team, get(team_of, Int(r.match_id), "?"), Float64[]), log(λh[i]))
        end
        per_team = [mean(v) for (_, v) in by_team if length(v) >= 2]
        println("  $(rpad(res.config.name, 20)) mean λ_goals(h)=$(_r(mean(λh))) " *
                "across-team sd log λ=$(_r(length(per_team) >= 3 ? std(per_team) : NaN, 4)) " *
                "($(length(per_team)) teams)")
    catch e
        @error "latents failed for $(res.config.name)" exception=(e, catch_backtrace())
    end
end

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]
R04_EVAL = nothing
try
    metric = Evaluation.AbstractScoringRule[Evaluation.RQR()]
    append!(metric, [Evaluation.LogLoss(s) for s in selections])
    append!(metric, [Evaluation.GLMEdge(s) for s in selections])
    global R04_EVAL = Evaluation.evaluate_experiments(metric, all_results, ds)

    present = sort(unique(R04_EVAL.model))
    function _col(df, model, colname)
        colname in names(df) || return NaN
        r = df[df.model .== model, colname]
        (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits=4)
    end
    println("\n", "="^70,
            "\n📉 LogLoss diff (model−market) per line — NEGATIVE = beats Bet365 close\n", "="^70)
    ll = DataFrame(model = present)
    for s in selections; ll[!, s] = [_col(R04_EVAL, m, "logloss_$(s)_overall_diff_ll") for m in present]; end
    show(ll; allrows=true, allcols=true, truncate=0); println()

    fam = Dict(:x12 => [:home, :draw, :away], :btts => [:btts_yes, :btts_no],
               :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
                           :over_35, :under_35, :over_45, :under_45])
    println("\n", "="^70, "\n📊 Family-pooled mean LogLoss diff " *
            "(Stage 1: funnel x12 0.0153 / btts −0.0078 / totals −0.0139; " *
            "none 0.0224 / −0.0082 / −0.0217)\n", "="^70)
    fm = DataFrame(model = present)
    for (fname, sels) in fam
        fm[!, fname] = [round(mean(filter(!isnan,
            [_col(R04_EVAL, m, "logloss_$(s)_overall_diff_ll") for s in sels])), digits=4) for m in present]
    end
    show(fm; allrows=true, allcols=true, truncate=0); println()

    hr = findfirst(occursin.("hier", fm.model)); nr = findfirst(m -> occursin("none", m), fm.model)
    fr = findfirst(m -> occursin("funnel_pois", m), fm.model)
    if hr !== nothing
        for (lbl, ref) in (("vs none_pois", nr), ("vs Stage-1 funnel", fr))
            ref === nothing && continue
            println("  >> hier − $(lbl) (negative = hier better): " *
                    join(["$(f) $(_r(fm[hr, f] - fm[ref, f], 4))" for f in keys(fam)], "  |  "))
        end
    end
    _mark("5. three-way eval table produced", length(present) == length(all_results))
catch e
    _mark("5. three-way eval table produced", false)
    @error "evaluate_experiments failed" exception=(e, catch_backtrace())
end

println("\n", "="^70, "\nR04 STAGE-2 SUMMARY\n", "="^70)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
println("\n$(count(last, verdicts))/$(length(verdicts)) checks passed.")
println("READ: (a) is σ_p2 materially above 0 — does finishing really vary by team? " *
        "(b) does totals recover vs Stage 1 while 1X2 holds?")
