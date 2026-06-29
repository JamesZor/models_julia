#=
SMOKE TEST — hierarchical-σ smile (l08 HierSmileDoublePoissonModel). Goal: does it RUN and CONVERGE?

Three cells (smile_weight=1, supremacy_weight=1, market_on=true), each a different σ hierarchy:
  - hsmile_perstrike : log σ = log_σ_base + τ_strike·z_strike[k]                 (delta per O/U line)
  - hsmile_perteam   : log σ = log_σ_base + τ_team·(z_team[home]+z_team[away])    (delta per team)
  - hsmile_both      : both terms

PRIMARY OUTPUT = convergence (R-hat / ESS), focused on the new hierarchy params:
  log_σ_base, τ_strike, τ_team, z_strike[*], z_team[*], plus log_φ[*] and σ_sup.
READ:
  • All R-hat ≤ ~1.05 + healthy ESS ⇒ it samples cleanly (non-centred parameterisation working).
  • τ_strike / τ_team → ~0 ⇒ no learned heterogeneity (σ collapses to the global scalar = l03).
  • τ_* with bad R-hat / tiny ESS / divergences ⇒ Neal's funnel — tighten tau_*_prior or reparameterise.
A light GLMEdge/LogLoss is printed too, but the point here is CONVERGENCE, not edge.

Ireland (small, fast). DEPENDS on l03 (smile feature + prediction) then l08.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r17_smoke_hier_smile.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")
include("current_development/split_market_pillar/l08_hier_smile_poisson.jl")

# ==========================================
# 1. DATA
# ==========================================
SEGMENT = Data.Ireland()
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/hier_smile_smoke_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (= r10, smaller sample count for a smoke test)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

samples        = 600
warmup         = 600        # extra warmup helps a fresh hierarchical geometry adapt
target_seasons  = ["2026"]
dynamics_col    = :match_week
chains         = 4
KMAX           = 4
warmup_period   = 21

_hier(per_strike, per_team) = HierSmileDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = MarketSmileFeature(Kmax = KMAX),
    market_on              = true,
    supremacy_weight       = 1.0,
    smile_weight           = 1.0,
    sigma_per_strike       = per_strike,
    sigma_per_team         = per_team,
)

specs = Tuple{String, Any}[
    ("hsmile_perstrike", _hier(true,  false)),
    ("hsmile_perteam",   _hier(false, true)),
    ("hsmile_both",      _hier(true,  true)),
]
println("[INFO] Hier-smile smoke ($seg_tag): $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 3. RUN + CONVERGENCE per cell
# ==========================================
all_results = Any[]
runs        = Tuple{String,Any,Any}[]   # (name, model, res) — for the σ-hierarchy read in §3b
for (name, model) in specs
    println("\n", "#"^72, "\n# RUN: $name\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = target_seasons,
            history_seasons = 2,
            warmup_period   = warmup_period,
            dynamics_col    = dynamics_col,
            samples         = samples,
            warmup          = warmup,
            chains          = chains,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(all_results, res)
        push!(runs, (name, model, res))

        chains_obj = Experiments.Diagnostics.extract_chains(ds, res)
        conv = Experiments.Diagnostics.check_convergence(chains_obj)
        # focus on the smile-σ hierarchy + shape + supremacy
        key = r"σ_base|log_σ_base|τ_strike|τ_team|z_strike|z_team|log_φ|σ_sup|σ_smile"
        hier = filter(r -> occursin(key, string(r.raw_symbol)) || occursin(key, string(r.parameter)),
                      conv.df)
        println("\n--- $name : hierarchy-param convergence (R-hat / ESS) ---")
        if isempty(hier)
            println("(no hierarchy rows matched — inspect conv.df raw_symbol column)")
            display(conv.df[:, [:raw_symbol, :mean, :std, :rhat, :ess]])
        else
            display(sort(hier[:, [:raw_symbol, :mean, :std, :rhat, :ess]], :rhat, rev=true))
        end
        worst = isempty(conv.df) ? NaN : maximum(skipmissing(conv.df.rhat))
        println("  >> max R-hat over ALL params: $(round(worst, digits=4)) " *
                (worst <= 1.05 ? "(OK)" : "(⚠ > 1.05 — inspect)"))
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
    end
end

# ==========================================
# 3b. SMILE-σ HIERARCHY — pull the deltas straight from the raw chains
# ==========================================
# check_convergence's conv.df is a curated table and drops these params, so read the raw Chains directly
# (same pattern as `chain = res.training_results.items[i][1]`). Pool posterior draws across ALL folds.
# strike k (1-based) → total-goals threshold K=k-1 → O/U line K+0.5  (KMAX=4 ⇒ lines 0.5..4.5)
strike_lines(nK) = [ (k - 1) + 0.5 for k in 1:nK ]

_chain1(res) = res.training_results.items[1][1]
_has(res, s) = Symbol(s) in keys(_chain1(res))
# pool draws of one symbol across every fold (same fold/iter order for all symbols ⇒ draws stay matched)
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
_count(res, stem) = count(i -> _has(res, "$stem[$i]"), 1:999)
_f(v) = (m = round(mean(v), digits=3), sd = round(std(v), digits=3),
         lo = round(quantile(v, 0.05), digits=3), hi = round(quantile(v, 0.95), digits=3))

function smile_sigma_report(name, model, res)
    println("\n", "="^72)
    println("[σ-hierarchy] $name   (per_strike=$(model.sigma_per_strike), per_team=$(model.sigma_per_team))")
    println("="^72)
    if !_has(res, "log_σ_base")
        println("  (log_σ_base not in chain — check items access / symbol names)"); return
    end

    lsb = _pool(res, "log_σ_base"); ts = _pool(res, "τ_strike"); tt = _pool(res, "τ_team")
    g = _f(exp.(lsb))
    println("  GLOBAL anchor  σ_base = exp(log_σ_base): mean=$(g.m)  90%=[$(g.lo), $(g.hi)]   (prior centre 0.15)")
    println("    └─ small σ = model hugs the market smile tightly (little edge budget); large = loose anchor.")

    # ---- δ[k]: per-O/U-line heterogeneity ----
    if model.sigma_per_strike
        ft = _f(ts); ppos = round(mean(ts .> 0.05), digits=2)
        println("  τ_strike (per-LINE σ spread): mean=$(ft.m) 90%=[$(ft.lo),$(ft.hi)]  P(τ>0.05)=$ppos " *
                (ppos ≥ 0.9 ? "← real heterogeneity" : ppos ≤ 0.3 ? "← ~flat (σ wants to be global)" : ""))
        nK = _count(res, "z_strike"); lines = strike_lines(nK)
        println("    per-line anchor  σ_k = exp(log_σ_base + τ_strike·z_strike[k])  (typical match, team term=0):")
        for k in 1:nK
            zk = _pool(res, "z_strike[$k]"); n = min(length(lsb), length(ts), length(zk))
            sk = _f(exp.(lsb[1:n] .+ ts[1:n] .* zk[1:n]))
            println("      O/U $(lines[k]):  σ=$(sk.m) 90%=[$(sk.lo),$(sk.hi)]   z̄=$(round(mean(zk),digits=3))")
        end
    else
        println("  τ_strike: GATED OFF (drew from prior only — ignore its value).")
    end

    # ---- δ[team]: per-team heterogeneity ----
    if model.sigma_per_team
        ftt = _f(tt); ppos = round(mean(tt .> 0.05), digits=2)
        println("  τ_team (per-TEAM σ spread): mean=$(ftt.m) 90%=[$(ftt.lo),$(ftt.hi)]  P(τ>0.05)=$ppos " *
                (ppos ≥ 0.9 ? "← real heterogeneity" : ppos ≤ 0.3 ? "← ~flat (σ wants to be global)" : ""))
        nT = _count(res, "z_team")
        zbar = [mean(_pool(res, "z_team[$t]")) for t in 1:nT]
        mult = exp.(mean(tt) .* zbar)   # per-team multiplicative σ effect at posterior-mean τ
        println("    team σ-multiplier exp(τ_team·z̄_team) across $nT teams: " *
                "min=$(round(minimum(mult),digits=3)) med=$(round(median(mult),digits=3)) max=$(round(maximum(mult),digits=3))")
    else
        println("  τ_team: GATED OFF (drew from prior only — ignore its value).")
    end

    # ---- the smile SHAPE φ(K)=exp(log_φ) — shared with l03, the headline curve ----
    nK = _count(res, "log_φ"); lines = strike_lines(nK)
    println("  smile shape φ(K)=exp(log_φ)  (φ≡1 ⇒ pure Poisson; <1 thins, >1 fattens that strike):")
    for k in 1:nK
        φk = _f(exp.(_pool(res, "log_φ[$k]")))
        flag = φk.lo > 1.0 ? " ↑ fat" : φk.hi < 1.0 ? " ↓ thin" : ""
        println("      O/U $(lines[k]):  φ=$(φk.m) 90%=[$(φk.lo),$(φk.hi)]$flag")
    end
    g2 = _f(exp.(_pool(res, "σ_sup")))   # σ_sup itself is sampled, not hierarchical — context
    println("  σ_sup (supremacy anchor, sampled scalar): mean=$(round(mean(_pool(res,"σ_sup")),digits=3))")
    return
end

println("\n", "█"^72, "\n  SMILE-σ HIERARCHY READ (raw chains)\n", "█"^72)
for (name, model, res) in runs
    try
        smile_sigma_report(name, model, res)
    catch e
        @error "σ-hierarchy report failed: $name" exception=(e, catch_backtrace())
    end
end
println("""
\n[READ] What to take from §3b:
 • σ_base = the learned global anchor tightness (replaces l03's fixed scalar). This alone is the headline:
   it says how hard the smile pillar pulls λ_tot toward the market totals ladder.
 • τ_strike / τ_team > ~0.05 with mass away from 0 ⇒ the data WANTS different σ per O/U line / per team
   (heterogeneity worth keeping). Sitting at ~0 ⇒ collapses to the global σ = l03 (hierarchy bought nothing).
 • The per-line σ_k ladder shows WHICH strikes get a looser anchor (more edge budget) — usually the tails.
 • φ(K) is the smile shape (same object as l03): ↑/↓ flags = strikes the market over/under-disperses vs Poisson.
""")

#=
--- hsmile_perstrike : hierarchy-param convergence (R-hat / ESS) ---
1×5 DataFrame                                                                                                        
 Row │ raw_symbol  mean      std        rhat     ess                                                                 
     │ Symbol      Float64   Float64    Float64  Float64                                                             
─────┼───────────────────────────────────────────────────                                                            
   1 │ σ_sup       0.474008  0.0377371  1.00045      NaN                                                             
  >> max R-hat over ALL params: 1.0097 (OK)
=#


#=
--- hsmile_perteam : hierarchy-param convergence (R-hat / ESS) ---                                                   
1×5 DataFrame                                                                                                        
 Row │ raw_symbol  mean      std        rhat     ess                                                                 
     │ Symbol      Float64   Float64    Float64  Float64                                                             
─────┼───────────────────────────────────────────────────                                                            
   1 │ σ_sup       0.472297  0.0367417  1.00317      NaN                                                             
  >> max R-hat over ALL params: 1.0086 (OK)
=#


#=
--- hsmile_both : hierarchy-param convergence (R-hat / ESS) ---
1×5 DataFrame
 Row │ raw_symbol  mean      std       rhat     ess     
     │ Symbol      Float64   Float64   Float64  Float64 
─────┼──────────────────────────────────────────────────
   1 │ σ_sup       0.471634  0.036551  1.00213      NaN
  >> max R-hat over ALL params: 1.006 (OK)
=#


# ==========================================
# 4. LIGHT EDGE CHECK (secondary — confirms prediction path works end-to-end)
# ==========================================
try
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)
    println("\n", "="^60, "\n📊 GLM Edge (Betfair) — hier-smile smoke\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)
    println("\n", "="^60, "\n📉 LogLoss (Betfair)\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)
catch e
    @error "Eval phase failed (chains are saved; this is the secondary check)" exception=(e, catch_backtrace())
end

println("""

[INFO] r17 hier-smile smoke complete ($seg_tag).
 • PASS = all R-hat ≤ ~1.05 with healthy ESS on log_σ_base / τ_strike / τ_team / z_* / log_φ.
 • If τ_strike or τ_team posterior sits near 0 → the data wants the global σ (no heterogeneity) — that is
   itself a clean result (hierarchy doesn't help here), not a bug.
 • If R-hat is bad / ESS tiny only on the τ_* + z_* block → funnel; tighten tau_*_prior (e.g. (0,0.15))
   or drop max_depth and re-smoke before scaling up samples.
""")



using Statistics

# rebuild runs from the already-trained objects (match by saved config name → no retrain)
runs = Tuple{String,Any,Any}[]
for (name, model) in specs
    idx = findfirst(r -> r.config.name == name, all_results)
    idx === nothing || push!(runs, (name, model, all_results[idx]))
end

strike_lines(nK) = [ (k - 1) + 0.5 for k in 1:nK ]
_chain1(res) = res.training_results.items[1][1]
_has(res, s) = Symbol(s) in keys(_chain1(res))
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    out
end
_count(res, stem) = count(i -> _has(res, "$stem[$i]"), 1:999)
_f(v) = (m=round(mean(v),digits=3), sd=round(std(v),digits=3),
         lo=round(quantile(v,0.05),digits=3), hi=round(quantile(v,0.95),digits=3))

function smile_sigma_report(name, model, res)
    println("\n", "="^72)
    println("[σ-hierarchy] $name   (per_strike=$(model.sigma_per_strike), per_team=$(model.sigma_per_team))")
    println("="^72)
    _has(res, "log_σ_base") || (println("  (log_σ_base not in chain)"); return)
    lsb = _pool(res,"log_σ_base"); ts = _pool(res,"τ_strike"); tt = _pool(res,"τ_team")
    g = _f(exp.(lsb))
    println("  GLOBAL σ_base = exp(log_σ_base): mean=$(g.m) 90%=[$(g.lo),$(g.hi)]  (prior centre 0.15)")
    if model.sigma_per_strike
        ft=_f(ts); pp=round(mean(ts.>0.05),digits=2)
        println("  τ_strike (per-LINE spread): mean=$(ft.m) 90%=[$(ft.lo),$(ft.hi)]  P(τ>0.05)=$pp " *
                (pp≥0.9 ? "← real heterogeneity" : pp≤0.3 ? "← ~flat (wants global σ)" : ""))
        nK=_count(res,"z_strike"); ln=strike_lines(nK)
        for k in 1:nK
            zk=_pool(res,"z_strike[$k]"); n=min(length(lsb),length(ts),length(zk))
            sk=_f(exp.(lsb[1:n].+ts[1:n].*zk[1:n]))
            println("      O/U $(ln[k]):  σ=$(sk.m) 90%=[$(sk.lo),$(sk.hi)]  z̄=$(round(mean(zk),digits=3))")
        end
    else
        println("  τ_strike: GATED OFF (prior only — ignore).")
    end
    if model.sigma_per_team
        ftt=_f(tt); pp=round(mean(tt.>0.05),digits=2)
        println("  τ_team (per-TEAM spread): mean=$(ftt.m) 90%=[$(ftt.lo),$(ftt.hi)]  P(τ>0.05)=$pp " *
                (pp≥0.9 ? "← real heterogeneity" : pp≤0.3 ? "← ~flat (wants global σ)" : ""))
        nT=_count(res,"z_team"); zbar=[mean(_pool(res,"z_team[$t]")) for t in 1:nT]
        mult=exp.(mean(tt).*zbar)
        println("    team σ-mult exp(τ_team·z̄) across $nT teams: min=$(round(minimum(mult),digits=3)) " *
                "med=$(round(median(mult),digits=3)) max=$(round(maximum(mult),digits=3))")
    else
        println("  τ_team: GATED OFF (prior only — ignore).")
    end
    nK=_count(res,"log_φ"); ln=strike_lines(nK)
    println("  smile shape φ(K)=exp(log_φ) (φ≡1 ⇒ Poisson; <1 thin, >1 fat):")
    for k in 1:nK
        φk=_f(exp.(_pool(res,"log_φ[$k]")))
        flag = φk.lo>1.0 ? " ↑ fat" : φk.hi<1.0 ? " ↓ thin" : ""
        println("      O/U $(ln[k]):  φ=$(φk.m) 90%=[$(φk.lo),$(φk.hi)]$flag")
    end
end

println("\n", "█"^72, "\n  SMILE-σ HIERARCHY READ (raw chains)\n", "█"^72)
for (name, model, res) in runs
    try; smile_sigma_report(name, model, res)
    catch e; @error "report failed: $name" exception=(e, catch_backtrace()); end
end

#=
========================================================================
[σ-hierarchy] hsmile_perstrike   (per_strike=true, per_team=false)
========================================================================
  GLOBAL σ_base = exp(log_σ_base): mean=0.051 90%=[0.047,0.057]  (prior centre 0.15)
  τ_strike (per-LINE spread): mean=0.065 90%=[0.004,0.182]  P(τ>0.05)=0.48 
      O/U 0.5:  σ=0.052 90%=[0.047,0.059]  z̄=0.233
      O/U 1.5:  σ=0.051 90%=[0.045,0.056]  z̄=-0.125
      O/U 2.5:  σ=0.051 90%=[0.046,0.057]  z̄=-0.068
      O/U 3.5:  σ=0.051 90%=[0.046,0.056]  z̄=-0.108
      O/U 4.5:  σ=0.05 90%=[0.045,0.056]  z̄=-0.208
  τ_team: GATED OFF (prior only — ignore).
  smile shape φ(K)=exp(log_φ) (φ≡1 ⇒ Poisson; <1 thin, >1 fat):
      O/U 0.5:  φ=0.932 90%=[0.826,1.045]
      O/U 1.5:  φ=0.994 90%=[0.883,1.115]
      O/U 2.5:  φ=1.011 90%=[0.897,1.133]
      O/U 3.5:  φ=1.034 90%=[0.92,1.16]
      O/U 4.5:  φ=1.052 90%=[0.933,1.178]

========================================================================
[σ-hierarchy] hsmile_perteam   (per_strike=false, per_team=true)
========================================================================
  GLOBAL σ_base = exp(log_σ_base): mean=0.052 90%=[0.046,0.059]  (prior centre 0.15)
  τ_strike: GATED OFF (prior only — ignore).
  τ_team (per-TEAM spread): mean=0.083 90%=[0.008,0.191]  P(τ>0.05)=0.66 
    team σ-mult exp(τ_team·z̄) across 11 teams: min=0.94 med=0.991 max=1.053
  smile shape φ(K)=exp(log_φ) (φ≡1 ⇒ Poisson; <1 thin, >1 fat):
      O/U 0.5:  φ=0.936 90%=[0.83,1.053]
      O/U 1.5:  φ=0.999 90%=[0.886,1.122]
      O/U 2.5:  φ=1.016 90%=[0.901,1.141]
      O/U 3.5:  φ=1.039 90%=[0.92,1.168]
      O/U 4.5:  φ=1.057 90%=[0.937,1.187]

========================================================================
[σ-hierarchy] hsmile_both   (per_strike=true, per_team=true)
========================================================================
  GLOBAL σ_base = exp(log_σ_base): mean=0.052 90%=[0.046,0.061]  (prior centre 0.15)
  τ_strike (per-LINE spread): mean=0.061 90%=[0.004,0.173]  P(τ>0.05)=0.46 
      O/U 0.5:  σ=0.053 90%=[0.046,0.062]  z̄=0.208
      O/U 1.5:  σ=0.052 90%=[0.045,0.061]  z̄=-0.106
      O/U 2.5:  σ=0.052 90%=[0.045,0.061]  z̄=-0.077
      O/U 3.5:  σ=0.052 90%=[0.044,0.061]  z̄=-0.139
      O/U 4.5:  σ=0.052 90%=[0.044,0.06]  z̄=-0.225
  τ_team (per-TEAM spread): mean=0.086 90%=[0.008,0.202]  P(τ>0.05)=0.68 
    team σ-mult exp(τ_team·z̄) across 11 teams: min=0.938 med=0.991 max=1.057
  smile shape φ(K)=exp(log_φ) (φ≡1 ⇒ Poisson; <1 thin, >1 fat):
      O/U 0.5:  φ=0.931 90%=[0.827,1.047]
      O/U 1.5:  φ=0.993 90%=[0.882,1.114]
      O/U 2.5:  φ=1.009 90%=[0.896,1.133]
      O/U 3.5:  φ=1.032 90%=[0.917,1.159]
      O/U 4.5:  φ=1.051 90%=[0.934,1.177]
=#

