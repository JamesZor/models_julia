#=
SMOKE TEST — hierarchical-σ ISO market pillar (l09 HierIsoDoublePoissonModel). Goal: does it RUN + CONVERGE?

Iso = the isotropic market pillar (anchors home & away log-rates to market with ONE σ), NOT the smile.
There are no O/U strikes here, so the only natural σ grouping is per-TEAM (+ an optional global home/away
offset δ_side). Three cells (market_weight=1, market_on=true):
  - hiso_perteam : log σ = log_σ_base + τ_team·z_team[side]                 (δ per team)
  - hiso_perside : log σ = log_σ_base ± δ_side                              (home vs away anchor tightness)
  - hiso_both    : both terms

PRIMARY OUTPUT = convergence (R-hat / ESS), focused on the new hierarchy params:
  log_σ_base, τ_team, δ_side, z_team[*].
READ:
  • All R-hat ≤ ~1.05 + healthy ESS ⇒ samples cleanly (non-centred parameterisation working).
  • τ_team → ~0 ⇒ no learned per-team heterogeneity (σ collapses to the global scalar = src iso).
  • τ_team with bad R-hat / tiny ESS / divergences ⇒ funnel — tighten tau_team_prior or reparameterise.
A light GLMEdge/LogLoss is printed too, but the point here is CONVERGENCE, not edge.

Ireland (small, fast). STANDALONE — include l09 only.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r18_smoke_hier_iso.jl")
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

include("current_development/split_market_pillar/l09_hier_iso_poisson.jl")

# ==========================================
# 1. DATA
# ==========================================
SEGMENT = Data.Ireland()
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/hier_iso_smoke_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (= r17, smaller sample count for a smoke test)
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
warmup_period   = 21
ISO_MW         = 1.0        # smoke: full market pillar (the GRID r19 fixes this to the better iso weight)

_hiso(per_team, per_side) = HierIsoDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    market_weight          = ISO_MW,
    sigma_per_team         = per_team,
    sigma_per_side         = per_side,
)

specs = Tuple{String, Any}[
    ("hiso_perteam", _hiso(true,  false)),
    ("hiso_perside", _hiso(false, true)),
    ("hiso_both",    _hiso(true,  true)),
]
println("[INFO] Hier-iso smoke ($seg_tag): $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 3. RUN + CONVERGENCE per cell
# ==========================================
all_results = Any[]
runs        = Tuple{String,Any,Any}[]
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
        key = r"σ_base|log_σ_base|τ_team|δ_side|z_team"
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



#=
--- hiso_perteam : hierarchy-param convergence (R-hat / ESS) ---
(no hierarchy rows matched — inspect conv.df raw_symbol column)
31×5 DataFrame
 Row │ raw_symbol            mean        std        rhat      ess     
     │ Symbol                Float64     Float64    Float64   Float64 
─────┼────────────────────────────────────────────────────────────────
   1 │ ν_xg                   3.25451    0.323053   1.00509       NaN
   2 │ ha.γ_team_raw[1]       0.226913   0.0491323  1.00127       NaN
   3 │ ha.γ_team_raw[2]       0.182344   0.0625141  1.0055        NaN
   4 │ ha.γ_team_raw[3]       0.226785   0.0515142  1.00022       NaN
   5 │ ha.γ_team_raw[4]       0.204462   0.0521897  1.00248       NaN
   6 │ ha.γ_team_raw[5]       0.215058   0.06076    1.00123       NaN
   7 │ ha.γ_team_raw[6]       0.232881   0.0541358  1.00118       NaN
   8 │ ha.γ_team_raw[7]       0.221905   0.0495294  1.00129       NaN
   9 │ ha.γ_team_raw[8]       0.226283   0.0524409  0.999905      NaN
  10 │ ha.γ_team_raw[9]       0.20657    0.048553   1.00359       NaN
  11 │ ha.γ_team_raw[10]      0.224299   0.0514859  1.00141       NaN
  12 │ ha.γ_team_raw[11]      0.204862   0.0507689  1.0015        NaN
  13 │ ha.γ_base              0.214959   0.0370986  1.00323       NaN
  14 │ ha.σ_γ                 0.0409172  0.0314973  1.00108       NaN
  15 │ kap.κ_team_raw[1]      1.12382    0.0783044  1.00164       NaN
  16 │ kap.κ_team_raw[2]      0.977943   0.0808267  1.00028       NaN
  17 │ kap.κ_team_raw[3]      1.11575    0.075499   1.00367       NaN
  18 │ kap.κ_team_raw[4]      1.03678    0.0715904  0.999742      NaN
  19 │ kap.κ_team_raw[5]      1.07985    0.0930985  1.00125       NaN
  20 │ kap.κ_team_raw[6]      1.07887    0.0679501  1.00106       NaN
  21 │ kap.κ_team_raw[7]      1.14067    0.0797342  1.00117       NaN
  22 │ kap.κ_team_raw[8]      1.12161    0.0772647  1.0004        NaN
  23 │ kap.κ_team_raw[9]      1.04792    0.0717493  1.00249       NaN
  24 │ kap.κ_team_raw[10]     1.11787    0.0781514  1.00035       NaN
  25 │ kap.κ_team_raw[11]     1.04075    0.0708246  1.00198       NaN
  26 │ kap.κ_base             0.671404   0.0865773  0.999991      NaN
  27 │ kap.σ_κ                0.107608   0.0508521  1.00245       NaN
  28 │ p_dyn.w_G_att         -0.0843375  0.077715   1.00019       NaN
  29 │ p_dyn.w_G_def          0.159937   0.0688271  1.00185       NaN
  30 │ p_dyn.w_Outfield_att   0.0853897  0.0178152  1.00457       NaN
  31 │ p_dyn.w_Outfield_def  -0.120985   0.014048   1.00049       NaN
  >> max R-hat over ALL params: 1.0055 (OK)
=#


#=
--- hiso_perside : hierarchy-param convergence (R-hat / ESS) ---
(no hierarchy rows matched — inspect conv.df raw_symbol column)
31×5 DataFrame
 Row │ raw_symbol            mean        std        rhat      ess     
     │ Symbol                Float64     Float64    Float64   Float64 
─────┼────────────────────────────────────────────────────────────────
   1 │ ν_xg                   3.25202    0.334606   1.00294       NaN
   2 │ ha.γ_team_raw[1]       0.227173   0.0516707  1.00004       NaN
   3 │ ha.γ_team_raw[2]       0.182997   0.0630377  1.00469       NaN
   4 │ ha.γ_team_raw[3]       0.228088   0.0530255  1.00105       NaN
   5 │ ha.γ_team_raw[4]       0.205346   0.052632   1.00132       NaN
   6 │ ha.γ_team_raw[5]       0.214428   0.0642587  1.00277       NaN
   7 │ ha.γ_team_raw[6]       0.23318    0.0531083  1.00019       NaN
   8 │ ha.γ_team_raw[7]       0.221205   0.0519484  1.00019       NaN
   9 │ ha.γ_team_raw[8]       0.226555   0.0524686  1.0043        NaN
  10 │ ha.γ_team_raw[9]       0.207974   0.0526463  1.00091       NaN
  11 │ ha.γ_team_raw[10]      0.222418   0.0540796  1.00246       NaN
  12 │ ha.γ_team_raw[11]      0.205424   0.0534065  1.00077       NaN
  13 │ ha.γ_base              0.215687   0.038652   1.00831       NaN
  14 │ ha.σ_γ                 0.0412895  0.0311611  1.00967       NaN
  15 │ kap.κ_team_raw[1]      1.1289     0.077957   1.00091       NaN
  16 │ kap.κ_team_raw[2]      0.974335   0.0831642  1.00034       NaN
  17 │ kap.κ_team_raw[3]      1.1201     0.0746309  1.00081       NaN
  18 │ kap.κ_team_raw[4]      1.03463    0.0734613  1.00301       NaN
  19 │ kap.κ_team_raw[5]      1.08463    0.0971583  1.00521       NaN
  20 │ kap.κ_team_raw[6]      1.08124    0.0701101  1.00099       NaN
  21 │ kap.κ_team_raw[7]      1.14516    0.0799698  0.998959      NaN
  22 │ kap.κ_team_raw[8]      1.12672    0.075973   1.00141       NaN
  23 │ kap.κ_team_raw[9]      1.04909    0.0703804  1.0021        NaN
  24 │ kap.κ_team_raw[10]     1.12382    0.0791408  1.00125       NaN
  25 │ kap.κ_team_raw[11]     1.04445    0.0737229  1.00138       NaN
  26 │ kap.κ_base             0.676226   0.0882933  0.999656      NaN
  27 │ kap.σ_κ                0.110564   0.0498555  1.00025       NaN
  28 │ p_dyn.w_G_att         -0.0821059  0.0759813  0.999047      NaN
  29 │ p_dyn.w_G_def          0.150772   0.0680191  1.00489       NaN
  30 │ p_dyn.w_Outfield_att   0.084491   0.0180225  1.00006       NaN
  31 │ p_dyn.w_Outfield_def  -0.120717   0.0144775  1.00191       NaN
  >> max R-hat over ALL params: 1.0097 (OK)
=#



#=
--- hiso_both : hierarchy-param convergence (R-hat / ESS) ---
(no hierarchy rows matched — inspect conv.df raw_symbol column)
31×5 DataFrame
 Row │ raw_symbol            mean        std        rhat      ess     
     │ Symbol                Float64     Float64    Float64   Float64 
─────┼────────────────────────────────────────────────────────────────
   1 │ ν_xg                   3.25157    0.325232   1.00248       NaN
   2 │ ha.γ_team_raw[1]       0.227374   0.052801   1.00366       NaN
   3 │ ha.γ_team_raw[2]       0.18612    0.0610976  1.00082       NaN
   4 │ ha.γ_team_raw[3]       0.227021   0.0522697  1.00397       NaN
   5 │ ha.γ_team_raw[4]       0.206554   0.0517697  1.00475       NaN
   6 │ ha.γ_team_raw[5]       0.215152   0.0613128  1.00284       NaN
   7 │ ha.γ_team_raw[6]       0.232158   0.0509069  1.00359       NaN
   8 │ ha.γ_team_raw[7]       0.221837   0.0512778  1.00229       NaN
   9 │ ha.γ_team_raw[8]       0.227937   0.0513702  0.999261      NaN
  10 │ ha.γ_team_raw[9]       0.206689   0.0505899  1.00072       NaN
  11 │ ha.γ_team_raw[10]      0.222551   0.0493135  1.00126       NaN
  12 │ ha.γ_team_raw[11]      0.205781   0.0515985  1.00481       NaN
  13 │ ha.γ_base              0.215868   0.0383362  1.001         NaN
  14 │ ha.σ_γ                 0.0397854  0.0302892  1.00572       NaN
  15 │ kap.κ_team_raw[1]      1.12416    0.0785697  1.00323       NaN
  16 │ kap.κ_team_raw[2]      0.979728   0.0839151  1.00191       NaN
  17 │ kap.κ_team_raw[3]      1.1169     0.0731507  1.00119       NaN
  18 │ kap.κ_team_raw[4]      1.03492    0.0736242  1.0057        NaN
  19 │ kap.κ_team_raw[5]      1.08411    0.100222   1.00058       NaN
  20 │ kap.κ_team_raw[6]      1.07959    0.0717075  1.00358       NaN
  21 │ kap.κ_team_raw[7]      1.14382    0.0811301  0.999567      NaN
  22 │ kap.κ_team_raw[8]      1.12426    0.0773876  1.00351       NaN
  23 │ kap.κ_team_raw[9]      1.04859    0.0692477  1.00083       NaN
  24 │ kap.κ_team_raw[10]     1.12087    0.078082   1.00041       NaN
  25 │ kap.κ_team_raw[11]     1.04156    0.0701894  0.999539      NaN
  26 │ kap.κ_base             0.674904   0.0898995  1.00067       NaN
  27 │ kap.σ_κ                0.10883    0.0529991  1.00657       NaN
  28 │ p_dyn.w_G_att         -0.0796539  0.0752559  1.00126       NaN
  29 │ p_dyn.w_G_def          0.158243   0.0700946  1.00081       NaN
  30 │ p_dyn.w_Outfield_att   0.085388   0.0183059  1.00158       NaN
  31 │ p_dyn.w_Outfield_def  -0.12175    0.0145142  1.00153       NaN
  >> max R-hat over ALL params: 1.0066 (OK)
=#


# ==========================================
# 3b. ISO-σ HIERARCHY — pull the deltas straight from the raw chains
# ==========================================
# (check_convergence's curated conv.df drops these; read raw Chains, pool draws across all folds)
_chain1(res) = res.training_results.items[1][1]
_has(res, s) = Symbol(s) in keys(_chain1(res))
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

function iso_sigma_report(name, model, res)
    println("\n", "="^72)
    println("[σ-hierarchy] $name   (per_team=$(model.sigma_per_team), per_side=$(model.sigma_per_side))")
    println("="^72)
    if !_has(res, "log_σ_base")
        println("  (log_σ_base not in chain — check items access / symbol names)"); return
    end
    lsb = _pool(res, "log_σ_base"); tt = _pool(res, "τ_team"); ds_ = _pool(res, "δ_side")
    g = _f(exp.(lsb))
    println("  GLOBAL anchor  σ_base = exp(log_σ_base): mean=$(g.m)  90%=[$(g.lo), $(g.hi)]   (prior centre 0.1)")
    println("    └─ small σ = model hugs the market rates tightly (little edge budget); large = loose anchor.")

    if model.sigma_per_team
        ft = _f(tt); ppos = round(mean(tt .> 0.05), digits=2)
        println("  τ_team (per-TEAM σ spread): mean=$(ft.m) 90%=[$(ft.lo),$(ft.hi)]  P(τ>0.05)=$ppos " *
                (ppos ≥ 0.9 ? "← real heterogeneity" : ppos ≤ 0.3 ? "← ~flat (σ wants to be global)" : ""))
        nT = _count(res, "z_team")
        zbar = [mean(_pool(res, "z_team[$t]")) for t in 1:nT]
        mult = exp.(mean(tt) .* zbar)
        println("    team σ-multiplier exp(τ_team·z̄_team) across $nT teams: " *
                "min=$(round(minimum(mult),digits=3)) med=$(round(median(mult),digits=3)) max=$(round(maximum(mult),digits=3))")
    else
        println("  τ_team: GATED OFF (drew from prior only — ignore its value).")
    end

    if model.sigma_per_side
        fd = _f(ds_)
        σ_h = exp.(mean(lsb) + mean(ds_)); σ_a = exp.(mean(lsb) - mean(ds_))
        println("  δ_side (home/away anchor offset): mean=$(fd.m) 90%=[$(fd.lo),$(fd.hi)]  " *
                "⇒ σ_home≈$(round(σ_h,digits=3)) vs σ_away≈$(round(σ_a,digits=3))")
    else
        println("  δ_side: GATED OFF (drew from prior only — ignore its value).")
    end
end

println("\n", "█"^72, "\n  ISO-σ HIERARCHY READ (raw chains)\n", "█"^72)
for (name, model, res) in runs
    try
        iso_sigma_report(name, model, res)
    catch e
        @error "σ-hierarchy report failed: $name" exception=(e, catch_backtrace())
    end
end
println("""
\n[READ] What to take from §3b:
 • σ_base = the learned global anchor tightness (replaces src iso's sampled scalar σ_market).
 • τ_team > ~0.05 with mass away from 0 ⇒ the data WANTS different σ per team (some teams priced more
   reliably than others). Sitting at ~0 ⇒ collapses to the global σ = src iso (hierarchy bought nothing).
 • δ_side ≠ 0 ⇒ home & away rates are anchored to the market with different tightness.
 (Compare to the smile-σ result [[hierarchical-smile-sigma-null]]: that one collapsed to global on Ireland.)
""")




#=
========================================================================
[σ-hierarchy] hiso_perteam   (per_team=true, per_side=false)
========================================================================
  GLOBAL anchor  σ_base = exp(log_σ_base): mean=0.197  90%=[0.172, 0.226]   (prior centre 0.1)
    └─ small σ = model hugs the market rates tightly (little edge budget); large = loose anchor.
  τ_team (per-TEAM σ spread): mean=0.097 90%=[0.008,0.245]  P(τ>0.05)=0.66 
    team σ-multiplier exp(τ_team·z̄_team) across 11 teams: min=0.969 med=1.003 max=1.039
  δ_side: GATED OFF (drew from prior only — ignore its value).

========================================================================
[σ-hierarchy] hiso_perside   (per_team=false, per_side=true)
========================================================================
  GLOBAL anchor  σ_base = exp(log_σ_base): mean=0.199  90%=[0.176, 0.225]   (prior centre 0.1)
    └─ small σ = model hugs the market rates tightly (little edge budget); large = loose anchor.
  τ_team: GATED OFF (drew from prior only — ignore its value).
  δ_side (home/away anchor offset): mean=0.043 90%=[-0.072,0.151]  ⇒ σ_home≈0.207 vs σ_away≈0.19

========================================================================
[σ-hierarchy] hiso_both   (per_team=true, per_side=true)
========================================================================
  GLOBAL anchor  σ_base = exp(log_σ_base): mean=0.198  90%=[0.172, 0.225]   (prior centre 0.1)
    └─ small σ = model hugs the market rates tightly (little edge budget); large = loose anchor.
  τ_team (per-TEAM σ spread): mean=0.099 90%=[0.006,0.247]  P(τ>0.05)=0.69 
    team σ-multiplier exp(τ_team·z̄_team) across 11 teams: min=0.967 med=1.001 max=1.035
  δ_side (home/away anchor offset): mean=0.041 90%=[-0.068,0.152]  ⇒ σ_home≈0.205 vs σ_away≈0.189
=#


# ==========================================
# 4. LIGHT EDGE CHECK (secondary — confirms the prediction path works end-to-end)
# ==========================================
try
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)
    println("\n", "="^60, "\n📊 GLM Edge (Betfair) — hier-iso smoke\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)
    println("\n", "="^60, "\n📉 LogLoss (Betfair)\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)
catch e
    @error "Eval phase failed (chains are saved; this is the secondary check)" exception=(e, catch_backtrace())
end

println("""

[INFO] r18 hier-iso smoke complete ($seg_tag).
 • PASS = all R-hat ≤ ~1.05 with healthy ESS on log_σ_base / τ_team / δ_side / z_team.
 • If τ_team posterior sits near 0 → the data wants the global scalar σ (no per-team heterogeneity) — a
   clean result, not a bug (same shape as the smile-σ smoke).
 • If R-hat is bad / ESS tiny only on the τ_team + z_team block → funnel; tighten tau_team_prior or drop
   max_depth and re-smoke before scaling up samples.
 • PASS ⇒ run r19_grid_hier_iso.jl for the per-line GLMEdge/LogLoss grid over {flat,perteam,perside,both}.
""")
