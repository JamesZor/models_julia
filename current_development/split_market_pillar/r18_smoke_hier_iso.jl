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
