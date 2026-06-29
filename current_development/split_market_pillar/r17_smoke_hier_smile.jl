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
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek
KMAX           = 4

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
for (name, model) in specs
    println("\n", "#"^72, "\n# RUN: $name\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = target_seasons,
            history_seasons = 2,
            warmup_period   = 0,
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
