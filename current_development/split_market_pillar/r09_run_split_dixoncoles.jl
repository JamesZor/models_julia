#=
RUNNER for l05_split_market_dixoncoles.jl — SplitMarketDixonColesModel.

Single-split diagnostic. The Dixon-Coles sibling of r02/r08: same split (level/supremacy) market
pillar on the RATES, but goals ~ DC-Poisson with a FREE low-score correlation ρ (the τ correction
reshapes 0-0/1-0/0-1/1-1 → BTTS / correct-score). ρ is NOT market-anchored: it's the structural
edge. Defaults to Ireland top flight (79); DC ρ is ≈0 there empirically, so expect a weak τ
correction (this engine ≈ double-Poisson on low-ρ leagues).

Goals: (1) converges R-hat ≤ ~1.05 incl. σ_sup, σ_lev, dc.ρ_base; (2) ρ_base posterior visible.

Run after: git push -> git pull (server) -> RESTART REPL, then:
    include("current_development/split_market_pillar/r09_run_split_dixoncoles.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

include("current_development/split_market_pillar/l05_split_market_dixoncoles.jl")

# ==========================================
# 1. DATA — Betfair market pillar
# ==========================================
ds = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_market = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, odds,
    ds.lineups, ds.incidents, ds.betfair_odds
)

save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dc_cfg    = PreGame.GlobalDixonColesConfig()      # single global ρ (robust); swap to Hierarchical for per-team
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

# ==========================================
# 3. THE SPLIT-MARKET DIXON-COLES MODEL
# ==========================================
model = SplitMarketDixonColesModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),   # rates only; ρ free
    dixon_coles_config     = dc_cfg,
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 1.0,    # both marginals anchored
)

task = Experiments.create_experiment_task(
    ds_market, model, "split_dc_r1", save_dir;
    target_seasons  = ["2026"],
    history_seasons = 2,
    warmup_period   = 21,
    dynamics_col    = :match_week,
    samples         = 1000,
    warmup          = 500,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,
)

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)

# ==========================================
# 4. CONVERGENCE + DIXON-COLES ρ DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)

chain = results.training_results.items[1][1]
println("\n--- Dixon-Coles correlation ρ (after 0.3·tanh squash) ---")
if Symbol("dc.ρ_base") in keys(chain)
    rb = vec(Array(chain[Symbol("dc.ρ_base")]))
    ρ  = 0.3 .* tanh.(rb)
    println("  ρ_base: mean=$(round(mean(rb),digits=4))  ->  ρ: mean=$(round(mean(ρ),digits=4))  " *
            "q=[$(round(quantile(ρ,0.05),digits=4)), $(round(quantile(ρ,0.95),digits=4))]")
    println("  (ρ≈0 = DC correction negligible ⇒ ≈ double-Poisson; |ρ|>~0.05 = real low-score structure)")
else
    println("  (dc.ρ_base not in chain — check raw_symbol names in conv.df)")
end
for p in (:σ_sup, :σ_lev, :ν_xg)
    if p in keys(chain)
        v = vec(Array(chain[p]))
        println("  $p: mean=$(round(mean(v),digits=4))  std=$(round(std(v),digits=4))")
    end
end
println("\nNext: full-CV + r06_per_line_eval.jl (point include at l05) — look for edge on BTTS /")
println("correct-score (where ρ reshapes the low-score cells), not totals.")
